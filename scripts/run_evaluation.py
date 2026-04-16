"""Evaluation-only entrypoint.

Loads a pre-calibrated format map (from ``run_calibration.py``) and
evaluates the quantized model on perplexity and/or zero-shot tasks.

Usage:
    python scripts/run_evaluation.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --format-map results/llama_moq/format_map.json \\
        --ppl-dataset wikitext2 \\
        --zero-shot hellaswag piqa arc_easy \\
        --output-dir results/llama_moq_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("moq.scripts.evaluation")


def _rebuild_format_map(
    format_map_raw: dict[str, dict],
) -> dict:
    """Reconstruct quantizer instances from a serialised format_map.json."""
    from moq.quantizers.int_quantizer import INTQuantizer
    from moq.quantizers.fp_quantizer import FPQuantizer

    format_map = {}
    for layer_name, cfg in format_map_raw.items():
        cls_name = cfg["class"]
        bits = cfg["bits"]

        if cls_name == "INTQuantizer":
            format_map[layer_name] = INTQuantizer(
                bits=bits,
                symmetric=cfg.get("symmetric", True),
                channel_wise=cfg.get("channel_wise", False),
                use_aciq=cfg.get("use_aciq", False),
            )
        elif cls_name in ("FPQuantizer", "E4M3Quantizer", "E5M2Quantizer",
                          "FP4E2M1Quantizer", "FP4E3M0Quantizer"):
            format_map[layer_name] = FPQuantizer(
                bits=bits,
                exp_bits=cfg["exp_bits"],
                channel_wise=cfg.get("channel_wise", False),
                use_aciq=cfg.get("use_aciq", False),
            )
        else:
            raise ValueError(f"Unknown quantizer class: {cls_name}")
    return format_map


def main():
    parser = argparse.ArgumentParser(description="MoQ Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--format-map", type=str, required=True,
                        help="Path to format_map.json from calibration")
    parser.add_argument("--ppl-dataset", type=str, nargs="*",
                        default=["wikitext2"],
                        help="PPL datasets (wikitext2, c4, ptb)")
    parser.add_argument("--ppl-seq-len", type=int, default=2048)
    parser.add_argument("--zero-shot", type=str, nargs="*",
                        default=None,
                        help="Zero-shot tasks (hellaswag, piqa, arc_easy, ...)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="./results/eval")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compare-fp", action="store_true",
                        help="Also eval full-precision model as baseline")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32,
        device_map=device,
    )
    model.eval()

    # Load format map
    logger.info("Loading format map: %s", args.format_map)
    with open(args.format_map) as f:
        format_map_raw = json.load(f)
    format_map = _rebuild_format_map(format_map_raw)
    logger.info("Loaded %d layer format assignments", len(format_map))

    results: dict[str, dict] = {"quantized": {}, "full_precision": {}}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Perplexity evaluation ──────────────────────────────────────
    if args.ppl_dataset:
        from moq.evaluation.ppl_evaluator import PPLEvaluator
        from moq.transform.hook_injector import HookQuantInjector

        for dataset_name in args.ppl_dataset:
            logger.info("PPL evaluation on %s (quantized)…", dataset_name)
            with HookQuantInjector(model, format_map):
                evaluator = PPLEvaluator(model, tokenizer, seq_len=args.ppl_seq_len)
                ppl_q = evaluator.evaluate(dataset_name)
            results["quantized"][f"ppl_{dataset_name}"] = ppl_q
            logger.info("  Quantized PPL (%s): %.2f", dataset_name, ppl_q)

            if args.compare_fp:
                logger.info("PPL evaluation on %s (full precision)…", dataset_name)
                evaluator = PPLEvaluator(model, tokenizer, seq_len=args.ppl_seq_len)
                ppl_fp = evaluator.evaluate(dataset_name)
                results["full_precision"][f"ppl_{dataset_name}"] = ppl_fp
                logger.info("  Full-prec PPL (%s): %.2f", dataset_name, ppl_fp)

    # ── Zero-shot evaluation ───────────────────────────────────────
    if args.zero_shot:
        from moq.evaluation.zero_shot_runner import ZeroShotRunner
        from moq.transform.hook_injector import HookQuantInjector

        logger.info("Zero-shot evaluation: tasks=%s", args.zero_shot)
        with HookQuantInjector(model, format_map):
            runner = ZeroShotRunner(model, tokenizer, batch_size=args.batch_size)
            zshot_q = runner.run(args.zero_shot)
        results["quantized"]["zero_shot"] = zshot_q
        for task, acc in zshot_q.items():
            logger.info("  Quantized %s: %.4f", task, acc)

        if args.compare_fp:
            runner = ZeroShotRunner(model, tokenizer, batch_size=args.batch_size)
            zshot_fp = runner.run(args.zero_shot)
            results["full_precision"]["zero_shot"] = zshot_fp

    # ── Save results ───────────────────────────────────────────────
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("MoQ Evaluation Results")
    print("=" * 60)
    for mode in ("quantized", "full_precision"):
        if results[mode]:
            print(f"\n  [{mode}]")
            for metric, value in results[mode].items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"    {k:30s} : {v:.4f}")
                else:
                    print(f"    {metric:30s} : {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
