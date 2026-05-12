"""Evaluation-only entrypoint.

Loads a pre-calibrated format map (from ``run_calibration.py``) and
evaluates the quantized model on perplexity and/or zero-shot tasks.

Supports both activation format maps and weight format maps for
combined quantization evaluation.

Usage:
    python scripts/run_evaluation.py \
        --model meta-llama/Meta-Llama-3-8B \
        --format-map results/llama_moq/format_map.json \
        --weight-format-map results/llama_moq/weight_format_map.json \
        --ppl-dataset wikitext2 \
        --zero-shot hellaswag piqa arc_easy \
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
    """Reconstruct quantizer instances from a serialised format_map.json.

    Supports all quantizer types: INT, FP (including FP6), MXFP, NVFP4, NF4.
    """
    from moq.quantizers.int_quantizer import INTQuantizer
    from moq.quantizers.fp_quantizer import FPQuantizer
    from moq.quantizers.mxfp_quantizer import MXFPQuantizer
    from moq.quantizers.nvfp4_quantizer import NVFP4Quantizer
    from moq.quantizers.nf_quantizer import NF4Quantizer

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
                          "FP4E2M1Quantizer", "FP4E3M0Quantizer",
                          "FP6E3M2Quantizer", "FP6E2M3Quantizer"):
            format_map[layer_name] = FPQuantizer(
                bits=bits,
                exp_bits=cfg["exp_bits"],
                channel_wise=cfg.get("channel_wise", False),
                use_aciq=cfg.get("use_aciq", False),
            )
        elif cls_name in ("MXFPQuantizer", "MXFP8E4M3Quantizer",
                          "MXFP8E5M2Quantizer", "MXFP6E3M2Quantizer",
                          "MXFP6E2M3Quantizer", "MXFP4Quantizer"):
            format_map[layer_name] = MXFPQuantizer(
                element_bits=cfg.get("element_bits", bits),
                element_exp_bits=cfg.get("element_exp_bits", 4),
                group_size=cfg.get("group_size", 32),
            )
        elif cls_name == "NVFP4Quantizer":
            format_map[layer_name] = NVFP4Quantizer(
                block_size=cfg.get("block_size", 16),
            )
        elif cls_name == "NF4Quantizer":
            format_map[layer_name] = NF4Quantizer(
                group_size=cfg.get("group_size", 64),
                double_quant=cfg.get("double_quant", False),
            )
        else:
            raise ValueError(f"Unknown quantizer class: {cls_name}")
    return format_map


def main():
    parser = argparse.ArgumentParser(description="MoQ Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--format-map", type=str, default=None,
                        help="Path to activation format_map.json from calibration")
    parser.add_argument("--weight-format-map", type=str, default=None,
                        help="Path to weight_format_map.json from calibration")
    parser.add_argument("--ppl-dataset", type=str, nargs="*",
                        default=None,
                        help="PPL datasets (wikitext2, c4, ptb)")
    parser.add_argument("--ppl-seq-len", type=int, default=2048)
    parser.add_argument("--image-dataset", type=str, nargs="*",
                        default=None,
                        help="Image classification datasets (imagenet, cifar10)")
    parser.add_argument("--zero-shot", type=str, nargs="*",
                        default=None,
                        help="Zero-shot tasks (hellaswag, piqa, arc_easy, ...)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="./results/eval")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compare-fp", action="store_true",
                        help="Also eval full-precision model as baseline")
    args = parser.parse_args()

    if args.format_map is None and args.weight_format_map is None:
        parser.error("At least one of --format-map or --weight-format-map is required")

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

    logger.info("Loading model: %s", args.model)
    is_vision = "vit" in args.model.lower() or "resnet" in args.model.lower()
    
    if is_vision:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(args.model)
        tokenizer = None
        model = AutoModelForImageClassification.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32,
            device_map=device,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        processor = None
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32,
            device_map=device,
        )
    model.eval()

    # Load activation format map (if provided)
    act_format_map = None
    if args.format_map:
        logger.info("Loading activation format map: %s", args.format_map)
        with open(args.format_map) as f:
            act_format_map = _rebuild_format_map(json.load(f))
        logger.info("Loaded %d activation layer format assignments", len(act_format_map))

    # Load weight format map (if provided)
    wt_format_map = None
    if args.weight_format_map:
        logger.info("Loading weight format map: %s", args.weight_format_map)
        with open(args.weight_format_map) as f:
            wt_format_map = _rebuild_format_map(json.load(f))
        logger.info("Loaded %d weight layer format assignments", len(wt_format_map))

    results: dict[str, dict] = {"quantized": {}, "full_precision": {}}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the quantized evaluation context
    from moq.transform.hook_injector import HookQuantInjector
    from contextlib import ExitStack

    def _run_with_quant(eval_fn):
        """Run eval_fn with both activation and weight hooks active."""
        with ExitStack() as stack:
            if act_format_map:
                stack.enter_context(HookQuantInjector(model, act_format_map))
            if wt_format_map:
                stack.enter_context(HookQuantInjector(model, wt_format_map, quantize_weight=True))
            return eval_fn()

    # -- Image Classification evaluation --
    if args.image_dataset and is_vision:
        from moq.evaluation.image_evaluator import ImageClassificationEvaluator

        for dataset_name in args.image_dataset:
            logger.info("Image Classification on %s (quantized)...", dataset_name)
            def _eval_img():
                evaluator = ImageClassificationEvaluator(model, processor, batch_size=args.batch_size)
                return evaluator.evaluate(dataset_name)
            acc_q = _run_with_quant(_eval_img)
            results["quantized"][f"acc_{dataset_name}"] = acc_q
            logger.info("  Quantized Accuracy (%s): %.4f", dataset_name, acc_q)

            if args.compare_fp:
                logger.info("Image Classification on %s (full precision)...", dataset_name)
                evaluator = ImageClassificationEvaluator(model, processor, batch_size=args.batch_size)
                acc_fp = evaluator.evaluate(dataset_name)
                results["full_precision"][f"acc_{dataset_name}"] = acc_fp
                logger.info("  Full-prec Accuracy (%s): %.4f", dataset_name, acc_fp)

    # -- Perplexity evaluation --
    if args.ppl_dataset and not is_vision:
        from moq.evaluation.ppl_evaluator import PPLEvaluator

        for dataset_name in args.ppl_dataset:
            logger.info("PPL evaluation on %s (quantized)...", dataset_name)
            def _eval_ppl(ds=dataset_name):
                evaluator = PPLEvaluator(model, tokenizer, seq_len=args.ppl_seq_len)
                return evaluator.evaluate(ds)
            ppl_q = _run_with_quant(_eval_ppl)
            results["quantized"][f"ppl_{dataset_name}"] = ppl_q
            logger.info("  Quantized PPL (%s): %.2f", dataset_name, ppl_q)

            if args.compare_fp:
                logger.info("PPL evaluation on %s (full precision)...", dataset_name)
                evaluator = PPLEvaluator(model, tokenizer, seq_len=args.ppl_seq_len)
                ppl_fp = evaluator.evaluate(dataset_name)
                results["full_precision"][f"ppl_{dataset_name}"] = ppl_fp
                logger.info("  Full-prec PPL (%s): %.2f", dataset_name, ppl_fp)

    # -- Zero-shot evaluation --
    if args.zero_shot and not is_vision:
        from moq.evaluation.zero_shot_runner import ZeroShotRunner

        logger.info("Zero-shot evaluation: tasks=%s", args.zero_shot)
        def _eval_zshot():
            runner = ZeroShotRunner(model, tokenizer, batch_size=args.batch_size)
            return runner.run(args.zero_shot)
        zshot_q = _run_with_quant(_eval_zshot)
        results["quantized"]["zero_shot"] = zshot_q
        for task, acc in zshot_q.items():
            logger.info("  Quantized %s: %.4f", task, acc)

        if args.compare_fp:
            runner = ZeroShotRunner(model, tokenizer, batch_size=args.batch_size)
            zshot_fp = runner.run(args.zero_shot)
            results["full_precision"]["zero_shot"] = zshot_fp

    # -- Save results --
    results_path = output_dir / "eval_results.json"
    
    # Merge with existing results if file exists
    if results_path.exists():
        logger.info("Existing results found at %s. Merging...", results_path)
        with open(results_path, "r") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = {"quantized": {}, "full_precision": {}}
        
        # Merge quantized and full_precision dicts
        for mode in ["quantized", "full_precision"]:
            if mode not in existing_results:
                existing_results[mode] = {}
            if mode in results:
                for k, v in results[mode].items():
                    if isinstance(v, dict) and isinstance(existing_results[mode].get(k), dict):
                        # Merge nested dicts (like zero_shot)
                        existing_results[mode][k].update(v)
                    else:
                        existing_results[mode][k] = v
        results = existing_results
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("MoQ Evaluation Results")
    print("=" * 60)
    quant_mode = []
    if act_format_map:
        quant_mode.append("activation")
    if wt_format_map:
        quant_mode.append("weight")
    print(f"  Quantization mode: {' + '.join(quant_mode)}")
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
