"""Reproduce MoQ paper results: OHR metric comparison.

Reproduces the core experiment from the MoQ paper:
  1. Load a model (ViT or Llama).
  2. Run exhaustive search (all formats × all layers) to compute the
     ground-truth optimal format per layer.
  3. Run MoQ calibration-based selection with varying calibration sizes.
  4. Run baseline strategies (static INT, static FP, intermediate MSE,
     cosine distance).
  5. Compute and compare OHR for each strategy.

Usage:
    python scripts/reproduce_paper.py \\
        --model google/vit-base-patch16-224 \\
        --bits 4 8 \\
        --calib-sizes 64 128 320 640 \\
        --output-dir results/paper_reproduction
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("moq.scripts.reproduce")


def _detect_adapter(model_name: str):
    """Auto-detect the correct adapter from the model name."""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        from moq.transform.adapters import LlamaQuantAdapter
        return LlamaQuantAdapter
    elif "bert" in model_name_lower or "roberta" in model_name_lower:
        from moq.transform.adapters import BERTQuantAdapter
        return BERTQuantAdapter
    elif "vit" in model_name_lower:
        from moq.transform.adapters import ViTQuantAdapter
        return ViTQuantAdapter
    else:
        raise ValueError(
            f"Cannot auto-detect adapter for {model_name!r}. "
            f"Use --adapter to specify manually."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce MoQ paper OHR results"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--adapter", type=str, default=None,
                        choices=["llama", "bert", "vit"],
                        help="Model adapter (auto-detected if not set)")
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8],
                        help="Bit budgets to test")
    parser.add_argument("--calib-sizes", type=int, nargs="+",
                        default=[64, 128, 320, 640],
                        help="Calibration set sizes to sweep")
    parser.add_argument("--calib-dataset", type=str, default=None,
                        help="Calibration dataset (default: cifar10 for vision, wikitext2 for NLP)")
    parser.add_argument("--ohr-tolerance", type=float, default=0.01,
                        help="OHR tolerance (default 1%%)")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Limit number of layers (for quick testing)")
    parser.add_argument("--output-dir", type=str,
                        default="./results/paper_reproduction")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load model
    logger.info("Loading model: %s", args.model)
    is_vision = "vit" in args.model.lower() or "resnet" in args.model.lower()

    if is_vision:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        model = AutoModelForImageClassification.from_pretrained(args.model).to(device)
        processor = AutoImageProcessor.from_pretrained(args.model)
        tokenizer = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).to(device)
        processor = None

    model.eval()

    # Adapter
    if args.adapter:
        adapter_map = {"llama": "LlamaQuantAdapter", "bert": "BERTQuantAdapter", "vit": "ViTQuantAdapter"}
        exec(f"from moq.transform.adapters import {adapter_map[args.adapter]}")
        adapter_cls = locals()[adapter_map[args.adapter]]
    else:
        adapter_cls = _detect_adapter(args.model)

    layer_names = adapter_cls.get_layer_names(model)
    if args.max_layers:
        layer_names = layer_names[: args.max_layers]
    logger.info("Found %d quantizable layers (using %d)",
                len(adapter_cls.get_layer_names(model)), len(layer_names))

    # Prepare output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {
        "model": args.model,
        "layers": len(layer_names),
        "ohr_tolerance": args.ohr_tolerance,
        "experiments": [],
    }

    # Import components
    from moq.calibration.moq_calibrator import MoQCalibrator
    from moq.calibration.search_strategies import (
        MoQEndToEndStrategy,
        IntermediateMSEStrategy,
        CosineDistanceStrategy,
        StaticFormatStrategy,
    )
    from moq.evaluation.ohr_evaluator import OHRMetricEvaluator
    from moq.quantizers.int_quantizer import INTQuantizer
    from moq.quantizers.fp_quantizer import FPQuantizer

    ohr_eval = OHRMetricEvaluator(tolerance=args.ohr_tolerance)

    for bits in args.bits:
        logger.info("=" * 60)
        logger.info("Bit budget: %d", bits)
        logger.info("=" * 60)

        candidates = MoQCalibrator.default_candidates(bits)
        logger.info("Candidate formats: %d", len(candidates))

        # ── Step 1: Load calibration data ──────────────────────────
        max_calib = max(args.calib_sizes)
        if is_vision:
            from moq.utils.data_utils import load_calib_data_vision
            calib_dataset = args.calib_dataset or "cifar10"
            calib_data_full = load_calib_data_vision(
                dataset_name=calib_dataset,
                processor=processor,
                n_samples=max_calib,
                seed=args.seed,
            )
        else:
            from moq.utils.data_utils import load_calib_data_text
            calib_dataset = args.calib_dataset or "wikitext2"
            calib_data_full = load_calib_data_text(
                dataset_name=calib_dataset,
                tokenizer=tokenizer,
                n_samples=max_calib,
                seed=args.seed,
            )

        logger.info("Loaded %d calibration samples from %s",
                     len(calib_data_full), calib_dataset)

        # ── Step 2: Exhaustive ground-truth search ─────────────────
        logger.info("Running exhaustive search (ground truth)…")

        def eval_fn(m, data):
            """Negative output MSE (higher is better)."""
            quant_parts = []
            with torch.no_grad():
                for batch in data:
                    if isinstance(batch, dict):
                        batch_dev = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        out = m(**batch_dev)
                    else:
                        out = m(batch.to(device))
                    if hasattr(out, "logits"):
                        out = out.logits
                    quant_parts.append(out.cpu().float())
            full_out = torch.cat(quant_parts)
            return -full_out.pow(2).mean().item()

        gt_accs = ohr_eval.exhaustive_search(
            model, layer_names, candidates, calib_data_full,
            eval_fn, show_progress=True,
        )

        # ── Step 3: Run each strategy ─────────────────────────────
        strategies = {
            "Static INT": StaticFormatStrategy(INTQuantizer(bits=bits)),
            "Static FP (max exp)": StaticFormatStrategy(
                FPQuantizer(bits=bits, exp_bits=max(1, bits - 2))
            ),
            "Intermediate MSE": IntermediateMSEStrategy(),
            "Cosine Distance": CosineDistanceStrategy(),
        }

        experiment = {"bits": bits, "strategies": {}}

        # Baselines (don't depend on calib size)
        for strat_name, strategy in strategies.items():
            calibrator = MoQCalibrator(model, candidates, strategy=strategy, device=device)
            fmt_map = calibrator.calibrate(calib_data_full, layer_names, show_progress=False)
            predicted = {name: repr(q) for name, q in fmt_map.items()}
            ohr = ohr_eval.compute_ohr(predicted, gt_accs)
            experiment["strategies"][strat_name] = {"ohr": ohr, "calib_size": max_calib}
            logger.info("  %-25s OHR = %.4f", strat_name, ohr)

        # MoQ end-to-end with varying calibration sizes
        for calib_size in args.calib_sizes:
            calib_subset = calib_data_full[:calib_size]
            strategy = MoQEndToEndStrategy()
            calibrator = MoQCalibrator(model, candidates, strategy=strategy, device=device)
            fmt_map = calibrator.calibrate(calib_subset, layer_names, show_progress=False)
            predicted = {name: repr(q) for name, q in fmt_map.items()}
            ohr = ohr_eval.compute_ohr(predicted, gt_accs)
            key = f"MoQ (n={calib_size})"
            experiment["strategies"][key] = {"ohr": ohr, "calib_size": calib_size}
            logger.info("  %-25s OHR = %.4f", key, ohr)

        all_results["experiments"].append(experiment)

    # ── Save & print summary ──────────────────────────────────────
    results_path = output_dir / "paper_reproduction_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    print("\n" + "=" * 70)
    print("MoQ Paper Reproduction — OHR Summary")
    print("=" * 70)
    for exp in all_results["experiments"]:
        print(f"\n  Bit Budget: {exp['bits']}")
        print(f"  {'Strategy':<30s} {'Calib Size':<15s} {'OHR':>8s}")
        print(f"  {'-'*30} {'-'*15} {'-'*8}")
        for strat, vals in exp["strategies"].items():
            print(f"  {strat:<30s} {vals['calib_size']:<15d} {vals['ohr']:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
