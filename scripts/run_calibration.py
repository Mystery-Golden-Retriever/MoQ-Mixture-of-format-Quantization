"""End-to-end calibration entrypoint.

Supports activation-only, weight-only, or combined calibration.

Usage:
    # Activation-only (default)
    python scripts/run_calibration.py --config configs/qwen_moq.yaml

    # Weight-only
    python scripts/run_calibration.py --config configs/qwen_moq.yaml --weight-only --weight-bits 8

    # Combined (activation + weight)
    python scripts/run_calibration.py --config configs/qwen_moq.yaml --enable-weight-quant

    # Weight-only with specific strategy
    python scripts/run_calibration.py --config configs/qwen_moq.yaml --weight-only --weight-bits 4 --weight-strategy intermediate_mse

    # Weight-only with specific static format
    python scripts/run_calibration.py --config configs/qwen_moq.yaml --weight-only --weight-bits 4 --weight-strategy static --weight-static-format nf4
"""

from __future__ import annotations

import argparse
import re
import json
import logging
import os
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("moq.scripts.calibration")


def main():
    parser = argparse.ArgumentParser(description="MoQ Calibration")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")

    # Activation calibration args
    parser.add_argument("--bits", type=int, default=None, help="Override activation bits")
    parser.add_argument("--strategy", type=str, default=None, help="Override activation strategy")

    # Weight calibration args
    parser.add_argument("--weight-only", action="store_true",
                        help="Run ONLY weight calibration (skip activation)")
    parser.add_argument("--enable-weight-quant", action="store_true",
                        help="Enable weight quantization (in addition to activation)")
    parser.add_argument("--weight-bits", type=int, default=None,
                        help="Override weight quantization bits")
    parser.add_argument("--weight-strategy", type=str, default=None,
                        help="Override weight strategy (moq_end_to_end|intermediate_mse|cosine|static)")
    parser.add_argument("--weight-static-format", type=str, default=None,
                        help="For static strategy: format name (e.g. int8, nf4, mxfp4, nvfp4)")
    parser.add_argument("--weight-candidates", type=str, nargs="*", default=None,
                        help="Explicit list of weight candidate format names")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load model
    model_name = args.model if args.model else cfg["model"]["name"]
    logger.info("Loading model: %s", model_name)
    
    is_vision = "vit" in model_name.lower() or "resnet" in model_name.lower()
    
    if is_vision:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        tokenizer = None
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32,
            device_map=device,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32,
            device_map=device,
        )
    model.eval()

    # Get layer names via adapter
    adapter_name = cfg["model"].get("adapter", "llama")
    if adapter_name == "llama":
        from moq.transform.adapters import LlamaQuantAdapter
        layer_names = LlamaQuantAdapter.get_layer_names(
            model,
            include_attn=cfg["quantization"].get("include_attn", True),
            include_mlp=cfg["quantization"].get("include_mlp", True),
        )
    elif adapter_name == "bert":
        from moq.transform.adapters import BERTQuantAdapter
        layer_names = BERTQuantAdapter.get_layer_names(model)
    elif adapter_name == "vit":
        from moq.transform.adapters import ViTQuantAdapter
        layer_names = ViTQuantAdapter.get_layer_names(model)
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")

    logger.info("Found %d quantizable layers", len(layer_names))

    # Load calibration data
    calib_cfg = cfg["calibration"]
    if is_vision:
        from moq.utils.data_utils import load_calib_data_vision
        calib_data = load_calib_data_vision(
            dataset_name=calib_cfg.get("dataset", "imagenet"),
            processor=processor,
            n_samples=calib_cfg.get("n_samples", 128),
            seed=calib_cfg.get("seed", 42),
        )
    else:
        from moq.utils.data_utils import load_calib_data_text
        calib_data = load_calib_data_text(
            dataset_name=calib_cfg.get("dataset", "wikitext2"),
            tokenizer=tokenizer,
            n_samples=calib_cfg.get("n_samples", 128),
            seq_len=calib_cfg.get("seq_len", 2048),
            seed=calib_cfg.get("seed", 42),
        )

    # ── Strategy builder helper ────────────────────────────────────
    from moq.calibration.search_strategies import (
        MoQEndToEndStrategy,
        IntermediateMSEStrategy,
        CosineDistanceStrategy,
        StaticFormatStrategy,
    )


    def _parse_static_format(fmt_name: str, default_bits: int = 8):
        """Parse a format name like 'int4', 'int4_aciq', 'mxfp4', 'nf4' into a quantizer."""
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.quantizers.fp_quantizer import FPQuantizer

        # Handle INT variants: int4, int8, int6, int4_aciq, int8_aciq
        m = re.match(r"int(\d+)(?:_(aciq))?$", fmt_name)
        if m:
            b = int(m.group(1))
            aciq = m.group(2) is not None
            return INTQuantizer(bits=b, use_aciq=aciq)

        # Try registry for everything else (mxfp4, nvfp4, nf4, fp6_e3m2, etc.)
        from moq.quantizers.registry import get_quantizer
        try:
            return get_quantizer(fmt_name)
        except KeyError:
            pass

        raise ValueError(f"Unknown static format: {fmt_name!r}. "
                         f"Use int<N>, int<N>_aciq, or a registered name.")

    def _build_strategy(strategy_name: str, target: str = "activation", bits: int = 8,
                        static_format: str | None = None):
        if strategy_name == "static":
            if static_format:
                fixed_q = _parse_static_format(static_format, bits)
            else:
                from moq.quantizers.int_quantizer import INTQuantizer
                fixed_q = INTQuantizer(bits=bits)
            return StaticFormatStrategy(fixed_quantizer=fixed_q)
        strategy_map = {
            "moq_end_to_end": MoQEndToEndStrategy,
            "intermediate_mse": IntermediateMSEStrategy,
            "cosine": CosineDistanceStrategy,
        }
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return strategy_map[strategy_name](target=target)

    # ── Output dir ─────────────────────────────────────────────────
    out_path = args.output_dir if args.output_dir else cfg.get("output", {}).get("output_dir", "./results")
    output_dir = Path(out_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Activation calibration ────────────────────────────
    act_format_map = {}
    if not args.weight_only:
        from moq.calibration.moq_calibrator import MoQCalibrator

        act_bits = args.bits if args.bits is not None else cfg["quantization"].get("activation_bits", 8)
        candidates = MoQCalibrator.default_candidates(act_bits)
        logger.info("=== Phase 1: Activation Calibration (%d-bit, %d candidates) ===",
                     act_bits, len(candidates))

        strategy_name = args.strategy if args.strategy else calib_cfg.get("strategy", "moq_end_to_end")
        strategy = _build_strategy(strategy_name, target="activation", bits=act_bits)

        calibrator = MoQCalibrator(model, candidates, strategy=strategy, device=device)
        act_format_map = calibrator.calibrate(calib_data, layer_names)

        # Save activation format map
        act_serialized = {name: q.get_config() for name, q in act_format_map.items()}
        act_path = output_dir / "format_map.json"
        with open(act_path, "w") as f:
            json.dump(act_serialized, f, indent=2)
        logger.info("Activation format map saved to %s", act_path)
    else:
        logger.info("=== Skipping activation calibration (--weight-only) ===")

    # ── Phase 2: Weight calibration ────────────────────────────────
    weight_cfg = cfg.get("weight_quantization", {})
    weight_enabled = (
        args.weight_only
        or args.enable_weight_quant
        or weight_cfg.get("enabled", False)
    )
    wt_format_map = {}

    if weight_enabled:
        from moq.calibration.weight_calibrator import WeightCalibrator

        wt_bits = args.weight_bits if args.weight_bits is not None else weight_cfg.get("bits", 8)
        wt_strategy_name = (
            args.weight_strategy
            if args.weight_strategy is not None
            else weight_cfg.get("strategy", "moq_end_to_end")
        )
        wt_static_format = args.weight_static_format
        wt_strategy = _build_strategy(wt_strategy_name, target="weight", bits=wt_bits,
                                       static_format=wt_static_format)

        # Build weight candidates
        if args.weight_candidates:
            from moq.quantizers.registry import get_quantizer
            wt_candidates = [get_quantizer(name) for name in args.weight_candidates]
        else:
            wt_candidates_cfg = weight_cfg.get("candidates", "hardware_default")
            if isinstance(wt_candidates_cfg, list):
                from moq.quantizers.registry import get_quantizer
                wt_candidates = [get_quantizer(name) for name in wt_candidates_cfg]
            else:
                wt_candidates = WeightCalibrator.default_weight_candidates(wt_bits)

        logger.info("=== Phase 2: Weight Calibration (%d-bit, %d candidates, strategy=%s) ===",
                     wt_bits, len(wt_candidates), wt_strategy_name)

        wt_calibrator = WeightCalibrator(model, wt_candidates, strategy=wt_strategy, device=device)
        wt_format_map = wt_calibrator.calibrate(calib_data, layer_names)

        # Save weight format map
        wt_serialized = {name: q.get_config() for name, q in wt_format_map.items()}
        wt_path = output_dir / "weight_format_map.json"
        with open(wt_path, "w") as f:
            json.dump(wt_serialized, f, indent=2)
        logger.info("Weight format map saved to %s", wt_path)
    else:
        logger.info("Weight quantization disabled")

    # ── Inline evaluation (optional) ──────────────────────────────
    if "evaluation" in cfg and (act_format_map or wt_format_map):
        eval_cfg = cfg["evaluation"]
        from moq.transform.hook_injector import HookQuantInjector
        from contextlib import ExitStack

        results = {}

        def _run_eval(eval_fn):
            with ExitStack() as stack:
                if act_format_map:
                    stack.enter_context(HookQuantInjector(model, act_format_map))
                if wt_format_map:
                    stack.enter_context(HookQuantInjector(model, wt_format_map, quantize_weight=True))
                return eval_fn()

        if is_vision:
            from moq.evaluation.image_evaluator import ImageClassificationEvaluator
            logger.info("Running Image Classification evaluation...")
            def _eval():
                evaluator = ImageClassificationEvaluator(model, processor, batch_size=eval_cfg.get("batch_size", 32))
                return evaluator.evaluate(eval_cfg.get("image_dataset", "imagenet"), max_samples=500)
            acc = _run_eval(_eval)
            logger.info("Accuracy: %.4f", acc)
            results["accuracy"] = acc
        else:
            from moq.evaluation.ppl_evaluator import PPLEvaluator
            logger.info("Running PPL evaluation...")
            def _eval():
                evaluator = PPLEvaluator(model, tokenizer, seq_len=eval_cfg.get("ppl_seq_len", 2048))
                return evaluator.evaluate(eval_cfg.get("ppl_dataset", "wikitext2"))
            ppl = _run_eval(_eval)
            logger.info("Perplexity: %.2f", ppl)
            results["ppl"] = ppl

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", results_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
