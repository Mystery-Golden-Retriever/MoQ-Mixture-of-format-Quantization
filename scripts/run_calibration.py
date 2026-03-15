"""End-to-end calibration entrypoint.

Usage:
    python -m scripts.run_calibration --config configs/llama_moq.yaml
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["name"]
    logger.info("Loading model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
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
    from moq.utils.data_utils import load_calib_data_text

    calib_cfg = cfg["calibration"]
    calib_data = load_calib_data_text(
        dataset_name=calib_cfg.get("dataset", "wikitext2"),
        tokenizer=tokenizer,
        n_samples=calib_cfg.get("n_samples", 128),
        seq_len=calib_cfg.get("seq_len", 2048),
        seed=calib_cfg.get("seed", 42),
    )

    # Build candidates
    from moq.calibration.moq_calibrator import MoQCalibrator

    bits = cfg["quantization"].get("activation_bits", 8)
    candidates = MoQCalibrator.default_candidates(bits)
    logger.info("Using %d candidate formats for %d-bit", len(candidates), bits)

    # Build strategy
    from moq.calibration.search_strategies import (
        MoQEndToEndStrategy,
        IntermediateMSEStrategy,
        CosineDistanceStrategy,
    )

    strategy_name = calib_cfg.get("strategy", "moq_end_to_end")
    strategy_map = {
        "moq_end_to_end": MoQEndToEndStrategy,
        "intermediate_mse": IntermediateMSEStrategy,
        "cosine": CosineDistanceStrategy,
    }
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    strategy = strategy_map[strategy_name]()

    # Run calibration
    calibrator = MoQCalibrator(model, candidates, strategy=strategy, device=device)
    format_map = calibrator.calibrate(calib_data, layer_names)

    # Save results
    output_dir = Path(cfg.get("output", {}).get("output_dir", "./results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Serialize format map
    serialized = {
        name: q.get_config() for name, q in format_map.items()
    }
    output_path = output_dir / "format_map.json"
    with open(output_path, "w") as f:
        json.dump(serialized, f, indent=2)
    logger.info("Format map saved to %s", output_path)

    # Optionally run evaluation
    if "evaluation" in cfg:
        eval_cfg = cfg["evaluation"]

        # PPL
        from moq.evaluation.ppl_evaluator import PPLEvaluator
        from moq.transform.hook_injector import HookQuantInjector

        logger.info("Running PPL evaluation with MoQ format map…")
        with HookQuantInjector(model, format_map):
            evaluator = PPLEvaluator(
                model, tokenizer,
                seq_len=eval_cfg.get("ppl_seq_len", 2048),
            )
            ppl = evaluator.evaluate(eval_cfg.get("ppl_dataset", "wikitext2"))
        logger.info("Perplexity: %.2f", ppl)

        results = {"ppl": ppl}
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
