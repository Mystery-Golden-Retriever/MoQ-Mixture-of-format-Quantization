"""Weight calibration engine -- per-layer greedy weight format selection.

Parallel to ``MoQCalibrator`` (which handles activations), this engine
selects the optimal quantization format for each layer's **weight**
tensor from a set of hardware-supported candidates.

The search uses the same ``BaseSearchStrategy`` interface as activation
calibration, but with ``target="weight"``.

Algorithm overview
------------------
1. Run calibration data through the model in **full precision** and
   record the reference output.
2. For each quantizable layer:
   a. Invoke the strategy to evaluate each weight format candidate.
   b. Record the winning weight format.
3. Return ``weight_format_map: {layer_name -> best_weight_quantizer}``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from moq.quantizers.base import BaseQuantizer
from moq.calibration.search_strategies import (
    BaseSearchStrategy,
    MoQEndToEndStrategy,
)

logger = logging.getLogger(__name__)


class WeightCalibrator:
    """Per-layer greedy weight format selection.

    Parameters
    ----------
    model : nn.Module
        The pretrained model (inference-only).
    candidates : list[BaseQuantizer]
        Candidate weight quantization formats to evaluate per layer.
    strategy : BaseSearchStrategy or None
        Format selection strategy (with ``target="weight"``).
        Defaults to ``MoQEndToEndStrategy(target="weight")``.
    device : torch.device or str or None
        Device for calibration.  Auto-detected if ``None``.

    Example
    -------
    >>> from moq.calibration import WeightCalibrator
    >>> candidates = WeightCalibrator.default_weight_candidates(bits=8)
    >>> wc = WeightCalibrator(model, candidates)
    >>> weight_format_map = wc.calibrate(calib_data, layer_names)
    """

    def __init__(
        self,
        model: nn.Module,
        candidates: list[BaseQuantizer],
        strategy: Optional[BaseSearchStrategy] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.candidates = candidates
        self.strategy = strategy or MoQEndToEndStrategy(target="weight")
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(
        self,
        calib_data: list,
        layer_names: list[str],
        show_progress: bool = True,
    ) -> dict[str, BaseQuantizer]:
        """Run greedy per-layer weight format selection.

        Parameters
        ----------
        calib_data : list
            List of input batches (tensors or dicts).
        layer_names : list[str]
            Fully-qualified names of nn.Linear layers to calibrate.
        show_progress : bool
            Show a tqdm progress bar.

        Returns
        -------
        dict[str, BaseQuantizer]
            Mapping {layer_name: best_weight_quantizer} for each layer.
        """
        self.model.eval()

        # Collect reference output (full precision)
        logger.info("Collecting reference output for weight calibration (full precision)...")
        reference_output = self._collect_output(calib_data)
        logger.info(
            "Reference output shape: %s, mean: %.4f",
            reference_output.shape,
            reference_output.mean().item(),
        )

        # Per-layer weight format selection
        weight_format_map: dict[str, BaseQuantizer] = {}
        scores: dict[str, float] = {}

        iterator = (
            tqdm(layer_names, desc="Weight Calibrating")
            if show_progress
            else layer_names
        )

        for layer_name in iterator:
            best_q, best_score = self.strategy.select_format(
                layer_name=layer_name,
                candidates=self.candidates,
                model=self.model,
                calib_data=calib_data,
                reference_output=reference_output,
            )
            weight_format_map[layer_name] = best_q
            scores[layer_name] = best_score

            logger.info(
                "Weight %-50s -> %s  (score=%.6f)",
                layer_name,
                best_q,
                best_score,
            )

        self._log_summary(weight_format_map, scores)
        return weight_format_map

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_output(self, calib_data: list) -> torch.Tensor:
        """Run calibration data through model, return concatenated output."""
        outputs = []
        self.model.eval()
        for batch in calib_data:
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = self.model(**batch)
            else:
                out = self.model(batch.to(self.device))

            if hasattr(out, "logits"):
                out = out.logits
            outputs.append(out.cpu().float())

        return torch.cat(outputs, dim=0)

    def _log_summary(
        self,
        format_map: dict[str, BaseQuantizer],
        scores: dict[str, float],
    ) -> None:
        """Log a summary of selected weight formats."""
        type_counts: dict[str, int] = {}
        for q in format_map.values():
            name = type(q).__name__
            type_counts[name] = type_counts.get(name, 0) + 1

        logger.info("=" * 60)
        logger.info("Weight Calibration Summary")
        logger.info("-" * 60)
        for fmt, count in sorted(type_counts.items()):
            logger.info("  %-30s : %d layers", fmt, count)
        logger.info(
            "  Mean score (weight MSE)     : %.6f",
            sum(scores.values()) / len(scores) if scores else 0.0,
        )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Convenience: default weight candidates per bit budget
    # ------------------------------------------------------------------

    @staticmethod
    def default_weight_candidates(bits: int) -> list[BaseQuantizer]:
        """Generate hardware-supported weight candidates for *bits*.

        Returns the standard set per the hardware format matrix:
          * 8-bit: INT8, INT8+ACIQ, FP8-E4M3, FP8-E5M2, MXFP8-E4M3, MXFP8-E5M2
          * 6-bit: INT6, INT6+ACIQ, FP6-E3M2, FP6-E2M3, MXFP6-E3M2, MXFP6-E2M3
          * 4-bit: INT4, INT4+ACIQ, MXFP4, NVFP4, NF4
        """
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.quantizers.fp_quantizer import FPQuantizer

        candidates: list[BaseQuantizer] = []

        if bits == 8:
            from moq.quantizers.mxfp_quantizer import MXFP8E4M3Quantizer, MXFP8E5M2Quantizer

            candidates = [
                INTQuantizer(bits=8),
                INTQuantizer(bits=8, use_aciq=True),
                FPQuantizer(bits=8, exp_bits=4),   # E4M3
                FPQuantizer(bits=8, exp_bits=5),   # E5M2
                MXFP8E4M3Quantizer(),
                MXFP8E5M2Quantizer(),
            ]
        elif bits == 6:
            from moq.quantizers.mxfp_quantizer import MXFP6E3M2Quantizer, MXFP6E2M3Quantizer

            candidates = [
                INTQuantizer(bits=6),
                INTQuantizer(bits=6, use_aciq=True),
                FPQuantizer(bits=6, exp_bits=3),   # E3M2
                FPQuantizer(bits=6, exp_bits=2),   # E2M3
                MXFP6E3M2Quantizer(),
                MXFP6E2M3Quantizer(),
            ]
        elif bits == 4:
            from moq.quantizers.mxfp_quantizer import MXFP4Quantizer
            from moq.quantizers.nvfp4_quantizer import NVFP4Quantizer
            from moq.quantizers.nf_quantizer import NF4Quantizer

            candidates = [
                INTQuantizer(bits=4),
                INTQuantizer(bits=4, use_aciq=True),
                MXFP4Quantizer(),
                NVFP4Quantizer(),
                NF4Quantizer(),
            ]
        else:
            # Fallback: INT + FP sweep (same as activation default)
            candidates.append(INTQuantizer(bits=bits))
            candidates.append(INTQuantizer(bits=bits, use_aciq=True))
            for exp_bits in range(1, bits - 1):
                candidates.append(FPQuantizer(bits=bits, exp_bits=exp_bits))

        return candidates
