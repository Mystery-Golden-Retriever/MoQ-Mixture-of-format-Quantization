"""MoQ Calibration engine — the core algorithm.

Orchestrates the per-layer greedy format search described in the MoQ
paper.  Delegates format selection to a pluggable ``BaseSearchStrategy``.

Algorithm overview
------------------
1. Run the calibration dataset through the model in **full precision**
   and record the reference output.
2. For each quantizable layer:
   a. Invoke the search strategy to evaluate all candidate formats.
   b. Record the winning format for that layer.
3. Return the ``format_map: {layer_name → best_quantizer}``.

The default strategy is ``MoQEndToEndStrategy`` which evaluates final
output MSE — the paper's core contribution.
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


class MoQCalibrator:
    """Core MoQ calibration engine.

    Parameters
    ----------
    model : nn.Module
        The pretrained model (inference-only).
    candidates : list[BaseQuantizer]
        Candidate quantization formats to evaluate per layer.
    strategy : BaseSearchStrategy or None
        The format selection strategy.  Defaults to
        ``MoQEndToEndStrategy`` (end-to-end output MSE).
    device : torch.device or str or None
        Device for calibration.  Auto-detected if ``None``.

    Example
    -------
    >>> from moq.quantizers import INTQuantizer, FPQuantizer
    >>> from moq.calibration import MoQCalibrator
    >>>
    >>> candidates = [
    ...     INTQuantizer(bits=8),
    ...     INTQuantizer(bits=8, use_aciq=True),
    ...     FPQuantizer(bits=8, exp_bits=4),  # E4M3
    ...     FPQuantizer(bits=8, exp_bits=5),  # E5M2
    ... ]
    >>> calibrator = MoQCalibrator(model, candidates)
    >>> format_map = calibrator.calibrate(calib_loader, layer_names)
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
        self.strategy = strategy or MoQEndToEndStrategy()
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
        """Run MoQ greedy per-layer format selection.

        Parameters
        ----------
        calib_data : list
            List of input batches (tensors or dicts with ``input_ids`` etc.).
        layer_names : list[str]
            Fully-qualified names of layers to calibrate.
        show_progress : bool
            Show a ``tqdm`` progress bar.

        Returns
        -------
        dict[str, BaseQuantizer]
            Mapping ``{layer_name: best_quantizer}`` for each layer.
        """
        self.model.eval()

        # Step 1: Collect reference output (full precision)
        logger.info("Collecting reference output (full precision)…")
        reference_output = self._collect_output(calib_data)
        logger.info(
            "Reference output shape: %s, mean: %.4f",
            reference_output.shape,
            reference_output.mean().item(),
        )

        # Step 2: Per-layer format selection
        format_map: dict[str, BaseQuantizer] = {}
        scores: dict[str, float] = {}

        iterator = tqdm(layer_names, desc="MoQ Calibrating") if show_progress else layer_names

        for layer_name in iterator:
            best_q, best_score = self.strategy.select_format(
                layer_name=layer_name,
                candidates=self.candidates,
                model=self.model,
                calib_data=calib_data,
                reference_output=reference_output,
            )
            format_map[layer_name] = best_q
            scores[layer_name] = best_score

            logger.info(
                "Layer %-50s → %s  (score=%.6f)",
                layer_name,
                best_q,
                best_score,
            )

        # Summary
        self._log_summary(format_map, scores)
        return format_map

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_output(self, calib_data: list) -> torch.Tensor:
        """Run calibration data through the model, return concatenated output."""
        outputs = []
        self.model.eval()
        for batch in calib_data:
            if isinstance(batch, dict):
                # Move dict values to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = self.model(**batch)
            else:
                out = self.model(batch.to(self.device))

            # Handle HuggingFace model outputs
            if hasattr(out, "logits"):
                out = out.logits
            outputs.append(out.cpu().float())

        return torch.cat(outputs, dim=0)

    def _log_summary(
        self,
        format_map: dict[str, BaseQuantizer],
        scores: dict[str, float],
    ) -> None:
        """Log a summary of selected formats."""
        # Count format types
        type_counts: dict[str, int] = {}
        for q in format_map.values():
            name = type(q).__name__
            type_counts[name] = type_counts.get(name, 0) + 1

        logger.info("=" * 60)
        logger.info("MoQ Calibration Summary")
        logger.info("-" * 60)
        for fmt, count in sorted(type_counts.items()):
            logger.info("  %-30s : %d layers", fmt, count)
        logger.info("  Mean score (output MSE)     : %.6f", 
                     sum(scores.values()) / len(scores) if scores else 0.0)
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Convenience: generate all standard candidates
    # ------------------------------------------------------------------

    @staticmethod
    def default_candidates(bits: int = 8) -> list[BaseQuantizer]:
        """Generate the standard MoQ candidate set for a given bit budget.

        Includes INT with/without ACIQ, and FP with all valid exponent
        allocations, with/without ACIQ.
        """
        from moq.quantizers.int_quantizer import INTQuantizer
        from moq.quantizers.fp_quantizer import FPQuantizer

        candidates: list[BaseQuantizer] = []

        # INT variants
        candidates.append(INTQuantizer(bits=bits))
        candidates.append(INTQuantizer(bits=bits, use_aciq=True))

        # FP variants: sweep exponent bits from 1 to bits-2
        for exp_bits in range(1, bits - 1):
            candidates.append(FPQuantizer(bits=bits, exp_bits=exp_bits))
            candidates.append(FPQuantizer(bits=bits, exp_bits=exp_bits, use_aciq=True))

        return candidates
