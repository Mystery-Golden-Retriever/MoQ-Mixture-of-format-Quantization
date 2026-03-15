"""Optimum Hit Rate (OHR) metric from the MoQ paper.

OHR measures the fraction of layers where the predicted format is within
a tolerance margin of the globally optimal format's accuracy.

    OHR = |{l : acc(π(l)) ≥ acc(π*(l)) - τ}| / L

where:
  * π(l) = predicted format for layer l
  * π*(l) = optimal format for layer l (from exhaustive search)
  * τ = tolerance (default 1% accuracy margin)

The exhaustive search baseline is expensive — it evaluates every format
at every layer independently — but is needed for faithful comparison.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from moq.quantizers.base import BaseQuantizer
from moq.transform.hook_injector import HookQuantInjector

logger = logging.getLogger(__name__)


class OHRMetricEvaluator:
    """Compute the Optimum Hit Rate (OHR) metric.

    Parameters
    ----------
    tolerance : float
        Accuracy margin for considering a format "optimal" (default 0.01 = 1%).

    Example
    -------
    >>> evaluator = OHRMetricEvaluator(tolerance=0.01)
    >>> ohr = evaluator.compute_ohr(predicted_formats, optimal_formats, layer_accuracies)
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    # ------------------------------------------------------------------

    def compute_ohr(
        self,
        predicted_formats: dict[str, str],
        layer_accuracies: dict[str, dict[str, float]],
    ) -> float:
        """Compute OHR from pre-computed per-layer per-format accuracies.

        Parameters
        ----------
        predicted_formats : dict[str, str]
            ``{layer_name: format_repr}`` — the MoQ-selected formats.
        layer_accuracies : dict[str, dict[str, float]]
            ``{layer_name: {format_repr: accuracy}}``.

        Returns
        -------
        float
            OHR in [0, 1].
        """
        hits = 0
        total = 0

        for layer, pred_fmt in predicted_formats.items():
            if layer not in layer_accuracies:
                logger.warning("Layer %s has no accuracy data — skipping", layer)
                continue

            accs = layer_accuracies[layer]
            optimal_acc = max(accs.values())
            pred_acc = accs.get(pred_fmt, 0.0)

            if pred_acc >= optimal_acc - self.tolerance:
                hits += 1
            total += 1

        ohr = hits / total if total > 0 else 0.0
        logger.info("OHR = %d / %d = %.4f (tolerance=%.2f%%)", hits, total, ohr, self.tolerance * 100)
        return ohr

    # ------------------------------------------------------------------

    @torch.no_grad()
    def exhaustive_search(
        self,
        model: nn.Module,
        layer_names: list[str],
        candidates: list[BaseQuantizer],
        calib_data: list,
        eval_fn: callable,
        show_progress: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Run exhaustive per-layer per-format evaluation to build the
        accuracy table needed for OHR computation.

        Parameters
        ----------
        model : nn.Module
            The pretrained model.
        layer_names : list[str]
            Layers to evaluate.
        candidates : list[BaseQuantizer]
            All candidate formats.
        calib_data : list
            Calibration/validation batches.
        eval_fn : callable
            ``eval_fn(model, calib_data) -> float`` — returns accuracy or
            negative-MSE (higher is better).
        show_progress : bool
            Show a ``tqdm`` progress bar.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{layer_name: {format_repr: accuracy}}``.
        """
        model.eval()
        layer_accuracies: dict[str, dict[str, float]] = {}

        total_evals = len(layer_names) * len(candidates)
        pbar = tqdm(total=total_evals, desc="OHR Exhaustive Search") if show_progress else None

        for layer_name in layer_names:
            layer_accuracies[layer_name] = {}
            for quantizer in candidates:
                fmt_key = repr(quantizer)
                with HookQuantInjector(model, {layer_name: quantizer}):
                    score = eval_fn(model, calib_data)
                layer_accuracies[layer_name][fmt_key] = score

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        return layer_accuracies
