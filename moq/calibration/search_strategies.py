"""Pluggable format-selection strategies (Strategy Pattern).

Each strategy decides which quantization format to assign to a given
layer.  The strategies range from the MoQ paper's end-to-end evaluation
(gold standard) to simple baselines used for ablation comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from moq.quantizers.base import BaseQuantizer
from moq.observer.statistics_observer import LayerStats
from moq.transform.hook_injector import HookQuantInjector


class BaseSearchStrategy(ABC):
    """Abstract base for format-selection strategies."""

    @abstractmethod
    def select_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
        calib_data: list[torch.Tensor],
        reference_output: torch.Tensor,
        stats: Optional[LayerStats] = None,
    ) -> tuple[BaseQuantizer, float]:
        """Select the best format for *layer_name*.

        Returns
        -------
        (best_quantizer, best_score) : tuple
            The selected quantizer and its score (lower is better).
        """
        ...


class MoQEndToEndStrategy(BaseSearchStrategy):
    """MoQ's gold standard: select by **end-to-end output MSE**.

    For each candidate format, inject it at the target layer via a
    forward hook, run the calibration data through the full model, and
    compare the final output to the full-precision reference.

    This is the paper's core contribution: intermediate-layer MSE does
    **not** correlate with final accuracy, but output MSE does.
    """

    @torch.no_grad()
    def select_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
        calib_data: list[torch.Tensor],
        reference_output: torch.Tensor,
        stats: Optional[LayerStats] = None,
    ) -> tuple[BaseQuantizer, float]:
        model.eval()
        device = next(model.parameters()).device
        best_q: Optional[BaseQuantizer] = None
        best_score = float("inf")

        for quantizer in candidates:
            # Inject quantizer at ONE layer and measure output MSE
            with HookQuantInjector(model, {layer_name: quantizer}):
                outputs = []
                for batch in calib_data:
                    if isinstance(batch, dict):
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        out = model(**batch)
                    else:
                        out = model(batch.to(device))
                    # Handle HuggingFace-style outputs
                    if hasattr(out, "logits"):
                        out = out.logits
                    outputs.append(out.cpu().float())

                pred = torch.cat(outputs, dim=0)

            score = F.mse_loss(pred, reference_output).item()

            if score < best_score:
                best_score = score
                best_q = quantizer

        assert best_q is not None
        return best_q, best_score


class IntermediateMSEStrategy(BaseSearchStrategy):
    """Baseline: select by per-layer local MSE.

    Quantize the layer's output and compare to the full-precision output
    **at that layer** (not the final model output).  The MoQ paper
    demonstrates this is suboptimal, but we include it for ablation.
    """

    @torch.no_grad()
    def select_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
        calib_data: list[torch.Tensor],
        reference_output: torch.Tensor,
        stats: Optional[LayerStats] = None,
    ) -> tuple[BaseQuantizer, float]:
        model.eval()
        device = next(model.parameters()).device

        # Collect full-precision activations at this layer
        fp_activations: list[torch.Tensor] = []
        name_to_module = dict(model.named_modules())
        target_module = name_to_module[layer_name]

        def _collect_hook(_mod, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            fp_activations.append(out.detach().cpu().float())

        hook = target_module.register_forward_hook(_collect_hook)
        for batch in calib_data:
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                model(**batch)
            else:
                model(batch.to(device))
        hook.remove()

        fp_acts = torch.cat(fp_activations, dim=0)

        # Evaluate each candidate
        best_q: Optional[BaseQuantizer] = None
        best_score = float("inf")

        for quantizer in candidates:
            q_acts = quantizer(fp_acts)
            score = F.mse_loss(q_acts, fp_acts).item()
            if score < best_score:
                best_score = score
                best_q = quantizer

        assert best_q is not None
        return best_q, best_score


class CosineDistanceStrategy(BaseSearchStrategy):
    """Baseline: select by cosine similarity of quantised vs. original
    activations at the target layer.
    """

    @torch.no_grad()
    def select_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
        calib_data: list[torch.Tensor],
        reference_output: torch.Tensor,
        stats: Optional[LayerStats] = None,
    ) -> tuple[BaseQuantizer, float]:
        model.eval()
        device = next(model.parameters()).device

        # Collect FP activations
        fp_activations: list[torch.Tensor] = []
        name_to_module = dict(model.named_modules())
        target_module = name_to_module[layer_name]

        def _collect_hook(_mod, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            fp_activations.append(out.detach().cpu().float())

        hook = target_module.register_forward_hook(_collect_hook)
        for batch in calib_data:
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                model(**batch)
            else:
                model(batch.to(device))
        hook.remove()

        fp_acts = torch.cat(fp_activations, dim=0).flatten()

        best_q: Optional[BaseQuantizer] = None
        best_score = float("inf")  # We minimise (1 - cosine_sim)

        for quantizer in candidates:
            q_acts = quantizer(fp_acts)
            sim = F.cosine_similarity(
                fp_acts.unsqueeze(0), q_acts.unsqueeze(0)
            ).item()
            score = 1.0 - sim  # lower is better
            if score < best_score:
                best_score = score
                best_q = quantizer

        assert best_q is not None
        return best_q, best_score


class StaticFormatStrategy(BaseSearchStrategy):
    """Baseline: always assign the same fixed format to every layer.

    Used to measure the cost of not adapting per-layer.
    """

    def __init__(self, fixed_quantizer: BaseQuantizer) -> None:
        self.fixed_quantizer = fixed_quantizer

    def select_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
        calib_data: list[torch.Tensor],
        reference_output: torch.Tensor,
        stats: Optional[LayerStats] = None,
    ) -> tuple[BaseQuantizer, float]:
        # No evaluation needed — always return the fixed format
        return self.fixed_quantizer, 0.0
