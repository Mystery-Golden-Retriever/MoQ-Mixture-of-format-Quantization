"""Pluggable format-selection strategies (Strategy Pattern).

Each strategy decides which quantization format to assign to a given
layer.  The strategies range from the MoQ paper's end-to-end evaluation
(gold standard) to simple baselines used for ablation comparison.

All strategies support a ``target`` parameter:
  * ``"activation"`` (default) -- quantize the layer's output activations.
  * ``"weight"`` -- quantize the layer's ``nn.Linear`` weight tensor.
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
    forward hook (activation) or temporary weight replacement (weight),
    run the calibration data through the full model, and compare the
    final output to the full-precision reference.

    Parameters
    ----------
    target : str
        ``"activation"`` (quantize layer output) or ``"weight"``
        (quantize layer weight tensor).
    """

    def __init__(self, target: str = "activation") -> None:
        if target not in ("activation", "weight"):
            raise ValueError(f"target must be 'activation' or 'weight', got {target!r}")
        self.target = target

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
            if self.target == "weight":
                # Temporarily replace weight with quantized version
                outputs = self._forward_with_quantized_weight(
                    model, layer_name, quantizer, calib_data, device
                )
            else:
                # Inject quantizer at ONE layer and measure output MSE
                with HookQuantInjector(model, {layer_name: quantizer}):
                    outputs = self._forward_batches(model, calib_data, device)

            pred = torch.cat(outputs, dim=0)
            score = F.mse_loss(pred, reference_output).item()

            if score < best_score:
                best_score = score
                best_q = quantizer

        assert best_q is not None
        return best_q, best_score

    @staticmethod
    def _forward_batches(model, calib_data, device):
        """Run batches through the model and collect outputs."""
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
            if hasattr(out, "logits"):
                out = out.logits
            outputs.append(out.cpu().float())
        return outputs

    @staticmethod
    def _forward_with_quantized_weight(model, layer_name, quantizer, calib_data, device):
        """Temporarily quantize a layer's weight and run forward."""
        name_to_module = dict(model.named_modules())
        target_module = name_to_module[layer_name]

        # Save original weight
        orig_weight = target_module.weight.data.clone()

        try:
            # Apply weight quantization
            target_module.weight.data = quantizer(orig_weight)

            outputs = MoQEndToEndStrategy._forward_batches(model, calib_data, device)
        finally:
            # Always restore original weight
            target_module.weight.data = orig_weight

        return outputs


class IntermediateMSEStrategy(BaseSearchStrategy):
    """Baseline: select by per-layer local MSE.

    Quantize the layer's output (activation mode) or weight (weight mode)
    and compare to the full-precision version.

    In weight mode, this simply computes ``MSE(Q(W), W)`` -- no forward
    pass needed.

    Parameters
    ----------
    target : str
        ``"activation"`` or ``"weight"``.
    """

    def __init__(self, target: str = "activation") -> None:
        if target not in ("activation", "weight"):
            raise ValueError(f"target must be 'activation' or 'weight', got {target!r}")
        self.target = target

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

        if self.target == "weight":
            return self._select_weight_format(layer_name, candidates, model)
        else:
            return self._select_activation_format(
                layer_name, candidates, model, calib_data, device
            )

    def _select_weight_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
    ) -> tuple[BaseQuantizer, float]:
        """Direct weight MSE: compare Q(W) vs W."""
        name_to_module = dict(model.named_modules())
        target_module = name_to_module[layer_name]
        weight = target_module.weight.data.float()

        best_q: Optional[BaseQuantizer] = None
        best_score = float("inf")

        for quantizer in candidates:
            w_q = quantizer(weight)
            score = F.mse_loss(w_q, weight).item()
            if score < best_score:
                best_score = score
                best_q = quantizer

        assert best_q is not None
        return best_q, best_score

    def _select_activation_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
        calib_data: list,
        device: torch.device,
    ) -> tuple[BaseQuantizer, float]:
        """Original activation-based intermediate MSE."""
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
    activations/weights at the target layer.
    """

    def __init__(self, target: str = "activation") -> None:
        if target not in ("activation", "weight"):
            raise ValueError(f"target must be 'activation' or 'weight', got {target!r}")
        self.target = target

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

        if self.target == "weight":
            return self._select_weight_format(layer_name, candidates, model)
        else:
            return self._select_activation_format(
                layer_name, candidates, model, calib_data, device
            )

    def _select_weight_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
    ) -> tuple[BaseQuantizer, float]:
        """Cosine distance on weight tensor."""
        name_to_module = dict(model.named_modules())
        target_module = name_to_module[layer_name]
        weight = target_module.weight.data.float().flatten()

        best_q: Optional[BaseQuantizer] = None
        best_score = float("inf")

        for quantizer in candidates:
            w_q = quantizer(weight.reshape(target_module.weight.shape)).flatten()
            sim = F.cosine_similarity(
                weight.unsqueeze(0), w_q.unsqueeze(0)
            ).item()
            score = 1.0 - sim
            if score < best_score:
                best_score = score
                best_q = quantizer

        assert best_q is not None
        return best_q, best_score

    def _select_activation_format(
        self,
        layer_name: str,
        candidates: list[BaseQuantizer],
        model: nn.Module,
        calib_data: list,
        device: torch.device,
    ) -> tuple[BaseQuantizer, float]:
        """Original activation-based cosine distance."""
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
        best_score = float("inf")

        for quantizer in candidates:
            q_acts = quantizer(fp_acts)
            sim = F.cosine_similarity(
                fp_acts.unsqueeze(0), q_acts.unsqueeze(0)
            ).item()
            score = 1.0 - sim
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
        # No evaluation needed -- always return the fixed format
        return self.fixed_quantizer, 0.0
