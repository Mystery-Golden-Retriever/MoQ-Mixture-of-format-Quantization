"""Symmetric / Asymmetric uniform INT fake-quantizer.

Supports:
  * Per-tensor or per-channel scales.
  * Optional ACIQ analytical clipping.
  * Bit widths 2–16.

All operations are pure PyTorch and ``torch.compile`` friendly.
"""

from __future__ import annotations

from typing import Any

import torch

from moq.quantizers.base import BaseQuantizer
from moq.quantizers.aciq import ACIQClipper
from moq.quantizers.registry import register_quantizer


@register_quantizer("int")
class INTQuantizer(BaseQuantizer):
    """Uniform integer fake-quantizer.

    Parameters
    ----------
    bits : int
        Bit budget (2–16).
    symmetric : bool
        If ``True`` the zero-point is fixed at 0 (symmetric around zero).
    channel_wise : bool
        If ``True`` use per-output-channel scales (dim 0).
    use_aciq : bool
        If ``True`` apply ACIQ analytical clipping before quantization.
    aciq_distribution : str
        Distribution prior for ACIQ (``"gaussian"`` or ``"laplacian"``).
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        channel_wise: bool = False,
        use_aciq: bool = False,
        aciq_distribution: str = "gaussian",
    ) -> None:
        super().__init__(bits, symmetric, channel_wise)
        self.use_aciq = use_aciq
        self._aciq = ACIQClipper(bits, aciq_distribution) if use_aciq else None

    # ------------------------------------------------------------------
    # Core quant path
    # ------------------------------------------------------------------

    def _compute_qparams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(scale, clip_val)`` for *x*.

        For per-channel mode, ``scale`` has shape ``(C,)`` with ``C = x.shape[0]``.
        """
        if self.symmetric:
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**self.bits - 1

        if self.use_aciq and self._aciq is not None:
            # ACIQ provides a tighter clip range than abs-max
            clip_val = self._aciq.compute_clip(x)
        else:
            if self.channel_wise and x.dim() >= 2:
                # Per-channel: reduce over all dims except dim-0
                reduce_dims = list(range(1, x.dim()))
                clip_val = x.abs().amax(dim=reduce_dims)
            else:
                clip_val = x.abs().amax()

        scale = clip_val / max(abs(qmin), abs(qmax))
        # Guard against zero scale (e.g. all-zero tensor)
        scale = scale.clamp(min=1e-10)
        return scale, clip_val

    def get_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the quantization scale factor(s) for *x*."""
        scale, _ = self._compute_qparams(x)
        return scale

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize *x* using uniform INT mapping.

        ``x_q = round(clip(x, -c, c) / s) * s``
        """
        scale, clip_val = self._compute_qparams(x)

        if self.symmetric:
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**self.bits - 1

        # Reshape scale for broadcasting in per-channel mode
        if self.channel_wise and x.dim() >= 2:
            shape = [1] * x.dim()
            shape[0] = -1
            scale = scale.view(shape)
            clip_val = clip_val.view(shape)

        # Clip → scale → round → de-scale
        x_clipped = x.clamp(-clip_val, clip_val)
        x_int = (x_clipped / scale).round().clamp(qmin, qmax)
        x_dequant = x_int * scale
        return x_dequant

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["use_aciq"] = self.use_aciq
        if self._aciq is not None:
            cfg["aciq_distribution"] = self._aciq.distribution
        return cfg
