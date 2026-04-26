"""OCP Microscaling (MXFP) fake-quantizer with E8M0 block scaling.

Implements the OCP MX specification (v1.0):
  * Elements are grouped into blocks of ``group_size`` (default 32).
  * Each block shares a single **E8M0** scale factor (power-of-two only).
  * Elements within a block are quantized to the target FP precision
    using the same rounding logic as ``FPQuantizer``.

Supported configurations:
  * MXFP8  — element E4M3 or E5M2, group_size=32
  * MXFP6  — element E3M2 or E2M3, group_size=32
  * MXFP4  — element E2M1,          group_size=32

Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec

All operations are vectorised PyTorch — ``torch.compile`` friendly.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from moq.quantizers.base import BaseQuantizer
from moq.quantizers.fp_quantizer import _fp_max_normal, _fp_min_subnormal
from moq.quantizers.registry import register_quantizer


# ======================================================================
# E8M0 scale quantization helpers
# ======================================================================

def _quantize_scale_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Quantize a scale tensor to E8M0 format (power-of-two only).

    E8M0 has 8 exponent bits and 0 mantissa bits, meaning it can only
    represent values of the form ``2^(e - 127)`` where ``e in [0, 254]``.
    This gives a range of ``[2^-127, 2^127]``.

    We round the log2 of the input to the nearest integer.
    """
    # Clamp to avoid log2(0) and stay within E8M0 range
    safe_scale = scale.clamp(min=2.0**-127, max=2.0**127)
    log2_scale = torch.log2(safe_scale)
    # Round to nearest integer (E8M0 = powers of two)
    exp = torch.round(log2_scale)
    # Clamp to E8M0 exponent range: bias=127, e in [0, 254]
    exp = exp.clamp(-127, 127)
    return torch.exp2(exp)


# ======================================================================
# Generic MXFP quantizer
# ======================================================================

@register_quantizer("mxfp")
class MXFPQuantizer(BaseQuantizer):
    """OCP Microscaling (MXFP) fake-quantizer.

    Parameters
    ----------
    element_bits : int
        Bit budget for each element (4, 6, or 8).
    element_exp_bits : int
        Exponent bits for the mini-float element.
        ``element_man_bits = element_bits - element_exp_bits - 1``.
    group_size : int
        Number of elements sharing one E8M0 scale (default 32, OCP standard).
    """

    def __init__(
        self,
        element_bits: int = 8,
        element_exp_bits: int = 4,
        group_size: int = 32,
    ) -> None:
        # The effective bit budget = element_bits (scale overhead is amortised)
        super().__init__(bits=element_bits, symmetric=True, channel_wise=False)

        man_bits = element_bits - element_exp_bits - 1  # 1 sign bit
        if man_bits < 0:
            raise ValueError(
                f"element_exp_bits={element_exp_bits} too large for "
                f"element_bits={element_bits}"
            )

        self.element_bits = element_bits
        self.element_exp_bits = element_exp_bits
        self.element_man_bits = man_bits
        self.group_size = group_size

        # Pre-compute element FP range
        self._max_normal = _fp_max_normal(element_exp_bits, man_bits)
        self._min_subnormal = _fp_min_subnormal(element_exp_bits, man_bits)
        self._bias = (1 << (element_exp_bits - 1)) - 1

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------

    def get_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-block E8M0 scales for *x*."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.group_size)
        block_max = x_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        # Scale so that block_max maps to max_normal
        raw_scale = block_max / self._max_normal
        return _quantize_scale_e8m0(raw_scale)

    # ------------------------------------------------------------------
    # Core quantize
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize *x* with MXFP block scaling."""
        orig_shape = x.shape
        numel = x.numel()

        # Pad to multiple of group_size if needed
        remainder = numel % self.group_size
        if remainder != 0:
            pad_size = self.group_size - remainder
            x_padded = torch.cat([x.reshape(-1), torch.zeros(pad_size, device=x.device, dtype=x.dtype)])
        else:
            pad_size = 0
            x_padded = x.reshape(-1)

        x_blocks = x_padded.reshape(-1, self.group_size)

        # Per-block E8M0 scale
        block_max = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        raw_scale = block_max / self._max_normal
        e8m0_scale = _quantize_scale_e8m0(raw_scale)

        # Scale elements into representable range
        x_scaled = x_blocks / e8m0_scale

        # Clamp to representable range
        x_scaled = x_scaled.clamp(-self._max_normal, self._max_normal)

        # Round elements to target FP precision
        x_fp = self._round_elements(x_scaled)

        # De-scale
        x_out = x_fp * e8m0_scale

        # Remove padding and reshape
        x_out = x_out.reshape(-1)[:numel].reshape(orig_shape)
        return x_out

    def _round_elements(self, x: torch.Tensor) -> torch.Tensor:
        """Round elements to (element_exp_bits, element_man_bits) FP."""
        sign = x.sign()
        x_abs = x.abs()

        nonzero_mask = x_abs > 0
        safe_x = torch.where(nonzero_mask, x_abs, torch.ones_like(x_abs))

        log2_x = torch.log2(safe_x)
        exp_floor = torch.floor(log2_x)

        max_exp = float((1 << self.element_exp_bits) - 2 - self._bias)
        min_exp = float(1 - self._bias)
        exp_clamped = exp_floor.clamp(min_exp, max_exp)

        pow2_exp = torch.exp2(exp_clamped)
        mantissa = safe_x / pow2_exp  # in [1, 2)

        factor = float(1 << self.element_man_bits)
        mantissa_rounded = torch.round(mantissa * factor) / factor

        # Handle mantissa overflow
        overflow = mantissa_rounded >= 2.0
        mantissa_rounded = torch.where(overflow, mantissa_rounded / 2.0, mantissa_rounded)
        exp_clamped = torch.where(overflow, exp_clamped + 1, exp_clamped)

        x_q = sign * mantissa_rounded * torch.exp2(exp_clamped)

        # Handle subnormals
        subnormal_threshold = float(2.0 ** min_exp)
        is_subnormal = x_abs < subnormal_threshold
        if is_subnormal.any():
            subnormal_step = float(2.0 ** (min_exp - self.element_man_bits))
            x_subnormal = torch.round(x_abs / subnormal_step) * subnormal_step * sign
            x_q = torch.where(is_subnormal, x_subnormal, x_q)

        x_q = torch.where(nonzero_mask, x_q, torch.zeros_like(x_q))
        return x_q

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["element_bits"] = self.element_bits
        cfg["element_exp_bits"] = self.element_exp_bits
        cfg["element_man_bits"] = self.element_man_bits
        cfg["group_size"] = self.group_size
        cfg["max_normal"] = self._max_normal
        return cfg


# ======================================================================
# Convenience subclasses for standard MXFP configurations
# ======================================================================

@register_quantizer("mxfp8_e4m3")
class MXFP8E4M3Quantizer(MXFPQuantizer):
    """MXFP8 with E4M3 elements (OCP standard)."""

    def __init__(self, group_size: int = 32) -> None:
        super().__init__(element_bits=8, element_exp_bits=4, group_size=group_size)


@register_quantizer("mxfp8_e5m2")
class MXFP8E5M2Quantizer(MXFPQuantizer):
    """MXFP8 with E5M2 elements (OCP standard)."""

    def __init__(self, group_size: int = 32) -> None:
        super().__init__(element_bits=8, element_exp_bits=5, group_size=group_size)


@register_quantizer("mxfp6_e3m2")
class MXFP6E3M2Quantizer(MXFPQuantizer):
    """MXFP6 with E3M2 elements (OCP standard)."""

    def __init__(self, group_size: int = 32) -> None:
        super().__init__(element_bits=6, element_exp_bits=3, group_size=group_size)


@register_quantizer("mxfp6_e2m3")
class MXFP6E2M3Quantizer(MXFPQuantizer):
    """MXFP6 with E2M3 elements (OCP standard, recommended for weights)."""

    def __init__(self, group_size: int = 32) -> None:
        super().__init__(element_bits=6, element_exp_bits=2, group_size=group_size)


@register_quantizer("mxfp4")
class MXFP4Quantizer(MXFPQuantizer):
    """MXFP4 with E2M1 elements (OCP standard)."""

    def __init__(self, group_size: int = 32) -> None:
        super().__init__(element_bits=4, element_exp_bits=2, group_size=group_size)
