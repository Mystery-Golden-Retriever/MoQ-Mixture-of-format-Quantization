"""Floating-point fake-quantizer with configurable exponent / mantissa split.

Supports arbitrary ``(exp_bits, man_bits)`` combinations as well as the
two IEEE 754 FP8 variants (E4M3, E5M2) via convenience constructors.

The core algorithm:
  1. Compute a per-tensor (or per-channel) scale so that ``x`` fits in
     the representable FP range.
  2. Decompose ``x_scaled`` into ``sign × 2^e × (1 + frac)``   (normal)
     or ``sign × 2^(1-bias) × frac``              (subnormal).
  3. Round ``frac`` to ``man_bits`` binary digits.
  4. Re-compose and de-scale.

All operations are vectorised PyTorch — no Python loops over elements.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from moq.quantizers.base import BaseQuantizer
from moq.quantizers.aciq import ACIQClipper
from moq.quantizers.registry import register_quantizer


# ======================================================================
# Pre-computed FP format specifications
# ======================================================================

def _fp_max_normal(exp_bits: int, man_bits: int) -> float:
    """Maximum *normal* representable value for (exp_bits, man_bits).

    Formula:  (2 - 2^{-m}) × 2^{(2^e - 2) - bias}
    where bias = 2^{e-1} - 1.
    """
    bias = (1 << (exp_bits - 1)) - 1
    max_exp = (1 << exp_bits) - 2  # all-ones exponent is reserved (Inf/NaN)
    return (2.0 - 2.0 ** (-man_bits)) * (2.0 ** (max_exp - bias))


def _fp_min_subnormal(exp_bits: int, man_bits: int) -> float:
    """Smallest positive *subnormal* representable value."""
    bias = (1 << (exp_bits - 1)) - 1
    return 2.0 ** (1 - bias - man_bits)


# ======================================================================
# Generic FP quantizer
# ======================================================================

@register_quantizer("fp")
class FPQuantizer(BaseQuantizer):
    """Floating-point fake-quantizer with arbitrary exponent/mantissa split.

    Parameters
    ----------
    bits : int
        Total bit budget (including 1 sign bit).
    exp_bits : int
        Number of exponent bits.  ``man_bits = bits - exp_bits - 1``.
    channel_wise : bool
        Per-channel (dim 0) scaling if ``True``.
    use_aciq : bool
        Apply ACIQ clipping to the input before FP quantization.
    """

    def __init__(
        self,
        bits: int = 8,
        exp_bits: int = 4,
        channel_wise: bool = False,
        use_aciq: bool = False,
        aciq_distribution: str = "gaussian",
    ) -> None:
        super().__init__(bits, symmetric=True, channel_wise=channel_wise)

        man_bits = bits - exp_bits - 1  # 1 sign bit
        if man_bits < 0:
            raise ValueError(
                f"exp_bits={exp_bits} is too large for bits={bits} "
                f"(need at least 1 sign bit, leaving man_bits<0)"
            )
        self.exp_bits = exp_bits
        self.man_bits = man_bits
        self.use_aciq = use_aciq
        self._aciq = ACIQClipper(bits, aciq_distribution) if use_aciq else None

        # Pre-compute representable range
        self._max_normal = _fp_max_normal(exp_bits, man_bits)
        self._min_subnormal = _fp_min_subnormal(exp_bits, man_bits)
        self._bias = (1 << (exp_bits - 1)) - 1

    # ------------------------------------------------------------------
    # Scale computation
    # ------------------------------------------------------------------

    def get_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Scale factor that maps ``|x|_max`` into the FP representable range."""
        if self.use_aciq and self._aciq is not None:
            abs_max = self._aciq.compute_clip(x)
        elif self.channel_wise and x.dim() >= 2:
            reduce_dims = list(range(1, x.dim()))
            abs_max = x.abs().amax(dim=reduce_dims)
        else:
            abs_max = x.abs().amax()

        abs_max = abs_max.clamp(min=1e-12)
        scale = abs_max / self._max_normal
        return scale

    # ------------------------------------------------------------------
    # Fake quantize
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize *x* in floating-point with ``(exp_bits, man_bits)``."""
        scale = self.get_scale(x)

        # Reshape for per-channel broadcasting
        if self.channel_wise and x.dim() >= 2:
            shape = [1] * x.dim()
            shape[0] = -1
            scale = scale.view(shape)

        # Scale to representable range
        x_scaled = x / scale

        # Optional ACIQ clipping (after scaling)
        if self.use_aciq and self._aciq is not None:
            clip = self._max_normal
            x_scaled = x_scaled.clamp(-clip, clip)

        # Apply FP precision rounding
        x_fp = self._round_to_fp(x_scaled)

        # De-scale
        return x_fp * scale

    def _round_to_fp(self, x: torch.Tensor) -> torch.Tensor:
        """Round *x* (assumed to be within representable range) to the
        nearest value representable by ``(exp_bits, man_bits)`` FP.

        The algorithm decomposes each value into sign, exponent, and mantissa,
        rounds the mantissa to ``man_bits`` fractional digits, and re-composes.
        """
        sign = x.sign()
        x_abs = x.abs()

        # Mask out exact zeros to avoid log2(0)
        nonzero_mask = x_abs > 0
        safe_x = torch.where(nonzero_mask, x_abs, torch.ones_like(x_abs))

        # Compute biased exponent (floor of log2)
        log2_x = torch.log2(safe_x)
        exp_floor = torch.floor(log2_x)  # unbiased exponent

        # Clamp exponent to representable range
        max_exp = float((1 << self.exp_bits) - 2 - self._bias)  # max normal exponent
        min_exp = float(1 - self._bias)  # min normal exponent
        exp_clamped = exp_floor.clamp(min_exp, max_exp)

        # Mantissa: x_abs = 2^e * (1 + frac)  →  frac = x_abs / 2^e - 1
        pow2_exp = torch.exp2(exp_clamped)
        mantissa = safe_x / pow2_exp  # in [1, 2)

        # Round mantissa to man_bits fractional digits
        factor = float(1 << self.man_bits)  # 2^man_bits
        mantissa_rounded = torch.round(mantissa * factor) / factor

        # Handle mantissa overflow (rounded up to 2.0 → bump exponent)
        overflow = mantissa_rounded >= 2.0
        mantissa_rounded = torch.where(overflow, mantissa_rounded / 2.0, mantissa_rounded)
        exp_clamped = torch.where(overflow, exp_clamped + 1, exp_clamped)

        # Re-compose
        x_q = sign * mantissa_rounded * torch.exp2(exp_clamped)

        # Handle subnormal values (very small magnitudes flushed toward zero)
        subnormal_threshold = float(2.0 ** min_exp)
        is_subnormal = x_abs < subnormal_threshold
        if is_subnormal.any():
            subnormal_step = float(2.0 ** (min_exp - self.man_bits))
            x_subnormal = torch.round(x_abs / subnormal_step) * subnormal_step * sign
            x_q = torch.where(is_subnormal, x_subnormal, x_q)

        # Zero out original zeros
        x_q = torch.where(nonzero_mask, x_q, torch.zeros_like(x_q))
        return x_q

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["exp_bits"] = self.exp_bits
        cfg["man_bits"] = self.man_bits
        cfg["use_aciq"] = self.use_aciq
        cfg["max_normal"] = self._max_normal
        return cfg


# ======================================================================
# Convenience subclasses for common IEEE FP8 variants
# ======================================================================

@register_quantizer("fp8_e4m3")
class E4M3Quantizer(FPQuantizer):
    """IEEE 754 FP8 E4M3 quantizer (higher precision, lower range).

    - 1 sign, 4 exponent, 3 mantissa bits
    - Max representable: 448.0
    - Best for weights and activations without extreme outliers.
    """

    def __init__(self, channel_wise: bool = False, use_aciq: bool = False) -> None:
        super().__init__(bits=8, exp_bits=4, channel_wise=channel_wise, use_aciq=use_aciq)


@register_quantizer("fp8_e5m2")
class E5M2Quantizer(FPQuantizer):
    """IEEE 754 FP8 E5M2 quantizer (lower precision, higher range).

    - 1 sign, 5 exponent, 2 mantissa bits
    - Max representable: 57344.0
    - Best for activations with high-magnitude outliers (e.g. attention).
    """

    def __init__(self, channel_wise: bool = False, use_aciq: bool = False) -> None:
        super().__init__(bits=8, exp_bits=5, channel_wise=channel_wise, use_aciq=use_aciq)


@register_quantizer("fp4_e2m1")
class FP4E2M1Quantizer(FPQuantizer):
    """4-bit FP with 2 exponent, 1 mantissa bits.

    Commonly used in MXFP4 formats.  8 representable values per sign.
    """

    def __init__(self, channel_wise: bool = False, use_aciq: bool = False) -> None:
        super().__init__(bits=4, exp_bits=2, channel_wise=channel_wise, use_aciq=use_aciq)


@register_quantizer("fp4_e3m0")
class FP4E3M0Quantizer(FPQuantizer):
    """4-bit FP with 3 exponent, 0 mantissa bits.

    Maximum dynamic range at 4 bits — values snap to exact powers of 2.
    """

    def __init__(self, channel_wise: bool = False, use_aciq: bool = False) -> None:
        super().__init__(bits=4, exp_bits=3, channel_wise=channel_wise, use_aciq=use_aciq)
