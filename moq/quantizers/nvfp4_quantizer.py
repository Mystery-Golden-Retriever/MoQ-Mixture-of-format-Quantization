"""NVIDIA NVFP4 fake-quantizer with two-level scaling.

NVFP4 is NVIDIA's proprietary 4-bit format for Blackwell Tensor Cores:
  * Elements: E2M1 (1 sign, 2 exponent, 1 mantissa).
  * Per-block scale: FP8-E4M3, block size = 16.
  * Per-tensor super-scale: FP32.

The two-level hierarchy provides FP32-level dynamic range while keeping
per-element precision at 4 bits.

Reference: NVIDIA Blackwell Architecture Technical Brief

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
# FP8-E4M3 scale quantization helper
# ======================================================================

# E4M3 constants
_E4M3_MAX = _fp_max_normal(4, 3)  # 448.0
_E4M3_BIAS = 7  # 2^(4-1) - 1


def _quantize_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP8-E4M3 precision (fake quantization).

    Used to quantize the per-block scale factors in NVFP4.
    """
    sign = x.sign()
    x_abs = x.abs()

    nonzero = x_abs > 0
    safe_x = torch.where(nonzero, x_abs, torch.ones_like(x_abs))

    log2_x = torch.log2(safe_x)
    exp_floor = torch.floor(log2_x)

    max_exp = float((1 << 4) - 2 - _E4M3_BIAS)  # 7
    min_exp = float(1 - _E4M3_BIAS)               # -6
    exp_clamped = exp_floor.clamp(min_exp, max_exp)

    pow2_exp = torch.exp2(exp_clamped)
    mantissa = safe_x / pow2_exp

    # Round mantissa to 3 bits
    factor = float(1 << 3)  # 8
    mantissa_rounded = torch.round(mantissa * factor) / factor

    # Overflow correction
    overflow = mantissa_rounded >= 2.0
    mantissa_rounded = torch.where(overflow, mantissa_rounded / 2.0, mantissa_rounded)
    exp_clamped = torch.where(overflow, exp_clamped + 1, exp_clamped)

    x_q = sign * mantissa_rounded * torch.exp2(exp_clamped)
    x_q = torch.where(nonzero, x_q, torch.zeros_like(x_q))

    # Clamp to E4M3 range
    return x_q.clamp(-_E4M3_MAX, _E4M3_MAX)


# ======================================================================
# NVFP4 quantizer
# ======================================================================

# E2M1 constants (NVFP4 elements)
_E2M1_MAX = _fp_max_normal(2, 1)      # (2 - 0.5) * 2^(2 - 1) = 3.0
_E2M1_BIAS = 1  # 2^(2-1) - 1


@register_quantizer("nvfp4")
class NVFP4Quantizer(BaseQuantizer):
    """NVIDIA NVFP4 fake-quantizer with two-level scaling.

    Parameters
    ----------
    block_size : int
        Number of elements sharing one FP8-E4M3 block scale.
        Default 16 (NVIDIA standard).
    """

    def __init__(self, block_size: int = 16) -> None:
        super().__init__(bits=4, symmetric=True, channel_wise=False)
        self.block_size = block_size

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------

    def get_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the per-tensor super-scale."""
        return x.abs().amax().clamp(min=1e-12)

    # ------------------------------------------------------------------
    # Core quantize
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize *x* with NVFP4 two-level scaling."""
        orig_shape = x.shape
        numel = x.numel()

        # Pad to multiple of block_size
        remainder = numel % self.block_size
        if remainder != 0:
            pad_size = self.block_size - remainder
            x_padded = torch.cat([
                x.reshape(-1),
                torch.zeros(pad_size, device=x.device, dtype=x.dtype),
            ])
        else:
            pad_size = 0
            x_padded = x.reshape(-1)

        x_blocks = x_padded.reshape(-1, self.block_size)

        # -- Level 1: Per-tensor FP32 super-scale --
        global_max = x_blocks.abs().amax().clamp(min=1e-12)
        super_scale = global_max / (_E4M3_MAX * _E2M1_MAX)
        super_scale = super_scale.clamp(min=1e-12)

        x_prescaled = x_blocks / super_scale

        # -- Level 2: Per-block FP8-E4M3 scale --
        block_max = x_prescaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        raw_block_scale = block_max / _E2M1_MAX
        fp8_block_scale = _quantize_to_fp8_e4m3(raw_block_scale)
        fp8_block_scale = fp8_block_scale.clamp(min=1e-12)

        # Scale elements to E2M1 range
        x_scaled = x_prescaled / fp8_block_scale
        x_scaled = x_scaled.clamp(-_E2M1_MAX, _E2M1_MAX)

        # -- Quantize elements to E2M1 --
        x_fp = self._round_e2m1(x_scaled)

        # -- Reconstruct --
        x_out = x_fp * fp8_block_scale * super_scale
        x_out = x_out.reshape(-1)[:numel].reshape(orig_shape)
        return x_out

    def _round_e2m1(self, x: torch.Tensor) -> torch.Tensor:
        """Round to E2M1 (2 exponent, 1 mantissa) precision."""
        sign = x.sign()
        x_abs = x.abs()

        nonzero = x_abs > 0
        safe_x = torch.where(nonzero, x_abs, torch.ones_like(x_abs))

        log2_x = torch.log2(safe_x)
        exp_floor = torch.floor(log2_x)

        max_exp = float((1 << 2) - 2 - _E2M1_BIAS)  # 0
        min_exp = float(1 - _E2M1_BIAS)               # 0
        exp_clamped = exp_floor.clamp(min_exp, max_exp)

        pow2_exp = torch.exp2(exp_clamped)
        mantissa = safe_x / pow2_exp

        # Round mantissa to 1 bit
        factor = 2.0  # 2^1
        mantissa_rounded = torch.round(mantissa * factor) / factor

        # Overflow correction
        overflow = mantissa_rounded >= 2.0
        mantissa_rounded = torch.where(overflow, mantissa_rounded / 2.0, mantissa_rounded)
        exp_clamped = torch.where(overflow, exp_clamped + 1, exp_clamped)

        x_q = sign * mantissa_rounded * torch.exp2(exp_clamped)
        x_q = torch.where(nonzero, x_q, torch.zeros_like(x_q))
        return x_q

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["block_size"] = self.block_size
        cfg["element_format"] = "E2M1"
        cfg["block_scale_format"] = "FP8-E4M3"
        cfg["super_scale_format"] = "FP32"
        return cfg
