"""NormalFloat (NF4) fake-quantizer -- quantile-based lookup table.

NF4 is the information-theoretically optimal 4-bit data type for
normally distributed weights, introduced in the QLoRA paper.

Core algorithm:
  1. Pre-compute a 16-level codebook from N(0, 1) quantiles.
  2. Divide the weight tensor into blocks of group_size (default 64).
  3. Normalise each block by its abs_max to [-1, 1].
  4. Map each value to the nearest codebook entry (vectorised).
  5. Reconstruct: codebook[index] * abs_max.

Reference: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized
           Large Language Models", NeurIPS 2023.

All operations are vectorised PyTorch -- torch.compile friendly.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from moq.quantizers.base import BaseQuantizer
from moq.quantizers.registry import register_quantizer


# ======================================================================
# NF4 codebook -- 16 values from N(0,1) quantiles
# ======================================================================

def _build_nf4_codebook() -> torch.Tensor:
    """Build the NF4 codebook: 16 values from standard normal quantiles.

    The values are symmetric around 0 and designed so each bin
    captures equal probability mass under N(0, 1).

    The codebook is the same as used in bitsandbytes:
    https://github.com/TimDettmers/bitsandbytes
    """
    nf4_values = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
    return torch.tensor(nf4_values, dtype=torch.float32)


# Pre-computed codebook (module-level constant)
_NF4_CODEBOOK = _build_nf4_codebook()


# ======================================================================
# NormalFloat quantizer
# ======================================================================

@register_quantizer("nf4")
class NF4Quantizer(BaseQuantizer):
    """NormalFloat 4-bit (NF4) fake-quantizer.

    Parameters
    ----------
    group_size : int
        Block size for per-block normalisation (default 64, bitsandbytes standard).
    double_quant : bool
        If True, quantise the per-block scales to FP8 (saves ~0.37 bits/param).
        Default False.
    """

    def __init__(
        self,
        group_size: int = 64,
        double_quant: bool = False,
    ) -> None:
        super().__init__(bits=4, symmetric=True, channel_wise=False)
        self.group_size = group_size
        self.double_quant = double_quant

    # ------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------

    def get_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-block abs-max scales."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.group_size)
        return x_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)

    # ------------------------------------------------------------------
    # Core quantize
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize *x* with NF4 block-wise lookup."""
        orig_shape = x.shape
        numel = x.numel()

        # Pad to multiple of group_size
        remainder = numel % self.group_size
        if remainder != 0:
            pad_size = self.group_size - remainder
            x_padded = torch.cat([
                x.reshape(-1),
                torch.zeros(pad_size, device=x.device, dtype=x.dtype),
            ])
        else:
            pad_size = 0
            x_padded = x.reshape(-1)

        x_blocks = x_padded.reshape(-1, self.group_size)

        # Per-block normalisation
        block_max = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)

        # Optional double quantization of scales
        if self.double_quant:
            scale_max = block_max.max().clamp(min=1e-12)
            block_max_norm = block_max / scale_max
            block_max_q = torch.round(block_max_norm * 255) / 255
            block_max = block_max_q * scale_max

        x_norm = x_blocks / block_max  # now in [-1, 1]

        # Move codebook to same device
        codebook = _NF4_CODEBOOK.to(device=x.device, dtype=x.dtype)

        # Nearest-neighbour lookup
        distances = (x_norm.unsqueeze(-1) - codebook.reshape(1, 1, -1)).abs()
        indices = distances.argmin(dim=-1)

        # Reconstruct from codebook
        x_q = codebook[indices]

        # De-normalise
        x_out = x_q * block_max

        # Remove padding and reshape
        x_out = x_out.reshape(-1)[:numel].reshape(orig_shape)
        return x_out

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        cfg = super().get_config()
        cfg["group_size"] = self.group_size
        cfg["double_quant"] = self.double_quant
        cfg["codebook_size"] = 16
        return cfg
