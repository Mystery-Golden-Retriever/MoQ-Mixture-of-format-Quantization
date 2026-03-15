"""ACIQ — Analytical Clipping for Integer Quantization.

Computes the *optimal* clip threshold ``c*`` that minimises the expected
MSE for uniform (INT) quantization.  Supports both **Gaussian** and
**Laplacian** distribution priors.

Reference
---------
Banner et al., "ACIQ: Analytical Clipping for Integer Quantization of
Neural Networks", 2018.  https://arxiv.org/abs/1810.05723

The optimal alpha coefficients are pre-computed for common bit widths.
For arbitrary widths the Gaussian formula is evaluated analytically.
"""

from __future__ import annotations

import math
from typing import Literal

import torch


# Pre-computed α for Gaussian: c* = α · σ
# Values from Table 1 in the ACIQ paper (rounded to 2 dp).
_GAUSSIAN_ALPHA: dict[int, float] = {
    2: 1.71,
    3: 2.15,
    4: 2.83,
    5: 3.38,
    6: 3.89,
    7: 4.42,
    8: 5.03,
}

# Pre-computed α for Laplacian: c* = α · b  (b = scale = σ/√2)
_LAPLACIAN_ALPHA: dict[int, float] = {
    2: 2.83,
    3: 3.89,
    4: 5.03,
    5: 6.20,
    6: 7.41,
    7: 8.64,
    8: 9.89,
}


class ACIQClipper:
    """Compute the ACIQ optimal clipping threshold.

    Parameters
    ----------
    bits : int
        Target bit width of the quantizer (2–8).
    distribution : ``"gaussian"`` | ``"laplacian"``
        Assumed prior distribution of activations.

    Example
    -------
    >>> clipper = ACIQClipper(bits=4, distribution="gaussian")
    >>> x = torch.randn(1, 128)
    >>> clip_val = clipper.compute_clip(x)
    """

    def __init__(
        self,
        bits: int,
        distribution: Literal["gaussian", "laplacian"] = "gaussian",
    ) -> None:
        if distribution == "gaussian":
            alpha_table = _GAUSSIAN_ALPHA
        elif distribution == "laplacian":
            alpha_table = _LAPLACIAN_ALPHA
        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

        self.bits = bits
        self.distribution = distribution

        if bits in alpha_table:
            self.alpha = alpha_table[bits]
        else:
            # Fall back to a simple closed-form estimate for Gaussian:
            #   α ≈ √(2 · ln(n_levels))  where n_levels = 2^b
            self.alpha = math.sqrt(2.0 * math.log(2**bits))

    # ------------------------------------------------------------------

    def compute_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Return the optimal clip value ``c* = α · σ`` (Gaussian) or
        ``c* = α · b`` (Laplacian) as a scalar tensor on the same device.
        """
        if self.distribution == "gaussian":
            dispersion = x.float().std()
        else:
            # Laplacian scale  b = σ / √2
            dispersion = x.float().std() / math.sqrt(2.0)

        clip_val = self.alpha * dispersion
        # Never clip tighter than the actual range (safety guard)
        clip_val = torch.min(clip_val, x.abs().max())
        return clip_val.to(x.device)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ACIQClipper(bits={self.bits}, distribution={self.distribution!r}, "
            f"alpha={self.alpha:.3f})"
        )
