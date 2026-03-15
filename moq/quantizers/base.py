"""Base quantizer abstract class.

All quantization formats (INT, FP, MXFP) inherit from ``BaseQuantizer``
and implement the ``quantize`` / ``get_scale`` contract.  This file is the
**Strategy interface** in the Open-Closed architecture: new formats only
need to subclass, never modify.

Design notes (inference-only):
  * No STE / gradient support — this is a pure inference framework.
  * ``forward()`` is the public API; it delegates to ``quantize()``.
  * ``__repr__`` encodes the full config for logging / config serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseQuantizer(ABC):
    """Strategy interface for all quantization formats.

    Parameters
    ----------
    bits : int
        Total bit budget (e.g. 4, 8).
    symmetric : bool
        Whether to use symmetric (zero-centred) quantization.
    channel_wise : bool
        If ``True``, compute per-channel (dim-0) scales instead of per-tensor.
    """

    def __init__(
        self,
        bits: int,
        symmetric: bool = True,
        channel_wise: bool = False,
    ) -> None:
        if bits < 2 or bits > 16:
            raise ValueError(f"bits must be in [2, 16], got {bits}")
        self.bits = bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize *x*: simulate precision loss, return FP tensor.

        The returned tensor has the same dtype and shape as *x*.
        """
        ...

    @abstractmethod
    def get_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the dynamic scaling factor(s) for *x*.

        Returns a scalar tensor (per-tensor) or a 1-D tensor (per-channel).
        """
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization (inference-only, no gradient)."""
        return self.quantize(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Convenience alias so quantizers are callable like ``nn.Module``."""
        return self.forward(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serialisable config dict for this quantizer."""
        return {
            "class": type(self).__name__,
            "bits": self.bits,
            "symmetric": self.symmetric,
            "channel_wise": self.channel_wise,
        }

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in self.get_config().items() if k != "class")
        return f"{type(self).__name__}({fields})"
