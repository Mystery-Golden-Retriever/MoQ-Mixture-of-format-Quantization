"""Statistics observer — activation distribution collection via forward hooks.

Uses **Welford's online algorithm** for numerically stable, O(1)-memory
running mean/variance.  Also collects min/max, kurtosis, and a 256-bin
histogram for distribution shape analysis.

The observer is non-destructive: it only reads activations via hooks and
never modifies the model or its outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LayerStats:
    """Aggregated activation statistics for one layer."""

    min_val: float = float("inf")
    max_val: float = float("-inf")
    mean: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    kurtosis: float = 0.0
    num_elements: int = 0
    num_batches: int = 0
    histogram: Optional[torch.Tensor] = field(default=None, repr=False)

    # Welford accumulators (not part of the public API)
    _m2: float = field(default=0.0, repr=False)
    _m4: float = field(default=0.0, repr=False)


class StatisticsObserver:
    """Collect per-layer activation statistics via forward hooks.

    Uses Welford's online algorithm for numerically stable incremental
    mean / variance / kurtosis — **no full-tensor storage**.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins (default 256).

    Example
    -------
    >>> observer = StatisticsObserver()
    >>> observer.attach(model, ["encoder.layer.0.output.dense"])
    >>> for batch in calib_loader:
    ...     model(batch)
    >>> stats = observer.get_stats()
    >>> observer.detach()
    """

    def __init__(self, n_bins: int = 256) -> None:
        self.n_bins = n_bins
        self._stats: dict[str, LayerStats] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._hist_ranges: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------

    def attach(self, model: nn.Module, layer_names: list[str]) -> "StatisticsObserver":
        """Register forward hooks on the specified layers.

        Parameters
        ----------
        model : nn.Module
            The model to observe.
        layer_names : list[str]
            Fully-qualified module names to observe.
        """
        name_to_module = dict(model.named_modules())
        for name in layer_names:
            if name not in name_to_module:
                raise KeyError(f"Module {name!r} not found in the model.")
            module = name_to_module[name]
            self._stats[name] = LayerStats()
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)
        return self

    def detach(self) -> None:
        """Remove all observer hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Hook factory
    # ------------------------------------------------------------------

    def _make_hook(self, name: str):
        """Create a forward hook that updates online statistics."""

        def hook(_module: nn.Module, _input, output):
            # Handle tuple outputs (e.g. attention layers)
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return

            x = output.detach().float().flatten()
            self._update_welford(name, x)
            self._update_histogram(name, x)

        return hook

    # ------------------------------------------------------------------
    # Welford's online algorithm
    # ------------------------------------------------------------------

    def _update_welford(self, name: str, x: torch.Tensor) -> None:
        """Incremental update of mean, variance, kurtosis (Welford)."""
        s = self._stats[name]
        n_new = x.numel()
        batch_mean = x.mean().item()
        batch_var = x.var().item() if n_new > 1 else 0.0

        # Update min / max
        s.min_val = min(s.min_val, x.min().item())
        s.max_val = max(s.max_val, x.max().item())

        n_old = s.num_elements
        n_total = n_old + n_new

        if n_old == 0:
            s.mean = batch_mean
            s.variance = batch_var
            s._m2 = batch_var * n_new
        else:
            # Combine two sets of statistics
            delta = batch_mean - s.mean
            new_mean = s.mean + delta * n_new / n_total
            new_m2 = s._m2 + batch_var * n_new + delta**2 * n_old * n_new / n_total
            s.mean = new_mean
            s._m2 = new_m2
            s.variance = s._m2 / n_total

        s.std = s.variance**0.5
        s.num_elements = n_total
        s.num_batches += 1

        # Fourth central moment (for kurtosis) — simplified batch estimate
        if n_new > 3:
            m4_batch = ((x - batch_mean) ** 4).mean().item()
            if s.variance > 1e-12:
                s.kurtosis = m4_batch / (s.variance**2) - 3.0  # Excess kurtosis

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def _update_histogram(self, name: str, x: torch.Tensor) -> None:
        """Accumulate per-layer histogram."""
        x_min, x_max = x.min().item(), x.max().item()

        if name not in self._hist_ranges:
            self._hist_ranges[name] = (x_min, x_max)
        else:
            old_min, old_max = self._hist_ranges[name]
            self._hist_ranges[name] = (min(old_min, x_min), max(old_max, x_max))

        lo, hi = self._hist_ranges[name]
        if hi - lo < 1e-10:
            hi = lo + 1.0

        hist = torch.histc(x, bins=self.n_bins, min=lo, max=hi)

        s = self._stats[name]
        if s.histogram is None:
            s.histogram = hist
        else:
            # Re-bin if range expanded (simple approach: restart)
            if s.histogram.shape == hist.shape:
                s.histogram = s.histogram + hist
            else:
                s.histogram = hist

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, LayerStats]:
        """Return collected statistics (read-only snapshot)."""
        return dict(self._stats)

    def get_layer_stats(self, name: str) -> LayerStats:
        """Return stats for a single layer."""
        if name not in self._stats:
            raise KeyError(f"No stats collected for {name!r}")
        return self._stats[name]

    def reset(self) -> None:
        """Clear all collected statistics (hooks remain attached)."""
        for name in self._stats:
            self._stats[name] = LayerStats()
        self._hist_ranges.clear()

    def __repr__(self) -> str:
        return (
            f"StatisticsObserver(layers={len(self._stats)}, "
            f"hooks={len(self._hooks)}, bins={self.n_bins})"
        )
