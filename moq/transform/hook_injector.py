"""Hook-based non-destructive quantization injector.

Registers ``forward_hook`` on target modules to fake-quantize their
outputs.  The original model weights and architecture are **unchanged** —
hooks can be added and removed freely.

Use cases:
  * Calibration phase — temporarily quantize one layer at a time.
  * Quick A/B evaluation without modifying the model graph.

Limitations:
  * ``torch.compile(fullgraph=True)`` may not inline hooks correctly.
    For compiled inference, prefer ``ModelQuantizer`` (module replacement).

This class implements the **context-manager** protocol so that hooks are
always cleaned up, even on exceptions:

    >>> with HookQuantInjector(model, format_map) as injector:
    ...     output = model(batch)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from moq.quantizers.base import BaseQuantizer


class HookQuantInjector:
    """Inject fake-quantization via PyTorch forward hooks.

    Parameters
    ----------
    model : nn.Module
        The pretrained model to instrument.
    format_map : dict[str, BaseQuantizer]
        Mapping ``{module_name: quantizer}``.  Only modules listed here
        will have hooks attached.
    quantize_input : bool
        If ``True`` the hook quantizes the module's **input** (activation
        entering the layer).  If ``False`` (default) it quantizes the
        module's **output** (activation leaving the layer).
    """

    def __init__(
        self,
        model: nn.Module,
        format_map: dict[str, BaseQuantizer],
        quantize_input: bool = False,
    ) -> None:
        self.model = model
        self.format_map = format_map
        self.quantize_input = quantize_input
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._enabled = False

    # ------------------------------------------------------------------
    # Hook builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_output_hook(quantizer: BaseQuantizer):
        """Return a hook function that quantizes the module output."""
        def hook(_module: nn.Module, _input, output: torch.Tensor):
            if isinstance(output, torch.Tensor):
                return quantizer(output)
            # Some modules return tuples (e.g. Attention returns (attn_out, weights))
            if isinstance(output, tuple):
                return (quantizer(output[0]),) + output[1:]
            return output
        return hook

    @staticmethod
    def _make_input_hook(quantizer: BaseQuantizer):
        """Return a hook function that quantizes the module input."""
        def hook(_module: nn.Module, args):
            quantized_args = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    quantized_args.append(quantizer(a))
                else:
                    quantized_args.append(a)
            return tuple(quantized_args)
        return hook

    # ------------------------------------------------------------------
    # Inject / remove
    # ------------------------------------------------------------------

    def inject(self) -> "HookQuantInjector":
        """Register all quantization hooks.  Idempotent."""
        if self._enabled:
            return self

        name_to_module = dict(self.model.named_modules())

        for name, quantizer in self.format_map.items():
            module = name_to_module.get(name)
            if module is None:
                raise KeyError(
                    f"Module {name!r} not found in the model. "
                    f"Available: {list(name_to_module.keys())[:20]}…"
                )
            if self.quantize_input:
                h = module.register_forward_pre_hook(
                    self._make_input_hook(quantizer)
                )
            else:
                h = module.register_forward_hook(
                    self._make_output_hook(quantizer)
                )
            self._hooks.append(h)

        self._enabled = True
        return self

    def remove(self) -> None:
        """Remove all hooks (restore original model).  Idempotent."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._enabled = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HookQuantInjector":
        self.inject()
        return self

    def __exit__(self, *_args) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def num_hooks(self) -> int:
        return len(self._hooks)

    def __repr__(self) -> str:
        return (
            f"HookQuantInjector(layers={len(self.format_map)}, "
            f"hooks_active={self.num_hooks}, "
            f"quantize={'input' if self.quantize_input else 'output'})"
        )
