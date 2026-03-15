"""Module-replacement based quantization injection.

Unlike ``HookQuantInjector`` this **modifies the model graph** by replacing
``nn.Linear`` (and optionally other layer types) with ``QuantizedLinear``
wrappers.  This is the preferred approach for:

  * ``torch.compile``-compatible inference.
  * Final deployment with a fixed format map.

The replacement is non-destructive to the original *weights* — they are
copied (not moved) into the new modules.
"""

from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from moq.quantizers.base import BaseQuantizer


# ======================================================================
# Quantized drop-in replacements
# ======================================================================

class QuantizedLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with fake weight + activation
    quantization (inference-only).

    Parameters
    ----------
    original : nn.Linear
        The original layer whose weights and bias are copied.
    weight_quantizer : BaseQuantizer or None
        Quantizer applied to ``self.weight`` before matrix multiply.
        ``None`` = leave weights in full precision.
    act_quantizer : BaseQuantizer or None
        Quantizer applied to the input activations.
        ``None`` = leave activations in full precision.
    """

    def __init__(
        self,
        original: nn.Linear,
        weight_quantizer: Optional[BaseQuantizer] = None,
        act_quantizer: Optional[BaseQuantizer] = None,
    ) -> None:
        super().__init__()
        # Copy weight and bias from the original layer
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        self.bias: Optional[nn.Parameter] = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)

        self.in_features = original.in_features
        self.out_features = original.out_features

        self.weight_quantizer = weight_quantizer
        self.act_quantizer = act_quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation quantization (input side)
        if self.act_quantizer is not None:
            x = self.act_quantizer(x)

        # Weight quantization (fake — compute on the fly)
        w = self.weight
        if self.weight_quantizer is not None:
            w = self.weight_quantizer(w)

        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"bias={self.bias is not None}",
        ]
        if self.weight_quantizer is not None:
            parts.append(f"weight_q={self.weight_quantizer}")
        if self.act_quantizer is not None:
            parts.append(f"act_q={self.act_quantizer}")
        return ", ".join(parts)


# ======================================================================
# Model-level replacer
# ======================================================================

class ModelQuantizer:
    """Replace target modules in a model with quantized variants.

    Parameters
    ----------
    target_types : set of types
        Module types eligible for replacement (default: ``{nn.Linear}``).

    Example
    -------
    >>> replacer = ModelQuantizer()
    >>> format_map = {
    ...     "encoder.layer.0.fc1": (act_q, wt_q),
    ...     "encoder.layer.0.fc2": (act_q, wt_q),
    ... }
    >>> model = replacer.replace(model, format_map)
    """

    DEFAULT_TARGETS: set[Type[nn.Module]] = {nn.Linear}

    def __init__(
        self,
        target_types: Optional[set[Type[nn.Module]]] = None,
    ) -> None:
        self.target_types = target_types or self.DEFAULT_TARGETS

    # ------------------------------------------------------------------

    def replace(
        self,
        model: nn.Module,
        format_map: dict[str, tuple[Optional[BaseQuantizer], Optional[BaseQuantizer]]],
    ) -> nn.Module:
        """Replace modules listed in *format_map* with quantized wrappers.

        Parameters
        ----------
        model : nn.Module
            The model to transform **in-place**.
        format_map : dict
            ``{module_name: (act_quantizer, weight_quantizer)}``.
            Either quantizer may be ``None`` to skip that pathway.

        Returns
        -------
        nn.Module
            The same model reference (modified in place).
        """
        name_to_module = dict(model.named_modules())

        for name, (act_q, wt_q) in format_map.items():
            module = name_to_module.get(name)
            if module is None:
                raise KeyError(f"Module {name!r} not found in the model.")
            if type(module) not in self.target_types:
                raise TypeError(
                    f"Module {name!r} is {type(module).__name__}, "
                    f"not in target_types {self.target_types}"
                )

            # Build the quantized replacement
            if isinstance(module, nn.Linear):
                replacement = QuantizedLinear(module, wt_q, act_q)
            else:
                raise NotImplementedError(
                    f"No quantized replacement for {type(module).__name__}"
                )

            # Graft the replacement into the parent module
            self._set_submodule(model, name, replacement)

        return model

    # ------------------------------------------------------------------

    def restore(self, model: nn.Module) -> nn.Module:
        """Undo quantization: replace ``QuantizedLinear`` back to ``nn.Linear``.

        This reconstructs regular ``nn.Linear`` from the stored parameters.
        """
        for name, module in list(model.named_modules()):
            if isinstance(module, QuantizedLinear):
                restored = nn.Linear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                )
                restored.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    restored.bias.data.copy_(module.bias.data)
                restored.weight.requires_grad_(False)
                if restored.bias is not None:
                    restored.bias.requires_grad_(False)
                self._set_submodule(model, name, restored)
        return model

    # ------------------------------------------------------------------

    @staticmethod
    def _set_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
        """Set a nested sub-module by dotted name."""
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], replacement)
