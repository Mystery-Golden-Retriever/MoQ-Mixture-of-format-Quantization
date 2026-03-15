"""Adapter for Llama-style decoder-only Transformer models.

Tested with:
  * ``meta-llama/Llama-2-7b-hf``
  * ``meta-llama/Meta-Llama-3-8B``
  * ``meta-llama/Meta-Llama-3.1-8B``

The adapter knows the canonical HuggingFace layer names and provides
utilities to enumerate quantizable ``nn.Linear`` modules.
"""

from __future__ import annotations

import torch.nn as nn


class LlamaQuantAdapter:
    """Adapter for Llama decoder-only architecture (HuggingFace Transformers).

    Layer naming convention (for layer index ``i``):

        model.layers.{i}.self_attn.q_proj
        model.layers.{i}.self_attn.k_proj
        model.layers.{i}.self_attn.v_proj
        model.layers.{i}.self_attn.o_proj
        model.layers.{i}.mlp.gate_proj
        model.layers.{i}.mlp.up_proj
        model.layers.{i}.mlp.down_proj
    """

    # Pattern templates — `{i}` is replaced with the layer index.
    ATTN_PROJ_PATTERNS = [
        "model.layers.{i}.self_attn.q_proj",
        "model.layers.{i}.self_attn.k_proj",
        "model.layers.{i}.self_attn.v_proj",
        "model.layers.{i}.self_attn.o_proj",
    ]

    MLP_PATTERNS = [
        "model.layers.{i}.mlp.gate_proj",
        "model.layers.{i}.mlp.up_proj",
        "model.layers.{i}.mlp.down_proj",
    ]

    ALL_PATTERNS = ATTN_PROJ_PATTERNS + MLP_PATTERNS

    @classmethod
    def get_num_layers(cls, model: nn.Module) -> int:
        """Return the number of decoder layers."""
        # HuggingFace Llama stores layers in model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        # If passed the inner model directly
        if hasattr(model, "layers"):
            return len(model.layers)
        raise AttributeError(
            "Cannot detect number of layers. Expected "
            "`model.model.layers` or `model.layers`."
        )

    @classmethod
    def get_layer_names(
        cls,
        model: nn.Module,
        include_attn: bool = True,
        include_mlp: bool = True,
    ) -> list[str]:
        """Enumerate all quantizable ``nn.Linear`` layer names.

        Parameters
        ----------
        model : nn.Module
            A HuggingFace ``LlamaForCausalLM`` (or its inner model).
        include_attn : bool
            Include Q/K/V/O attention projections.
        include_mlp : bool
            Include gate/up/down MLP projections.

        Returns
        -------
        list[str]
            Sorted list of fully-qualified module names.
        """
        n_layers = cls.get_num_layers(model)
        patterns: list[str] = []
        if include_attn:
            patterns.extend(cls.ATTN_PROJ_PATTERNS)
        if include_mlp:
            patterns.extend(cls.MLP_PATTERNS)

        names = []
        for i in range(n_layers):
            for pattern in patterns:
                names.append(pattern.format(i=i))
        return names

    @classmethod
    def get_layer_groups(cls, model: nn.Module) -> dict[str, list[str]]:
        """Group quantizable layers by decoder block.

        Returns ``{"layer_0": ["...q_proj", "...k_proj", ...], ...}``.
        """
        n_layers = cls.get_num_layers(model)
        groups = {}
        for i in range(n_layers):
            key = f"layer_{i}"
            groups[key] = [p.format(i=i) for p in cls.ALL_PATTERNS]
        return groups
