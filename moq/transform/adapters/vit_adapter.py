"""Adapter for ViT (Vision Transformer) models.

Tested with:
  * ``google/vit-base-patch16-224``
  * ``google/vit-large-patch16-224``

The adapter enumerates the quantizable ``nn.Linear`` sub-modules inside
each encoder layer's self-attention and MLP blocks.
"""

from __future__ import annotations

import torch.nn as nn


class ViTQuantAdapter:
    """Adapter for ViT encoder-only architecture (HuggingFace Transformers).

    Layer naming convention (for layer index ``i``):

        vit.encoder.layer.{i}.attention.attention.query
        vit.encoder.layer.{i}.attention.attention.key
        vit.encoder.layer.{i}.attention.attention.value
        vit.encoder.layer.{i}.attention.output.dense
        vit.encoder.layer.{i}.intermediate.dense
        vit.encoder.layer.{i}.output.dense
    """

    ATTN_PATTERNS = [
        "vit.encoder.layer.{i}.attention.attention.query",
        "vit.encoder.layer.{i}.attention.attention.key",
        "vit.encoder.layer.{i}.attention.attention.value",
        "vit.encoder.layer.{i}.attention.output.dense",
    ]

    MLP_PATTERNS = [
        "vit.encoder.layer.{i}.intermediate.dense",
        "vit.encoder.layer.{i}.output.dense",
    ]

    ALL_PATTERNS = ATTN_PATTERNS + MLP_PATTERNS

    @classmethod
    def get_num_layers(cls, model: nn.Module) -> int:
        """Return the number of encoder layers."""
        if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
            return len(model.vit.encoder.layer)
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return len(model.encoder.layer)
        raise AttributeError(
            "Cannot detect number of layers. Expected "
            "`model.vit.encoder.layer` or `model.encoder.layer`."
        )

    @classmethod
    def get_layer_names(
        cls,
        model: nn.Module,
        include_attn: bool = True,
        include_mlp: bool = True,
    ) -> list[str]:
        """Enumerate all quantizable ``nn.Linear`` layer names."""
        n_layers = cls.get_num_layers(model)
        patterns: list[str] = []
        if include_attn:
            patterns.extend(cls.ATTN_PATTERNS)
        if include_mlp:
            patterns.extend(cls.MLP_PATTERNS)

        names = []
        for i in range(n_layers):
            for pattern in patterns:
                names.append(pattern.format(i=i))
        return names

    @classmethod
    def get_layer_groups(cls, model: nn.Module) -> dict[str, list[str]]:
        """Group layers by encoder block index."""
        n_layers = cls.get_num_layers(model)
        groups = {}
        for i in range(n_layers):
            groups[f"layer_{i}"] = [p.format(i=i) for p in cls.ALL_PATTERNS]
        return groups
