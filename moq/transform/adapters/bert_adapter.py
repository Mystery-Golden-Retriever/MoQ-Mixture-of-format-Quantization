"""Adapter for BERT-style encoder-only Transformer models.

Tested with:
  * ``bert-base-uncased``
  * ``bert-large-uncased``
  * ``roberta-base``

The adapter enumerates the quantizable ``nn.Linear`` sub-modules inside
each encoder layer's self-attention and feed-forward blocks.
"""

from __future__ import annotations

import torch.nn as nn


class BERTQuantAdapter:
    """Adapter for BERT encoder-only architecture (HuggingFace Transformers).

    Layer naming convention (for layer index ``i``):

        bert.encoder.layer.{i}.attention.self.query
        bert.encoder.layer.{i}.attention.self.key
        bert.encoder.layer.{i}.attention.self.value
        bert.encoder.layer.{i}.attention.output.dense
        bert.encoder.layer.{i}.intermediate.dense
        bert.encoder.layer.{i}.output.dense
    """

    ATTN_PATTERNS = [
        "bert.encoder.layer.{i}.attention.self.query",
        "bert.encoder.layer.{i}.attention.self.key",
        "bert.encoder.layer.{i}.attention.self.value",
        "bert.encoder.layer.{i}.attention.output.dense",
    ]

    FFN_PATTERNS = [
        "bert.encoder.layer.{i}.intermediate.dense",
        "bert.encoder.layer.{i}.output.dense",
    ]

    ALL_PATTERNS = ATTN_PATTERNS + FFN_PATTERNS

    @classmethod
    def get_num_layers(cls, model: nn.Module) -> int:
        """Return the number of encoder layers."""
        if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
            return len(model.bert.encoder.layer)
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return len(model.encoder.layer)
        raise AttributeError(
            "Cannot detect number of layers. Expected "
            "`model.bert.encoder.layer` or `model.encoder.layer`."
        )

    @classmethod
    def get_layer_names(
        cls,
        model: nn.Module,
        include_attn: bool = True,
        include_ffn: bool = True,
    ) -> list[str]:
        """Enumerate all quantizable ``nn.Linear`` layer names."""
        n_layers = cls.get_num_layers(model)
        patterns: list[str] = []
        if include_attn:
            patterns.extend(cls.ATTN_PATTERNS)
        if include_ffn:
            patterns.extend(cls.FFN_PATTERNS)

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
