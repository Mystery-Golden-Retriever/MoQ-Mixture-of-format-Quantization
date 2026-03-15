"""Model adapters — know the quantizable layer names for each architecture."""

from moq.transform.adapters.llama_adapter import LlamaQuantAdapter
from moq.transform.adapters.bert_adapter import BERTQuantAdapter
from moq.transform.adapters.vit_adapter import ViTQuantAdapter

__all__ = [
    "LlamaQuantAdapter",
    "BERTQuantAdapter",
    "ViTQuantAdapter",
]
