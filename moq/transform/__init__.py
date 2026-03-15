"""Transformation backbone: hook injection and module replacement."""

from moq.transform.hook_injector import HookQuantInjector
from moq.transform.module_replacer import QuantizedLinear, ModelQuantizer

__all__ = [
    "HookQuantInjector",
    "QuantizedLinear",
    "ModelQuantizer",
]
