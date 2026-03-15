"""Quantizer registry — dynamic lookup by name.

Every quantizer decorated with ``@register_quantizer("name")`` is
automatically added to the global registry.  The calibrator and config
system resolve quantiser instances by name at runtime.

The registry is intentionally a plain ``dict`` (not an ``enum``) so that
**downstream users can add new formats** without editing this module.

Usage
-----
>>> from moq.quantizers.registry import get_quantizer, list_quantizers
>>> q = get_quantizer("int", bits=4, use_aciq=True)
>>> q
INTQuantizer(bits=4, symmetric=True, channel_wise=False, use_aciq=True)
>>> list_quantizers()
['int', 'fp', 'fp8_e4m3', 'fp8_e5m2', 'fp4_e2m1', 'fp4_e3m0']
"""

from __future__ import annotations

from typing import Any, Type

from moq.quantizers.base import BaseQuantizer

# Global mutable registry
_QUANTIZER_REGISTRY: dict[str, Type[BaseQuantizer]] = {}


def register_quantizer(name: str):
    """Class decorator that registers a ``BaseQuantizer`` subclass.

    Parameters
    ----------
    name : str
        Short, unique key.  Convention: ``"int"``, ``"fp"``,
        ``"fp8_e4m3"``, ``"fp4_e2m1"``, ``"nf4"`` …

    Raises
    ------
    ValueError
        If *name* is already registered (prevents silent overwrites).
    """

    def decorator(cls: Type[BaseQuantizer]) -> Type[BaseQuantizer]:
        if name in _QUANTIZER_REGISTRY:
            raise ValueError(
                f"Quantizer name {name!r} is already registered to "
                f"{_QUANTIZER_REGISTRY[name].__name__}"
            )
        _QUANTIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_quantizer(name: str, **kwargs: Any) -> BaseQuantizer:
    """Instantiate a registered quantizer by *name*.

    Extra *kwargs* are forwarded to the quantizer constructor.

    Raises
    ------
    KeyError
        If no quantizer is registered under *name*.
    """
    if name not in _QUANTIZER_REGISTRY:
        available = ", ".join(sorted(_QUANTIZER_REGISTRY.keys()))
        raise KeyError(
            f"Quantizer {name!r} not found.  Available: [{available}]"
        )
    return _QUANTIZER_REGISTRY[name](**kwargs)


def list_quantizers() -> list[str]:
    """Return all registered quantizer names (sorted)."""
    return sorted(_QUANTIZER_REGISTRY.keys())
