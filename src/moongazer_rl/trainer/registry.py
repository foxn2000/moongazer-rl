"""Name -> lazy TrainerBackend factory. Importing this module never imports torch."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

_BACKENDS: dict[str, str] = {
    "unsloth-native": "moongazer_rl.trainer.unsloth_native.backend:UnslothNativeBackend",
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def resolve_backend(name: str) -> Callable[..., Any]:
    """Return the backend class for ``name`` (imports it on first use)."""
    try:
        target = _BACKENDS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown trainer backend {name!r}; available: {available_backends()}"
        ) from exc
    module_name, _, attr = target.partition(":")
    module = import_module(module_name)
    return getattr(module, attr)


__all__ = ["available_backends", "resolve_backend"]
