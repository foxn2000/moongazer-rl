"""``unsloth-native``: TRL ``GRPOTrainer`` (Unsloth-patched when installed) whose generation
is replaced by canonical rollouts. Importing this package pulls in torch; the registry stays
lazy on purpose."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from moongazer_rl.trainer.unsloth_native.backend import (
        BACKEND_NAME,
        UnslothNativeBackend,
    )

__all__ = ["BACKEND_NAME", "UnslothNativeBackend"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from moongazer_rl.trainer.unsloth_native import backend

        return getattr(backend, name)
    raise AttributeError(name)
