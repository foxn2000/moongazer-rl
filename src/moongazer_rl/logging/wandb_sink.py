"""Optional Weights & Biases sink. A no-op unless ``wandb`` is importable and enabled.

``wandb`` is never added as a project dependency (see pyproject.toml): it is imported
lazily, inside ``WandbSink.__init__``, only when the caller
asks for it (``enabled=True``, the default) — so importing this module, or using
``RunLogger`` without a sink, never requires ``wandb`` to be installed.
"""

from __future__ import annotations

from typing import Any


class WandbSink:
    """Thin wrapper around ``wandb.init``/``wandb.log``/``wandb.finish``.

    Every method swallows sink-side failures (missing package, no network, a raised
    ``wandb`` error) rather than propagating them: logging metrics to an optional sink
    must never break a training run. Check ``.enabled`` if the caller wants to know
    whether logging is actually happening.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        project: str | None = None,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        **init_kwargs: Any,
    ) -> None:
        self.enabled = False
        self._wandb: Any = None
        self._run: Any = None
        if not enabled:
            return
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError:
            return
        try:
            self._run = wandb.init(project=project, name=run_name, config=config, **init_kwargs)
        except Exception:
            return
        self._wandb = wandb
        self.enabled = True

    def log(self, step: int, data: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._wandb.log(data, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass


__all__ = ["WandbSink"]
