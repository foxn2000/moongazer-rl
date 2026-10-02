"""Version-specific glue for the TRL ``GRPOTrainer`` (and Unsloth's patched copy of it).

Everything that depends on TRL/Unsloth *internals* lives behind ``select_shim()`` so a
version bump is one new module plus one entry in ``_SHIMS``, never a diff scattered through
the backend.

What to re-check when TRL or Unsloth move (each item has a matching ``# COMPAT(...)``
comment in the shim module):

1. ``GRPOTrainer._generate_single_turn(prompts, images)`` still returns
   ``(prompt_ids, completion_ids, logprobs, forward_kwargs)`` and is still the only place
   generation happens.
2. ``_generate_and_score_completions`` still returns a dict with ``prompt_ids``,
   ``prompt_mask``, ``completion_ids``, ``completion_mask``, ``advantages``,
   ``num_items_in_batch`` (+ ``old_per_token_logps`` / ``ref_per_token_logps`` /
   ``importance_sampling_ratio``), and still computes advantages over
   ``num_generations`` consecutive rows.
3. ``self.use_vllm and self.vllm_importance_sampling_correction`` is still what turns on
   ``old_per_token_logps`` + truncated importance sampling in the loss.
4. Unsloth still reads ``inputs["tool_mask"]`` in its patched ``compute_loss`` and still
   passes it to ``unsloth_zoo.rl_replacements.grpo_accumulated_loss``.
5. ``_compute_loss`` still builds the model attention mask as
   ``cat([prompt_mask, completion_mask])`` and calls
   ``self._get_per_token_logps_and_entropies(model, input_ids, attention_mask, ...)``.
6. ``GRPOConfig.__post_init__`` still derives ``generation_batch_size`` /
   ``steps_per_generation`` the same way.
7. The private ``Trainer`` internals listed in ``SUPPORTED_TRANSFORMERS`` still exist and still
   mean the same thing.
"""

from __future__ import annotations

import os
from types import ModuleType

SUPPORTED_TRL: dict[str, str] = {
    "0.24": "moongazer_rl.trainer.unsloth_native.compat.trl_0_24",
}
"""``major.minor`` of ``trl.__version__`` -> shim module."""

SUPPORTED_TRANSFORMERS = (">=5.2", "<=5.16.1")
"""Inclusive version window for ``transformers``, as (min, max) for humans; enforced below.

The backend replays ``Trainer``'s own optimizer step, so it calls private methods that only
exist in transformers 5.x:

* ``Trainer._clip_grad_norm`` / ``Trainer._get_grad_norm`` (``trainer.py:2489-2504``) — absent
  in 4.x, where the loop inlined ``accelerator.clip_grad_norm_``;
* ``Trainer.current_gradient_accumulation_steps`` (``trainer.py:1705``), which GRPO's
  ``_compute_loss`` divides by;
* the sync-step block we mirror (``trainer.py:1753-1772``) and the
  ``accelerator.prepare(optimizer)`` -> ``create_scheduler`` ordering (``trainer.py:1600-1605``).

Upper bound: 5.5.0 is what every line number above was read from and
what ``uv sync`` resolves (unsloth 2026.9.2 pins ``transformers<=5.5.0``). Every internal above
is also present in transformers 5.16.1 (``trainer.py:1320`` / ``:1739`` / ``:2544-2550``), so
the window extends to it."""

_TRANSFORMERS_MIN = "5.2"
_TRANSFORMERS_MAX = "5.16.1"

ALLOW_UNSUPPORTED_ENV = "MOONGAZER_RL_ALLOW_UNSUPPORTED_TRL"
"""Set to ``1`` to proceed on an unverified TRL **or** transformers version. Untested; loud."""


_unsloth_import_done = False


def ensure_unsloth_first() -> bool:
    """Import ``unsloth`` before anything imports ``transformers`` or ``trl``.

    Unsloth patches transformers/trl at import time and its own loader asserts it comes first;
    imported afterwards, some patches simply do not land. The backend touches
    ``trl.GRPOConfig`` in ``validate_args`` — long before ``setup()`` reaches
    ``_load_unsloth()`` — so the ordering has to be enforced at every entry point that can
    reach TRL, not just in ``setup()``.

    A no-op wherever unsloth is not installed (``find_spec`` is ``None``), which is every
    non-Linux machine, so it costs one ``find_spec`` there. Returns whether unsloth is imported.
    """
    global _unsloth_import_done
    import sys

    if _unsloth_import_done:
        return "unsloth" in sys.modules
    if "unsloth" in sys.modules:
        _unsloth_import_done = True
        return True

    import importlib
    import importlib.util

    try:
        spec = importlib.util.find_spec("unsloth")
    except (ImportError, ValueError):  # pragma: no cover - broken/partial install
        _unsloth_import_done = True
        return False
    if spec is None:
        _unsloth_import_done = True
        return False

    # Installed: import it now, before trl/transformers get a chance. Failures are loud —
    # a broken unsloth on a GPU box must not silently fall back to the plain path.
    importlib.import_module("unsloth")
    _unsloth_import_done = True
    return True


def ensure_trl_importable() -> None:
    """Make ``from trl import GRPOTrainer`` work on transformers 5.x.

    COMPAT(trl==0.24, transformers==5.5): ``trl/import_utils.py:29-42`` calls
    ``transformers.utils.import_utils._is_package_available(pkg)``, which in transformers 5.x
    always returns a ``(bool, version)`` **tuple** — so every ``trl.import_utils.is_*_available()``
    is truthy. The first casualty is ``trl/mergekit_utils.py:21``: ``if is_mergekit_available():
    from mergekit.config import ...`` runs unconditionally, and since
    ``trl/trainer/callbacks.py:40`` imports that module, ``trl.trainer.grpo_trainer`` cannot be
    imported at all without ``mergekit`` installed. Coercing the cached flags back to bools fixes
    that and every other mis-detected optional dependency in one place.

    Remove when TRL ships a release whose ``import_utils`` unpacks the tuple (or when the project
    pins transformers < 5). Idempotent, and a no-op on a TRL that already stores bools.
    """
    ensure_unsloth_first()
    import trl.import_utils as import_utils

    for name, value in list(vars(import_utils).items()):
        if name.startswith("_") and name.endswith("_available") and isinstance(value, tuple):
            setattr(import_utils, name, bool(value[0]))


def transformers_version() -> str:
    ensure_unsloth_first()
    import transformers

    return str(transformers.__version__)


def ensure_supported_transformers(version: str | None = None) -> None:
    """Refuse a transformers whose private ``Trainer`` internals we have not verified.

    See ``SUPPORTED_TRANSFORMERS`` for exactly which internals and why.
    """
    from packaging.version import InvalidVersion, Version

    version = version or transformers_version()
    try:
        parsed = Version(version)
    except InvalidVersion:  # pragma: no cover - a dev build; let it through, loudly
        import warnings

        warnings.warn(
            f"could not parse transformers version {version!r}; "
            f"unsloth-native expects {' '.join(SUPPORTED_TRANSFORMERS)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if Version(_TRANSFORMERS_MIN) <= parsed <= Version(_TRANSFORMERS_MAX):
        return

    message = (
        f"unsloth-native supports transformers {' '.join(SUPPORTED_TRANSFORMERS)} "
        f"(found {version}). It replays Trainer's own optimizer step and calls private 5.x "
        "internals — Trainer._clip_grad_norm / _get_grad_norm (trainer.py:2489-2504), "
        "current_gradient_accumulation_steps (trainer.py:1705), and the sync-step / "
        "accelerator.prepare ordering (trainer.py:1753-1772, 1600-1605) — none of which exist "
        "or behave the same in 4.x. Pin transformers, or verify those call sites and widen "
        f"SUPPORTED_TRANSFORMERS. {ALLOW_UNSUPPORTED_ENV}=1 proceeds anyway (untested)."
    )
    if os.environ.get(ALLOW_UNSUPPORTED_ENV) == "1":
        import warnings

        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return
    raise RuntimeError(message)


def trl_version() -> str:
    import trl

    return str(trl.__version__)


def select_shim(version: str | None = None) -> ModuleType:
    """Return the compat module for the installed (or given) TRL version."""
    from importlib import import_module

    ensure_unsloth_first()
    ensure_supported_transformers()
    ensure_trl_importable()
    version = version or trl_version()
    key = ".".join(version.split(".")[:2])
    target = SUPPORTED_TRL.get(key)
    if target is None:
        newest = SUPPORTED_TRL[max(SUPPORTED_TRL)]
        if os.environ.get(ALLOW_UNSUPPORTED_ENV) == "1":
            import warnings

            warnings.warn(
                f"trl=={version} is not a version this backend was verified against; "
                f"falling back to {newest} because {ALLOW_UNSUPPORTED_ENV}=1. "
                "Re-check the six invariants in trainer/unsloth_native/compat/__init__.py.",
                RuntimeWarning,
                stacklevel=2,
            )
            return import_module(newest)
        raise RuntimeError(
            f"unsloth-native supports trl {sorted(SUPPORTED_TRL)} (found {version}). "
            "The backend overrides GRPOTrainer internals, so a different minor version is "
            "refused rather than silently mis-training. Either pin trl, add a shim module "
            f"next to {newest}, or set {ALLOW_UNSUPPORTED_ENV}=1 to try the newest shim "
            "(untested, check the invariants listed in this module's docstring first)."
        )
    return import_module(target)


__all__ = [
    "ALLOW_UNSUPPORTED_ENV",
    "SUPPORTED_TRANSFORMERS",
    "SUPPORTED_TRL",
    "ensure_supported_transformers",
    "ensure_unsloth_first",
    "ensure_trl_importable",
    "select_shim",
    "transformers_version",
    "trl_version",
]
