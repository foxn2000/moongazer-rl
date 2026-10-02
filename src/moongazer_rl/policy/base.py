"""PolicyBackend contract.

A policy backend owns *generation only*: token IDs in, token IDs + sampling logprobs out.
Rendering messages to tokens and parsing completions back to messages is the job of a
``renderers.Renderer`` (Prime Intellect's ``renderers`` package, a verifiers dependency), so
this module never sees chat templates or text.

Implementations:

* ``moongazer_rl.policy.hf.TransformersPolicy``: ``model.generate`` with scores on the learner's
  own weights (in-process fallback sampler).
* ``moongazer_rl.policy.vllm_server.VllmServerPolicy``: an external ``vllm serve`` process.

The trainer backend returns the policy in use from ``TrainerBackend.policy()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from moongazer_rl.rollout.schema import FinishReason


@dataclass(frozen=True)
class SamplingParams:
    """Sampling knobs forwarded to the engine. Unknown engine-specific keys go in ``extra``."""

    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SamplingParams:
        data = dict(raw)
        known = {
            k: data.pop(k)
            for k in ("temperature", "top_p", "top_k", "max_tokens", "seed")
            if k in data
        }
        stop = tuple(data.pop("stop_token_ids", ()) or ())
        if "max_completion_tokens" in data and "max_tokens" not in known:
            known["max_tokens"] = data.pop("max_completion_tokens")
        return cls(stop_token_ids=stop, extra=data, **known)


@dataclass(frozen=True)
class GenerateRequest:
    prompt_ids: list[int]
    sampling: SamplingParams
    session_id: str | None = None
    """verifiers trace id: identical for every call of one rollout."""
    request_id: str | None = None
    policy_version: int | None = None
    """Adapter version this call must be sampled with (asynchronous path: the episode's
    lease). ``None`` means "whatever the policy currently serves". A policy that serves a
    single version must raise ``ValueError`` when asked for another one."""


@dataclass(frozen=True)
class GenerateResult:
    completion_ids: list[int]
    logprobs: list[float]
    """Sampling logprob per completion token. Same length as ``completion_ids``; a backend
    that cannot produce them must return ``[]`` and the runtime records the gap."""
    finish_reason: FinishReason
    policy_version: int
    request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PolicyBackend(Protocol):
    """Generation-only policy. Implementations must be safe to call concurrently from asyncio."""

    @property
    def name(self) -> str: ...

    @property
    def weight_version(self) -> int:
        """Monotonic version of the weights currently served. Bumped by ``sync_weights``."""
        ...

    async def generate(self, request: GenerateRequest) -> GenerateResult: ...

    async def sync_weights(self, version: int) -> None:
        """Make weights of ``version`` visible to generation (no-op if the policy shares
        the learner's parameters in-process)."""
        ...

    async def close(self) -> None: ...


def resolve_sampling_temperature(sampling: Any, default: float = 1.0) -> float:
    """The temperature the engine will actually sample at, from a verifiers ``SamplingConfig``
    (or the plain mapping a config file produced).

    Mirrors ``PolicyClient._sampling_params``: ``extra_body`` is unpacked first and the typed
    fields are applied on top, so a top-level ``temperature`` wins over
    ``extra_body.temperature``. Trainers need the same answer as the sampler — the learner
    divides its logits by it — hence one shared helper rather than two readings of the config.

    Accepts a mapping or any object with the attributes; missing/``None`` values fall through
    to ``default``.
    """
    top_level = _lookup(sampling, "temperature")
    if top_level is not None:
        return float(top_level)
    nested = _lookup(_lookup(sampling, "extra_body"), "temperature")
    if nested is not None:
        return float(nested)
    return float(default)


def _lookup(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


CallUnit = Literal["sequence", "call"]
"""Which canonical unit a trainer consumes: verifiers Branch sequences or raw call segments."""

__all__ = [
    "CallUnit",
    "GenerateRequest",
    "GenerateResult",
    "PolicyBackend",
    "SamplingParams",
    "resolve_sampling_temperature",
]
