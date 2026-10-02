"""``PolicyClient``: a ``verifiers.v1`` model client backed by a local ``PolicyBackend``.

Verifiers owns the rollout loop and calls a ``Client`` once per model turn. The upstream
training client (``verifiers.v1.clients.train.TrainClient``) renders the typed prompt to
token IDs with a ``renderers.Renderer`` and POSTs them to a vLLM
``/inference/v1/generate`` engine. This client keeps that path byte for byte and swaps
only the HTTP call for ``PolicyBackend.generate`` — so nothing is ever re-tokenized and
the exact sampled token IDs + logprobs reach the trainer.

Reused verbatim from upstream (never reimplemented here):

* ``verifiers.v1.clients.train.response_from_generate`` — the ``generate`` result dict
  -> typed ``Response`` (including ``TurnTokens`` and the per-message token spans).
* ``verifiers.v1.clients.train.serialize_completion`` — ``Response`` -> the OpenAI
  ``chat.completion`` dict the harness program's SDK expects (``Response.raw``).
* ``verifiers.v1.clients.train.tool_to_wire`` — typed ``Tool`` -> OpenAI tool schema.
* ``verifiers.v1.clients.train._is_valid_incremental_tail`` /
  ``_has_multimodal_content`` — the cheap guards that decide whether the renderer's
  ``bridge_to_next_turn`` may be attempted.
* ``verifiers.v1.dialects.chat.message_to_wire`` / ``parse_tools`` — wire translation.
* The ``renderers.client.generate`` finish-reason rule: promote ``stop`` ->
  ``tool_calls`` when the renderer parsed at least one well-formed tool call.

Every turn is also recorded as a canonical :class:`~moongazer_rl.rollout.schema.ModelCall`
in a :class:`CallStore`, keyed by the verifiers session id (== trace id).
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from collections.abc import Mapping
from typing import Any, cast

from renderers.base import Message, RenderedTokens, Renderer, ToolCallParseStatus, ToolSpec
from verifiers.v1.clients.client import Client
from verifiers.v1.clients.train import (
    _has_multimodal_content,
    _is_valid_incremental_tail,
    response_from_generate,
    serialize_completion,
    tool_to_wire,
)
from verifiers.v1.dialects import ChatDialect, Dialect, parse_tools
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import ProviderError
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, SamplingConfig

from moongazer_rl.policy.base import GenerateRequest, PolicyBackend, SamplingParams
from moongazer_rl.rollout.schema import FinishReason, ModelCall

__all__ = [
    "CallStore",
    "IncompatibleRequestError",
    "MissingLogprobsError",
    "PolicyClient",
    "structural_thinking_leak_as_reasoning",
]


class IncompatibleRequestError(ProviderError):
    """The harness asked for something this trainable client cannot serve losslessly
    (a non-chat dialect, streaming, or ``n > 1``). A ``ProviderError`` so verifiers
    records it on the trace as a model error instead of crashing the rollout."""


class MissingLogprobsError(ProviderError):
    """The policy returned no (or misaligned) sampling logprobs while they were required."""


class CallStore:
    """Per-session ``ModelCall`` collection, filled while verifiers owns the loop.

    Keyed by the verifiers session id (the trace id), which is identical for every turn
    of one rollout. The runtime pops each trace's calls once its episode finishes and
    discards them if the episode blew up, so nothing leaks across steps.
    """

    def __init__(self) -> None:
        self._calls: dict[str, list[ModelCall]] = {}

    def append(self, session_id: str, call: ModelCall) -> None:
        self._calls.setdefault(session_id, []).append(call)

    def count(self, session_id: str) -> int:
        """Calls already recorded for ``session_id`` — the next call's ``index``."""
        return len(self._calls.get(session_id, ()))

    def peek(self, session_id: str) -> list[ModelCall]:
        return list(self._calls.get(session_id, ()))

    def pop(self, session_id: str) -> list[ModelCall]:
        return self._calls.pop(session_id, [])

    def discard(self, session_id: str) -> None:
        self._calls.pop(session_id, None)

    def clear(self) -> None:
        self._calls.clear()

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(self._calls)

    def __len__(self) -> int:
        return len(self._calls)


def _sampling_params(sampling_args: SamplingConfig, stop_token_ids: list[int]) -> SamplingParams:
    """Flatten a verifiers ``SamplingConfig`` (+ its ``extra_body``) into ``SamplingParams``.

    Mirrors ``TrainClient.get_response``: ``extra_body`` is unpacked first so the typed
    fields win, and ``chat_template_kwargs`` is dropped — it is a renderer-construction
    knob, not a sampling knob, so it must never reach the engine.
    """
    raw = sampling_args.model_dump(exclude_none=True)
    flat: dict[str, Any] = dict(raw.pop("extra_body", None) or {})
    flat.pop("chat_template_kwargs", None)
    flat.update(raw)
    flat["stop_token_ids"] = stop_token_ids
    return SamplingParams.from_mapping(flat)


class PolicyClient(Client):
    """Render -> ``PolicyBackend.generate`` -> parse, recording every call losslessly."""

    def __init__(
        self,
        policy: PolicyBackend,
        renderer: Renderer,
        store: CallStore,
        *,
        require_logprobs: bool = True,
        policy_version: int | None = None,
    ) -> None:
        self.policy = policy
        self.renderer = renderer
        self.store = store
        self.require_logprobs = require_logprobs
        self.policy_version = policy_version

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        del headers  # nothing to relay: we generate locally.
        if not isinstance(dialect, ChatDialect):
            raise IncompatibleRequestError(
                "PolicyClient renders a chat template, so it only supports the "
                f"chat-completions dialect, got {type(dialect).__name__}.",
                status_code=400,
            )
        if not session_id:
            raise IncompatibleRequestError(
                "verifiers did not supply a rollout session id; PolicyClient needs it to "
                "attribute model calls to a trace.",
                status_code=400,
            )
        if body.get("stream") is True:
            raise IncompatibleRequestError(
                "streaming model turns are not trainable by this client", status_code=400
            )
        if body.get("n") not in (None, 1):
            raise IncompatibleRequestError(
                "set n=1; verifiers owns rollout-group sampling", status_code=400
            )

        # Intercepted turns already own the typed prompt; only their tools need parsing.
        if turn is not None:
            prompt = turn.prompt
            tools = parse_tools(body.get("tools"))
        else:
            prompt, tools = dialect.parse_request(body)

        # `tool_to_wire` / `message_to_wire` produce exactly the OpenAI shapes the renderer
        # protocol names as `ToolSpec` / `Message`; both are plain TypedDicts, so this is a
        # static-only cast with no runtime conversion (and no copy of the token payload).
        wire_tools = cast(
            "list[ToolSpec] | None", [tool_to_wire(t) for t in tools] if tools else None
        )
        wire_messages = cast(
            "list[Message]", [message_to_wire(m) for m in turn.tail] if turn is not None else []
        )

        prompt_ids: list[int] | None = None
        prompt_attribution: RenderedTokens | None = None
        bridged_turn: PendingTurn | None = None

        # Only build the O(context) previous-turn ids once the cheap guards pass, exactly
        # like TrainClient: multimodal prompts and non-`[tool*, user?]` tails cannot bridge.
        can_bridge = (
            turn is not None
            and not _has_multimodal_content(prompt)
            and _is_valid_incremental_tail(cast("list[dict[str, Any]]", wire_messages))
        )
        previous_ids = turn.previous_token_ids() if can_bridge and turn is not None else None
        if previous_ids is not None:
            previous_prompt_ids, previous_completion_ids = previous_ids
            bridged = self.renderer.bridge_to_next_turn(
                previous_prompt_ids,
                previous_completion_ids,
                wire_messages,
                tools=wire_tools,
            )
            if bridged is not None:
                prompt_ids = bridged.token_ids
                prompt_attribution = bridged
                bridged_turn = turn

        if prompt_ids is None:
            # Bridging failed (or was impossible): re-render the whole typed prompt. A
            # prior assistant turn that re-tokenizes differently makes verifiers fork the
            # graph, which is why the trainer consumes branches rather than one sequence.
            wire_messages = cast("list[Message]", [message_to_wire(m) for m in prompt])
            rendered = self.renderer.render(
                wire_messages, tools=wire_tools, add_generation_prompt=True
            )
            prompt_ids = rendered.token_ids
            prompt_attribution = rendered

        sampling = _sampling_params(sampling_args, self.renderer.get_stop_token_ids())
        request_id = uuid.uuid4().hex
        started_at = time.time()
        result = await self.policy.generate(
            GenerateRequest(
                prompt_ids=prompt_ids,
                sampling=sampling,
                session_id=session_id,
                request_id=request_id,
                policy_version=self.policy_version,
            )
        )
        ended_at = time.time()

        completion_ids = result.completion_ids
        logprobs = self._check_logprobs(result.logprobs, completion_ids)

        parsed = self.renderer.parse_response(completion_ids, tools=wire_tools)
        finish_reason: FinishReason = result.finish_reason
        parsed, thinking_leak = structural_thinking_leak_as_reasoning(
            parsed, completion_ids, self.renderer
        )
        parsed = unfinished_thinking_as_reasoning(parsed, finish_reason, self.renderer)
        ok_tool_calls = [tc for tc in parsed.tool_calls if tc.status == ToolCallParseStatus.OK]
        if ok_tool_calls and finish_reason == "stop":
            # renderers.client.generate's rule: an engine reports stop/length, never
            # "tool_calls", so promote it client-side or OpenAI-shaped agent loops end early.
            finish_reason = "tool_calls"

        generate_result: dict[str, Any] = {
            "request_id": result.request_id or request_id,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "completion_logprobs": logprobs,
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": parsed.tool_calls,
            "finish_reason": finish_reason,
            "routed_experts": None,
            "multi_modal_data": None,
            "prompt_attribution": prompt_attribution,
        }
        response = response_from_generate(generate_result, model, bridged_turn)
        # We generated, so there is no provider response to relay: synthesize the
        # chat.completion the interception server hands back to the harness program.
        response.raw = serialize_completion(response, model)

        call_extra = dict(result.extra)
        call_extra["thinking_leak"] = thinking_leak
        self.store.append(
            session_id,
            ModelCall(
                index=self.store.count(session_id),
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                sampling_logprobs=logprobs,
                finish_reason=finish_reason,
                policy_version=result.policy_version,
                started_at=started_at,
                ended_at=ended_at,
                request_id=result.request_id or request_id,
                extra=call_extra,
            ),
        )
        return response

    def _check_logprobs(self, logprobs: list[float], completion_ids: list[int]) -> list[float]:
        if not logprobs:
            if self.require_logprobs and completion_ids:
                raise MissingLogprobsError(
                    f"policy {self.policy.name!r} returned no sampling logprobs for "
                    f"{len(completion_ids)} completion tokens, but require_logprobs=True. "
                    "Use a backend that reports them, or set require_logprobs=False to "
                    "record the gap instead.",
                    status_code=400,
                )
            return []
        if len(logprobs) != len(completion_ids):
            raise MissingLogprobsError(
                f"policy {self.policy.name!r} returned {len(logprobs)} sampling logprobs "
                f"for {len(completion_ids)} completion tokens; they must align 1:1.",
                status_code=400,
            )
        return logprobs

    async def close(self) -> None:
        # The runtime owns the policy's lifecycle (it may be the trainer's own weights).
        return None


def structural_thinking_leak_as_reasoning(
    parsed: Any, completion_ids: list[int], renderer: Renderer
) -> tuple[Any, dict[str, bool | str]]:
    """Keep structurally malformed Gemma 4 thinking out of the graded answer.

    Gemma 4's renderer exposes the token IDs it uses for ``<|channel>`` and
    ``<channel|>`` as ``_channel_start`` and ``_channel_end``. Its parser splits at the
    first closing marker, so a second close, or a new channel opener after that close,
    lands in parsed ``content``. Once that happens the first boundary cannot be trusted, so
    move the whole parsed answer to reasoning instead of guessing which fragment is safe to
    grade.

    Sampled token IDs and logprobs are not touched; this changes only the parsed message
    seen by the environment.
    """
    metadata: dict[str, bool | str] = {"detected": False, "reason": "none"}
    if not getattr(getattr(renderer, "config", None), "enable_thinking", False):
        return parsed, metadata

    channel_start = getattr(renderer, "_channel_start", None)
    channel_end = getattr(renderer, "_channel_end", None)
    if not isinstance(channel_start, int) or not isinstance(channel_end, int):
        return parsed, metadata

    end_positions = [i for i, token_id in enumerate(completion_ids) if token_id == channel_end]
    reasons: list[str] = []
    if len(end_positions) >= 2:
        reasons.append("thinking_channel_end_repeated")
    if end_positions and channel_start in completion_ids[end_positions[0] + 1 :]:
        reasons.append("thinking_channel_start_in_answer")
    if not reasons:
        return parsed, metadata

    metadata = {"detected": True, "reason": ",".join(reasons)}
    content = getattr(parsed, "content", None)
    if not content:
        return parsed, metadata
    prior_reasoning = getattr(parsed, "reasoning_content", None)
    reasoning = f"{prior_reasoning}\n{content}" if prior_reasoning else content
    return dataclasses.replace(parsed, content="", reasoning_content=reasoning), metadata


def unfinished_thinking_as_reasoning(
    parsed: Any, finish_reason: str | None, renderer: Renderer
) -> Any:
    """A length-truncated completion whose thinking never closed is reasoning, not an answer.

    Thinking renderers (Qwen3 / Qwen3.5 / Gemma 4 with ``enable_thinking``) put the opening
    think marker in the *generation prompt*, so the sampled ids carry only the thoughts and,
    when the cap hits first, no closing marker either. ``renderers`` parses such a completion
    as plain ``content`` (no marker in the completion -> nothing to split;
    renderers/parsing.py parse_qwen35), the environment then grades the unfinished thoughts as
    the final reply and the reference judge is asked about them. Recognise the case by
    its signature - finish_reason ``length``, no ``reasoning_content`` parsed, renderer config
    with thinking on - and move the text to ``reasoning_content``. Token ids and logprobs are
    untouched, so training rows do not change; only the graded ``content`` does."""
    if finish_reason != "length" or getattr(parsed, "reasoning_content", None) is not None:
        return parsed
    if getattr(renderer, "parses_thinking_by_tokens", False):
        # The renderer separates thinking from the answer on token structure (Harmony channels):
        # a capped completion with no reasoning_content really went straight to its answer.
        return parsed
    if getattr(parsed, "tool_calls", None):
        return parsed
    if not getattr(getattr(renderer, "config", None), "enable_thinking", False):
        return parsed
    content = getattr(parsed, "content", None)
    if not content:
        return parsed
    return dataclasses.replace(parsed, content="", reasoning_content=content)
