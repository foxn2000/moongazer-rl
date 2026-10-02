"""Renderer for LLM-jp-4 instruct / thinking checkpoints (OpenAI Harmony format, own tokenizer).

LLM-jp-4 (`llm-jp/llm-jp-4-8b-thinking`, `-32b-a3b-thinking`, `-*-instruct`) answers in Harmony:

    <|start|>assistant<|channel|>analysis<|message|>{thoughts}<|end|>
    <|start|>assistant<|channel|>final<|message|>{answer}<|return|>

with its own special-token ids (`<|start|>`=10, `<|channel|>`=9, `<|message|>`=12, `<|end|>`=11,
`<|return|>`=2 = eos, `<|call|>`=13, `<|constrain|>`=8 on the 8B tokenizer). The ``renderers``
package's ``gpt-oss`` renderer cannot be used: it renders message bodies with ``openai_harmony``,
i.e. with the gpt-oss o200k vocabulary, not with this tokenizer. The stock ``DefaultRenderer``
renders the prompt correctly (the bundled chat template is the Harmony template, including the
``reasoning_effort`` kwarg) but its text-based reasoning parser only knows ``<think>``.

This renderer keeps ``DefaultRenderer``'s rendering and replaces parsing with a token-id walk of the
Harmony structure, the same lexical analysis as ``llmjp4_harmony.HarmonyMessageParser`` bundled with
the checkpoints and used by the llm-jp-4-cookbook vLLM reasoning parser
(https://github.com/llm-jp/llm-jp-4-cookbook, ``llmjp4_vllm/llmjp4_harmony.py``):

* ``analysis`` channel  -> ``reasoning_content``
* ``final`` channel     -> ``content``
* ``commentary`` to a ``functions.*`` recipient -> a tool call (JSON arguments); other
  commentary -> content
* a completion that stops inside ``analysis`` (cap hit before ``final``) has empty ``content`` and
  its thoughts as ``reasoning_content`` — the environment must not grade unfinished thoughts.

Stop tokens are ``<|return|>`` (end of turn) and ``<|call|>`` (tool call); ``<|end|>`` is a message
separator, not a stop. Config: ``model.renderer: {name: llm-jp-4, reasoning_effort: medium}`` —
every other key is forwarded to ``apply_chat_template`` like ``DefaultRendererConfig`` does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from renderers.base import ParsedResponse, ParsedToolCall, ToolCallParseStatus, ToolSpec
from renderers.configs import DefaultRendererConfig
from renderers.default import DefaultRenderer

RENDERER_NAME = "llm-jp-4"

_SPECIALS = (
    "<|start|>",
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
)


@dataclass(frozen=True)
class HarmonyMessage:
    role: list[int] | None
    channel: list[int] | None
    constrain: list[int] | None
    content: list[int] | None
    end: str  # "end" | "return" | "call" | "incomplete"


def harmony_token_ids(tokenizer: Any) -> dict[str, int]:
    """The special-token ids this tokenizer uses for the Harmony markers (fails if absent)."""
    ids: dict[str, int] = {}
    unk = getattr(tokenizer, "unk_token_id", None)
    for token in _SPECIALS:
        tid = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(tid, int) or tid is None or (unk is not None and tid == unk):
            raise ValueError(
                f"tokenizer has no special token {token!r}; not a Harmony (LLM-jp-4) tokenizer"
            )
        ids[token] = tid
    return ids


def iter_harmony_messages(token_ids: list[int], ids: dict[str, int]) -> list[HarmonyMessage]:
    """Lexical walk over Harmony tokens (port of llmjp4_harmony.HarmonyMessageParser)."""
    begin = {
        ids["<|start|>"]: "role",
        ids["<|channel|>"]: "channel",
        ids["<|constrain|>"]: "constrain",
        ids["<|message|>"]: "content",
    }
    end = {ids["<|end|>"]: "end", ids["<|return|>"]: "return", ids["<|call|>"]: "call"}
    messages: list[HarmonyMessage] = []
    fields: dict[str, list[int]] = {}
    section: str | None = None
    text: list[int] = []
    for tid in token_ids:
        if tid in begin:
            if section is not None:
                fields[section] = text
            section = begin[tid]
            text = []
        elif tid in end:
            if section is not None:
                fields[section] = text
            messages.append(_message(fields, end[tid]))
            fields, section, text = {}, None, []
        elif section is not None:
            text.append(tid)
    if section is not None:
        fields[section] = text
        messages.append(_message(fields, "incomplete"))
    return messages


def _message(fields: dict[str, list[int]], end: str) -> HarmonyMessage:
    return HarmonyMessage(
        role=fields.get("role"),
        channel=fields.get("channel"),
        constrain=fields.get("constrain"),
        content=fields.get("content"),
        end=end,
    )


class LlmJp4Renderer(DefaultRenderer):
    """``DefaultRenderer`` rendering + token-level Harmony parsing for LLM-jp-4."""

    def __init__(self, tokenizer: Any, config: DefaultRendererConfig | None = None) -> None:
        cfg = config or DefaultRendererConfig()
        if cfg.reasoning_parser is not None or cfg.tool_parser is not None:
            raise ValueError(
                "llm-jp-4 renderer parses Harmony itself; drop reasoning_parser / tool_parser"
            )
        super().__init__(tokenizer, cfg)
        self._ids = harmony_token_ids(tokenizer)
        # ``unfinished_thinking_as_reasoning`` (rollout/client.py) keys on this attribute; LLM-jp-4
        # thinking checkpoints always reason first, so a capped completion is never an answer.
        # DefaultRendererConfig is frozen: expose a view with the flag set.
        self.config = _ConfigView(cfg, enable_thinking=True)

    # The rollout client's generic "capped completion with no parsed reasoning is reasoning" rule
    # (rollout/client.py) exists for renderers that only see text; this renderer already knows from
    # the token structure whether the cap fell inside analysis or inside the final channel.
    parses_thinking_by_tokens = True

    @property
    def supports_tools(self) -> bool:
        return True

    def _apply(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        # The Harmony template does `... in message.content` (a None content raises TypeError) and
        # reads earlier thinking from `message.thinking`; OpenAI-shaped history carries it as
        # `reasoning_content` and uses None for tool-call-only turns.
        fixed = []
        for message in messages:
            m = dict(message)
            if m.get("content") is None:
                m["content"] = ""
            if (
                m.get("role") == "assistant"
                and m.get("reasoning_content")
                and not m.get("thinking")
            ):
                m["thinking"] = m["reasoning_content"]
            fixed.append(m)
        return super()._apply(fixed, tools=tools, add_generation_prompt=add_generation_prompt)

    def get_stop_token_ids(self) -> list[int]:
        return [self._ids["<|return|>"], self._ids["<|call|>"]]

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 - arguments are JSON, schema not needed
    ) -> ParsedResponse:
        # The generation prompt ends with ``<|start|>assistant``; the sampled ids continue with
        # ``<|channel|>...``. Prepend a start marker so the first message parses like the others.
        ids = self._ids
        messages = iter_harmony_messages([ids["<|start|>"], *token_ids], ids)
        reasoning: list[str] = []
        content: list[str] = []
        tool_calls: list[ParsedToolCall] = []
        for message in messages:
            if message.end not in ("end", "incomplete") and not (
                message.channel or message.content
            ):
                # a bare terminator right after the prefill: nothing sampled
                break
            channel = self._decode(message.channel).strip() if message.channel else ""
            recipient = None
            if " to=" in channel:
                channel, _, recipient = channel.partition(" to=")
            elif message.role and " to=" in self._decode(message.role):
                recipient = self._decode(message.role).split(" to=", 1)[1].strip()
            body = self._decode(message.content) if message.content else ""
            if channel.startswith("analysis"):
                reasoning.append(body)
            elif channel.startswith("final"):
                content.append(body)
            elif (
                channel.startswith("commentary")
                and recipient
                and recipient.startswith("functions.")
            ):
                name = recipient[len("functions.") :].strip()
                try:
                    arguments: Any = json.loads(body or "{}")
                    status = ToolCallParseStatus.OK
                except json.JSONDecodeError:
                    arguments, status = None, ToolCallParseStatus.INVALID_JSON
                tool_calls.append(
                    ParsedToolCall(raw=body, name=name, arguments=arguments, status=status)
                )
            else:
                content.append(body)
            if message.end == "return":
                # <|return|> closes the turn; anything after it (only possible when the engine did
                # not stop on it) is not part of this response. <|call|> only closes one tool call
                # and a turn may carry several, so it does not end the walk.
                break
        return ParsedResponse(
            content="".join(content).strip(),
            reasoning_content="".join(reasoning).strip() or None,
            tool_calls=tool_calls,
        )

    def bridge_to_next_turn(self, *args: Any, **kwargs: Any) -> Any:
        # Multi-turn Harmony re-rendering drops analysis messages of earlier turns (the template
        # does the same), so the client re-renders the whole prompt: no incremental bridge.
        return None

    def _decode(self, ids: list[int] | None) -> str:
        if not ids:
            return ""
        return self._tokenizer.decode(ids, skip_special_tokens=False)


class _ConfigView:
    """Frozen-config view that adds ``enable_thinking`` for the rollout client's truncation rule."""

    def __init__(self, config: DefaultRendererConfig, **extra: Any) -> None:
        self._config = config
        self._extra = extra

    def __getattr__(self, name: str) -> Any:
        if name in self._extra:
            return self._extra[name]
        return getattr(self._config, name)


def build_llmjp4_renderer(tokenizer: Any, raw: dict[str, Any]) -> LlmJp4Renderer:
    """``model.renderer`` -> renderer. ``name`` is dropped; the rest are template kwargs
    (``reasoning_effort``, ...), validated by ``DefaultRendererConfig`` (``extra="allow"``)."""
    rest = {k: v for k, v in raw.items() if k != "name"}
    return LlmJp4Renderer(tokenizer, DefaultRendererConfig(**rest))


__all__ = [
    "RENDERER_NAME",
    "HarmonyMessage",
    "LlmJp4Renderer",
    "build_llmjp4_renderer",
    "harmony_token_ids",
    "iter_harmony_messages",
]
