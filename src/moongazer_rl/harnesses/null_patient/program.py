# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "mcp"]
# ///
# Portions of this file are derived from verifiers 0.2.0
# (https://github.com/PrimeIntellect-ai/verifiers), licensed under the MIT License:
#
# Copyright (c) 2026 Prime Intellect
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Derived from verifiers 0.2.0 ``verifiers/v1/harnesses/null/program.py`` (full copy, 2026-09-08).
Differences:
  * ``--timeout-s`` / ``--max-retries`` argv options feed
    ``AsyncOpenAI(timeout=..., max_retries=...)``.
    The stock program uses the SDK defaults (600 s, 2 retries): a thinking completion that takes
    longer than 600 s is silently re-requested, the first sample still
    finishes on the server, and the trajectory ends up with two independent samples of the same
    prompt sharing one reward.
The interception endpoint and secret arrive through argv rather than the environment."""

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack

from openai import AsyncOpenAI


async def chat(
    client: AsyncOpenAI, model: str, messages: list[dict], tools: list[dict]
):
    completion = await client.chat.completions.create(
        model=model, messages=messages, tools=tools or None
    )
    return completion.choices[0].message


async def connect_mcp(stack: AsyncExitStack, config: dict) -> tuple[list[dict], dict]:
    """Connect to each configured MCP server (a streamable-HTTP `url`); return
    (tool schemas, dispatch mapping `<server>_<tool>` -> (session, raw tool name))."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    tool_schemas: list[dict] = []
    dispatch: dict[str, tuple] = {}
    for name, spec in config.get("mcpServers", {}).items():
        http_client = await stack.enter_async_context(
            create_mcp_http_client(headers=spec.get("headers") or None)
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(spec["url"], http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            full = f"{name}_{tool.name}"
            tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": full,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
            dispatch[full] = (session, tool.name)
    return tool_schemas, dispatch


def mcp_content_to_chat_content(blocks) -> str | list[dict]:
    parts = []
    for block in blocks:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            url = f"data:{block.mimeType};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            parts.append({"type": "text", "text": str(block)})
    if not parts:
        return str(blocks)
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


async def call_mcp(dispatch: dict, name: str, arguments: dict) -> str | list[dict]:
    session, raw = dispatch[name]
    result = await session.call_tool(raw, arguments)
    return mcp_content_to_chat_content(result.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--mcp-config", default="")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=0)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout_s,
        max_retries=args.max_retries,
    )
    config = json.loads(args.mcp_config or "{}")
    async with AsyncExitStack() as stack:
        tools, dispatch = (
            await connect_mcp(stack, config) if config.get("mcpServers") else ([], {})
        )
        messages = (
            [{"role": "system", "content": args.system_prompt}]
            if args.system_prompt
            else []
        )
        # A Messages prompt (e.g. an image-bearing prompt) arrives pre-built as OpenAI wire dicts
        # via INITIAL_MESSAGES (kept in env: it can be large multimodal content that overflows
        # argv, and it's prompt content, not a credential); otherwise --prompt is the opening
        # message. Both empty means the task has no prompt — the user simulator seeds the opening.
        initial = json.loads(os.environ.get("INITIAL_MESSAGES", "[]"))
        if initial:
            messages.extend(initial)
        elif args.prompt:
            messages.append({"role": "user", "content": args.prompt})
        while True:
            message = await chat(client, args.model, messages, tools)
            messages.append(message.model_dump(exclude_none=True))
            if not message.tool_calls:
                break
            for call in message.tool_calls:
                name = call.function.name
                try:
                    tool_args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON",
                        }
                    )
                    continue
                # Valid JSON can still be a non-object (`[]`, `42`, `null`); the MCP dispatch
                # assumes a dict, so reject anything else as a tool error rather than crashing.
                if not isinstance(tool_args, dict):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": f"error: tool arguments must be a JSON object, got {type(tool_args).__name__}; resend as an object",
                        }
                    )
                    continue
                if name in dispatch:
                    content = await call_mcp(dispatch, name, tool_args)
                else:
                    content = f"error: unknown tool {name!r}"
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )


if __name__ == "__main__":
    asyncio.run(main())
