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
"""Derived from verifiers 0.2.0 ``verifiers/v1/harnesses/null/harness.py`` (full copy, 2026-09-08).
Differences:
  * ``NullPatientHarnessConfig`` adds ``timeout_s`` (default None = no client-side deadline) and
    ``max_retries`` (default 0), forwarded to the sibling ``program.py`` as argv.
  * ``PROGRAM_SOURCE`` reads the sibling (derived) program.
Use as ``harness: {id: null_patient, timeout_s: 7200}`` (the flat ``null_patient`` top-level module
re-exports this plugin because verifiers cannot resolve dotted local ids)."""
import json
from pathlib import Path

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class NullPatientHarnessConfig(HarnessConfig):
    timeout_s: float | None = None
    """Client-side deadline of one model call in the harness program (None = wait forever; the
    policy's own ``timeout_s`` still bounds the vLLM request)."""
    max_retries: int = 0
    """OpenAI SDK retries per model call. Keep 0 for RL: a retried call is a second sample."""


class NullPatientHarness(Harness[NullPatientHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--max-retries={self.config.max_retries}",
        ]
        if self.config.timeout_s is not None:
            args.append(f"--timeout-s={self.config.timeout_s}")
        if system_prompt:
            args.append(f"--system-prompt={system_prompt}")
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            args.append(
                "--mcp-config="
                + json.dumps(
                    {
                        "mcpServers": {
                            name: {"url": url} for name, url in mcp_urls.items()
                        }
                    }
                )
            )
        # A Messages prompt (e.g. an image-bearing prompt) seeds the chat loop directly via env
        # (it can be large multimodal content that overflows argv); a plain string is the single
        # first user message; None means the task has no prompt and the user simulator opens it.
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            env["INITIAL_MESSAGES"] = json.dumps([message_to_wire(m) for m in prompt])
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program([*program, *args], env)
