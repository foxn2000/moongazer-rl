"""``PolicyBackend`` over an external vLLM OpenAI-compatible server (separate process/venv).

Why a separate server: Unsloth's in-process vLLM (``fast_generate``) needs a vllm wheel built
against the exact torch/transformers that unsloth pins (torch<2.12, transformers<=5.5 for
unsloth 2026.9.x), and no such wheel exists for recent vllm releases. A standalone
``vllm serve`` in its own venv has no such coupling, batches concurrent rollouts, and hot-loads
LoRA adapters, so the learner exports its PEFT adapter after every optimizer step and the
sampler switches to it.

Token fidelity: the request carries the renderer's prompt *token ids* (``prompt: list[int]``)
and asks for ``return_token_ids`` so the sampled ids come back verbatim; nothing is
re-tokenised. ``logprobs: 0`` returns the sampled token's logprob only.

Logprob convention: the runtime treats ``GenerateResult.logprobs`` as the log-probability of
each sampled token under the *distribution it was sampled from* (``TransformersPolicy`` computes
``log_softmax(logits / temperature)``). vLLM reports that only when the server runs with
``--logprobs-mode processed_logprobs`` (logprobs after temperature and top-k/top-p). The
default ``raw_logprobs`` mode is accepted only for ``temperature == 1`` with no top-k/top-p, in
which case the two coincide.

Server launch::

    VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve <model> --enable-lora \
        --max-lora-rank <r> --logprobs-mode processed_logprobs ...
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from moongazer_rl.policy.base import GenerateRequest, GenerateResult, SamplingParams
from moongazer_rl.policy.hf import NON_GENERATION_EXTRA_KEYS
from moongazer_rl.rollout.schema import FinishReason

LOGPROBS_MODES = ("processed_logprobs", "raw_logprobs")

# Extras that leave the sampled distribution untouched (ignore_eos only changes when generation
# stops). Everything else in _SAMPLING_EXTRA_PASSTHROUGH reshapes the logits and is refused under
# raw_logprobs, where vLLM would report pre-processing logprobs.
_RAW_LOGPROBS_SAFE_EXTRAS = frozenset({"ignore_eos"})

_SAMPLING_EXTRA_PASSTHROUGH = frozenset(
    {
        "min_tokens",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "min_p",
        "ignore_eos",
        "logit_bias",
        "bad_words",
        "allowed_token_ids",
    }
)


def _finish_reason(raw: str | None, completion_ids: list[int], stop_ids: set[int]) -> FinishReason:
    if raw == "length":
        return "length"
    if raw == "stop" or (completion_ids and completion_ids[-1] in stop_ids):
        return "stop"
    return "stop" if raw is None else "length"


class VllmServerPolicy:
    """Generation through ``POST {base_url}/v1/completions`` with token-id prompts.

    ``export_adapter(version) -> Path`` is supplied by the trainer backend; it writes the PEFT
    adapter for ``version`` to disk (the server must see the same filesystem). ``None`` means
    the served weights are never updated (frozen-policy evaluation).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        export_adapter: Callable[[int], Path | str] | None = None,
        weight_version: int = 0,
        logprobs_mode: str = "processed_logprobs",
        timeout_s: float = 3600.0,
        admin_timeout_s: float = 60.0,
        max_connections: int = 128,
        lora_name_prefix: str = "policy",
        keep_adapters: int = 2,
        api_key_var: str | None = None,
        name: str = "vllm-server",
    ) -> None:
        if logprobs_mode not in LOGPROBS_MODES:
            raise ValueError(
                f"logprobs_mode must be one of {LOGPROBS_MODES}, got {logprobs_mode!r}"
            )
        if keep_adapters < 1:
            raise ValueError("keep_adapters must be >= 1")
        self._base_url = base_url.rstrip("/")
        self._base_model = model
        self._export_adapter = export_adapter
        self._weight_version = int(weight_version)
        self._logprobs_mode = logprobs_mode
        self._timeout_s = float(timeout_s)
        if admin_timeout_s <= 0:
            raise ValueError("admin_timeout_s must be > 0")
        self._admin_timeout_s = float(admin_timeout_s)
        self._max_connections = int(max_connections)
        self._lora_name_prefix = lora_name_prefix
        self._keep_adapters = int(keep_adapters)
        self._name = name
        self._headers: dict[str, str] = {}
        if api_key_var:
            key = os.environ.get(api_key_var)
            if not key:
                raise ValueError(f"environment variable {api_key_var} is not set")
            self._headers["Authorization"] = f"Bearer {key}"
        self._served_model = model
        self._loaded_adapters: list[str] = []
        self._adapter_paths: dict[str, Path] = {}
        self._admin_lock = threading.RLock()
        self._async_client: Any = None
        self._closed = False

    # -- PolicyBackend -------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._base_model

    @property
    def weight_version(self) -> int:
        return self._weight_version

    @property
    def served_model(self) -> str:
        """Model name sent with each request: the base model, or the current LoRA adapter."""
        return self._served_model

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        if self._closed:
            raise RuntimeError("VllmServerPolicy is closed")
        if request.policy_version is not None and request.policy_version != self._weight_version:
            raise ValueError(
                f"{self.name} serves weight version {self._weight_version} and cannot sample "
                f"version {request.policy_version}"
            )
        if not request.prompt_ids:
            raise ValueError("GenerateRequest.prompt_ids must not be empty")
        body = self.build_body(request)
        version = self._weight_version
        client = self._client()
        response = await client.post("/v1/completions", json=body, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM /v1/completions returned {response.status_code}: {response.text[:500]}"
            )
        return self.parse_response(response.json(), request, version)

    async def sync_weights(self, version: int) -> None:
        await asyncio.to_thread(self.set_weight_version, version)

    async def close(self) -> None:
        self._closed = True
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # -- request / response --------------------------------------------------------
    def build_body(self, request: GenerateRequest, *, model: str | None = None) -> dict[str, Any]:
        sampling = request.sampling
        self._check_sampling(sampling)
        body: dict[str, Any] = {
            "model": self._served_model if model is None else model,
            "prompt": [int(i) for i in request.prompt_ids],
            "n": 1,
            "stream": False,
            "logprobs": 0,
            "return_token_ids": True,
            "skip_special_tokens": False,
            "temperature": float(sampling.temperature),
        }
        if sampling.max_tokens is not None:
            body["max_tokens"] = int(sampling.max_tokens)
        # Always explicit: vLLM otherwise applies the model's generation_config.json defaults
        # (Gemma 4 ships top_k=64, top_p=0.95), so an omitted knob would silently change the
        # sampling distribution away from what the run config declares.
        body["top_p"] = float(sampling.top_p) if sampling.top_p is not None else 1.0
        body["top_k"] = int(sampling.top_k) if sampling.top_k is not None else -1
        if sampling.seed is not None:
            body["seed"] = int(sampling.seed)
        if sampling.stop_token_ids:
            body["stop_token_ids"] = [int(i) for i in sampling.stop_token_ids]
        # n / reasoning_effort / chat_template_kwargs are applied by the runtime or the renderer
        # (same exclusion as TransformersPolicy); only genuinely unsupported engine knobs fail.
        unknown = sorted(
            k
            for k in sampling.extra
            if k not in _SAMPLING_EXTRA_PASSTHROUGH and k not in NON_GENERATION_EXTRA_KEYS
        )
        if unknown:
            raise ValueError(
                f"vllm-server policy does not forward these sampling keys: {unknown}; "
                f"allowed extras: {sorted(_SAMPLING_EXTRA_PASSTHROUGH)}"
            )
        for key in _SAMPLING_EXTRA_PASSTHROUGH:
            if key in sampling.extra:
                body[key] = sampling.extra[key]
        return body

    def parse_response(
        self,
        payload: dict[str, Any],
        request: GenerateRequest,
        version: int,
        *,
        served_model: str | None = None,
    ) -> GenerateResult:
        choices = payload.get("choices") or []
        if len(choices) != 1:
            raise RuntimeError(f"expected exactly one choice, got {len(choices)}")
        choice = choices[0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise RuntimeError(
                "vLLM response has no token_ids; the server must support return_token_ids "
                "(vllm >= 0.10.2)"
            )
        completion_ids = [int(i) for i in token_ids]
        logprobs_block = choice.get("logprobs") or {}
        raw_logprobs = logprobs_block.get("token_logprobs") or []
        logprobs: list[float] = []
        if len(raw_logprobs) == len(completion_ids) and all(
            isinstance(v, (int, float)) for v in raw_logprobs
        ):
            logprobs = [float(v) for v in raw_logprobs]
        stop_ids = {int(i) for i in request.sampling.stop_token_ids}
        finish = _finish_reason(choice.get("finish_reason"), completion_ids, stop_ids)
        usage = payload.get("usage") or {}
        return GenerateResult(
            completion_ids=completion_ids,
            logprobs=logprobs,
            finish_reason=finish,
            policy_version=version,
            request_id=request.request_id,
            extra={
                "num_prompt_tokens": int(usage.get("prompt_tokens") or len(request.prompt_ids)),
                "served_model": self._served_model if served_model is None else served_model,
                "vllm_finish_reason": choice.get("finish_reason"),
                "vllm_stop_reason": choice.get("stop_reason"),
            },
        )

    def _check_sampling(self, sampling: SamplingParams) -> None:
        if self._logprobs_mode == "processed_logprobs":
            return
        # raw_logprobs: the reported logprob ignores temperature, top-k/top-p and every other
        # logits processor, so it is the sampling distribution only when none of them is active.
        distorting = sorted(
            k
            for k in sampling.extra
            if k not in _RAW_LOGPROBS_SAFE_EXTRAS and k not in NON_GENERATION_EXTRA_KEYS
        )
        if (
            sampling.temperature != 1.0
            or sampling.top_p not in (None, 1.0)
            or sampling.top_k not in (None, 0, -1)
            or distorting
        ):
            raise ValueError(
                "the vLLM server runs with --logprobs-mode raw_logprobs, which reports logprobs "
                "before temperature/top-k/top-p/penalties; use temperature=1.0 without top_p, "
                f"top_k or distribution-changing extras (got {distorting or 'none'}), or start "
                "the server with --logprobs-mode processed_logprobs"
            )

    # -- weight sync ---------------------------------------------------------------
    def set_weight_version(self, version: int) -> None:
        """Export the adapter for ``version`` and make the server sample from it.

        Adapters are registered as ``<prefix>-v<version>``; the newest ``keep_adapters`` stay
        registered so requests still in flight on the previous version keep working."""
        version = int(version)
        with self._admin_lock:
            if self._export_adapter is None:
                self._weight_version = version
                return
            lora_name = f"{self._lora_name_prefix}-v{version}"
            if version == self._weight_version and lora_name in self._loaded_adapters:
                # Already exported and registered by this process (the backend pushes the
                # version and the orchestrator syncs it again): nothing to do.
                return
            path = Path(self._export_adapter(version))
            if not (path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"no adapter_config.json under exported adapter {path}")
            # A previous process (crashed run, resume) may have left the same name registered:
            # always replace it so the server never serves a stale adapter under this name.
            self._admin_post("/v1/unload_lora_adapter", {"lora_name": lora_name}, ok_404=True)
            if lora_name in self._loaded_adapters:
                self._loaded_adapters.remove(lora_name)
            self._admin_post(
                "/v1/load_lora_adapter", {"lora_name": lora_name, "lora_path": str(path.resolve())}
            )
            self._loaded_adapters.append(lora_name)
            self._adapter_paths[lora_name] = path
            self._served_model = lora_name
            self._weight_version = version
            while len(self._loaded_adapters) > self._keep_adapters:
                stale = self._loaded_adapters.pop(0)
                self._admin_post("/v1/unload_lora_adapter", {"lora_name": stale}, ok_404=True)
                # The exported directory is only a transport to the server; once unloaded it
                # would otherwise accumulate one copy per optimizer step for the whole run.
                stale_path = self._adapter_paths.pop(stale, None)
                if stale_path is not None and stale_path.is_dir():
                    shutil.rmtree(stale_path, ignore_errors=True)

    def _admin_post(self, route: str, body: dict[str, Any], *, ok_404: bool = False) -> None:
        import httpx

        with httpx.Client(base_url=self._base_url, timeout=self._admin_timeout_s) as client:
            response = client.post(route, json=body, headers=self._headers)
        if response.status_code == 200:
            return
        if ok_404 and response.status_code in (400, 404):
            # vLLM answers "not found" for an unknown adapter with 404 or 400 depending on
            # version; an unload of something that is not loaded is a no-op for us.
            return
        raise RuntimeError(
            f"vLLM {route} returned {response.status_code}: {response.text[:500]} "
            "(is the server started with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True and --enable-lora?)"
        )

    # -- health --------------------------------------------------------------------
    def check_server(self) -> dict[str, Any]:
        """Blocking ``GET /v1/models``; raises if the server is unreachable or the base model is
        not served. Returns the models payload."""
        import httpx

        with httpx.Client(base_url=self._base_url, timeout=self._admin_timeout_s) as client:
            response = client.get("/v1/models", headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM GET /v1/models returned {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
        served = {m.get("id") for m in payload.get("data", [])}
        if self._base_model not in served:
            raise RuntimeError(
                f"vLLM server at {self._base_url} serves {sorted(served)}, not "
                f"{self._base_model!r}; set policy.args.model to the served name or restart "
                "the server with this model"
            )
        return payload

    def _client(self) -> Any:
        if self._async_client is None:
            import httpx

            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s, connect=30.0),
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_connections,
                ),
            )
        return self._async_client


class _SharedAdapterExporter:
    """Cache one adapter export while replicas register the same version."""

    def __init__(self, export_adapter: Callable[[int], Path | str]) -> None:
        self._export_adapter = export_adapter
        self._lock = threading.RLock()
        self._paths: dict[int, Path] = {}

    def __call__(self, version: int) -> Path:
        with self._lock:
            cached = self._paths.get(version)
            if cached is not None and (cached / "adapter_config.json").is_file():
                return cached
            path = Path(self._export_adapter(version))
            self._paths[version] = path
            return path


class VllmServerPolicyGroup:
    """Least-loaded routing over synchronous-path vLLM server replicas."""

    def __init__(
        self,
        base_urls: list[str],
        model: str,
        *,
        export_adapter: Callable[[int], Path | str] | None = None,
        weight_version: int = 0,
        **policy_kwargs: Any,
    ) -> None:
        urls = [url.rstrip("/") for url in base_urls]
        if len(urls) < 2:
            raise ValueError("base_urls must contain at least 2 URLs")
        if any(not url for url in urls):
            raise ValueError("base_urls must contain non-empty URLs")
        if len(set(urls)) != len(urls):
            raise ValueError("base_urls must not contain duplicates")
        shared_export = (
            _SharedAdapterExporter(export_adapter) if export_adapter is not None else None
        )
        self._replicas = [
            VllmServerPolicy(
                url,
                model,
                export_adapter=shared_export,
                weight_version=weight_version,
                **policy_kwargs,
            )
            for url in urls
        ]
        self._base_urls = urls
        self._model = model
        self._weight_version = int(weight_version)
        self._in_flight = [0] * len(urls)
        self._next_index = 0
        self._route_lock = asyncio.Lock()
        self._admin_lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._replicas[0].name

    @property
    def base_url(self) -> str:
        """Compatibility value for callers that expect one representative URL."""
        return self._base_urls[0]

    @property
    def base_urls(self) -> list[str]:
        return list(self._base_urls)

    @property
    def model(self) -> str:
        return self._model

    @property
    def weight_version(self) -> int:
        return self._weight_version

    @property
    def served_model(self) -> str:
        return self._replicas[0].served_model

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        async with self._route_lock:
            minimum = min(self._in_flight)
            candidates = {i for i, count in enumerate(self._in_flight) if count == minimum}
            count = len(self._replicas)
            index = next(
                (self._next_index + offset) % count
                for offset in range(count)
                if (self._next_index + offset) % count in candidates
            )
            self._next_index = (index + 1) % count
            self._in_flight[index] += 1
        try:
            return await self._replicas[index].generate(request)
        finally:
            async with self._route_lock:
                self._in_flight[index] -= 1

    def set_weight_version(self, version: int) -> None:
        version = int(version)
        with self._admin_lock:
            errors: list[Exception] = []
            for replica in self._replicas:
                try:
                    replica.set_weight_version(version)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
            self._weight_version = version

    async def sync_weights(self, version: int) -> None:
        await asyncio.to_thread(self.set_weight_version, version)

    def check_server(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        errors: list[Exception] = []
        for replica in self._replicas:
            try:
                results.append(replica.check_server())
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        return results

    async def close(self) -> None:
        await asyncio.gather(*(replica.close() for replica in self._replicas))

    async def aclose(self) -> None:
        await self.close()


__all__ = ["LOGPROBS_MODES", "VllmServerPolicy", "VllmServerPolicyGroup"]
