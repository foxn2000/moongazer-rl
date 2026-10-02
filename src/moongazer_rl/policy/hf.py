"""``TransformersPolicy``: generation with a plain HuggingFace causal LM.

Token IDs in, token IDs + per-token sampling logprobs out. Nothing is re-tokenized: the
prompt IDs handed in by the renderer are fed to ``model.generate`` verbatim and the sampled
IDs come straight back.

Logprob convention (must match the trainer's learner logprobs, otherwise the importance
ratio compares two different distributions):

    logprob[t] = log_softmax(raw_logits[t] / temperature)[sampled_id[t]]

i.e. the *raw* model logits divided by the sampling temperature, with no top-p / top-k
renormalisation. That is what vLLM reports as ``Logprob.logprob`` for a sampled token and
what ``GRPOTrainer._get_per_token_logps_and_entropies`` computes (it also divides by
``args.temperature``). ``output_scores`` would give the post-warper distribution and would
not match; we therefore use ``output_logits``.

The policy shares the learner's parameters in-process, so ``sync_weights`` only bumps the
version counter.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from moongazer_rl.policy.base import GenerateRequest, GenerateResult, SamplingParams
from moongazer_rl.rollout.schema import FinishReason

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


DEFAULT_MAX_NEW_TOKENS = 256

NON_GENERATION_EXTRA_KEYS = frozenset({"reasoning_effort", "chat_template_kwargs", "n"})
"""``SamplingParams.extra`` keys that legitimately arrive from a verifiers ``SamplingConfig``
but are not engine sampling knobs: ``reasoning_effort`` and ``chat_template_kwargs`` are
applied by the *renderer* when the prompt is built, and ``n`` is the episode's rollout count,
which the runtime handles by issuing separate calls. Ignored on purpose, by every policy."""

GENERATE_EXTRA_KEYS: dict[str, str] = {
    "repetition_penalty": "repetition_penalty",
    "min_p": "min_p",
    "min_tokens": "min_new_tokens",
    "min_new_tokens": "min_new_tokens",
    "no_repeat_ngram_size": "no_repeat_ngram_size",
    "length_penalty": "length_penalty",
    "bad_words_ids": "bad_words_ids",
    "typical_p": "typical_p",
    "epsilon_cutoff": "epsilon_cutoff",
    "eta_cutoff": "eta_cutoff",
}
"""``SamplingParams.extra`` key -> ``transformers.GenerationConfig`` keyword. Anything outside
this map (and ``NON_GENERATION_EXTRA_KEYS``) is an error: a sampling setting the caller
validated must never be silently dropped."""


def resolve_extra_generate_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    """Map ``SamplingParams.extra`` onto ``model.generate`` keywords.

    Raises ``ValueError`` naming every key this backend cannot honour, rather than ignoring it.
    """
    resolved: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in extra.items():
        if key in NON_GENERATION_EXTRA_KEYS:
            continue
        target = GENERATE_EXTRA_KEYS.get(key)
        if target is None:
            unknown.append(key)
        elif value is not None:
            resolved[target] = value
    if unknown:
        raise ValueError(
            "TransformersPolicy cannot honour these sampling keys: "
            + ", ".join(sorted(unknown))
            + ". transformers' generate() has no equivalent (text `stop` strings are not "
            "supported either: the policy contract is token IDs, use stop_token_ids). "
            "Supported extras: "
            + ", ".join(sorted(GENERATE_EXTRA_KEYS))
            + "; ignored as non-generation keys: "
            + ", ".join(sorted(NON_GENERATION_EXTRA_KEYS))
            + "."
        )
    return resolved


class TransformersPolicy:
    """``PolicyBackend`` backed by ``transformers`` ``model.generate``.

    Safe to call from asyncio: every request runs in a worker thread (``asyncio.to_thread``)
    and ``lock`` serialises the actual forward passes, because a single ``nn.Module`` is not
    safe to run concurrently while the learner may be stepping it.

    ``lock`` should be the **same object** the trainer backend takes before its optimizer step
    (``UnslothNativeBackend.model_lock`` passes its own). That matters for cancellation:
    ``asyncio.to_thread`` cannot interrupt a worker thread, so when the verifiers harness times
    out a rollout the awaiting coroutine raises ``CancelledError`` while ``model.generate`` is
    still running. Without a shared lock the orchestrator would then start ``train_step`` and
    mutate the parameters (and the train/eval flag) underneath that thread. With it, the step
    simply blocks until the orphaned generation finishes. An ``RLock`` so the two never
    self-deadlock if a backend ever calls the policy from inside its own locked region.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        device: str | torch.device | None = None,
        *,
        weight_version: int = 0,
        default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        name: str = "transformers",
        lock: threading.RLock | None = None,
        before_generate: Callable[[], Any] | None = None,
        after_generate: Callable[[], Any] | None = None,
    ) -> None:
        import torch

        self._model = model
        # Optional mode switches run inside the lock around each generate (e.g. Unsloth's
        # for_inference / for_training). When unset, plain eval()/train(was_training) is used.
        self._before_generate = before_generate
        self._after_generate = after_generate
        self._tokenizer = tokenizer
        resolved: str | torch.device = torch.device("cpu")
        if device is not None:
            resolved = device
        else:
            try:
                resolved = next(model.parameters()).device
            except StopIteration:  # pragma: no cover - model without parameters
                pass
        self._device = torch.device(resolved)
        self._weight_version = int(weight_version)
        self._default_max_new_tokens = int(default_max_new_tokens)
        self._name = name
        self._lock = lock if lock is not None else threading.RLock()
        self._closed = False

        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            eos = getattr(getattr(model, "config", None), "eos_token_id", None)
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(getattr(model, "config", None), "pad_token_id", None)
        if pad is None:
            pad = eos
        self._eos_token_id = eos
        self._pad_token_id = pad

    # -- PolicyBackend -------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def weight_version(self) -> int:
        return self._weight_version

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any | None:
        return self._tokenizer

    @property
    def lock(self) -> Any:
        """The lock serialising model access. Shared with the trainer backend."""
        return self._lock

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        if self._closed:
            raise RuntimeError("TransformersPolicy is closed")
        return await asyncio.to_thread(self.generate_sync, request)

    async def sync_weights(self, version: int) -> None:
        self.set_weight_version(version)

    def set_weight_version(self, version: int) -> None:
        """Synchronous ``sync_weights``: the policy shares the learner's parameters, so a new
        version is visible the moment the optimizer steps. Used by the trainer backend."""
        self._weight_version = int(version)

    async def close(self) -> None:
        self._closed = True

    # -- implementation ------------------------------------------------------------
    def generate_sync(self, request: GenerateRequest) -> GenerateResult:
        """Blocking generation. Exposed for tests and for synchronous callers."""
        version = self._weight_version
        if request.policy_version is not None and request.policy_version != version:
            raise ValueError(
                f"{self.name} serves weight version {version} and cannot sample version "
                f"{request.policy_version}"
            )
        import torch

        sampling = request.sampling
        if not request.prompt_ids:
            raise ValueError("GenerateRequest.prompt_ids must not be empty")

        stop_ids = self._stop_ids(sampling)
        max_new_tokens = sampling.max_tokens or self._default_max_new_tokens
        # Validate before taking the lock so a bad request fails fast and touches no weights.
        extra_kwargs = resolve_extra_generate_kwargs(sampling.extra)

        # Held for the whole forward pass, and shared with the trainer backend, so an optimizer
        # step can never overlap a generation (including one whose awaiting task was cancelled).
        with self._lock:
            if sampling.seed is not None:
                torch.manual_seed(int(sampling.seed))
            was_training = bool(getattr(self._model, "training", False))
            if self._before_generate is not None:
                self._before_generate()
            else:
                self._model.eval()
            try:
                completion_ids, logits = self._run_generate(
                    request.prompt_ids, sampling, stop_ids, max_new_tokens, extra_kwargs
                )
            finally:
                # Restore deterministically: leaving the model in eval() would silently disable
                # dropout for the next optimizer step.
                if self._after_generate is not None:
                    self._after_generate()
                else:
                    self._model.train(was_training)

        logprobs = self._sampled_logprobs(completion_ids, logits, sampling.temperature)
        finish_reason: FinishReason = "length"
        if completion_ids and completion_ids[-1] in stop_ids:
            finish_reason = "stop"

        return GenerateResult(
            completion_ids=completion_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
            policy_version=version,
            request_id=request.request_id,
            extra={"num_prompt_tokens": len(request.prompt_ids)},
        )

    def _stop_ids(self, sampling: SamplingParams) -> set[int]:
        stop = {int(i) for i in sampling.stop_token_ids}
        if not stop and self._eos_token_id is not None:
            stop = {int(self._eos_token_id)}
        return stop

    def _run_generate(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        stop_ids: set[int],
        max_new_tokens: int,
        extra_kwargs: dict[str, Any],
    ) -> tuple[list[int], list[torch.Tensor]]:
        import torch

        input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_ids)

        kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "return_dict_in_generate": True,
            "output_logits": True,
            "use_cache": True,
        }
        if sampling.temperature and sampling.temperature > 0.0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(sampling.temperature)
        else:
            kwargs["do_sample"] = False
        # Always explicit: model.generate falls back to the checkpoint's generation_config.json
        # (Gemma 4: top_k=64, top_p=0.95), which would truncate the distribution the logprobs
        # below are computed from. top_k=0 disables top-k in transformers.
        kwargs["top_p"] = float(sampling.top_p) if sampling.top_p is not None else 1.0
        kwargs["top_k"] = int(sampling.top_k) if sampling.top_k is not None else 0
        if stop_ids:
            kwargs["eos_token_id"] = sorted(stop_ids)
        if self._pad_token_id is not None:
            kwargs["pad_token_id"] = int(self._pad_token_id)
        kwargs.update(extra_kwargs)

        with torch.no_grad():
            out = self._model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        sequences = out.sequences  # (1, prompt + generated)
        generated = sequences[0, input_ids.shape[1] :].tolist()
        logits = list(out.logits)  # one (1, vocab) tensor per generated step

        # transformers pads with pad_token_id after a stop token only for batched generation;
        # for batch size 1 the sequence simply ends. Trim anything past the first stop id so
        # completion_ids and logprobs stay 1:1 with what was actually sampled.
        completion_ids: list[int] = []
        for token in generated:
            completion_ids.append(int(token))
            if token in stop_ids:
                break
        return completion_ids, logits[: len(completion_ids)]

    def _sampled_logprobs(
        self,
        completion_ids: list[int],
        logits: list[torch.Tensor],
        temperature: float,
    ) -> list[float]:
        import torch

        if not completion_ids:
            return []
        temp = float(temperature) if temperature and temperature > 0.0 else 1.0
        out: list[float] = []
        with torch.no_grad():
            for token, step_logits in zip(completion_ids, logits, strict=True):
                row = step_logits[0].to(torch.float32) / temp
                out.append(float(torch.log_softmax(row, dim=-1)[token]))
        return out


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "GENERATE_EXTRA_KEYS",
    "NON_GENERATION_EXTRA_KEYS",
    "TransformersPolicy",
    "resolve_extra_generate_kwargs",
]
