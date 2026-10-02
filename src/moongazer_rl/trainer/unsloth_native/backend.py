"""``unsloth-native`` trainer backend: TRL ``GRPOTrainer`` (Unsloth-patched when available)
driven one optimizer step at a time by canonical ``RolloutGroup``s.

Why one manual step instead of ``trainer.train()``
--------------------------------------------------
``TrainerBackend.train_step(groups)`` is synchronous and must perform exactly one optimizer
step. ``Trainer.train()`` owns its own dataloader loop and cannot be resumed mid-epoch, so
driving it would need ART's background-task + queue + sentinel machinery
(``ref/ART/src/art/unsloth/train.py``: a 10M-row dummy dataset, ``_prepare_inputs`` replaced
by a blocking queue read, ``nest_asyncio``). We instead call ``trainer.training_step`` — which
still runs ``_prepare_inputs`` -> ``_generate_and_score_completions`` -> ``compute_loss`` ->
``accelerator.backward``, i.e. TRL's *and* Unsloth's real loss path — and then replay the ~8
lines ``Trainer._inner_training_loop`` runs on a sync step (clip, optimizer, scheduler,
zero_grad, global_step). The optimizer and scheduler are created once and persist.

Trade-off: ``TrainerCallback``s, ``trainer.log()`` and ``Trainer``'s own checkpointing never
fire. The run logger owns those anyway (``report_to`` is forced empty), and
``save_checkpoint`` writes the adapter + optimizer + scheduler explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from moongazer_rl.policy.base import CallUnit, PolicyBackend, resolve_sampling_temperature
from moongazer_rl.rollout.schema import RolloutGroup
from moongazer_rl.trainer.base import CheckpointInfo, StepResult
from moongazer_rl.trainer.unsloth_native.compat import (
    ensure_trl_importable,
    ensure_unsloth_first,
    select_shim,
)
from moongazer_rl.trainer.unsloth_native.dataset import RolloutBatch, build_batch
from moongazer_rl.trainer.unsloth_native.objective import (
    global_denominator,
    requires_equal_chunks,
    resolve_algorithm,
)

if TYPE_CHECKING:
    from moongazer_rl.trainer.unsloth_native.adapter_export import AdapterSnapshot

BACKEND_NAME = "unsloth-native"

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_METRIC_ALIASES = {
    "entropy": "entropy",
    "kl": "kl",
    "clip_ratio/region_mean": "clipped_token_fraction",
}
"""TRL ``_metrics['train']`` key -> canonical ``StepResult.metrics`` key.
Sources: ``trl/trainer/grpo_trainer.py:1758`` (kl), ``:1761`` (entropy), ``:1776``
(clip_ratio/region_mean)."""


def _get(config: Any, path: str, default: Any = None) -> Any:
    """Duck-typed ``config.a.b.c``; also accepts mappings at any level."""
    current: Any = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return default if current is None else current


def set_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if not hasattr(model, "_arl_gc_on_mode"):
        model._arl_gc_on_mode = getattr(model, "_unsloth_gradient_checkpointing", True)
    if getattr(model, "_arl_gc_enabled", None) is enabled:
        return
    value = model._arl_gc_on_mode if enabled else False
    model._unsloth_gradient_checkpointing = value
    for module in model.modules():
        if "_unsloth_gradient_checkpointing" in module.__dict__:
            module._unsloth_gradient_checkpointing = value
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
    model._arl_gc_enabled = enabled


def unsloth_available() -> bool:
    from importlib.util import find_spec

    try:
        return find_spec("unsloth") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


class UnslothNativeBackend:
    """``TrainerBackend`` over TRL/Unsloth GRPO with canonical rollouts as the generation."""

    def __init__(self, config: Any) -> None:
        self._config = config
        # Must happen before anything reaches transformers/trl: unsloth patches them at import
        # time and has to be imported first. select_shim() below already touches both.
        ensure_unsloth_first()
        self._shim = select_shim()

        self._model_name: str = str(_get(config, "model.name", ""))
        self._load_in_4bit: bool = bool(_get(config, "model.load_in_4bit", False))
        self._text_only_offload: bool = bool(_get(config, "model.text_only_offload", True))
        self._max_seq_length: int = int(_get(config, "model.max_seq_length", 2048))
        self._gradient_checkpointing: str = str(
            _get(config, "model.gradient_checkpointing", "unsloth")
        )
        self._gradient_checkpointing_auto_max_tokens = int(
            _get(config, "model.gradient_checkpointing_auto_max_tokens", 4096)
        )
        self._lora: dict[str, Any] = {
            "r": int(_get(config, "model.lora.r", 16)),
            # float, not int: LoraConfig.lora_alpha only feeds scaling = alpha / r, and peft
            # keeps whatever number it is given (verified with peft 0.20). int() would turn a
            # fractional alpha such as 0.5 into 0 and silently disable the adapter.
            "lora_alpha": float(_get(config, "model.lora.alpha", 32)),
            "lora_dropout": float(_get(config, "model.lora.dropout", 0.0)),
            "target_modules": list(
                _get(config, "model.lora.target_modules", DEFAULT_LORA_TARGET_MODULES)
            ),
        }
        raw_args = _get(config, "trainer.args", {}) or {}
        self._args: dict[str, Any] = dict(raw_args)
        self._algorithm_section = _get(config, "trainer.algorithm")
        # Backend emits the BNPO migration warning at its first split step, preserving
        # the legacy warning lifecycle. Standalone resolve_algorithm still warns directly.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="bnpo maps to global action_token")
            self._algorithm, self._algorithm_problems = resolve_algorithm(
                self._algorithm_section,
                self._args,
                max_completion_length=self._max_seq_length,
                lora_dropout=self._lora["lora_dropout"],
            )
        self._num_generations: int = int(_get(config, "rollout.num_generations", 2))
        # The learner divides its logits by GRPOConfig.temperature
        # (grpo_trainer.py:786-858), so it must be the temperature the rollouts sampled at or
        # the importance ratio is wrong even for a perfectly on-policy sample.
        # Same precedence PolicyClient._sampling_params uses (top-level beats extra_body), so
        # the learner divides by exactly the temperature the rollouts sampled at.
        self._sampling_temperature: float = resolve_sampling_temperature(
            _get(config, "rollout.sampling", {})
        )
        self._seed: int = int(_get(config, "run.seed", 0))
        self._output_dir: str = str(_get(config, "logging.output_dir", "runs/unsloth-native"))
        # Optional, additive: how canonical rollouts become rows. Absent from RunConfig today.
        self._call_unit: CallUnit = str(_get(config, "trainer.call_unit", "sequence"))  # type: ignore[assignment]
        self._fork_policy: str = str(_get(config, "trainer.fork_policy", "error"))

        self._model: Any = None
        self._tokenizer: Any = None
        self._trainer: Any = None
        self._policy: PolicyBackend | None = None
        self._global_step = 0
        self._weight_version = 0
        self._is_unsloth = False
        self._policy_backend: str = str(_get(config, "policy.backend", "auto"))
        # Memory-efficiency switches (config trainer.memory; duck-typed so tests can pass a
        # SimpleNamespace or nothing). All exact.
        mem = _get(config, "trainer.memory", None)
        self._memory: dict[str, Any] = {
            "blocked_global_attention": bool(_get(mem, "blocked_global_attention", True)),
            "global_attention": str(_get(mem, "global_attention", "flex")),
            "sliding_attention": str(_get(mem, "sliding_attention", "flex")),
            "materialize_sliding_mask": bool(_get(mem, "materialize_sliding_mask", False)),
            "query_chunk_size": int(_get(mem, "query_chunk_size", 1024)),
            "min_global_q_len": int(_get(mem, "min_global_q_len", 1024)),
            "token_chunked_logprobs": bool(_get(mem, "token_chunked_logprobs", True)),
            "logprob_chunk_size": int(_get(mem, "logprob_chunk_size", 64)),
            "score_rows_one_at_a_time": bool(_get(mem, "score_rows_one_at_a_time", True)),
        }
        self._trainable_params: list[Any] | None = None
        self._learner_peak_memory_gb = 0.0
        self._uninstall_attention: Any = None
        self._uninstall_sliding_mask: Any = None
        self._uninstall_text_mm: Any = None
        self._uninstall_text_only_offload: Any = None
        self._adapter_snapshot_s = 0.0
        self._policy_args: dict[str, Any] = dict(_get(config, "policy.args", {}) or {})
        self._warned_bnpo = False
        self._planned_steps: int | None = None
        self._lr_schedule_horizon: int | None = None
        # One lock for "somebody is touching the model": handed to the policy and taken by
        # train_step. See TransformersPolicy's docstring for the cancellation case it covers.
        self._model_lock = threading.RLock()

    # -- identity ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return BACKEND_NAME

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def weight_version(self) -> int:
        return self._weight_version

    @property
    def is_unsloth(self) -> bool:
        """True when the trainer class came from Unsloth's ``PatchFastRL``."""
        return self._is_unsloth

    @property
    def model_lock(self) -> Any:
        """Re-entrant lock guarding the model. Held by ``train_step`` and by the policy's
        generation, so an optimizer step never overlaps a rollout — including a rollout whose
        awaiting task was cancelled while its worker thread kept running."""
        return self._model_lock

    @property
    def trainer(self) -> Any:
        """The upstream (shimmed) ``GRPOTrainer``. Valid after ``setup``."""
        if self._trainer is None:
            raise RuntimeError("UnslothNativeBackend.setup() has not run yet")
        return self._trainer

    @property
    def tokenizer(self) -> Any:
        """The tokenizer the trainer uses. Valid after ``setup``.

        The orchestrator builds its ``renderers`` renderer from this exact object, so that
        rendering, sampling and training all agree on the token IDs.
        """
        if self._tokenizer is None:
            raise RuntimeError("UnslothNativeBackend.setup() has not run yet")
        return self._tokenizer

    def tokenizer_fingerprint(self) -> dict[str, Any]:
        """Identity of the tokenizer/chat template, for ``runtime.json``."""
        tok = self.tokenizer
        template = getattr(tok, "chat_template", None)
        digest = None
        if isinstance(template, str) and template:
            digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        return {
            "name_or_path": getattr(tok, "name_or_path", None),
            "class": type(tok).__name__,
            "vocab_size": int(getattr(tok, "vocab_size", 0) or 0),
            "len": len(tok) if hasattr(tok, "__len__") else None,
            "chat_template_sha256": digest,
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "pad_token_id": getattr(tok, "pad_token_id", None),
            "bos_token_id": getattr(tok, "bos_token_id", None),
        }

    def set_planned_steps(self, steps: int | None) -> None:
        """Tell the backend how many optimizer steps the run plans to take.

        Additive extension to ``TrainerBackend``: the orchestrator may call it **before**
        ``setup()`` when it knows the horizon (``train.max_steps``, or dataset x epochs). It is
        what a decaying LR schedule is built against; without it (and without
        ``trainer.args.max_steps``) the backend falls back to a constant LR rather than guessing
        a horizon the schedule would decay to zero over. Calling it after ``setup()`` is an
        error, because the scheduler is already built.
        """
        if self._trainer is not None:
            raise RuntimeError(
                "set_planned_steps() must be called before setup(); the LR scheduler is built "
                f"there (current horizon: {self._lr_schedule_horizon})"
            )
        if steps is not None and steps <= 0:
            raise ValueError(f"planned steps must be positive or None (got {steps})")
        self._planned_steps = steps

    @property
    def planned_steps(self) -> int | None:
        return self._planned_steps

    @property
    def lr_schedule_horizon(self) -> int | None:
        """Number of optimizer steps the LR schedule was built for; ``None`` = constant LR."""
        return self._lr_schedule_horizon

    # -- validation ----------------------------------------------------------------
    def validate_args(self, args: dict[str, Any]) -> list[str]:
        problems = list(self._shim.validate_args(args, temperature=self._sampling_temperature))
        if self._sampling_temperature <= 0.0:
            problems.append(
                "rollout.sampling.temperature must be > 0 for GRPO: the learner divides its "
                f"logits by it (got {self._sampling_temperature!r})"
            )
        problems.extend(self._shim.validate_lr_schedule(args, planned_steps=self._planned_steps))
        problems.extend(_validate_single_process(args))
        problems.extend(_validate_gemma4_micro_batch(self._model_name, args, self._num_generations))
        problems.extend(self._validate_precision(args))
        if self._memory["token_chunked_logprobs"] != self._memory["score_rows_one_at_a_time"]:
            problems.append(
                "trainer.memory.token_chunked_logprobs and score_rows_one_at_a_time must match: "
                "loss and scoring must use the same trimmed positions"
            )
        problems.extend(self._algorithm_problems)
        if self._num_generations < 2 and self._algorithm.advantage_estimator in {
            "group_zscore",
            "group_center",
            "rloo",
        }:
            problems.append(
                f"rollout.num_generations must be >= 2 for GRPO (got {self._num_generations})"
            )
        if self._call_unit not in ("sequence", "call"):
            problems.append(
                f"trainer.call_unit must be 'sequence' or 'call' (got {self._call_unit!r})"
            )
        if self._fork_policy not in ("error", "all", "longest"):
            problems.append(
                "trainer.fork_policy must be 'error', 'all' or 'longest' "
                f"(got {self._fork_policy!r})"
            )
        return problems

    # -- setup ---------------------------------------------------------------------
    def setup(self) -> None:
        if self._trainer is not None:
            return
        problems = self.validate_args(self._args)
        if problems:
            raise ValueError(
                "invalid trainer configuration for unsloth-native:\n  - " + "\n  - ".join(problems)
            )

        if unsloth_available():
            self._model, self._tokenizer = self._load_unsloth()
            self._is_unsloth = True
        else:
            self._model, self._tokenizer = self._load_transformers()
        # COMPAT(trl==0.24, transformers==5.5, peft==0.20): GRPOTrainer.__init__ writes
        # model.warnings_issued["estimate_tokens"] (grpo_trainer.py:406), but transformers 5.x
        # PreTrainedModel no longer defines it and PeftModel.__getattr__ therefore raises.
        # ART works around the same thing (ref/ART/src/art/unsloth/train.py:721).
        if not hasattr(self._model, "warnings_issued"):
            self._model.warnings_issued = {}

        ensure_trl_importable()
        # Imported from the top-level package AFTER PatchFastRL, so that on the GPU path we
        # subclass Unsloth's UnslothGRPOTrainer (unsloth/models/rl.py:3387 rebinds trl.GRPOTrainer).
        from trl import GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

        horizon = self._resolve_schedule_horizon()
        overrides: dict[str, Any] = {}
        overrides["gradient_checkpointing"] = self._gradient_checkpointing != "off"
        if horizon is None:
            overrides["lr_scheduler_type"] = self._shim.DEFAULT_LR_SCHEDULER
        if not self._is_unsloth:
            # GRPOConfig.__post_init__ resolves bf16 to True whenever fp16 is False
            # (grpo_config.py:640), and Trainer then builds an Accelerator with
            # mixed_precision="bf16" — but nothing on this path ever enters an autocast region
            # (see _validate_precision). Force both off so the resolved config, the accelerator
            # and the actual arithmetic agree on fp32.
            overrides["bf16"] = False
            overrides["fp16"] = False
        grpo_config = self._shim.build_grpo_config(
            self._args,
            num_generations=self._num_generations,
            output_dir=self._output_dir,
            seed=self._seed,
            max_completion_length=self._max_seq_length,
            temperature=self._sampling_temperature,
            overrides=overrides,
            spec=self._algorithm,
        )
        trainer_cls = self._shim.make_trainer_class(GRPOTrainer)
        self._trainer = trainer_cls(
            model=self._model,
            reward_funcs=[_precomputed_reward],
            args=grpo_config,
            train_dataset=None,
            processing_class=self._tokenizer,
        )
        self._trainer._arl_spec = self._algorithm
        if self._gradient_checkpointing == "auto":
            self._trainer._arl_gc_auto_max_tokens = self._gradient_checkpointing_auto_max_tokens
        self._lr_schedule_horizon = horizon
        self._install_memory_settings(self._trainer)
        # COMPAT(transformers==5.5): mirror _inner_training_loop's non-distributed branch
        # (trainer.py:1600-1605): prepare the optimizer through the accelerator, THEN build the
        # scheduler on top of it. Without the prepare, `accelerator._optimizers` stays empty, so
        # `clip_grad_norm_`'s unscale_gradients() has nothing to unscale, AcceleratedOptimizer's
        # `scaler.step()/update()` never runs and `optimizer_step_was_skipped` is always False —
        # i.e. fp16 loss scaling would be silently broken. The model is deliberately NOT
        # prepared: Unsloth does its own wrapping and `training_step` already opens the autocast
        # context; DeepSpeed/FSDP, which do need a prepared model, are rejected in validate_args.
        trainer = self._trainer
        trainer.create_optimizer()
        trainer.optimizer = trainer.accelerator.prepare(trainer.optimizer)
        trainer.create_scheduler(
            num_training_steps=horizon if horizon is not None else 1, optimizer=trainer.optimizer
        )

    def _validate_precision(self, args: dict[str, Any]) -> list[str]:
        """Refuse mixed precision on the plain-transformers path, where it would do nothing.

        transformers 5.5 delegates autocast entirely to accelerate —
        ``Trainer.autocast_smart_context_manager`` returns ``contextlib.nullcontext()``
        (``trainer.py:2034-2039``, "We rely on accelerate for autocast") and the wrapper comes
        from ``accelerator.prepare_model``, which this backend deliberately does not call
        (see ``setup``). So ``bf16=True`` / ``fp16=True`` would leave every forward in fp32
        while the config claimed otherwise. On the Unsloth path precision is Unsloth's own
        business (``trainer._autocast_dtype`` and the autocast inside
        ``unsloth_zoo.rl_replacements.grpo_accumulated_loss``), so both are left alone there.
        """
        if unsloth_available():
            return []
        problems: list[str] = []
        for key in ("fp16", "bf16"):
            if args.get(key):
                problems.append(
                    f"trainer.args.{key}=True has no effect without unsloth installed: "
                    "transformers 5.5 gets its autocast from accelerator.prepare_model, which "
                    "this backend does not call (autocast_smart_context_manager is a "
                    "nullcontext, trainer.py:2034-2039). The plain-transformers path therefore "
                    "trains in fp32 and forces both flags off. Install the unsloth extra for "
                    "mixed-precision training; Unsloth handles precision itself."
                )
        return problems

    def _resolve_schedule_horizon(self) -> int | None:
        """``set_planned_steps`` wins, then ``trainer.args.max_steps``, else ``None``.

        ``None`` means "no horizon is known": the backend then uses a constant LR
        (``compat.DEFAULT_LR_SCHEDULER``) instead of decaying over a made-up step count.
        ``validate_args`` has already rejected a user-requested decaying schedule in that case.
        """
        if self._planned_steps is not None and self._planned_steps > 0:
            return int(self._planned_steps)
        max_steps = self._args.get("max_steps")
        if isinstance(max_steps, int) and max_steps > 0:
            return int(max_steps)
        return None

    def _load_transformers(self) -> tuple[Any, Any]:
        """CPU / no-Unsloth path: plain transformers + a peft LoRA adapter."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        # Deterministic LoRA init from run.seed. GRPOTrainer.__init__ seeds too
        # (grpo_trainer.py:~555), but only after get_peft_model has drawn lora_A.
        set_seed(self._seed)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        _warn_if_pad_is_eos(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(self._model_name, dtype=torch.float32)
        model.config.use_cache = False
        peft_config = LoraConfig(task_type="CAUSAL_LM", bias="none", **self._lora)
        model = get_peft_model(model, peft_config)
        if self._gradient_checkpointing == "off":
            model.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_enable()
        return model, tokenizer

    def _load_unsloth(self) -> tuple[Any, Any]:  # pragma: no cover - needs a GPU + unsloth
        """GPU path. Never exercised on macOS/CPU."""
        # Unsloth must be imported before trl/transformers so its patches land.
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel, PatchFastRL

        PatchFastRL("GRPO", FastLanguageModel)
        # No in-process vLLM engine (fast_inference): the production sampler is an external
        # `vllm serve` (policy.backend: vllm-server); otherwise HF generate on the same weights.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._model_name,
            max_seq_length=self._max_seq_length,
            load_in_4bit=self._load_in_4bit,
            fast_inference=False,
        )
        # Multimodal checkpoints (e.g. Gemma 4) come back with a Processor; the renderer, TRL's
        # processing_class and the policy all need the text tokenizer underneath it.
        if not hasattr(tokenizer, "encode") and hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        model = FastLanguageModel.get_peft_model(
            model,
            r=self._lora["r"],
            lora_alpha=self._lora["lora_alpha"],
            lora_dropout=self._lora["lora_dropout"],
            target_modules=self._lora["target_modules"],
            use_gradient_checkpointing={
                "unsloth": "unsloth",
                "auto": "unsloth",
                "torch": True,
                "off": False,
            }[self._gradient_checkpointing],
            random_state=self._seed,
        )
        from moongazer_rl.trainer.unsloth_native import gemma4_text

        self._uninstall_text_mm = gemma4_text.install_text_only_mm_token_type_ids(model)
        if self._text_only_offload:
            self._uninstall_text_only_offload = gemma4_text.offload_text_only_modules(model)
        # Unsloth's own sliding path misbehaves with a None mask, so only skip the dense mask
        # when the sliding layers run on flex.
        if not self._memory["materialize_sliding_mask"] and self._memory["sliding_attention"] == "flex":
            from moongazer_rl.trainer.unsloth_native.attention import (
                install_unmaterialized_sliding_mask,
            )

            self._uninstall_sliding_mask = install_unmaterialized_sliding_mask()
        # After every Unsloth patch has landed (PatchFastRL + the model-load patches), chain our
        # exact global/sliding attention on top of the SDPA router.
        if self._memory["blocked_global_attention"] or self._memory["sliding_attention"] == "flex":
            from moongazer_rl.trainer.unsloth_native.attention import (
                install_blocked_global_attention,
            )

            self._uninstall_attention = install_blocked_global_attention(
                global_attention=self._memory["global_attention"],
                sliding_attention=self._memory["sliding_attention"],
                blocked_global_attention=self._memory["blocked_global_attention"],
                min_q_len=self._memory["min_global_q_len"],
                query_chunk_size=self._memory["query_chunk_size"],
            )
            from moongazer_rl.trainer.unsloth_native.attention import blocked_attention_stats

            self._attention_engaged_previous = blocked_attention_stats()["engaged"]
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        _warn_if_pad_is_eos(tokenizer)
        return model, tokenizer

    # -- policy --------------------------------------------------------------------
    def attach_policy(self, policy: PolicyBackend) -> None:
        """Install the lease-capable vLLM policy used by the async orchestrator.

        This keeps ``_push_weight_version`` connected to that external policy after each
        learner update instead of constructing the synchronous policy backend.
        """
        if self._trainer is None:
            raise RuntimeError("UnslothNativeBackend.setup() has not run yet")
        if self._policy is not None:
            raise RuntimeError("a policy has already been attached or created")
        self._policy = policy

    def policy(self) -> PolicyBackend:
        if self._policy is not None:
            return self._policy
        if self._model is None:
            raise RuntimeError("UnslothNativeBackend.setup() has not run yet")
        if self._policy_backend == "vllm-server":
            self._policy = self._build_vllm_server_policy()
        elif self._is_unsloth:  # pragma: no cover - GPU only
            from unsloth import FastLanguageModel

            from moongazer_rl.policy.hf import TransformersPolicy

            model = self._model
            # Unsloth's forward needs its inference flags (no gradient checkpointing, use_cache
            # restored, _flag_for_generation) around generate, and the training flags back
            # afterwards; plain eval()/train() is not enough.
            self._policy = TransformersPolicy(
                model,
                self._tokenizer,
                weight_version=self._weight_version,
                lock=self._model_lock,
                name="unsloth-hf",
                before_generate=lambda: FastLanguageModel.for_inference(model),
                after_generate=lambda: FastLanguageModel.for_training(model),
            )
        else:
            from moongazer_rl.policy.hf import TransformersPolicy

            # Shares the learner's parameters in-process: nothing to export, the constructor's
            # weight_version is already the current (possibly restored) one.
            self._policy = TransformersPolicy(
                self._model,
                self._tokenizer,
                weight_version=self._weight_version,
                lock=self._model_lock,
            )
        return self._policy

    def _build_vllm_server_policy(self) -> PolicyBackend:
        """External ``vllm serve`` sampler; the learner exports its LoRA adapter per version."""
        from moongazer_rl.policy.vllm_server import VllmServerPolicy, VllmServerPolicyGroup

        args = dict(self._policy_args)
        args.setdefault("model", self._model_name)
        if "base_urls" in args:
            if "base_url" in args:
                raise ValueError("policy.args must set exactly one of base_url or base_urls")
            policy = VllmServerPolicyGroup(
                weight_version=self._weight_version,
                export_adapter=self.export_adapter_path,
                **args,
            )
        else:
            policy = VllmServerPolicy(
                weight_version=self._weight_version,
                export_adapter=self.export_adapter_path,
                **args,
            )
        policy.check_server()
        # Register the adapter for the current version right away (also covers resume, where
        # load_checkpoint() ran before the policy existed), so sampling never falls back to
        # the base model while an adapter is trained.
        policy.set_weight_version(self._weight_version)
        return policy

    def export_adapter_path(self, version: int) -> Path:
        """Write the PEFT adapter for ``version`` under ``<output_dir>/.lora-sync/v<version>``.
        Taken under the model lock so no optimizer step interleaves with the export."""
        if self._model is None:
            raise RuntimeError("UnslothNativeBackend.setup() has not run yet")
        target = Path(self._output_dir) / ".lora-sync" / f"v{int(version)}"
        tmp = target.with_name(target.name + ".tmp")
        with self._model_lock:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(str(tmp))
            if target.exists():
                shutil.rmtree(target)
            tmp.rename(target)
        return target

    @property
    def adapter_export_root(self) -> Path:
        return Path(self._output_dir) / ".lora-sync"

    def _export_adapter(self, version: int) -> AdapterSnapshot:
        """Take a CPU adapter snapshot under the learner lock; publisher owns disk I/O."""
        from moongazer_rl.trainer.unsloth_native.adapter_export import snapshot_lora

        with self._model_lock:
            if self._model is None:
                raise RuntimeError("UnslothNativeBackend.setup() has not run yet")
            snapshot = snapshot_lora(self._model, version)
            self._adapter_snapshot_s = snapshot.snapshot_s
            return snapshot

    def export_adapter(self, version: int) -> AdapterSnapshot:
        """Async publication API: return AdapterSnapshot; use export_adapter_path for sync."""
        return self._export_adapter(version)

    def device_info(self) -> dict[str, Any]:
        """Return best-effort learner placement metadata for ``runtime.json``."""
        info: dict[str, Any] = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": None,
            "device_name": None,
        }
        try:
            device = next(self._model.parameters()).device
            info["device"] = str(device)
            if getattr(device, "type", None) == "cuda":
                try:
                    import torch

                    info["device_name"] = torch.cuda.get_device_name(device)
                except Exception:  # noqa: BLE001 - diagnostics must never break a run
                    pass
        except Exception:  # noqa: BLE001 - diagnostics must never break a run
            pass
        return info

    # -- training ------------------------------------------------------------------
    @property
    def early_drop_safe(self) -> bool:
        """Whether one group's rows determine its trainability without the rest of a batch."""
        return (
            self._algorithm.advantage_estimator
            in {"group_zscore", "group_center", "rloo"}
            and self._algorithm.std_scope == "group"
            and self._algorithm.beta == 0
        )

    def would_train(self, group: RolloutGroup) -> bool:
        """Apply the real row builder so dynamic sampling cannot mistake metadata for loss.

        This deliberately avoids the model and tensors: it is safe before ``setup()`` and cheap
        enough for the learner's dequeue loop. After setup it mirrors ``train_step``'s resolved
        reward scaling, although trainability itself is independent of the scaling mode.
        """
        from moongazer_rl.trainer.unsloth_native.dataset import would_train_group

        return would_train_group(
            group,
            unit=self._call_unit,
            fork_policy=self._fork_policy,
            max_seq_length=self._max_seq_length,
            advantage=self._advantage_spec(),
            mask_truncated=bool(self._algorithm.mask_truncated_completions),
            kl_active=self._algorithm.beta != 0,
        )

    def _advantage_spec(self):
        from moongazer_rl.trainer.unsloth_native.dataset import AdvantageSpec

        return AdvantageSpec(
            advantage_estimator=self._algorithm.advantage_estimator,
            std_scope=self._algorithm.std_scope,
        )

    def train_step(self, groups: list[RolloutGroup]) -> StepResult:
        started = time.perf_counter()
        _ = self.trainer  # require setup before building a step
        batch = build_batch(
            groups,
            unit=self._call_unit,
            advantage=self._advantage_spec(),
            fork_policy=self._fork_policy,  # type: ignore[arg-type]
            max_seq_length=self._max_seq_length,
            sequence_overflow=_get(
                getattr(self, "_config", None), "trainer.sequence_overflow", "error"
            ),
        )
        if batch.num_dropped_fidelity:
            dropped_ids = [trajectory_id for trajectory_id, _ in batch.fidelity_dropped]
            warnings.warn(
                f"dropped {batch.num_dropped_fidelity} trajectory/trajectories with training "
                f"fidelity problems: {dropped_ids}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not batch.rows:
            return self._empty_step(batch, started)

        # Blocking: an in-flight generation (possibly one whose awaiting task was cancelled)
        # must finish before any parameter changes.
        with self._model_lock:
            self._adapter_snapshot_s = 0.0
            result = self._locked_train_step(batch, started)
            self._last_step_result = result
            return result

    def _locked_train_step(self, batch: RolloutBatch, started: float) -> StepResult:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if (
            self._algorithm.beta == 0
            and batch.rows
            and batch.num_zero_advantage_groups == batch.num_groups_used
        ):
            return self._skipped_step(batch, started, reason="zero_advantage_batch")
        trainer = self.trainer
        learner_version_before = self._weight_version
        # Score the whole batch once (advantages need the full group, and old/ref logps are
        # already chunked internally by per_device_train_batch_size), then back-propagate in
        # micro-batches so the learner forward never holds every sequence at once.
        from moongazer_rl.trainer.unsloth_native.planner import (
            BatchBudget,
            describe_plan,
            plan_microbatches,
        )
        per_device = int(getattr(trainer.args, "per_device_train_batch_size", 1) or 1)
        settings = _get(self._config, "trainer.batch", {})
        explicit_batch = (
            settings.model_dump(exclude_unset=True) if hasattr(settings, "model_dump") else settings
        )
        algorithm_section = self._algorithm_section
        explicit_algorithm = (
            algorithm_section.model_dump(exclude_unset=True)
            if hasattr(algorithm_section, "model_dump")
            else algorithm_section
        )
        legacy_batch = not explicit_batch and not explicit_algorithm
        # Rows whose loss is identically zero (truncated rows under mask_truncated_completions,
        # rows without action tokens, or zero advantages with beta=0) still shape the group
        # advantages and the DAPO normaliser, but forwarding them only burns learner time.
        # Keep them in the batch, leave them out of the micro-batch plan.
        # Legacy loss_type="grpo" averages over the forwarded rows, so dropping a zero-advantage
        # row there would rescale the gradient; every other normalisation uses a batch-wide
        # denominator that the forward plan does not change.
        skip_zero_advantage = self._algorithm.beta == 0 and not (
            legacy_batch and self._args.get("loss_type") == "grpo"
        )
        mask_truncated = bool(trainer._mask_truncated_requested())
        zero_loss = [
            i
            for i, row in enumerate(batch.rows)
            if (mask_truncated and row.truncated)
            or row.num_action_tokens == 0
            or (skip_zero_advantage and row.advantage == 0.0)
        ]
        active = [i for i in range(batch.size) if i not in set(zero_loss)]
        if not active:
            return self._skipped_step(batch, started, reason="every row carries zero loss")
        budget = BatchBudget(
            max_model_tokens=_get(settings, "max_model_tokens"),
            max_action_tokens=_get(settings, "max_action_tokens"),
            max_rows=_get(settings, "max_rows") or per_device,
            max_groups=_get(settings, "max_groups"),
            keep_groups_whole=_get(settings, "keep_groups_whole", True),
            length_bucketing=_get(settings, "length_bucketing", True),
        )
        if legacy_batch:
            count = self._shim.micro_batch_chunks(len(active), per_device)
            width = len(active) // count
            plan = [active[i : i + width] for i in range(0, len(active), width)]
        else:
            plan = [
                [active[j] for j in chunk]
                for chunk in plan_microbatches([batch.rows[i] for i in active], budget)
            ]
        if requires_equal_chunks(self._algorithm) and len({len(chunk) for chunk in plan}) > 1:
            warnings.warn(
                "sequence normalization with beta != 0 requires equal micro-batches; "
                "falling back to legacy equal chunks (group/token budgets may be overridden)",
                UserWarning,
                stacklevel=2,
            )
            count = self._shim.micro_batch_chunks(len(active), budget.max_rows or per_device)
            width = len(active) // count
            plan = [active[i : i + width] for i in range(0, len(active), width)]
        num_chunks = len(plan)
        denominator = global_denominator(self._algorithm, batch.rows)
        if legacy_batch:
            # Legacy sequence reduction is a native per-row mean / GAS even for beta=0.
            # Retain the public num_chunks hook and old GRPO semantics for existing YAMLs.
            if self._args.get("loss_type") == "grpo":
                trainer._arl_spec = None
                trainer.args.loss_type = "grpo"
                trainer.loss_type = "grpo"
            trainer.begin_step(batch, num_chunks=num_chunks, active_rows=active)
        else:
            trainer.begin_step(batch, plan=plan, spec=self._algorithm, denominator=denominator)
        trainer._arl_row_staleness = {
            index: learner_version_before - row.behavior_version
            for index, row in enumerate(batch.rows)
        }
        trainer._arl_stats["zero_loss_rows_skipped"] = float(len(zero_loss))
        trainer._arl_stats["zero_loss_model_tokens_skipped"] = float(
            sum(batch.rows[i].total_len for i in zero_loss)
        )
        trainer._arl_stats["num_loss_action_tokens"] = float(
            sum(batch.rows[i].num_action_tokens for i in active)
        )
        trainer._arl_stats["num_forward_tokens"] = float(
            sum(batch.rows[i].total_len for i in active)
        )
        trainer._arl_stats["num_skipped_zero_loss_action_tokens"] = float(
            sum(batch.rows[i].num_action_tokens for i in zero_loss)
        )
        if self._args.get("loss_type") == "bnpo" and num_chunks > 1 and not self._warned_bnpo:
            warnings.warn(
                "bnpo now uses global action-token normalization across micro-batches",
                UserWarning,
                stacklevel=2,
            )
            self._warned_bnpo = True
        model = trainer.model_wrapped
        rows = batch.to_trl_inputs()
        loss = None
        backward_started = _interval_boundary(torch)
        for _ in range(num_chunks):
            # Only the first call generates and scores; the rest read TRL's buffered chunks.
            # training_step runs compute_loss + accelerator.backward, so gradients accumulate.
            chunk_loss = trainer.training_step(model, rows, num_items_in_batch=None)
            chunk_loss = chunk_loss.detach().float()
            loss = chunk_loss if loss is None else loss + chunk_loss

        # COMPAT(transformers==5.5): the sync-step block of Trainer._inner_training_loop
        # (trainer.py:1753-1772), minus callbacks/logging which the run logger owns. It runs
        # once, after every micro-batch's backward, so grad_norm sees the accumulated gradient.
        grad_clip_started = _interval_boundary(torch)
        backward_s = grad_clip_started - backward_started
        max_grad_norm = float(getattr(trainer.args, "max_grad_norm", 0.0) or 0.0)
        grad_norm = trainer._clip_grad_norm(model) if max_grad_norm > 0 else None
        grad_norm = trainer._get_grad_norm(model, grad_norm=grad_norm)
        norm_value = float(grad_norm.item()) if hasattr(grad_norm, "item") else grad_norm
        grad_clip_ended = _interval_boundary(torch)
        grad_clip_s = grad_clip_ended - grad_clip_started
        if norm_value is not None and not math.isfinite(float(norm_value)):
            # A non-finite gradient must neither reach the optimizer nor be published as a new
            # policy version; report it as a skipped step so the run can stop or continue
            # deliberately.
            model.zero_grad(set_to_none=True)
            warnings.warn(
                f"non-finite gradient norm ({norm_value!r}) at global step "
                f"{self._global_step + 1}: optimizer step skipped, weights unchanged",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._skipped_step(batch, started, reason="non_finite_gradient")
        optimizer_diag: dict[str, float] = {}
        learning_rate = trainer._get_learning_rate()
        optimizer_started = _interval_boundary(torch)
        trainer.optimizer.step()
        if trainer.accelerator.optimizer_step_was_skipped:
            model.zero_grad(set_to_none=True)
            warnings.warn(
                f"AMP overflow at global step {self._global_step + 1}: "
                "optimizer step skipped, weights unchanged",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._skipped_step(batch, started, reason="amp_overflow")
        trainer.lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        optimizer_s = _interval_boundary(torch) - optimizer_started
        trainer.state.global_step += 1
        self._global_step += 1
        self._weight_version += 1
        self._push_weight_version()

        metrics: dict[str, float] = {}
        trl_metrics: dict[str, float] = {}
        extra: dict[str, Any] = {"trl_metrics": trl_metrics}
        try:
            if torch.cuda.is_available():
                step_peak = torch.cuda.max_memory_allocated() / 1e9
                self._learner_peak_memory_gb = max(self._learner_peak_memory_gb, step_peak)
                optimizer_diag["learner_step_peak_memory_gb"] = step_peak
                optimizer_diag["learner_peak_memory_gb"] = self._learner_peak_memory_gb
            loss_stats_started = _interval_boundary(torch)
            metrics = {
                "loss": float(loss.cpu()),
                "num_micro_batches": float(num_chunks),
                "learning_rate": float(learning_rate),
                "num_action_tokens": float(batch.num_action_tokens),
                "num_tokens": float(batch.num_tokens),
                "reward_mean": batch.group_reward_mean,
                "reward_std_mean": batch.group_reward_std_mean,
            }
            if norm_value is not None and math.isfinite(float(norm_value)):
                metrics["grad_norm"] = float(norm_value)
            metrics.update(optimizer_diag)
            metrics.update(trainer.pop_shim_stats())
            metrics.setdefault(
                "proximal_ratio_identity",
                float(
                    self._algorithm.skip_old_logprob_forward
                    or self._algorithm.ratio_reference == "behavior"
                    or self._algorithm.policy_objective == "reinforce"
                ),
            )
            if self._algorithm.beta == 0:
                metrics.pop("kl", None)
            metrics["time/forward_backward_s"] = max(
                0.0, backward_s - metrics.get("time/score_forward_s", 0.0)
            )
            metrics["time/optimizer_s"] = optimizer_s
            trl_metrics = self._drain_trl_metrics(trainer)
            for source, canonical in _METRIC_ALIASES.items():
                if canonical == "kl" and float(self._algorithm.beta) == 0.0:
                    continue  # no reference policy: TRL's bucket holds a NaN placeholder
                if source in trl_metrics and canonical not in metrics:
                    metrics[canonical] = trl_metrics[source]

            extra.update({
                "trl_metrics": trl_metrics,
                "algorithm": asdict(self._algorithm),
                "adapter_snapshot_s": self._adapter_snapshot_s,
                "num_zero_advantage_groups": batch.num_zero_advantage_groups,
                "microbatches": describe_plan(batch.rows, plan),
                "skip_old_logprob_forward": self._algorithm.skip_old_logprob_forward,
                "off_policy_by_staleness": trainer._arl_off_policy_by_staleness,
                "off_policy_by_staleness_partial": (trainer._arl_off_policy_by_staleness_partial),
                "behavior_versions": sorted({row.behavior_version for row in batch.rows}),
                "learner_version_before": learner_version_before,
                **{
                    key: getattr(self._model, flag) is not None
                    for key, flag in (
                        ("prefix_grouper_verified", "_unsloth_prefix_grouper_grad_verified"),
                        ("seq_packing_verified", "_unsloth_seq_packing_grad_ok"),
                    )
                    if hasattr(self._model, flag)
                },
                "rows_per_group": batch.rows_per_group,
                "num_dropped_untrainable": batch.num_dropped_untrainable,
                "num_dropped_fidelity": batch.num_dropped_fidelity,
                "fidelity_dropped_trajectory_ids": [
                    trajectory_id for trajectory_id, _ in batch.fidelity_dropped
                ],
                "num_dropped_small_group": batch.num_dropped_small_group,
                "num_zero_std_groups": batch.num_zero_std_groups,
                "num_rows_over_max_seq_length": batch.num_rows_over_max_len,
                "num_dropped_over_max_seq_length": batch.num_dropped_over_max_len,
                "num_forked_trajectories": batch.num_forked_trajectories,
                "num_dropped_forked_branches": batch.num_dropped_forked_branches,
                "call_unit": self._call_unit,
                "lr_schedule_horizon": self._lr_schedule_horizon,
                "sampling_temperature": self._sampling_temperature,
                "lr_scheduler_type": _scheduler_type_name(trainer.args),
                # Unsloth's patched GRPOConfig flips this on for loss_type=dapo ("The DAPO paper
                # recommends mask_truncated_completions = True"), so completions that hit
                # max_tokens contribute no loss. Surface the effective value per step.
                "mask_truncated_completions": bool(
                    getattr(trainer.args, "mask_truncated_completions", False)
                ),
                "num_truncated_rows": batch.num_truncated_rows,
                "warmup_steps": getattr(trainer.args, "warmup_steps", None),
                "memory": self._memory_step_extra(),
            })
            loss_stats_ended = _interval_boundary(torch)
            metrics["time/grad_clip_s"] = grad_clip_s
            metrics["time/loss_stats_s"] = loss_stats_ended - loss_stats_started
        except Exception as exc:  # noqa: BLE001 - diagnostics must not invalidate an update
            warnings.warn(
                f"step diagnostics failed at global step {self._global_step}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            metrics["diagnostic_error"] = 1.0
            extra["diagnostic_error"] = str(exc)

        wall_time_s = time.perf_counter() - started
        metrics["time/step_s"] = wall_time_s
        return StepResult(
            global_step=self._global_step,
            weight_version=self._weight_version,
            metrics=metrics,
            num_groups_received=batch.num_groups_received,
            num_groups_used=batch.num_groups_used,
            num_trajectories_used=batch.num_trajectories_used,
            num_sequences_used=batch.size,
            num_action_tokens=batch.num_action_tokens,
            wall_time_s=wall_time_s,
            trajectory_ids_used=batch.trajectory_ids,
            extra=extra,
        )

    def _memory_step_extra(self) -> dict[str, Any]:
        """Separate configured switches from engagement; deltas exclude preceding steps.

        Attention counters are process-wide, so concurrent backends share their increments.
        """
        extra: dict[str, Any] = dict(self._memory)
        extra["gradient_checkpointing"] = self._gradient_checkpointing
        extra["gradient_checkpointing_auto_max_tokens"] = (
            self._gradient_checkpointing_auto_max_tokens
        )
        extra["active"] = bool(getattr(self._trainer, "_arl_memory_unsloth", False))
        extra["compile_disabled"] = getattr(
            self, "_memory_compile_disabled", os.environ.get("UNSLOTH_COMPILE_DISABLE") == "1"
        )
        if self._uninstall_attention is not None:
            from moongazer_rl.trainer.unsloth_native.attention import blocked_attention_stats

            stats = blocked_attention_stats()
            stats["engaged_delta"] = stats["engaged"] - getattr(
                self, "_attention_engaged_previous", stats["engaged"]
            )
            self._attention_engaged_previous = stats["engaged"]
            extra["blocked_attention"] = stats
        return extra

    def _warn_once_about_bnpo(self, trainer: Any, num_chunks: int) -> None:
        """``bnpo`` normalises by the *micro-batch* token count, so splitting the backward pass
        changes its value slightly. ``grpo``, ``dr_grpo`` and ``dapo`` (the default) are exact.
        This is upstream TRL behaviour under gradient accumulation, not something the shim
        introduces, but it is worth saying out loud once per run."""
        if self._warned_bnpo or num_chunks <= 1:
            return
        if str(getattr(trainer.args, "loss_type", "")) != "bnpo":
            return
        self._warned_bnpo = True
        warnings.warn(
            f"loss_type='bnpo' with {num_chunks} micro-batches per step: bnpo normalises by the "
            "tokens of each micro-batch, so the loss differs slightly from an un-chunked step "
            "(same as TRL under gradient accumulation). Raise per_device_train_batch_size to "
            "fit the whole step in one micro-batch, or use dapo/dr_grpo/grpo, which are exact.",
            UserWarning,
            stacklevel=2,
        )

    def _push_weight_version(self) -> None:
        """The policy shares the learner's parameters, so a step makes new weights live."""
        sync = getattr(self._policy, "set_weight_version", None)
        if callable(sync):
            sync(self._weight_version)

    def _skipped_step(self, batch: RolloutBatch, started: float, *, reason: str) -> StepResult:
        """A batch was scored but the update was refused: no optimizer step, no weight bump."""
        result = self._empty_step(batch, started)
        if getattr(self, "_trainer", None) is not None:
            self._drain_trl_metrics(self._trainer)  # do not leak this step's TRL metrics
        result.extra["skipped"] = reason
        result.num_groups_received = batch.num_groups_received
        return result

    def _empty_step(self, batch: RolloutBatch, started: float) -> StepResult:
        """Nothing trainable arrived: no optimizer step, no weight bump, counts reported."""
        return StepResult(
            global_step=self._global_step,
            weight_version=self._weight_version,
            metrics={},
            num_groups_received=batch.num_groups_received,
            num_groups_used=0,
            num_trajectories_used=0,
            num_sequences_used=0,
            num_action_tokens=0,
            wall_time_s=time.perf_counter() - started,
            trajectory_ids_used=[],
            extra={
                "skipped": "no trainable rows",
                "num_dropped_untrainable": batch.num_dropped_untrainable,
                "num_dropped_fidelity": batch.num_dropped_fidelity,
                "fidelity_dropped_trajectory_ids": [
                    trajectory_id for trajectory_id, _ in batch.fidelity_dropped
                ],
                "num_dropped_small_group": batch.num_dropped_small_group,
                "num_rows_over_max_seq_length": batch.num_rows_over_max_len,
                "num_dropped_over_max_seq_length": batch.num_dropped_over_max_len,
            },
        )

    @staticmethod
    def _drain_trl_metrics(trainer: Any) -> dict[str, float]:
        """Average and clear TRL's ``_metrics['train']`` (normally done by ``trainer.log``)."""
        bucket = getattr(trainer, "_metrics", {}).get("train", {})
        out: dict[str, float] = {}
        for key, values in bucket.items():
            if isinstance(values, list) and values:
                try:
                    out[key] = float(sum(values) / len(values))
                except TypeError:  # pragma: no cover - non-numeric metric
                    continue
        bucket.clear()
        return out

    # -- optimizer -------------------------------------------------------------------
    def _trainable_parameters(self) -> list[Any]:
        if self._trainable_params is None:
            self._trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        return self._trainable_params

    def optimizer_info(self) -> dict[str, Any]:
        """What optimizer this learner runs: TRL's ``optim`` name, the torch class, the first
        param group's hyper-parameters and trainable-parameter count.
        Valid after ``setup``; logged as the ``optimizer`` event and into checkpoints."""
        trainer = self.trainer
        optimizer = trainer.optimizer
        inner = getattr(optimizer, "optimizer", optimizer)  # AcceleratedOptimizer -> torch
        group = dict(optimizer.param_groups[0]) if optimizer.param_groups else {}
        hyper = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in group.items()
            if key != "params" and isinstance(value, (int, float, bool, tuple, str))
        }
        optim_name = getattr(trainer.args, "optim", None)
        return {
            "optim": str(getattr(optim_name, "value", optim_name)),
            "optim_args": getattr(trainer.args, "optim_args", None),
            "class": type(inner).__name__,
            "hyperparameters": hyper,
            "configured_learning_rate": float(getattr(trainer.args, "learning_rate", 0.0)),
            "lr_scheduler_type": _scheduler_type_name(trainer.args),
            "num_trainable_parameters": int(
                sum(p.numel() for p in self._trainable_parameters())
            ),
        }

    # -- checkpoints ---------------------------------------------------------------
    def save_checkpoint(self, directory: Path) -> CheckpointInfo:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        trainer = self.trainer
        trainer.save_model(str(directory / "model"))
        torch.save(
            {
                "optimizer": trainer.optimizer.state_dict(),
                "lr_scheduler": trainer.lr_scheduler.state_dict()
                if trainer.lr_scheduler is not None
                else None,
            },
            directory / "optimizer.pt",
        )
        meta = {
            "backend": BACKEND_NAME,
            "advantage_estimator": self._algorithm.advantage_estimator,
            "global_step": self._global_step,
            "weight_version": self._weight_version,
            "trl_global_step": int(trainer.state.global_step),
            "model_name": self._model_name,
            "is_unsloth": self._is_unsloth,
            "lr_schedule_horizon": self._lr_schedule_horizon,
            "lr_scheduler_type": _scheduler_type_name(trainer.args),
            "sampling_temperature": self._sampling_temperature,
            "tokenizer": self.tokenizer_fingerprint(),
            "optimizer": self.optimizer_info(),
        }
        (directory / "trainer_state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return CheckpointInfo(
            path=directory,
            global_step=self._global_step,
            weight_version=self._weight_version,
            extra=meta,
        )

    def load_checkpoint(self, directory: Path) -> CheckpointInfo:
        import torch
        from peft import load_peft_weights, set_peft_model_state_dict

        directory = Path(directory)
        trainer = self.trainer
        adapter_dir = directory / "model"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"no adapter directory at {adapter_dir}")
        set_peft_model_state_dict(self._model, load_peft_weights(str(adapter_dir)))

        optimizer_file = directory / "optimizer.pt"
        if optimizer_file.exists():
            state = torch.load(optimizer_file, map_location="cpu", weights_only=False)
            trainer.optimizer.load_state_dict(state["optimizer"])
            if state.get("lr_scheduler") is not None and trainer.lr_scheduler is not None:
                trainer.lr_scheduler.load_state_dict(state["lr_scheduler"])
            # Order matters: adopt the configured base LR first, then recompute the current one
            # from it.
            self._apply_configured_learning_rate(trainer)
            self._resync_learning_rate(trainer, saved_meta(directory))

        meta: dict[str, Any] = {}
        meta_file = directory / "trainer_state.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            saved_estimator = meta.get("advantage_estimator")
            if saved_estimator != self._algorithm.advantage_estimator:
                warnings.warn(
                    f"checkpoint advantage_estimator {saved_estimator!r} differs from "
                    f"configured {self._algorithm.advantage_estimator!r}",
                    UserWarning,
                    stacklevel=2,
                )
            self._global_step = int(meta.get("global_step", self._global_step))
            self._weight_version = int(meta.get("weight_version", self._weight_version))
            trainer.state.global_step = int(meta.get("trl_global_step", trainer.state.global_step))
        self._push_weight_version()
        return CheckpointInfo(
            path=directory,
            global_step=self._global_step,
            weight_version=self._weight_version,
            extra=meta,
        )

    def _apply_configured_learning_rate(self, trainer: Any) -> None:
        """Let ``trainer.args.learning_rate`` win over the checkpoint's base LR.

        ``optimizer.load_state_dict`` restores the saved ``param_groups`` (``lr`` and
        ``initial_lr``) and ``LRScheduler.load_state_dict`` restores ``base_lrs``, so a resume
        that changes the learning rate — ``--set trainer.args.learning_rate=1e-5`` — would keep
        training at the old one and ``_resync_learning_rate`` would recompute from the *old*
        base. Overwriting both here makes the effective LR match the config that the run
        directory records.
        """
        configured = getattr(trainer.args, "learning_rate", None)
        if configured is None:
            return
        configured = float(configured)
        scheduler = trainer.lr_scheduler
        groups = list(trainer.optimizer.param_groups)
        restored = list(getattr(scheduler, "base_lrs", None) or []) or [
            float(group.get("initial_lr", group["lr"])) for group in groups
        ]
        if not restored or all(abs(float(base) - configured) <= 1e-12 for base in restored):
            return

        warnings.warn(
            f"resuming with a different learning rate: the checkpoint's base LR is "
            f"{restored[0]!r}, trainer.args.learning_rate is {configured!r}. The configured "
            "value wins; the schedule is re-based on it at the restored step.",
            UserWarning,
            stacklevel=2,
        )
        for group in groups:
            group["initial_lr"] = configured
            group["lr"] = configured
        if getattr(scheduler, "base_lrs", None):
            scheduler.base_lrs = [configured] * len(scheduler.base_lrs)

    def _resync_learning_rate(self, trainer: Any, saved: dict[str, Any]) -> None:
        """Recompute the param-group LRs from the *current* schedule at the restored position.

        ``optimizer.load_state_dict`` writes the saved ``param_groups`` back, including the
        ``lr`` the old schedule had reached, and ``LRScheduler.load_state_dict`` only restores
        ``last_epoch``/``base_lrs`` — it never touches the optimizer's ``lr``
        (torch 2.11 ``lr_scheduler.py``: ``self.__dict__.update(state_dict)``). With an
        unchanged horizon those two agree, so this is a no-op. With a *different* horizon they
        do not: a run that finished a 2-step linear decay saves ``lr == 0``, and resuming it
        with a 4-step horizon would keep stepping at 0 forever, because the LR is only rewritten
        by the *next* ``scheduler.step()`` — after the optimizer has already stepped.

        Rewinding ``last_epoch`` by one and calling ``step()`` puts it back where it was and
        writes ``base_lr * lambda_new(last_epoch)`` into every param group, using only the
        public API (so no ``get_lr()``-outside-``step()`` warning).
        """
        scheduler = trainer.lr_scheduler
        if scheduler is None or not hasattr(scheduler, "last_epoch"):
            return
        saved_horizon = saved.get("lr_schedule_horizon")
        saved_type = saved.get("lr_scheduler_type")
        current_type = _scheduler_type_name(trainer.args)
        if saved_horizon != self._lr_schedule_horizon or (
            saved_type is not None and saved_type != current_type
        ):
            warnings.warn(
                f"resuming with a different LR schedule: checkpoint had "
                f"{saved_type!r} over {saved_horizon!r} steps, this run has {current_type!r} "
                f"over {self._lr_schedule_horizon!r}. The learning rate is recomputed from the "
                "new schedule at the restored step, not carried over from the checkpoint.",
                UserWarning,
                stacklevel=2,
            )
        restored_epoch = int(scheduler.last_epoch)
        scheduler.last_epoch = restored_epoch - 1
        scheduler.step()

    def close(self) -> None:
        if self._uninstall_text_only_offload is not None:
            try:
                self._uninstall_text_only_offload()
            finally:
                self._uninstall_text_only_offload = None
        if self._uninstall_text_mm is not None:
            try:
                self._uninstall_text_mm()
            finally:
                self._uninstall_text_mm = None
        if self._uninstall_attention is not None:
            try:
                self._uninstall_attention()
            finally:
                self._uninstall_attention = None
        if self._uninstall_sliding_mask is not None:
            try:
                self._uninstall_sliding_mask()
            finally:
                self._uninstall_sliding_mask = None
        self._trainer = None
        self._policy = None
        self._model = None

    def _install_memory_settings(self, trainer: Any) -> None:
        """Hand the token-chunked logprob / one-row scoring switches to the shim (a no-op on
        trainers that predate the memory module)."""
        setter = getattr(trainer, "set_memory_settings", None)
        if not callable(setter):
            return
        from moongazer_rl.trainer.unsloth_native.memory import MemorySettings

        settings = MemorySettings(
            token_chunked_logprobs=self._memory["token_chunked_logprobs"],
            logprob_chunk_size=self._memory["logprob_chunk_size"],
            score_rows_one_at_a_time=self._memory["score_rows_one_at_a_time"],
        )
        setter(settings if settings.token_chunked_logprobs else None)
        self._memory_compile_disabled = os.environ.get("UNSLOTH_COMPILE_DISABLE") == "1"
        if (
            (
                getattr(trainer, "_arl_memory_unsloth", False)
                or self._uninstall_attention is not None
            )
            and not self._memory_compile_disabled
            and not getattr(self, "_warned_memory_compile", False)
        ):
            warnings.warn(
                "Memory primitives are eager boundaries (torch.compiler.disable) inside the "
                "Unsloth-compiled model.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_memory_compile = True

    @property
    def memory_settings(self) -> dict[str, Any]:
        return dict(self._memory)


def _validate_gemma4_micro_batch(
    model_name: str, args: dict[str, Any], num_generations: int
) -> list[str]:
    """Gemma 4 E-series + transformers 5.5.0/5.5.1: Unsloth's shared-KV carrier is module-scoped,
    so a micro-batch with more than one row (several forwards before one backward under
    gradient checkpointing) is refused by Unsloth at step 1 — after the rollouts were already
    spent. Refuse it at config time instead."""
    if "gemma-4-e" not in model_name.lower():
        return []
    try:
        from importlib.metadata import version

        parts = tuple(int(x) for x in version("transformers").split(".")[:3])
    except Exception:  # pragma: no cover - version unknown: do not block
        return []
    if parts >= (5, 5, 2):
        return []
    per_device = args.get("per_device_train_batch_size", num_generations)
    if int(per_device) == 1:
        return []
    return [
        "trainer.args.per_device_train_batch_size must be 1 for Gemma 4 E-series on transformers "
        f"{'.'.join(map(str, parts))}: Unsloth's shared-KV carrier cannot run several forwards "
        "before one backward under gradient checkpointing (transformers >= 5.5.2 lifts this)"
    ]


def _validate_single_process(args: dict[str, Any]) -> list[str]:
    """This backend drives one optimizer step itself and never prepares the model through
    accelerate, so the model-parallel launchers are out of scope. Reject them loudly rather
    than half-working."""
    problems: list[str] = []
    for key in ("deepspeed", "fsdp", "fsdp_config"):
        if args.get(key):
            problems.append(
                f"trainer.args.{key} is not supported by the unsloth-native backend: it replays "
                "Trainer's single-process step and does not run accelerator.prepare() on the "
                "model, which DeepSpeed/FSDP require."
            )
    return problems


def _interval_boundary(torch: Any) -> float:
    """Synchronize CUDA once at a timing boundary and return the host timestamp."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def saved_meta(directory: Path) -> dict[str, Any]:
    """``trainer_state.json`` of a checkpoint, or ``{}`` for one written before it existed."""
    meta_file = Path(directory) / "trainer_state.json"
    if not meta_file.exists():
        return {}
    try:
        loaded = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - unreadable checkpoint metadata
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _scheduler_type_name(args: Any) -> str:
    raw = getattr(args, "lr_scheduler_type", "")
    return str(getattr(raw, "value", raw))


def _scale_rewards(args: Any) -> str:
    """``GRPOConfig.scale_rewards`` accepts the legacy bools (grpo_config.py:644)."""
    raw = getattr(args, "scale_rewards", "group")
    if raw is True:
        return "group"
    if raw is False:
        return "none"
    return str(raw)


def _precomputed_reward(prompts: Any, completions: Any, completion_ids: Any, **kwargs: Any):
    """Reward function that returns the rollout reward carried by each row.

    ``GRPOTrainer._calculate_rewards`` forwards every dataset column that is not
    ``prompt``/``completion``/``completion_ids`` as a keyword argument
    (``trl/trainer/grpo_trainer.py:1022-1023``), so ``reward`` arrives as a list of floats,
    one per row, in row order. Verifiers already composed it; nothing is recomputed here.
    """
    rewards = kwargs.get("reward")
    if rewards is None:
        raise KeyError("rows must carry a 'reward' column")
    return [float(r) for r in rewards]


def _warn_if_pad_is_eos(tokenizer: Any) -> None:
    """unsloth_zoo's GRPO loss tells padding from tokens by id (left_pack_padding and
    create_completion_attention_mask compare against pad_token_id), so a tokenizer whose pad
    token is the eos token loses the loss on the final eos and can move an in-context eos as if
    it were padding. Gemma 4, Qwen 3.5 and llm-jp-4 ship dedicated pad tokens; warn for others."""
    pad = getattr(tokenizer, "pad_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    if pad is not None and pad in eos_ids:
        warnings.warn(
            f"pad_token_id ({pad}) equals eos_token_id: Unsloth's GRPO loss identifies padding "
            "by token id, so the final eos gets no loss and eos tokens inside the context are "
            "treated as padding. Use a tokenizer with a dedicated pad token (or set one) for "
            "training.",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = ["BACKEND_NAME", "UnslothNativeBackend", "unsloth_available"]
