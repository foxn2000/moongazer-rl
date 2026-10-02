"""Design B: Gemma 4 hidden-only GRPO, with bounded vocabulary intermediates.

Only the final projection is token-chunked. The transformer and GRPO sequence axes
stay intact. Unsloth's generated modules and the patched Gemma forwards are never edited.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Any

import torch
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class MemorySettings:
    token_chunked_logprobs: bool = True
    logprob_chunk_size: int = 64
    score_rows_one_at_a_time: bool = True

    def __post_init__(self) -> None:
        # Both paths must use trimmed positions; the padded Unsloth fallback differs.
        if self.token_chunked_logprobs != self.score_rows_one_at_a_time:
            raise ValueError("token_chunked_logprobs and score_rows_one_at_a_time must match")
        if self.logprob_chunk_size < 1:
            raise ValueError("logprob_chunk_size must be positive")


def _project_logprobs(hidden, weight, ids, temperature, softcap):
    logits = hidden @ weight.t()
    # Gemma4ForConditionalGeneration.forward, transformers 5.5.0:2447-2451.
    if softcap is not None:
        logits = logits / softcap
        logits = torch.tanh(logits)
        logits = logits * softcap
    logits = logits.float() / temperature
    return logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)


class TokenLogprobs(torch.autograd.Function):
    """Save hidden/weight/ids only; finish each chunk's recompute/backward in turn."""

    @staticmethod
    def forward(ctx: Any, hidden, weight, ids, temperature, softcap, chunk_size):
        if weight.requires_grad:
            raise NotImplementedError("TokenLogprobs requires a frozen lm_head weight")
        ctx.save_for_backward(hidden, weight, ids)
        ctx.meta = temperature, softcap, chunk_size
        ctx.device_type = hidden.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)
        flat = hidden.reshape(-1, hidden.shape[-1])
        targets = ids.reshape(-1)
        output = torch.empty(targets.shape, device=hidden.device, dtype=torch.float32)
        for start in range(0, targets.numel(), chunk_size):
            sl = slice(start, start + chunk_size)
            output[sl] = _project_logprobs(flat[sl], weight, targets[sl], temperature, softcap)
        return output.reshape(ids.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_logps):
        hidden, weight, ids = ctx.saved_tensors
        temperature, softcap, chunk_size = ctx.meta
        flat = hidden.reshape(-1, hidden.shape[-1])
        targets = ids.reshape(-1)
        upstream = grad_logps.reshape(-1)
        grad_hidden = torch.empty_like(flat)
        for start in range(0, targets.numel(), chunk_size):
            sl = slice(start, start + chunk_size)
            with (
                torch.enable_grad(),
                torch.autocast(
                    ctx.device_type, dtype=ctx.autocast_dtype, enabled=ctx.autocast_enabled
                ),
            ):
                h = flat[sl].detach().requires_grad_(True)
                lp = _project_logprobs(h, weight, targets[sl], temperature, softcap)
                (dh,) = torch.autograd.grad(lp, h, upstream[sl])
            grad_hidden[sl] = dh
            del h, lp, dh
        return grad_hidden.reshape(hidden.shape), None, None, None, None, None


@torch.compiler.disable()
def token_chunked_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    temperature: float,
    softcap: float | None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """fp32 selected logprobs; projection/softcap retain the caller's autocast dtype."""
    if chunk_size < 1 or temperature <= 0 or (softcap is not None and softcap <= 0):
        raise ValueError("chunk_size, temperature and any softcap must be positive")
    if hidden.ndim not in (2, 3) or target_ids.shape != hidden.shape[:-1]:
        raise ValueError("hidden must be [N,H] or [B,W,H], with matching target_ids")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("lm_head must be [V,H] with the same hidden dimension")
    return TokenLogprobs.apply(hidden, weight, target_ids, temperature, softcap, chunk_size)


def _gemma4_base(model: Any) -> Any | None:
    from peft import PeftModel

    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    if not isinstance(model, PeftModel):
        return None
    base = model.get_base_model()
    if getattr(base.config, "model_type", None) != "gemma4":
        return None
    from transformers import Gemma4ForConditionalGeneration

    return base if isinstance(base, Gemma4ForConditionalGeneration) else None


def is_gemma4_peft(model: Any) -> bool:
    return _gemma4_base(model) is not None


def resolve_lm_head(model: Any) -> tuple[torch.Tensor, float | None]:
    base = _gemma4_base(model)
    if base is None:
        raise NotImplementedError("memory GRPO requires a Gemma4 conditional-generation PEFT model")
    head = base.get_output_embeddings()
    # LoRA's base weight can be frozen while its A/B matrices are trainable.
    if any(p.requires_grad for p in head.parameters()) or hasattr(head, "lora_A"):
        raise NotImplementedError("memory GRPO does not support trainable lm_head / lm_head LoRA")
    if getattr(head, "bias", None) is not None:
        raise NotImplementedError("memory GRPO requires a bias-free lm_head")
    text_config = base.config.get_text_config()
    if head.weight.ndim != 2 or head.weight.shape[1] != text_config.hidden_size:
        raise ValueError("lm_head must be [V,H] with H equal to the model hidden_size")
    return head.weight, text_config.final_logit_softcapping


def gemma4_text_hidden_forward(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Call the live patched inner model, preserving LoRA and the shared-KV carrier.

    Boundary: transformers/models/gemma4/modeling_gemma4.py:2425-2444.
    KV carrier/non-reentrant wrappers: unsloth_zoo/temporary_patches/gemma4.py:
    576-658,844-856. No adapter switch, no all-layer hidden-state collection.
    """
    base = _gemma4_base(model)
    if base is None:
        raise NotImplementedError("hidden-only forward requires Gemma4 conditional-generation PEFT")
    return base.model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
    ).last_hidden_state


def _window(input_ids, attention_mask, logits_to_keep, pad_id, width=None):
    # Mirror Zoo rl_replacements.py:170-213 and U/models/rl_replacements.py:1527-1534:
    # count pad IDs in the prompt BEFORE the row loop (not per-row max padding).
    if not 0 < logits_to_keep < input_ids.shape[1]:
        raise ValueError("logits_to_keep must be positive and smaller than sequence length")
    nonpad = input_ids != pad_id
    left_pad = (~nonpad[:, :-logits_to_keep]).sum(1)
    if width is None:
        width = logits_to_keep + int(left_pad.max())
    extra = width - logits_to_keep
    if extra < int(left_pad.max()) or width >= input_ids.shape[1]:
        raise ValueError("old/ref logps have an incompatible global window width")
    if attention_mask is not None and not torch.equal(attention_mask.bool(), nonpad):
        raise NotImplementedError(
            "memory GRPO requires an attention mask matching pad tokens; memory scoring needs "
            "a dedicated pad token: disable trainer.memory or set a dedicated pad token"
        )
    lengths = nonpad.sum(1)
    for i in range(input_ids.shape[0]):
        left, length = int(left_pad[i]), int(lengths[i])
        if length == 0 or not bool(nonpad[i, left : left + length].all()):
            raise NotImplementedError("memory GRPO requires contiguous left/right padding")
    return left_pad, lengths, width


def _trainer_autocast(trainer: Any, device: torch.device):
    # Reuse the generated precision decision (U/models/rl_replacements.py:916).
    original = getattr(
        trainer, "_arl_original_compute_loss", getattr(trainer, "compute_loss", None)
    )
    if original is None:
        return nullcontext()
    fn = inspect.unwrap(getattr(original, "__func__", original))
    init = getattr(fn, "__globals__", {}).get("_unsloth_grpo_autocast")
    if init is not None:
        init(trainer)
    if device.type != "cuda" or not hasattr(trainer, "_autocast_dtype"):
        return nullcontext()
    dtype = trainer._autocast_dtype
    return torch.autocast(
        device.type,
        dtype=dtype,
        enabled=dtype is not None and getattr(trainer, "_autocast_enabled", True),
    )


def _row_logps(model, ids, left, length, width, weight, softcap, temperature, chunk_size):
    # The packed window starts at S-W for EVERY row. Include its valid prompt columns.
    start = ids.shape[1] - width
    trimmed = ids[:, left : left + length]
    hidden = gemma4_text_hidden_forward(model, trimmed)
    if length <= start:
        return hidden.sum().float().mul(0).expand(1, width)
    logps = token_chunked_logprobs(
        hidden[:, start - 1 : length - 1].to(device=weight.device, dtype=weight.dtype),
        weight,
        trimmed[:, start:length].to(weight.device),
        temperature=temperature,
        softcap=softcap,
        chunk_size=chunk_size,
    ).to(ids.device)
    return torch.nn.functional.pad(logps, (0, width - logps.shape[1]))


@torch.no_grad()
@torch.compiler.disable()
def score_per_token_logps(
    trainer: Any,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """One model row at a time; return fp32 [B,C+global_max_left_pad], pad columns zero."""
    weight, softcap = resolve_lm_head(model)
    left_pad, lengths, width = _window(
        input_ids, attention_mask, logits_to_keep, trainer.processing_class.pad_token_id
    )
    output = torch.zeros((input_ids.shape[0], width), dtype=torch.float32, device=input_ids.device)
    with _trainer_autocast(trainer, input_ids.device):
        for i in range(input_ids.shape[0]):
            output[i : i + 1] = _row_logps(
                model,
                input_ids[i : i + 1],
                int(left_pad[i]),
                int(lengths[i]),
                width,
                weight,
                softcap,
                trainer.temperature,
                chunk_size,
            )
    return output


@torch.compiler.disable()
def accumulated_loss(
    trainer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_logps: torch.Tensor | None,
    ref_logps: torch.Tensor | None,
    n_chunks: int = -1,
    tool_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Shadow Zoo rl_replacements.py:807-819, exact argument/default and 7-tuple contract.

    grpo_accumulated_loss(trainer, input_ids, attention_mask, logits_to_keep,
    completion_mask, advantages, old_logps, ref_logps, n_chunks=-1, tool_mask=None, **kwargs).
    Run each trimmed row separately so projection workspace is independent of B.
    Assemble the global [B,W] window before Zoo's grpo_compute_loss (:425-648),
    preserving batch/sequence normalisers even when prompt and completion lengths differ.
    """
    for name in (
        "pixel_values",
        "pixel_values_videos",
        "input_features",
        "image_grid_thw",
        "pixel_attention_mask",
        "image_sizes",
        "num_images",
        "token_type_ids",
        "mm_token_type_ids",
    ):
        if kwargs.get(name) is not None:
            raise NotImplementedError("memory GRPO currently supports text-only inputs")
    if kwargs.get("logit_scale_multiply", 0) or kwargs.get("logit_scale_divide", 0):
        raise NotImplementedError("memory GRPO supports Gemma4 softcapping only")
    zoo = importlib.import_module("unsloth_zoo.rl_replacements")
    weight, softcap = resolve_lm_head(trainer.model)
    if "logit_softcapping" in kwargs and kwargs["logit_softcapping"] != softcap:
        raise ValueError(
            f"logit_softcapping={kwargs['logit_softcapping']!r} disagrees with "
            f"lm_head softcap={softcap!r}"
        )
    prior = old_logps if old_logps is not None else ref_logps
    left_pad, lengths, width = _window(
        input_ids,
        attention_mask,
        logits_to_keep,
        trainer.processing_class.pad_token_id,
        None if prior is None else prior.shape[1],
    )
    for logps in (old_logps, ref_logps):
        if logps is not None and logps.shape != (input_ids.shape[0], width):
            raise ValueError("old/ref logps window shapes disagree")
    # Zoo :915-938; tool tokens stay in the transformer context.
    packed = zoo.left_pack_padding(input_ids, trainer.processing_class.pad_token_id)
    completion_ids = packed[:, -width:]
    mask = zoo.create_completion_attention_mask(
        completion_ids, left_pad, width - logits_to_keep, trainer.processing_class.pad_token_id
    ).to(attention_mask.dtype)
    sampling = kwargs.pop("sampling_per_token_logps", None)
    diagnostic_sampling = None if sampling is None else zoo.align_logprobs_with_mask(sampling, mask)
    if (
        trainer.use_vllm
        and getattr(trainer, "vllm_importance_sampling_correction", False)
        and sampling is not None
    ):
        sampling = diagnostic_sampling
    else:
        sampling = None
    mask = zoo.align_completion_tool_mask(tool_mask, mask)
    # Zoo :841-867, including TRL 0.24's legacy token-truncate/cap resolution.
    cap = getattr(trainer.args, "vllm_importance_sampling_cap", None)
    clip_min = getattr(trainer.args, "vllm_importance_sampling_clip_min", None)
    clip_max = getattr(trainer.args, "vllm_importance_sampling_clip_max", None)
    if clip_max is None and cap is not None:
        clip_min, clip_max = 0, cap
    kwargs.update(
        use_vllm=trainer.use_vllm,
        vllm_importance_sampling_cap=cap,
        vllm_importance_sampling_mode=(
            getattr(trainer.args, "vllm_importance_sampling_mode", None) or "token_truncate"
        ),
        vllm_importance_sampling_clip_min=clip_min,
        vllm_importance_sampling_clip_max=clip_max,
    )
    for name in ("get_sapo_token_loss", "get_gamma_weights", "get_off_policy_mask"):
        kwargs[name] = getattr(trainer, name, None)
    for name, default in (
        ("sapo_temperature_pos", None),
        ("sapo_temperature_neg", None),
        ("vespo_k_pos", 2.0),
        ("vespo_k_neg", 3.0),
        ("vespo_lambda_pos", 3.0),
        ("vespo_lambda_neg", 2.0),
        ("off_policy_mask_threshold", None),
    ):
        kwargs[name] = getattr(trainer.args, name, default)
    with _trainer_autocast(trainer, input_ids.device):
        new = torch.cat(
            [
                _row_logps(
                    trainer.model,
                    input_ids[i : i + 1],
                    int(left_pad[i]),
                    int(lengths[i]),
                    width,
                    weight,
                    softcap,
                    kwargs.get("temperature", 1.0),
                    trainer.memory.logprob_chunk_size,
                )
                for i in range(input_ids.shape[0])
            ],
            dim=0,
        )
        spec = getattr(trainer, "_arl_spec", None)
        if spec is not None:
            if spec.ratio_reference == "behavior" or spec.policy_objective == "reinforce":
                old_logps = None
        if spec is not None and old_logps is None and sampling is not None:
            old_logps = new.detach()
        record = getattr(trainer, "_arl_record_logps", None)
        if record is not None:
            record(new, old_logps, ref_logps, diagnostic_sampling, mask, advantages)
        loss, length, kl, delta, ratios, coef, _ = zoo.grpo_compute_loss(
            ref_logps,
            new,
            old_logps,
            sampling,
            completion_ids,
            mask,
            trainer.beta,
            advantages,
            **kwargs,
        )
    # UnslothEfficientGRPO :778-782: absent sampling => None; metrics carry no loss graph.
    # TRL 0.24 sets compute_loss_func != None, so Trainer.training_step does not
    # divide again. Zoo owns GRPO accumulation / DAPO token normalisation; Accelerator
    # still owns mixed-precision scaling.
    return (
        loss,
        length.detach(),
        kl.detach(),
        None if sampling is None else delta.detach(),
        None if sampling is None else ratios.detach(),
        coef.detach(),
        mask,
    )


def has_unsloth_loss(trainer: Any) -> bool:
    original = getattr(
        trainer, "_arl_original_compute_loss", getattr(trainer, "compute_loss", None)
    )
    if original is None:
        return False
    fn = inspect.unwrap(getattr(original, "__func__", original))
    return (
        isinstance(fn, FunctionType) and "grpo_accumulated_loss" in fn.__code__.co_names
    ) or any(cls.__name__ == "UnslothGRPOTrainer" for cls in type(trainer).__mro__)


def make_memory_efficient_compute_loss(trainer: Any) -> Callable:
    """Copy generated compute_loss; replace ONE global in the private globals dictionary.

    U/models/rl_replacements.py:2491-2504 generates independent globals; :2766-2798
    calls grpo_accumulated_loss. Changing the installed Zoo symbol cannot intercept it.
    """
    original = getattr(
        trainer, "_arl_original_compute_loss", getattr(trainer, "compute_loss", None)
    )
    if original is None:
        raise RuntimeError("unsupported Unsloth compute_loss: direct grpo_accumulated_loss missing")
    fn = inspect.unwrap(getattr(original, "__func__", original))
    if not isinstance(fn, FunctionType) or "grpo_accumulated_loss" not in fn.__code__.co_names:
        raise RuntimeError("unsupported Unsloth compute_loss: direct grpo_accumulated_loss missing")
    namespace = fn.__globals__.copy()
    upstream = namespace.get("grpo_accumulated_loss")
    actual = inspect.signature(upstream).parameters if callable(upstream) else {}
    required = inspect.signature(accumulated_loss).parameters
    if list(actual) != list(required) or any(
        actual[k].kind != p.kind or actual[k].default != p.default
        for k, p in required.items()
        if k in actual
    ):
        raise RuntimeError("unsupported Unsloth grpo_accumulated_loss signature")
    if not callable(namespace.get("grpo_compute_loss")):
        raise RuntimeError("unsupported Unsloth globals: grpo_compute_loss missing")
    namespace["grpo_accumulated_loss"] = accumulated_loss
    return MethodType(copy_function_globals(fn, namespace), trainer)


def copy_function_globals(fn: Callable, replacements: dict[str, Any]) -> Callable:
    """Runtime function clone; never mutate a generated module's shared globals."""
    fn = inspect.unwrap(getattr(fn, "__func__", fn))
    namespace = {**fn.__globals__, **replacements}
    copied = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    copied.__kwdefaults__ = fn.__kwdefaults__
    copied.__annotations__ = fn.__annotations__.copy()
    copied.__dict__.update(fn.__dict__)
    copied.__qualname__ = fn.__qualname__
    copied.__doc__ = fn.__doc__
    return copied
