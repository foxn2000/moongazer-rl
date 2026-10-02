"""Exact flex or query-blocked causal attention for Gemma 4 global text layers.

Motivation: with torch 2.11 cu130, head_dim=512 has no flash, memory-efficient, or
cuDNN SDPA kernel, so math SDPA materializes quadratic fp32 scores. E2B uses Hq=8, Hkv=1
and scale=1.0 in these layers.

In the blocked path, each query block attends its entire causal prefix, using absolute positions;
softmax is never split across keys. Backward recomputes that same probability
matrix and sums shared-KV gradients, so this is ordinary causal attention up to
floating-point rounding. Only q/k/v are saved: O(S*(Hq+Hkv)*D) storage and
O(Hq*Cq*S) score workspace per batch row, with quadratic arithmetic unchanged.
K/V are never repeated across query heads. Accumulation is fp32 (fp64 for fp64).

Flex also supports causal sliding windows. Non-goals: padding/packed/prefix
masks, dropout, second-order gradients, and UNSLOTH_FORCE_FLOAT32, whose raw
SDPA path bypasses the registry.
Install after Unsloth patches and before training compilation. The primitive is
an eager boundary; outer fullgraph compilation is not supported across it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


def _compatible_qkv(q: Tensor, k: Tensor, v: Tensor) -> bool:
    return (
        q.ndim == k.ndim == v.ndim == 4
        and k.shape == v.shape
        and q.shape[0] == k.shape[0]
        and q.shape[2:] == k.shape[2:]
        and q.shape[1] > 0
        and k.shape[1] > 0
        and q.shape[1] % k.shape[1] == 0
        and q.shape[2] > 0
        and q.shape[3] > 0
        and q.dtype == k.dtype == v.dtype
        and q.device == k.device == v.device
        and q.device.type in ("cpu", "cuda")
    )


def _block_probabilities(
    q: Tensor, k: Tensor, start: int, end: int, scale: float
) -> tuple[Tensor, Tensor]:
    """Fold query groups into rows so matmul never broadcasts/repeats K/V."""
    batch, heads, _, dim = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    width = end - start
    qb = q[:, :, start:end, :].to(k.dtype).reshape(batch, kv_heads, groups * width, dim)
    scores = torch.matmul(qb, k[:, :, :end, :].transpose(-2, -1)) * scale
    absolute_queries = torch.arange(start, end, device=q.device)
    keys = torch.arange(end, device=q.device)
    scores.view(batch, kv_heads, groups, width, end).masked_fill_(
        keys[None, :] > absolute_queries[:, None], float("-inf")
    )
    return qb, scores.softmax(dim=-1)


class BlockedCausalAttention(torch.autograd.Function):
    """First-order causal attention; save only inputs and recompute each tile."""

    @staticmethod
    def forward(
        ctx: Any, q: Tensor, k: Tensor, v: Tensor, scale: float, query_chunk_size: int
    ) -> Tensor:
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if not _compatible_qkv(q, k, v) or q.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Expected compatible floating-point [B,H,S,D] CPU/CUDA GQA inputs")
        ctx.save_for_backward(q, k, v)
        ctx.meta = scale, query_chunk_size
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        batch, heads, length, dim = q.shape
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        with torch.autocast(device_type=q.device.type, enabled=False):
            ka, va = k.to(dtype), v.to(dtype)
            for start in range(0, length, query_chunk_size):
                end = min(start + query_chunk_size, length)
                qb, p = _block_probabilities(q, ka, start, end, scale)
                out[:, :, start:end, :] = torch.matmul(p, va[:, :, :end, :]).reshape(
                    batch, heads, end - start, dim
                )
                del qb, p
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dout: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None]:
        q, k, v = ctx.saved_tensors
        scale, query_chunk_size = ctx.meta
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        batch, heads, length, dim = q.shape
        kv_heads = k.shape[1]
        dq = torch.empty(q.shape, dtype=dtype, device=q.device)
        dk = torch.zeros(k.shape, dtype=dtype, device=k.device)
        dv = torch.zeros(v.shape, dtype=dtype, device=v.device)
        with torch.autocast(device_type=q.device.type, enabled=False):
            ka, va = k.to(dtype), v.to(dtype)
            for start in range(0, length, query_chunk_size):
                end = min(start + query_chunk_size, length)
                qb, p = _block_probabilities(q, ka, start, end, scale)
                do = dout[:, :, start:end, :].to(dtype).reshape(batch, kv_heads, -1, dim)
                dp = torch.matmul(do, va[:, :, :end, :].transpose(-2, -1))
                ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
                dq[:, :, start:end, :] = (torch.matmul(ds, ka[:, :, :end, :]) * scale).reshape(
                    batch, heads, end - start, dim
                )
                # The row axis includes all query heads sharing this KV head.
                dk[:, :, :end, :] += torch.matmul(ds.transpose(-2, -1), qb) * scale
                dv[:, :, :end, :] += torch.matmul(p.transpose(-2, -1), do)
                del qb, p, do, dp, ds
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


@torch.compiler.disable()
def blocked_causal_attention(
    q: Tensor, k: Tensor, v: Tensor, *, scale: float, query_chunk_size: int = 128
) -> Tensor:
    """Return [B,Hq,S,D] attention in q.dtype, without compiling the block loop."""
    return BlockedCausalAttention.apply(q, k, v, scale, query_chunk_size)


_compiled_flex_attention: Callable[..., Tensor] | None = None
_causal_block_masks: dict[torch.device, dict[tuple[int, int | None], Any]] = {}
_flex_failed = False
_sliding_flex_failed = False


def _flex_mask_mod(window: int | None) -> Callable:
    def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        # HF sliding_window_overlay: ki > qi - window (self counts in window).
        causal = ki <= qi
        return causal if window is None else causal & (qi - ki < window)

    return mask_mod


def _flex_kernel_options(device: torch.device, head_dim: int = 512) -> dict[str, int]:
    if (
        device.type == "cuda"
        and getattr(torch.cuda.get_device_properties(device), "shared_memory_per_block_optin", 0)
        < 128 * 1024
    ):
        return {
            name: 32 if head_dim <= 256 else 16
            for name in ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")
        }
    return {}


# Skip padding/cache/fallback in Unsloth's outer graph, but allow the explicitly
# compiled flex and mask builder to compile: recursive=True would disable callees.
@torch.compiler.disable(recursive=False)
def flex_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    scale: float,
    query_chunk_size: int = 128,
    window: int | None = None,
) -> Tensor:
    """Pad causal/windowed attention; sliding failures are re-raised for the router to handle."""
    global _compiled_flex_attention, _flex_failed
    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError("window must be a positive integer or None")
    if window is not None or not _flex_failed:
        try:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention

            if _compiled_flex_attention is None:
                _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
            length = q.shape[2]
            padded_length = (length + 127) // 128 * 128
            masks = _causal_block_masks.setdefault(q.device, {})
            cache_key = (padded_length, window)
            if cache_key not in masks:
                # Causality is batch/head independent; None broadcasts the same
                # mask even when batch size changes within a sequence bucket.
                masks[cache_key] = create_block_mask(
                    _flex_mask_mod(window),
                    None,
                    None,
                    padded_length,
                    padded_length,
                    device=q.device,
                    _compile=True,
                )
            padding = padded_length - length
            qp, kp, vp = (
                torch.nn.functional.pad(t, (0, 0, 0, padding)) if padding else t for t in (q, k, v)
            )
            out = _compiled_flex_attention(
                qp,
                kp,
                vp,
                block_mask=masks[cache_key],
                scale=scale,
                enable_gqa=True,
                kernel_options=_flex_kernel_options(q.device, q.shape[-1]),
            )
            # Real queries cannot see padded keys; slicing also drops padded dQ.
            return out[:, :, :length, :]
        except Exception as exc:
            if window is not None:
                raise  # Only the router can preserve Unsloth's sliding semantics.
            _flex_failed = True
            warnings.warn(
                f"Flex attention failed; using blocked attention for this process: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return blocked_causal_attention(q, k, v, scale=scale, query_chunk_size=query_chunk_size)


_engaged = 0
_deferred = 0
_last_shape: tuple[int, ...] | None = None


def install_unmaterialized_sliding_mask() -> Callable[[], None]:
    """Skip the dense SDPA mask that Unsloth's banded sliding attention does not use."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    previous = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(previous, "_arl_unmaterialized_sliding_mask", False):
        return previous.__dict__["_arl_unmaterialized_sliding_mask_acquire"]()
    users = 0

    def mask_router(**kwargs: Any) -> Tensor | None:
        q_length = kwargs.get("q_length")
        kv_length = kwargs.get("kv_length")
        local_size = kwargs.get("local_size")
        if (
            local_size is not None
            and not _sliding_flex_failed  # Unsloth's sliding path needs the dense mask
            and kwargs.get("attention_mask") is None
            and kwargs.get("allow_is_causal_skip", True) is True
            and q_length == kv_length
            and q_length >= local_size
        ):
            return None
        return previous(**kwargs)

    def acquire() -> Callable[[], None]:
        nonlocal users
        users += 1
        released = False

        def uninstall() -> None:
            nonlocal users, released
            if not released:
                users -= 1
                released = True
            if users == 0 and ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] is mask_router:
                ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = previous

        return uninstall

    mask_router.__dict__["_arl_unmaterialized_sliding_mask"] = True
    mask_router.__dict__["_arl_unmaterialized_sliding_mask_acquire"] = acquire
    ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = mask_router
    return acquire()


def blocked_attention_stats() -> dict[str, Any]:
    """Process-lifetime counts and the last engaged query shape [B,Hq,S,D]."""
    return {"engaged": _engaged, "deferred": _deferred, "last_shape": _last_shape}


def install_blocked_global_attention(
    *,
    min_head_dim: int = 257,
    min_q_len: int = 1024,
    query_chunk_size: int = 128,
    global_attention: str = "flex",
    sliding_attention: str = "flex",
    blocked_global_attention: bool = True,
) -> Callable[[], None]:
    """Share the SDPA router until every caller releases its own installation.

    Repeated installation keeps the first thresholds/mode and warns once on conflict.
    Reference counting prevents one backend's teardown from disabling another.
    Restore only while this wrapper is current, preserving later patches.
    Transformers is imported here so importing the primitive does not load it
    before Unsloth. Callers must check registry ownership again after model load.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if min_head_dim <= 0 or min_q_len <= 0 or query_chunk_size <= 0:
        raise ValueError("Attention thresholds and query_chunk_size must be positive")
    if global_attention not in ("flex", "blocked"):
        raise ValueError("global_attention must be 'flex' or 'blocked'")
    if sliding_attention not in ("flex", "unsloth"):
        raise ValueError("sliding_attention must be 'flex' or 'unsloth'")
    previous = ALL_ATTENTION_FUNCTIONS["sdpa"]
    thresholds = (
        min_head_dim,
        min_q_len,
        query_chunk_size,
        global_attention,
        sliding_attention,
        blocked_global_attention,
    )
    if getattr(previous, "_arl_blocked_global_attention", False):
        return previous.__dict__["_arl_blocked_global_attention_acquire"](thresholds)
    users = 0
    warned_thresholds = False

    def router(
        module: torch.nn.Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None]:
        global _engaged, _deferred, _last_shape
        global _sliding_flex_failed
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        if (
            type(module).__name__.startswith("Gemma4")
            and causal is True
            and attention_mask is None
            and dropout == 0.0
            and _compatible_qkv(query, key, value)
            and query.shape[2] >= min_q_len
            and query.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            # None follows SDPA's default; Gemma 4 explicitly supplies 1.0.
            scale = scaling if scaling is not None else query.shape[-1] ** -0.5
            out = None
            if getattr(module, "is_sliding", False):
                window = kwargs.get("sliding_window")
                if window is None:
                    window = getattr(module, "sliding_window", None)
                if window is None:
                    window = getattr(getattr(module, "config", None), "sliding_window", None)
                if (
                    sliding_attention == "flex"
                    and not _sliding_flex_failed
                    and query.shape[-1] <= 256
                    and isinstance(window, int)
                    and window > 0
                ):
                    try:
                        out = flex_causal_attention(query, key, value, scale=scale, window=window)
                    except Exception as exc:
                        _sliding_flex_failed = True
                        warnings.warn(
                            f"Sliding flex attention failed; using previous Unsloth router "
                            f"for this process: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            elif blocked_global_attention and query.shape[-1] >= min_head_dim:
                attention = (
                    flex_causal_attention
                    if global_attention == "flex"
                    else blocked_causal_attention
                )
                out = attention(query, key, value, scale=scale, query_chunk_size=query_chunk_size)
            if out is not None:
                _engaged += 1
                _last_shape = tuple(query.shape)
                return out.transpose(1, 2).contiguous(), None
        _deferred += 1
        return previous(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    def acquire(requested: tuple[int, int, int, str, str, bool]) -> Callable[[], None]:
        nonlocal users, warned_thresholds
        if requested != thresholds and not warned_thresholds:
            warnings.warn(
                f"Blocked attention already installed with thresholds {thresholds}; "
                f"keeping them instead of requested {requested}",
                RuntimeWarning,
                stacklevel=3,
            )
            warned_thresholds = True
        users += 1
        released = False

        def uninstall() -> None:
            nonlocal users, released
            if not released:
                users -= 1
                released = True
            if users == 0 and ALL_ATTENTION_FUNCTIONS["sdpa"] is router:
                ALL_ATTENTION_FUNCTIONS["sdpa"] = previous

        return uninstall

    # Load-bearing: gemma4_flash_sliding's re-entry guard reads the registry entry.
    # Without its sentinel, re-patching sets _ORIG_SDPA[0] to our router and recurses.
    for name, sentinel in getattr(previous, "__dict__", {}).items():
        if name.startswith(("_unsloth", "__unsloth")):
            setattr(router, name, sentinel)
    router.__dict__["_arl_blocked_global_attention"] = True
    router.__dict__["_arl_blocked_global_attention_acquire"] = acquire
    ALL_ATTENTION_FUNCTIONS["sdpa"] = router
    return acquire(thresholds)
