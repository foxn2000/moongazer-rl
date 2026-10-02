"""Text-only training on the larger Gemma 4 checkpoints (transformers 5.5).

``Gemma4Model.forward`` routes through ``create_causal_mask_mapping`` when the text config has
``use_bidirectional_attention == "vision"`` (the 26B/31B checkpoints; the E-series does not), and
that helper raises ``"mm_token_type_ids is required as a model input when training"`` whenever
the module is in training mode and no ``mm_token_type_ids`` were passed
(``transformers/models/gemma4/modeling_gemma4.py:2004-2005``). TRL and Unsloth only pass the key
for multimodal rows, so a text-only training forward of these checkpoints raises.

For text-only rows the correct value is all zeros (type 0 = text token), so a forward pre-hook on
the multimodal ``Gemma4Model`` supplies it when absent. Rows carrying real multimodal type ids
are untouched; models whose config does not require it get no hook.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _gemma4_model(model: Any) -> Any | None:
    """The multimodal ``Gemma4Model`` behind a (PEFT-wrapped) ``Gemma4ForConditionalGeneration``."""
    current = model
    while hasattr(current, "module") or hasattr(current, "_orig_mod"):
        current = current.module if hasattr(current, "module") else current._orig_mod
    if hasattr(current, "get_base_model"):
        current = current.get_base_model()
    if getattr(getattr(current, "config", None), "model_type", None) != "gemma4":
        return None
    inner = getattr(current, "model", None)
    if inner is None or type(inner).__name__ != "Gemma4Model":
        return None
    return inner


def needs_text_mm_token_type_ids(model: Any) -> bool:
    inner = _gemma4_model(model)
    if inner is None:
        return False
    text_config = inner.config.get_text_config()
    return getattr(text_config, "use_bidirectional_attention", None) == "vision"


def install_text_only_mm_token_type_ids(model: Any) -> Callable[[], None] | None:
    """Register the pre-hook when the checkpoint needs it. Returns the uninstall callable, or
    ``None`` when the model does not route text through the multimodal mask path."""
    if not needs_text_mm_token_type_ids(model):
        return None
    inner = _gemma4_model(model)

    def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if kwargs.get("mm_token_type_ids") is not None or kwargs.get("pixel_values") is not None:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            kwargs["mm_token_type_ids"] = torch.zeros_like(input_ids, dtype=torch.long)
            return args, kwargs
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None:
            kwargs["mm_token_type_ids"] = torch.zeros(
                embeds.shape[:2], dtype=torch.long, device=embeds.device
            )
            return args, kwargs
        return None

    handle = inner.register_forward_pre_hook(hook, with_kwargs=True)
    return handle.remove


def offload_text_only_modules(model: Any) -> Callable[[], None] | None:
    """Move Gemma 4 E-series modules unused by text-only training to CPU."""
    if not hasattr(model, "named_modules"):
        return None
    per_layer_embedding = next(
        (
            module
            for name, module in model.named_modules()
            if name.rsplit(".", 1)[-1] == "embed_tokens_per_layer"
        ),
        None,
    )
    if per_layer_embedding is None or per_layer_embedding.weight.requires_grad:
        return None  # only a frozen table may live on the CPU

    original_device = per_layer_embedding.weight.device
    original_forward = per_layer_embedding.forward
    had_instance_forward = "forward" in per_layer_embedding.__dict__
    instance_forward = per_layer_embedding.__dict__.get("forward")
    per_layer_embedding.to("cpu")
    if torch.cuda.is_available():
        per_layer_embedding.weight.data = per_layer_embedding.weight.detach().pin_memory()

    def cpu_forward(input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        input_device = input_ids.device
        output = original_forward(input_ids.to("cpu", non_blocking=True), *args, **kwargs)
        return output.to(input_device, non_blocking=True)

    per_layer_embedding.forward = cpu_forward

    offloaded: list[tuple[Any, torch.device]] = []
    tower_names = {"vision_tower", "audio_tower", "embed_vision", "embed_audio"}
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in tower_names:
            continue
        tensor = next(iter(module.parameters()), None)
        if tensor is None:
            tensor = next(iter(module.buffers()), None)
        if tensor is None:
            continue
        offloaded.append((module, tensor.device))
        module.to("cpu")

    uninstalled = False

    def uninstall() -> None:
        nonlocal uninstalled
        if uninstalled:
            return
        uninstalled = True
        per_layer_embedding.to(original_device)
        if had_instance_forward:
            per_layer_embedding.forward = instance_forward
        else:
            per_layer_embedding.__dict__.pop("forward", None)
        for module, device in offloaded:
            module.to(device)

    return uninstall


__all__ = [
    "install_text_only_mm_token_type_ids",
    "needs_text_mm_token_type_ids",
    "offload_text_only_modules",
]
