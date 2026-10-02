"""Snapshot PEFT LoRA weights separately from their on-disk representation."""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import save_file


@dataclass(frozen=True)
class AdapterSnapshot:
    version: int
    tensors: dict[str, torch.Tensor]
    adapter_config: dict[str, Any]
    taken_at: float
    snapshot_s: float


def _adapter_config_dict(model: Any) -> dict[str, Any]:
    """Mirror the config preparation in ``PeftModel.save_pretrained`` without mutating it."""
    try:
        config = copy.deepcopy(model.peft_config["default"])
    except (AttributeError, KeyError) as exc:
        raise TypeError("model has no default PEFT adapter") from exc

    if config.base_model_name_or_path is None:
        config.base_model_name_or_path = (
            model.base_model.__dict__.get("name_or_path", None)
            if config.is_prompt_learning
            else model.base_model.model.__dict__.get("name_or_path", None)
        )
    config.inference_mode = True
    output = config.to_dict()
    for key, value in output.items():
        if isinstance(value, set):
            output[key] = list(value)

    if config.task_type is None:
        base_model_class = model._get_base_model_class(is_prompt_tuning=config.is_prompt_learning)
        output["auto_mapping"] = {
            "base_model_class": base_model_class.__name__,
            "parent_library": base_model_class.__module__,
        }
    return output


def snapshot_lora(model: Any, version: int) -> AdapterSnapshot:
    """Clone the default PEFT adapter to detached, contiguous CPU tensors."""
    started_at = time.time()
    started_mono = time.monotonic()
    state = get_peft_model_state_dict(model)
    tensors = {
        name: tensor.detach().to("cpu", copy=True).contiguous() for name, tensor in state.items()
    }
    adapter_config = _adapter_config_dict(model)
    return AdapterSnapshot(
        version=int(version),
        tensors=tensors,
        adapter_config=adapter_config,
        taken_at=started_at,
        snapshot_s=time.monotonic() - started_mono,
    )


def write_adapter(snapshot: AdapterSnapshot, target: Path) -> Path:
    """Materialize ``snapshot`` through a sibling temporary directory and rename it."""
    target = Path(target)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        save_file(
            snapshot.tensors,
            tmp / "adapter_model.safetensors",
            metadata={"format": "pt"},
        )
        (tmp / "adapter_config.json").write_text(
            json.dumps(snapshot.adapter_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


@runtime_checkable
class WeightTransport(Protocol):
    def snapshot(self, version: int) -> AdapterSnapshot: ...

    def materialize(self, snapshot: AdapterSnapshot, target: Path) -> Path: ...


class LocalFileTransport:
    def __init__(self, model: Any) -> None:
        self.model = model

    def snapshot(self, version: int) -> AdapterSnapshot:
        return snapshot_lora(self.model, version)

    def materialize(self, snapshot: AdapterSnapshot, target: Path) -> Path:
        return write_adapter(snapshot, target)


__all__ = [
    "AdapterSnapshot",
    "LocalFileTransport",
    "WeightTransport",
    "snapshot_lora",
    "write_adapter",
]
