"""Trainer backends. Only the protocol is imported eagerly; backends are lazy (see registry)."""

from moongazer_rl.trainer.base import CheckpointInfo, StepResult, TrainerBackend

__all__ = ["CheckpointInfo", "StepResult", "TrainerBackend"]
