"""Thin launcher config: YAML -> ``RunConfig``. See ``config.schema`` for the shape."""

from moongazer_rl.config.loader import (
    ConfigError,
    apply_overrides,
    dump_resolved,
    load_config,
    load_raw,
    resolved_dict,
)
from moongazer_rl.config.schema import AsyncSection, RunConfig
from moongazer_rl.config.validate import (
    ValidationReport,
    grpo_config_field_report,
    validate_config,
)

__all__ = [
    "AsyncSection",
    "ConfigError",
    "RunConfig",
    "ValidationReport",
    "apply_overrides",
    "dump_resolved",
    "grpo_config_field_report",
    "load_config",
    "load_raw",
    "resolved_dict",
    "validate_config",
]
