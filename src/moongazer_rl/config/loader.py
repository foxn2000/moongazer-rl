"""Load, override, and re-dump ``RunConfig`` YAML files.

``uv run mg-rl <cmd> config.yaml --set k=v`` maps to::

    config = load_config(path, overrides=["k=v"])

Override keys are dotted paths into the YAML tree (``trainer.args.learning_rate=1e-5``);
values are parsed as YAML scalars/collections (``4`` -> ``int``, ``1e-5`` -> ``float``,
``eval`` -> the string ``"eval"``, ``[1, 2]`` -> a list, ``{a: 1}`` -> a dict).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from moongazer_rl.config.schema import RunConfig


class ConfigError(ValueError):
    """A readable, user-facing config problem: which file/key, and why."""


class _Loader(yaml.SafeLoader):
    """``SafeLoader`` with one fix: PyYAML's stock float resolver requires a decimal point,
    so a bare scientific-notation scalar like ``5e-6`` (as used throughout our trainer.args
    examples, e.g. ``learning_rate: 5e-6``) loads as the *string* ``"5e-6"``, not a float —
    a well-known PyYAML gotcha (yaml/pyyaml#173). This resolver is the standard fix: same
    float grammar, plus exponents without a mandatory dot."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_raw(path: str | Path) -> dict[str, Any]:
    """Parse a YAML file into a plain dict. No defaults, no validation."""
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as exc:
        raise ConfigError(f"cannot read config file {p}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=_Loader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.load(raw, Loader=_Loader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse override value {raw!r} as YAML: {exc}") from exc


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not dotted_key or any(not p for p in parts):
        raise ConfigError(f"malformed override key {dotted_key!r}: expected dotted.path=value")
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if nxt is None:
            nxt = {}
            node[part] = nxt
        elif not isinstance(nxt, dict):
            raise ConfigError(
                f"cannot set {dotted_key!r}: {'.'.join(parts[: parts.index(part) + 1])} "
                f"is not a mapping (found {type(nxt).__name__})"
            )
        node = nxt
    node[parts[-1]] = value


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Return a new dict with ``dotted.key=value`` overrides merged in. ``data`` is not
    mutated."""
    import copy

    merged = copy.deepcopy(data)
    for item in overrides or []:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ConfigError(f"malformed override {item!r}: expected dotted.path=value")
        _set_dotted(merged, key.strip(), _parse_override_value(raw_value))
    return merged


def _format_validation_error(exc: ValidationError, source: str) -> str:
    lines = [f"invalid config ({source}):"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load ``path``, apply ``--set`` overrides, and validate into a ``RunConfig``.

    Raises ``ConfigError`` with a message naming the offending key(s) on any failure
    (missing file, bad YAML, malformed override, or schema violation).
    """
    raw = load_raw(path)
    merged = apply_overrides(raw, overrides)
    try:
        return RunConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, str(path))) from exc


def resolved_dict(config: RunConfig) -> dict[str, Any]:
    """JSON-safe dict of ``config`` after defaults + overrides, suitable for ``yaml.safe_dump``
    (this is what ``config.resolved.yaml`` holds)."""
    return config.model_dump(mode="json", by_alias=True)


def dump_resolved(config: RunConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(resolved_dict(config), sort_keys=False, allow_unicode=True))


__all__ = [
    "ConfigError",
    "apply_overrides",
    "dump_resolved",
    "load_config",
    "load_raw",
    "resolved_dict",
]
