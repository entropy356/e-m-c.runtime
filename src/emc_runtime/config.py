"""Loading and validation of the JSON runtime configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import ConfigError, SCOPES

SUPPORTED_SCHEMA_VERSION = 1
PROVIDER_KINDS = ("offline", "openai-compatible")
COGNITION_KINDS = ("direct",)

DEFAULT_SYSTEM_PROMPT = "You are a helpful conversational agent."
DEFAULT_TRIGGER_PROMPT = (
    "(internal trigger) This turn was started by an internal event, not by a user "
    "message. Continue the conversation as appropriate."
)


@dataclass(frozen=True)
class ModuleSpec:
    """One module entry from the configuration file."""

    module_id: str
    entrypoint: str
    scope: str
    config: dict


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: int
    database: str
    provider_kind: str
    provider_params: dict
    cognition_kind: str
    cognition_params: dict
    system_prompt: str
    internal_trigger_prompt: str
    modules: tuple[ModuleSpec, ...]
    raw: dict


def _require_mapping(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{what} must be a JSON object")
    return value


def _parse_modules(value: Any) -> tuple[ModuleSpec, ...]:
    if not isinstance(value, list):
        raise ConfigError("modules must be a list")
    specs: list[ModuleSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item = _require_mapping(item, f"modules[{index}]")
        module_id = item.get("id")
        if not isinstance(module_id, str) or not module_id.strip():
            raise ConfigError(f"modules[{index}].id must be nonempty text")
        module_id = module_id.strip()
        if module_id in seen:
            raise ConfigError(f"duplicate module id {module_id!r} in configuration")
        seen.add(module_id)
        entrypoint = item.get("entrypoint")
        if not isinstance(entrypoint, str) or ":" not in entrypoint:
            raise ConfigError(
                f"module {module_id!r} entrypoint must look like "
                f"'package.module:ClassName', got {entrypoint!r}"
            )
        scope = item.get("scope", "conversation")
        if scope not in SCOPES:
            raise ConfigError(
                f"module {module_id!r} scope must be one of {SCOPES}, got {scope!r}"
            )
        config = item.get("config", {})
        config = _require_mapping(config, f"module {module_id!r} config")
        specs.append(
            ModuleSpec(
                module_id=module_id,
                entrypoint=entrypoint.strip(),
                scope=scope,
                config=dict(config),
            )
        )
    return tuple(specs)


def parse_config(data: Any) -> RuntimeConfig:
    data = _require_mapping(data, "configuration")
    schema_version = data.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}, got {schema_version!r}"
        )
    database = data.get("database")
    if not isinstance(database, str) or not database.strip():
        raise ConfigError("database must be a nonempty file path")
    provider = _require_mapping(data.get("provider"), "provider")
    provider_kind = provider.get("kind")
    if provider_kind not in PROVIDER_KINDS:
        raise ConfigError(
            f"provider.kind must be one of {PROVIDER_KINDS}, got {provider_kind!r}"
        )
    cognition = _require_mapping(data.get("cognition", {"kind": "direct"}), "cognition")
    cognition_kind = cognition.get("kind", "direct")
    if cognition_kind not in COGNITION_KINDS:
        raise ConfigError(
            f"cognition.kind must be one of {COGNITION_KINDS}, got {cognition_kind!r}"
        )
    system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ConfigError("system_prompt must be nonempty text")
    trigger_prompt = data.get("internal_trigger_prompt", DEFAULT_TRIGGER_PROMPT)
    if not isinstance(trigger_prompt, str) or not trigger_prompt.strip():
        raise ConfigError("internal_trigger_prompt must be nonempty text")
    return RuntimeConfig(
        schema_version=schema_version,
        database=database.strip(),
        provider_kind=provider_kind,
        provider_params=dict(provider),
        cognition_kind=cognition_kind,
        cognition_params=dict(cognition),
        system_prompt=system_prompt,
        internal_trigger_prompt=trigger_prompt,
        modules=_parse_modules(data.get("modules", [])),
        raw=data,
    )


def load_config(path: str | Path) -> RuntimeConfig:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(f"configuration {path} is not valid JSON: {error}") from error
    return parse_config(data)


def merge_defaults(data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict:
    """Shallow-merge overrides onto a mapping (used by tests and tooling)."""
    merged = dict(data)
    merged.update(overrides)
    return merged
