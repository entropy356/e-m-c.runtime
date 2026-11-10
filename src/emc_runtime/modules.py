"""Explicit module loading, registration and scope handling.

Modules are named by trusted local configuration (``package.module:ClassName``);
there is no directory scanning or auto-discovery. The registry only verifies
the contract surface and keeps the configured order.
"""

from __future__ import annotations

import importlib
from types import MappingProxyType
from typing import Any, Mapping

from .config import ModuleSpec
from .contracts import (
    AGENT_SCOPE_KEY,
    ConfigError,
    Event,
    Module,
    ModuleContext,
    ModuleContractError,
    SCOPES,
)

_REQUIRED_METHODS = (
    "initial_state",
    "expose",
    "prepare",
    "finalize",
    "inspect",
)


def scope_key_for(scope: str, conversation_id: str) -> str:
    """Storage key for one module's state under the given scope."""
    if scope == "agent":
        return AGENT_SCOPE_KEY
    if scope == "conversation":
        return conversation_id
    raise ConfigError(f"unknown scope {scope!r}")


class ModuleRegistry:
    """Loaded module instances in configuration order."""

    def __init__(self, specs: tuple[ModuleSpec, ...]):
        self._specs = specs
        self._modules: list[Module] = []
        for spec in specs:
            self._modules.append(self._load(spec))

    @staticmethod
    def _load(spec: ModuleSpec) -> Module:
        module_path, _, class_name = spec.entrypoint.partition(":")
        if not module_path or not class_name:
            raise ConfigError(
                f"module {spec.module_id!r} entrypoint must look like "
                f"'package.module:ClassName', got {spec.entrypoint!r}"
            )
        try:
            module = importlib.import_module(module_path)
        except ImportError as error:
            raise ConfigError(
                f"module {spec.module_id!r}: cannot import {module_path!r} ({error})"
            ) from error
        try:
            cls = getattr(module, class_name)
        except AttributeError as error:
            raise ConfigError(
                f"module {spec.module_id!r}: {module_path!r} has no class {class_name!r}"
            ) from error
        try:
            instance = cls(spec.config, spec.module_id, spec.scope)
        except TypeError as error:
            raise ConfigError(
                f"module {spec.module_id!r}: constructor must accept "
                f"(config, module_id, scope) ({error})"
            ) from error
        ModuleRegistry._check(instance, spec)
        return instance

    @staticmethod
    def _check(instance: Any, spec: ModuleSpec) -> None:
        def fail(message: str):
            raise ConfigError(f"module {spec.module_id!r}: {message}")

        if getattr(instance, "module_id", None) != spec.module_id:
            fail(
                f"declared module_id {getattr(instance, 'module_id', None)!r} "
                f"does not match configuration id"
            )
        if getattr(instance, "scope", None) != spec.scope:
            fail(
                f"declared scope {getattr(instance, 'scope', None)!r} does not "
                f"match configured scope {spec.scope!r}"
            )
        state_version = getattr(instance, "state_version", None)
        if not isinstance(state_version, int) or isinstance(state_version, bool) or state_version < 1:
            fail("state_version must be a positive integer")
        for method in _REQUIRED_METHODS:
            if not callable(getattr(instance, method, None)):
                fail(f"missing required method {method!r}")

    @property
    def modules(self) -> tuple[Module, ...]:
        return tuple(self._modules)

    def module_keys(self, conversation_id: str) -> list[tuple[str, str]]:
        """(module_id, scope_key) pairs for one conversation snapshot."""
        return [
            (module.module_id, scope_key_for(module.scope, conversation_id))
            for module in self._modules
        ]

    def describe(self) -> list[dict]:
        return [
            {
                "module_id": module.module_id,
                "scope": module.scope,
                "state_version": module.state_version,
            }
            for module in self._modules
        ]

    @staticmethod
    def make_context(event: Event, history, exports: Mapping[str, Any]) -> ModuleContext:
        """Frozen, read-only view for one module call."""
        return ModuleContext(
            event=event,
            history=tuple(history),
            exports=MappingProxyType(dict(exports)),
        )
