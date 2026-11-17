"""Research workflows are ordinary Python classes implementing run(ctx)."""
from __future__ import annotations

import importlib
from typing import Any, Mapping

from .contracts import ConfigError


class DirectStrategy:
    """Default chat workflow; module helpers are optional workflow operations."""

    def __init__(self, options=None, history_limit=None):
        self.options = dict(options or {})
        self.history_limit = history_limit
        if history_limit is not None and (type(history_limit) is not int or history_limit < 0):
            raise ConfigError('cognition.history_limit must be a nonnegative integer')

    def run(self, ctx):
        blocks = ctx.prepare_modules()
        messages = ctx.assemble(blocks=blocks, history_limit=self.history_limit)
        reply = ctx.models.chat(messages, options=self.options, purpose='reply')
        ctx.finalize_modules(reply)
        return reply


def make_cognition(kind: str, params: Mapping[str, Any]):
    if kind == 'direct':
        options = params.get('options', {})
        if not isinstance(options, dict):
            raise ConfigError('cognition.options must be an object')
        return DirectStrategy(options, params.get('history_limit'))
    if kind == 'python':
        entrypoint = params.get('entrypoint', '')
        module_name, _, class_name = entrypoint.partition(':')
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            strategy = cls(dict(params.get('config', {})))
        except Exception as error:
            raise ConfigError(f'cannot load cognition {entrypoint!r}: {error}') from error
        if not callable(getattr(strategy, 'run', None)):
            raise ConfigError('cognition must implement run(ctx)')
        return strategy
    raise ConfigError(f'unknown cognition kind {kind!r}')
