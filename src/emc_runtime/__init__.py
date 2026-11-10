"""emc-runtime: a light, single-process, modular agent experiment runtime."""

from .config import RuntimeConfig, load_config
from .contracts import (
    AGENT_SCOPE_KEY,
    AssistantReply,
    ConcurrencyConflict,
    ContextBlock,
    EMCError,
    Event,
    Finalized,
    Message,
    Module,
    ModuleContext,
    ModuleContractError,
    ModuleStateVersionError,
    Prepared,
    ProviderError,
    RunResult,
    validate_event,
)
from .engine import Engine, build_engine
from .events import EventBus
from .storage import Storage

__version__ = "0.1.0"

__all__ = [
    "AGENT_SCOPE_KEY",
    "AssistantReply",
    "ConcurrencyConflict",
    "ContextBlock",
    "EMCError",
    "Engine",
    "Event",
    "EventBus",
    "Finalized",
    "Message",
    "Module",
    "ModuleContext",
    "ModuleContractError",
    "ModuleStateVersionError",
    "Prepared",
    "ProviderError",
    "RunResult",
    "RuntimeConfig",
    "Storage",
    "build_engine",
    "load_config",
    "validate_event",
]
