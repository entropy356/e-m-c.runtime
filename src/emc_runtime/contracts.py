"""Public types shared by the engine, modules, providers and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

#: Reserved scope key for agent-wide module state. Conversation ids that would
#: collide with it are rejected at validation time.
AGENT_SCOPE_KEY = "__agent__"

EVENT_KINDS = ("user_message", "internal_trigger")
SCOPES = ("agent", "conversation")


class EMCError(Exception):
    """Base class for all runtime errors with a user-readable message."""


class ConfigError(EMCError):
    """The runtime configuration is invalid."""


class EventValidationError(EMCError):
    """An input event does not satisfy the event contract."""


class ProviderError(EMCError):
    """A chat/embedding provider failed or returned an unusable response."""


class ConcurrencyConflict(EMCError):
    """The database revision changed between snapshot and commit.

    The turn is not retried automatically: retrying would replay a model
    request that was already paid for and may not be deterministic.
    """


class ModuleStateVersionError(EMCError):
    """Stored module state was written by a different state structure version."""


class ModuleContractError(EMCError):
    """A module violated the module contract (bad types, spoofed source...)."""


def dumps_json(value: Any, what: str) -> str:
    """Serialize ``value`` strictly; NaN/Inf and non-JSON types are rejected."""
    import json

    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ModuleContractError(
            f"{what} is not strictly JSON-serializable ({error}); the turn was aborted"
        ) from error


def check_json_value(value: Any, what: str) -> None:
    """Fail fast when a value cannot be stored as strict JSON."""
    dumps_json(value, what)


@dataclass(frozen=True)
class Event:
    """One input to a turn. Events describe what happened; they never schedule."""

    kind: Literal["user_message", "internal_trigger"]
    conversation_id: str = "default"
    content: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = ""


@dataclass(frozen=True)
class Message:
    """A stored conversation message, ordered by ``seq`` within its conversation."""

    conversation_id: str
    seq: int
    role: str  # "user" | "assistant"
    content: str
    turn_id: str


@dataclass(frozen=True)
class ContextBlock:
    """A module's contribution to the LLM context, treated as data."""

    source: str
    content: str


@dataclass
class Prepared:
    """Result of Module.prepare: candidate state plus context contribution."""

    working_state: Any
    context_blocks: list[ContextBlock]
    trace: dict


@dataclass(frozen=True)
class Finalized:
    """Result of Module.finalize: the state to commit atomically."""

    state: Any
    trace: dict


@dataclass(frozen=True)
class ModuleContext:
    """Read-only per-turn view handed to modules.

    Modules must never receive the engine, the storage connection or another
    module's mutable state.
    """

    event: Event
    history: tuple[Message, ...]
    exports: Mapping[str, Any]


@dataclass(frozen=True)
class AssistantReply:
    """A model reply. The first version is plain text only."""

    content: str


@dataclass(frozen=True)
class RunResult:
    """Everything a caller needs after one committed turn."""

    turn_id: str
    event_id: str
    conversation_id: str
    event_kind: str
    revision: int
    reply: str
    module_traces: dict

    def to_json_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "event_kind": self.event_kind,
            "revision": self.revision,
            "reply": self.reply,
            "module_traces": self.module_traces,
        }


@runtime_checkable
class Module(Protocol):
    """The minimal module contract implemented by counter, memory, emotion..."""

    module_id: str
    scope: Literal["agent", "conversation"]
    state_version: int

    def initial_state(self) -> Any: ...

    def expose(self, state: Any) -> Any: ...

    def prepare(self, event: Event, state: Any, ctx: ModuleContext) -> Prepared: ...

    def finalize(
        self, event: Event, prepared: Prepared, reply: AssistantReply, ctx: ModuleContext
    ) -> Finalized: ...

    def inspect(self, state: Any) -> Any: ...


@runtime_checkable
class ChatProvider(Protocol):
    """Transport to a chat model. No memory protocol, no provider fallback."""

    def generate(self, messages: Sequence[dict], options: Mapping[str, Any]) -> AssistantReply: ...


@runtime_checkable
class CognitionStrategy(Protocol):
    """Turns an assembled context into a reply using a chat provider."""

    def run(
        self,
        messages: Sequence[dict],
        provider: ChatProvider,
        options: Mapping[str, Any],
    ) -> AssistantReply: ...


def validate_event(event: Event) -> Event:
    """Normalize and validate one input event; fills event_id when missing."""
    import re
    import uuid

    if event.kind not in EVENT_KINDS:
        raise EventValidationError(
            f"event.kind must be one of {EVENT_KINDS}, got {event.kind!r}"
        )
    if not isinstance(event.conversation_id, str):
        raise EventValidationError("event.conversation_id must be text")
    conversation_id = event.conversation_id.strip()
    if not conversation_id or len(conversation_id) > 128:
        raise EventValidationError("conversation_id must be 1-128 characters")
    if re.search(r"\s", conversation_id):
        raise EventValidationError("conversation_id cannot contain whitespace")
    if conversation_id == AGENT_SCOPE_KEY:
        raise EventValidationError(f"conversation_id {AGENT_SCOPE_KEY!r} is reserved")
    content = event.content
    if event.kind == "user_message":
        if not isinstance(content, str) or not content.strip():
            raise EventValidationError("user_message requires nonempty content text")
        if len(content) > 12000:
            raise EventValidationError("user_message content exceeds 12000 characters")
        content = content.strip()
    else:
        if content is not None and not isinstance(content, str):
            raise EventValidationError("internal_trigger content must be text or null")
    metadata = dict(event.metadata or {})
    if not isinstance(metadata, dict):
        raise EventValidationError("event.metadata must be a JSON object")
    check_json_value(metadata, "event.metadata")
    event_id = event.event_id or uuid.uuid4().hex
    if not isinstance(event_id, str) or not event_id.strip():
        raise EventValidationError("event.event_id must be nonempty text")
    return Event(
        kind=event.kind,
        conversation_id=conversation_id,
        content=content,
        metadata=metadata,
        event_id=event_id.strip(),
    )
