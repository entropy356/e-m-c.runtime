"""UTC time sources, injectable for accelerated and deterministic experiments."""
from datetime import datetime, timezone
from typing import Protocol

from .contracts import EventValidationError


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def timestamp(value: datetime | str) -> str:
    try:
        instant = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
        if not isinstance(instant, datetime) or instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError('timezone required')
        return instant.astimezone(timezone.utc).isoformat(timespec='microseconds')
    except (TypeError, ValueError, OverflowError) as error:
        raise EventValidationError('time must be a timezone-aware datetime or ISO 8601 string') from error
