"""Shared datetime helpers for point-in-time prediction features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


def parse_dt(value: object) -> datetime | None:
    """Parse ISO-8601 timestamps; naive values are treated as UTC."""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class MaintainerSet:
    """CODEOWNERS/maintainer logins valid at a pinned snapshot — not main@HEAD."""

    logins: frozenset[str]
    as_of: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", ensure_utc(self.as_of))

    def for_snapshot(self, at: datetime) -> MaintainerSet:
        at_utc = ensure_utc(at)
        if self.as_of > at_utc:
            raise ValueError(
                f"MaintainerSet as_of ({self.as_of.isoformat()}) is after snapshot "
                f"({at_utc.isoformat()}) — reading CODEOWNERS from the future"
            )
        return self

    def contains(self, login: str) -> bool:
        return login.lower() in {entry.lower() for entry in self.logins}
