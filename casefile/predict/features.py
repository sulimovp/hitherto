"""Point-in-time engagement features from timeline + reactions list endpoints.

Never use the aggregate issue.comments count or reactions summary block —
those are current-state and leak future information into a snapshot at T.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from casefile.predict.labels import filter_events_le
from casefile.predict.time import MaintainerSet, parse_dt


@dataclass(frozen=True)
class EngagementAtT:
    n_comments_le_t: int
    n_participants_le_t: int
    maintainer_replied_le_t: bool
    hours_to_first_maintainer_reply: float | None
    reactions_plus1_le_t: int
    reactions_heart_le_t: int
    reactions_eyes_le_t: int
    reactions_minus1_le_t: int


def engagement_at_t(
    *,
    created_at: datetime,
    timeline: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    at: datetime,
    maintainers: MaintainerSet,
) -> EngagementAtT:
    maintainers.for_snapshot(at)
    events = filter_events_le(timeline, at)
    comment_keys: list[str] = []
    first_maintainer: datetime | None = None
    for event in events:
        if event.get("event") != "commented":
            continue
        actor = event.get("actor") if isinstance(event.get("actor"), dict) else {}
        login = str(actor.get("login") or "").strip()
        actor_id = actor.get("id")
        key = login.lower() if login else f"anon:{actor_id or 'unknown'}"
        comment_keys.append(key)
        when = parse_dt(event.get("created_at"))
        if login and maintainers.contains(login) and when is not None:
            if first_maintainer is None or when < first_maintainer:
                first_maintainer = when

    participants = set(comment_keys)
    hours: float | None = None
    if first_maintainer is not None:
        hours = (first_maintainer - created_at).total_seconds() / 3600.0

    rx = filter_events_le(reactions, at)
    counts = {"+1": 0, "heart": 0, "eyes": 0, "-1": 0}
    for reaction in rx:
        content = str(reaction.get("content") or "")
        if content in counts:
            counts[content] += 1

    return EngagementAtT(
        n_comments_le_t=len(comment_keys),
        n_participants_le_t=len(participants),
        maintainer_replied_le_t=first_maintainer is not None,
        hours_to_first_maintainer_reply=hours,
        reactions_plus1_le_t=counts["+1"],
        reactions_heart_le_t=counts["heart"],
        reactions_eyes_le_t=counts["eyes"],
        reactions_minus1_le_t=counts["-1"],
    )
