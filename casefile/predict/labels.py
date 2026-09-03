"""Competing-risk labels for issue / discussion resolution.

R1 — substantive resolution (what the hazard model predicts).
R2 — administrative closure (competing event, not censoring).
R3 — censored (still open at observation end).

See casefile/docs/PREDICT.md §2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from casefile.predict.time import parse_dt

_ADMIN_LABELS = frozenset(
    {
        "stale",
        "duplicate",
        "wontfix",
        "won't fix",
        "invalid",
        "no-response",
        "no response",
        "not planned",
        "spam",
    }
)

_BOT_SUFFIXES = ("[bot]", "-bot")
_BOT_LOGINS = frozenset(
    {
        "github-actions",
        "dependabot",
        "renovate",
        "stale",
        "codecov",
        "pytorch-bot",
        "facebook-github-bot",
        "pytorchmergebot",
    }
)


class Outcome(str, Enum):
    R1 = "substantive_resolution"
    R2 = "administrative_closure"
    R3 = "censored"


@dataclass(frozen=True)
class LinkedMergedPR:
    number: int
    merged_at: datetime | None
    paths_touched: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkedMergedPRs:
    """Merged PRs only — never bare cross-referenced closed issues."""

    prs: tuple[LinkedMergedPR, ...]

    def numbers(self) -> tuple[int, ...]:
        return tuple(sorted({pr.number for pr in self.prs}))

    def substantive_for_topic(self, topic_paths: tuple[str, ...]) -> tuple[int, ...]:
        if not topic_paths:
            return ()
        hits: list[int] = []
        for pr in self.prs:
            if not pr.paths_touched:
                continue
            if _paths_touch_topic(pr.paths_touched, topic_paths):
                hits.append(pr.number)
        return tuple(sorted(set(hits)))


@dataclass(frozen=True)
class ItemOutcome:
    outcome: Outcome
    closed_at: datetime | None
    reason: str
    linked_pr_numbers: tuple[int, ...] = ()


def label_issue_outcome(
    *,
    state: str,
    closed_at: datetime | None,
    closed_by_login: str | None,
    labels_at_close: list[str],
    linked_merged: LinkedMergedPRs | None,
    topic_paths: tuple[str, ...] = (),
    maintainer_answered: bool,
    observation_end: datetime,
    created_at: datetime,
    unlabeled_closed_as_r2: bool = True,
) -> ItemOutcome:
    """Assign R1 / R2 / R3 for one item as of observation_end."""
    if state != "closed" or closed_at is None or closed_at > observation_end:
        return ItemOutcome(
            Outcome.R3,
            None,
            "still_open_or_closed_after_observation_end",
        )

    merged = linked_merged or LinkedMergedPRs(())
    substantive = merged.substantive_for_topic(topic_paths)
    if substantive:
        return ItemOutcome(
            Outcome.R1,
            closed_at,
            "linked_merged_pr_touching_topic_paths",
            substantive,
        )

    if maintainer_answered:
        return ItemOutcome(Outcome.R1, closed_at, "maintainer_answer")

    label_hits = [lab for lab in labels_at_close if lab.lower().strip() in _ADMIN_LABELS]
    if label_hits:
        return ItemOutcome(
            Outcome.R2,
            closed_at,
            f"admin_label:{','.join(label_hits[:3])}",
        )

    if closed_by_login and _is_bot(closed_by_login):
        return ItemOutcome(Outcome.R2, closed_at, f"closed_by_bot:{closed_by_login}")

    if not unlabeled_closed_as_r2:
        return ItemOutcome(
            Outcome.R3,
            closed_at,
            "unlabeled_closed_not_classified_as_r2",
        )

    return ItemOutcome(
        Outcome.R2,
        closed_at,
        "closed_without_code_or_maintainer_answer",
    )


def labels_at_time(timeline: list[dict[str, Any]], at: datetime) -> list[str]:
    """Replay labeled/unlabeled events with created_at <= at."""
    current: set[str] = set()
    for event in sorted(timeline, key=lambda e: str(e.get("created_at") or "")):
        when = parse_dt(event.get("created_at"))
        if when is None or when > at:
            continue
        etype = event.get("event")
        label = event.get("label") if isinstance(event.get("label"), dict) else {}
        name = str(label.get("name") or "").strip()
        if not name:
            continue
        if etype == "labeled":
            current.add(name)
        elif etype == "unlabeled":
            current.discard(name)
    return sorted(current)


def linked_merged_prs_le(
    timeline: list[dict[str, Any]],
    at: datetime,
    *,
    pr_paths: dict[int, tuple[str, ...]] | None = None,
) -> LinkedMergedPRs:
    """Return only PRs with an explicit merged event ≤ at — not closed issues."""
    merged_at: dict[int, datetime | None] = {}
    pr_paths = pr_paths or {}

    for event in timeline:
        when = parse_dt(event.get("created_at"))
        if when is None or when > at:
            continue
        etype = str(event.get("event") or "")

        if etype == "merged":
            pr = event.get("pull_request") if isinstance(event.get("pull_request"), dict) else {}
            num = pr.get("number")
            if num is not None:
                merged_at[int(num)] = when
            continue

        if etype not in {"cross-referenced", "connected", "referenced"}:
            continue

        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        issue = source.get("issue") if isinstance(source.get("issue"), dict) else {}
        if not issue.get("pull_request"):
            continue
        number = issue.get("number")
        if number is None:
            continue
        num = int(number)
        if num in merged_at:
            continue
        # Cross-reference alone is not enough — require merged event later in timeline.
        if _has_merged_event(timeline, num, at):
            merged_at[num] = _merged_time(timeline, num, at)

    prs = tuple(
        LinkedMergedPR(
            number=num,
            merged_at=merged_at.get(num),
            paths_touched=pr_paths.get(num, ()),
        )
        for num in sorted(merged_at)
    )
    return LinkedMergedPRs(prs=prs)


def filter_events_le(events: list[dict[str, Any]], at: datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        when = parse_dt(event.get("created_at"))
        if when is not None and when <= at:
            out.append(event)
    return out


def _has_merged_event(timeline: list[dict[str, Any]], number: int, at: datetime) -> bool:
    return _merged_time(timeline, number, at) is not None


def _merged_time(
    timeline: list[dict[str, Any]], number: int, at: datetime
) -> datetime | None:
    for event in timeline:
        when = parse_dt(event.get("created_at"))
        if when is None or when > at:
            continue
        if str(event.get("event") or "") != "merged":
            continue
        pr = event.get("pull_request") if isinstance(event.get("pull_request"), dict) else {}
        if int(pr.get("number") or -1) == number:
            return when
    return None


def _paths_touch_topic(pr_paths: tuple[str, ...], topic_paths: tuple[str, ...]) -> bool:
    topic = {p.rstrip("/") for p in topic_paths if p}
    for raw in pr_paths:
        path = raw.rstrip("/")
        for t in topic:
            if path == t or path.startswith(f"{t}/"):
                return True
    return False


def _is_bot(login: str) -> bool:
    lower = login.lower()
    if lower in _BOT_LOGINS:
        return True
    return any(lower.endswith(suf) for suf in _BOT_SUFFIXES)
