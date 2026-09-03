"""Compute every input the demand×supply rollup needs from GitHub APIs.

One module, one dataclass, one async entry point. See PREDICT_QUADRANT.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from casefile.predict.assignment_score import automatic_assign
from casefile.predict.labels import (
    LinkedMergedPR,
    LinkedMergedPRs,
    Outcome,
    label_issue_outcome,
    labels_at_time,
    linked_merged_prs_le,
)
from casefile.predict.time import MaintainerSet, ensure_utc, parse_dt

if TYPE_CHECKING:
    from casefile.clients.github import GitHubClient

_RECENT_DAYS = 180
_BASELINE_DAYS = 730
_RECENT_MONTHS = 6.0
_BASELINE_MONTHS = 18.0
_R1_SAMPLE_CAP = 50
_INFLOW_FETCH_CAP = 100
_MAX_PR_FILE_LOOKUPS_PER_ISSUE = 3
_SEARCH_TOTAL_CAP = 1000
_MAX_EXCLUSION_FRACTION = 0.25


@dataclass(frozen=True)
class TopicHistory:
    path: str
    demand_recent_per_month: float | None
    demand_baseline_per_month: float | None
    realized_r1_rate_recent: float | None
    realized_r1_rate_baseline: float | None
    n_resolved_items: int | None
    topic_first_seen: datetime | None
    inflow_recent_total: int | None
    inflow_baseline_total: int | None
    r1_sampled_n: int = 0
    r1_sampled_n_recent: int = 0
    r1_sampled_n_baseline: int = 0
    r1_excluded_n: int = 0
    search_truncated: bool = False
    r1_truncated: bool = False
    fetch_errors: list[str] = field(default_factory=list)


def topic_windows(now: datetime) -> tuple[datetime, datetime, datetime]:
    """Non-overlapping windows: (baseline_start, recent_start, now).

    recent = [T − 180d, T] (6 months); baseline = [T − 730d, T − 180d) (18 months).
    GitHub `created:A..B` is inclusive on both ends, so search queries end the
    baseline window on (recent_start − 1 day).
    """
    now_utc = ensure_utc(now)
    recent_start = now_utc - timedelta(days=_RECENT_DAYS)
    baseline_start = now_utc - timedelta(days=_BASELINE_DAYS)
    return baseline_start, recent_start, now_utc


def _nonoverlap_search_bounds(
    baseline_start: datetime, recent_start: datetime, now: datetime
) -> tuple[str, str, str, str]:
    """GitHub `created:A..B` is inclusive on both ends. Baseline ends yesterday relative to recent."""
    baseline_end = recent_start - timedelta(days=1)
    return (
        baseline_start.strftime("%Y-%m-%d"),
        baseline_end.strftime("%Y-%m-%d"),
        recent_start.strftime("%Y-%m-%d"),
        now.strftime("%Y-%m-%d"),
    )


def inflow_month_divisors() -> tuple[float, float]:
    """(recent_months, baseline_months) for rate conversion."""
    return _RECENT_MONTHS, _BASELINE_MONTHS


async def compute_topic_history(
    gh: GitHubClient,
    *,
    repo: str,
    path: str,
    topic_paths: tuple[str, ...],
    now: datetime,
    maintainers: MaintainerSet,
    synonyms: tuple[str, ...] = (),
) -> TopicHistory:
    owner, name = repo.split("/", 1)
    baseline_start, recent_start, now_utc = topic_windows(now)
    errors: list[str] = []
    if not maintainers.logins:
        errors.append("maintainer set empty — R1-by-answer disabled")

    inflow_recent, inflow_baseline, search_truncated = await _inflow(
        gh,
        repo=repo,
        path=path,
        now=now_utc,
        recent_start=recent_start,
        baseline_start=baseline_start,
        synonyms=synonyms,
        errors=errors,
    )

    demand_recent: float | None = None
    demand_baseline: float | None = None
    if inflow_recent is not None and not search_truncated:
        demand_recent = inflow_recent / _RECENT_MONTHS
    if inflow_baseline is not None and not search_truncated:
        demand_baseline = inflow_baseline / _BASELINE_MONTHS

    (
        r1_rate_recent,
        r1_rate_baseline,
        n_resolved,
        sampled_recent,
        sampled_baseline,
        excluded_n,
        r1_truncated,
    ) = await _r1_rates(
        gh,
        owner=owner,
        name=name,
        repo=repo,
        path=path,
        topic_paths=topic_paths,
        now=now_utc,
        recent_start=recent_start,
        baseline_start=baseline_start,
        maintainers=maintainers,
        synonyms=synonyms,
        errors=errors,
    )

    topic_first_seen = await _topic_first_seen(
        gh, owner=owner, name=name, path=path, errors=errors
    )

    return TopicHistory(
        path=path,
        demand_recent_per_month=demand_recent,
        demand_baseline_per_month=demand_baseline,
        realized_r1_rate_recent=r1_rate_recent,
        realized_r1_rate_baseline=r1_rate_baseline,
        n_resolved_items=n_resolved,
        topic_first_seen=topic_first_seen,
        inflow_recent_total=inflow_recent,
        inflow_baseline_total=inflow_baseline,
        r1_sampled_n=sampled_recent + sampled_baseline,
        r1_sampled_n_recent=sampled_recent,
        r1_sampled_n_baseline=sampled_baseline,
        r1_excluded_n=excluded_n,
        search_truncated=search_truncated,
        r1_truncated=r1_truncated,
        fetch_errors=errors,
    )


def _issue_assigned(
    item: dict, path: str, repo: str, synonyms: tuple[str, ...]
) -> bool:
    return automatic_assign(
        str(item.get("title") or ""),
        str(item.get("body") or ""),
        path,
        synonyms=synonyms,
        repo=repo,
    )


async def _assigned_inflow_count(
    gh: GitHubClient,
    *,
    query: str,
    path: str,
    repo: str,
    synonyms: tuple[str, ...],
    label: str,
    errors: list[str],
) -> tuple[int | None, bool]:
    try:
        page = await gh.search_issues_page_sorted(query, per_page=_INFLOW_FETCH_CAP)
    except Exception as exc:  # noqa: BLE001 — leave field None; never impute zero
        errors.append(f"{label} search failed: {exc}")
        return None, False

    total = page["total_count"]
    items = page["items"]
    if total >= _SEARCH_TOTAL_CAP or total > _INFLOW_FETCH_CAP:
        errors.append(
            f"{label}: search total {total} exceeds fetch cap {_INFLOW_FETCH_CAP}"
        )
        return None, True
    if len(items) < total:
        errors.append(f"{label}: fetched {len(items)} of {total}")
        return None, True
    assigned = sum(1 for item in items if _issue_assigned(item, path, repo, synonyms))
    return assigned, False


async def _inflow(
    gh: GitHubClient,
    *,
    repo: str,
    path: str,
    now: datetime,
    recent_start: datetime,
    baseline_start: datetime,
    synonyms: tuple[str, ...],
    errors: list[str],
) -> tuple[int | None, int | None, bool]:
    b0, b1, r0, r1 = _nonoverlap_search_bounds(baseline_start, recent_start, now)

    recent_total, recent_trunc = await _assigned_inflow_count(
        gh,
        query=f'repo:{repo} is:issue "{path}" created:{r0}..{r1}',
        path=path,
        repo=repo,
        synonyms=synonyms,
        label="inflow recent",
        errors=errors,
    )
    baseline_total, baseline_trunc = await _assigned_inflow_count(
        gh,
        query=f'repo:{repo} is:issue "{path}" created:{b0}..{b1}',
        path=path,
        repo=repo,
        synonyms=synonyms,
        label="inflow baseline",
        errors=errors,
    )
    truncated = recent_trunc or baseline_trunc
    if truncated:
        return None, None, True
    return recent_total, baseline_total, False


async def _r1_rates(
    gh: GitHubClient,
    *,
    owner: str,
    name: str,
    repo: str,
    path: str,
    topic_paths: tuple[str, ...],
    now: datetime,
    recent_start: datetime,
    baseline_start: datetime,
    maintainers: MaintainerSet,
    synonyms: tuple[str, ...],
    errors: list[str],
) -> tuple[float | None, float | None, int | None, int, int, int, bool]:
    """Return (recent_rate, baseline_rate, n_resolved, sampled_r, sampled_b, excluded, truncated)."""
    b0, b1, r0, r1 = _nonoverlap_search_bounds(baseline_start, recent_start, now)
    r1_truncated = False

    async def rate_for_window(
        start_str: str, end_str: str
    ) -> tuple[float | None, int, int, int | None]:
        nonlocal r1_truncated
        try:
            page = await gh.search_issues_page_sorted(
                f'repo:{repo} is:issue is:closed "{path}" closed:{start_str}..{end_str}',
                sort="created",
                order="desc",
                per_page=_R1_SAMPLE_CAP,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"R1 sample search failed ({start_str}..{end_str}): {exc}")
            return None, 0, 0, None

        items = page["items"]
        window_total = page["total_count"]
        if window_total > _R1_SAMPLE_CAP:
            r1_truncated = True
            errors.append(
                f"R1 sample truncated: {window_total} closed issues in "
                f"{start_str}..{end_str}, cap {_R1_SAMPLE_CAP}"
            )
            return None, 0, 0, window_total

        r1_count = 0
        denominator = 0
        excluded = 0
        assigned_n = 0

        for issue in items:
            if not _issue_assigned(issue, path, repo, synonyms):
                continue
            assigned_n += 1
            issue_number = issue.get("number")
            if issue_number is None:
                continue

            closed_at_raw = parse_dt(issue.get("closed_at"))
            created_at_raw = parse_dt(issue.get("created_at"))
            if closed_at_raw is None or created_at_raw is None:
                continue

            try:
                timeline = await gh.list_issue_timeline(owner, name, issue_number)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"timeline #{issue_number}: {exc}")
                excluded += 1
                continue

            labels_close = labels_at_time(timeline, closed_at_raw)
            merged = await _resolve_merged_prs(gh, owner, name, timeline, closed_at_raw)

            if merged is None:
                excluded += 1
                continue

            if maintainers.logins:
                try:
                    maintainers.for_snapshot(closed_at_raw)
                    m_answered = _maintainer_answered(
                        timeline, closed_at_raw, maintainers
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                    m_answered = False
            else:
                m_answered = False

            closed_by = issue.get("closed_by")
            closed_by_login = (
                closed_by.get("login") if isinstance(closed_by, dict) else None
            )

            outcome = label_issue_outcome(
                state="closed",
                closed_at=closed_at_raw,
                closed_by_login=closed_by_login,
                labels_at_close=labels_close,
                linked_merged=merged,
                topic_paths=topic_paths,
                maintainer_answered=m_answered,
                observation_end=now,
                created_at=created_at_raw,
            )

            denominator += 1
            if outcome.outcome == Outcome.R1:
                r1_count += 1

        if excluded > _MAX_EXCLUSION_FRACTION * max(assigned_n, 1):
            return None, assigned_n, excluded, window_total

        rate = r1_count / denominator if denominator > 0 else None
        return rate, assigned_n, excluded, window_total

    recent_rate, s1, e1, t1 = await rate_for_window(r0, r1)
    baseline_rate, s2, e2, t2 = await rate_for_window(b0, b1)

    n_resolved: int | None = None
    if t1 is not None and t2 is not None:
        n_resolved = t1 + t2
    elif t1 is not None:
        n_resolved = t1
    elif t2 is not None:
        n_resolved = t2

    if r1_truncated:
        return None, None, n_resolved, s1, s2, e1 + e2, True
    return recent_rate, baseline_rate, n_resolved, s1, s2, e1 + e2, False


async def _resolve_merged_prs(
    gh: GitHubClient,
    owner: str,
    name: str,
    timeline: list[dict],
    at: datetime,
) -> LinkedMergedPRs | None:
    """Resolve PR paths for merged PRs. Returns None if any PR's paths are unresolvable."""
    merged = linked_merged_prs_le(timeline, at)
    if not merged.prs:
        return merged

    if len(merged.prs) > _MAX_PR_FILE_LOOKUPS_PER_ISSUE:
        return None

    resolved: list[LinkedMergedPR] = []
    for pr in merged.prs:
        try:
            files = await gh.list_pr_files(owner, name, pr.number)
        except Exception:  # noqa: BLE001 — fetch failure must not become R2
            return None
        if not files:
            return None
        resolved.append(
            LinkedMergedPR(
                number=pr.number,
                merged_at=pr.merged_at,
                paths_touched=tuple(files),
            )
        )
    return LinkedMergedPRs(prs=tuple(resolved))


def _maintainer_answered(
    timeline: list[dict], at: datetime, maintainers: MaintainerSet
) -> bool:
    for event in timeline:
        if event.get("event") != "commented":
            continue
        when = parse_dt(event.get("created_at"))
        if when is None or when > at:
            continue
        actor = event.get("actor") if isinstance(event.get("actor"), dict) else {}
        login = str(actor.get("login") or "").strip()
        if login and maintainers.contains(login):
            return True
    return False


async def _topic_first_seen(
    gh: GitHubClient,
    *,
    owner: str,
    name: str,
    path: str,
    errors: list[str],
) -> datetime | None:
    try:
        date_str = await gh.get_earliest_commit_date(owner, name, path=path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"topic_first_seen: {exc}")
        return None
    if date_str is None:
        errors.append("topic_first_seen: no commits for path")
        return None
    return parse_dt(date_str)
