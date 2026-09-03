"""Deterministic module vital signs — baseline any forecast must beat."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec

_COMMIT_PAGE_SIZE = 100
_COMMIT_MAX_PAGES = 5


@dataclass
class VitalSigns:
    path: str
    commits_3m: int = 0
    commits_6m: int = 0
    commits_12m: int = 0
    commits_fetched: int = 0
    commits_truncated: bool = False
    trend: str = "unknown"  # rising | flat | falling | unknown
    distinct_committers_12m: int = 0
    top_committer: str | None = None
    # True if top path committer authored a commit on THIS path in the last 6 months.
    top_committer_active_on_path_6m: bool | None = None
    open_issues_total: int | None = None
    closed_issues_total: int | None = None
    open_issues_sample: int = 0
    median_open_issue_age_days: float | None = None
    closure_rate: float | None = None
    has_codeowners: bool | None = None
    codeowners_mentions_path: bool | None = None
    prototype_or_stale_label_hits: int = 0
    evidence_urls: dict[str, str] = field(default_factory=dict)
    fetch_errors: list[str] = field(default_factory=list)
    issue_search_ok: bool = False
    refused: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


class VitalSignsRetriever:
    name = "vital_signs"
    tier = 1

    def plan(
        self,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        plan: RetrievalPlan,
    ) -> RetrievalSpec | None:
        for spec in plan.specs:
            if spec.retriever == self.name:
                return spec
        return None

    async def fetch(
        self,
        spec: RetrievalSpec,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        clients: ClientBundle,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        for path in spec.paths:
            signs = await compute_vital_signs(clients, request, path)
            items.append(_to_evidence(signs, request.repo))
        return items


async def compute_vital_signs(
    clients: ClientBundle,
    request: AssessmentRequest,
    path: str,
) -> VitalSigns:
    now = datetime.now(UTC)
    since_12 = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
    signs = VitalSigns(path=path)

    try:
        commits, truncated = await clients.github.list_commits(
            request.owner,
            request.name,
            path=path,
            per_page=_COMMIT_PAGE_SIZE,
            since=since_12,
            max_pages=_COMMIT_MAX_PAGES,
        )
    except httpx.HTTPError as exc:
        signs.fetch_errors.append(f"commits fetch failed: {exc}")
        commits, truncated = [], False

    signs.commits_fetched = len(commits)
    signs.commits_truncated = truncated

    window_3 = now - timedelta(days=90)
    window_6 = now - timedelta(days=180)
    window_12 = now - timedelta(days=365)
    authors: dict[str, int] = {}
    last_author_dates: dict[str, datetime] = {}

    for entry in commits:
        commit = entry.get("commit") if isinstance(entry.get("commit"), dict) else {}
        author_block = commit.get("author") if isinstance(commit.get("author"), dict) else {}
        date_str = author_block.get("date")
        when = _parse_dt(date_str)
        login = None
        if isinstance(entry.get("author"), dict):
            login = entry["author"].get("login")
        if login:
            authors[login] = authors.get(login, 0) + 1
            if when and (login not in last_author_dates or when > last_author_dates[login]):
                last_author_dates[login] = when
        if when is None:
            continue
        if when >= window_12:
            signs.commits_12m += 1
        if when >= window_6:
            signs.commits_6m += 1
        if when >= window_3:
            signs.commits_3m += 1

    signs.distinct_committers_12m = len(authors)
    if authors:
        top = max(authors.items(), key=lambda x: x[1])[0]
        signs.top_committer = top
        last = last_author_dates.get(top)
        signs.top_committer_active_on_path_6m = bool(last and last >= window_6)

    early = signs.commits_12m - signs.commits_6m
    late = signs.commits_6m
    if signs.commits_truncated:
        signs.trend = "unknown"
    elif signs.commits_12m == 0:
        signs.trend = "unknown"
    elif late > early * 1.25:
        signs.trend = "rising"
    elif late < early * 0.75:
        signs.trend = "falling"
    else:
        signs.trend = "flat"

    signs.evidence_urls["commits"] = f"https://github.com/{request.repo}/commits/{path}"

    # Keep the path as a quoted phrase — do not split torch/masked into free-text tokens.
    path_phrase = f'"{path}"'
    open_q = f"repo:{request.repo} is:issue is:open {path_phrase}"
    closed_q = f"repo:{request.repo} is:issue is:closed {path_phrase}"
    try:
        open_page = await clients.github.search_issues_page(open_q, per_page=30)
        closed_page = await clients.github.search_issues_page(closed_q, per_page=30)
        signs.issue_search_ok = True
        open_items = open_page["items"]
        closed_items = closed_page["items"]
        signs.open_issues_total = open_page["total_count"]
        signs.closed_issues_total = closed_page["total_count"]
        signs.open_issues_sample = len(open_items)
        open_t = signs.open_issues_total
        closed_t = signs.closed_issues_total
        total = open_t + closed_t
        if total > 0:
            signs.closure_rate = round(closed_t / total, 3)
    except httpx.HTTPError as exc:
        signs.fetch_errors.append(f"issue search failed: {exc}")
        signs.issue_search_ok = False
        open_items = []

    ages = []
    for issue in open_items:
        created = _parse_dt(issue.get("created_at"))
        if created:
            ages.append((now - created).days)
        labels = [
            str(lbl.get("name", "")).lower()
            for lbl in issue.get("labels", [])
            if isinstance(lbl, dict)
        ]
        if any(
            lbl in {"prototype", "stale"} or "prototype" in lbl or "stale" in lbl
            for lbl in labels
        ):
            signs.prototype_or_stale_label_hits += 1
    if ages:
        ages_sorted = sorted(ages)
        mid = len(ages_sorted) // 2
        signs.median_open_issue_age_days = float(
            ages_sorted[mid]
            if len(ages_sorted) % 2
            else (ages_sorted[mid - 1] + ages_sorted[mid]) / 2
        )
    signs.evidence_urls["issues"] = (
        f"https://github.com/{request.repo}/issues?q={quote(path_phrase)}"
    )

    try:
        codeowners = await clients.github.get_file_content(
            request.owner, request.name, ".github/CODEOWNERS"
        )
        if codeowners is None:
            codeowners = await clients.github.get_file_content(
                request.owner, request.name, "CODEOWNERS"
            )
        signs.has_codeowners = codeowners is not None
        if codeowners is not None:
            signs.codeowners_mentions_path = (
                path in codeowners or path.split("/")[0] in codeowners
            )
            signs.evidence_urls["codeowners"] = (
                f"https://github.com/{request.repo}/blob/main/.github/CODEOWNERS"
            )
    except httpx.HTTPError as exc:
        signs.fetch_errors.append(f"CODEOWNERS fetch failed: {exc}")
        signs.has_codeowners = None

    return signs


def _to_evidence(signs: VitalSigns, repo: str) -> EvidenceItem:
    trunc = " truncated" if signs.commits_truncated else ""
    err = f" errors={len(signs.fetch_errors)}" if signs.fetch_errors else ""
    snippet = (
        f"`{signs.path}`: commits 3/6/12mo = {signs.commits_3m}/{signs.commits_6m}/"
        f"{signs.commits_12m}{trunc} ({signs.trend}); "
        f"committers={signs.distinct_committers_12m}; "
        f"open_issues_total={signs.open_issues_total}; "
        f"closure_rate={signs.closure_rate}; CODEOWNERS={signs.has_codeowners}.{err}"
    )
    url = signs.evidence_urls.get("commits") or f"https://github.com/{repo}"
    return EvidenceItem(
        id=f"vitals-{signs.path.replace('/', '-')}",
        kind=EvidenceKind.VITAL_SIGNS,
        title=f"Module vital signs: {signs.path}",
        url=url,
        snippet=snippet[:500],
        source_retriever="vital_signs",
        relevance_score=0.92,
        metadata=signs.to_metadata(),
    )


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError:
        return None
