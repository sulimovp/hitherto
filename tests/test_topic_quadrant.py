"""Demand×supply quadrant wiring — PREDICT_QUADRANT.md §5."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from casefile.models.assessment import AssessmentReport, AssessmentRequest
from casefile.models.evidence import EvidenceBundle, EvidenceItem, EvidenceKind
from casefile.models.profile import AssignmentPrecision, EcosystemProfile
from casefile.predict.time import MaintainerSet
from casefile.predict.topic_hazard import (
    assess_topic_hazard,
    build_topic_provenance,
    topic_hazard_to_dict,
)
from casefile.predict.topic_history import (
    compute_topic_history,
    inflow_month_divisors,
    topic_windows,
)


def _inflow_page(n: int) -> dict:
    return {
        "items": [
            {
                "number": i,
                "title": f"torch/masked issue {i}",
                "body": "",
            }
            for i in range(1, n + 1)
        ],
        "total_count": n,
    }


def _closed_issue(number: int, **extra) -> dict:
    item = {
        "number": number,
        "title": f"torch/masked issue {number}",
        "body": "",
        "closed_at": "2026-07-01T00:00:00Z",
        "created_at": "2026-06-01T00:00:00Z",
        "closed_by": {"login": "alice"},
    }
    item.update(extra)
    return item
from casefile.render.markdown import render_markdown


def test_inflow_windows_do_not_overlap():
    now = datetime(2026, 8, 24, tzinfo=UTC)
    baseline_start, recent_start, end = topic_windows(now)
    assert end == now
    assert recent_start == now - timedelta(days=180)
    assert baseline_start == now - timedelta(days=730)
    # Search queries must not share an inclusive GitHub calendar day.
    from casefile.predict.topic_history import _nonoverlap_search_bounds

    b0, b1, r0, r1 = _nonoverlap_search_bounds(baseline_start, recent_start, end)
    assert b1 != r0
    assert b1 == (recent_start - timedelta(days=1)).strftime("%Y-%m-%d")
    recent_m, baseline_m = inflow_month_divisors()
    assert recent_m == 6.0
    assert baseline_m == 18.0


@pytest.mark.asyncio
async def test_inflow_saturation_refuses():
    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(
        side_effect=[
            {"items": [], "total_count": 1000},
            _inflow_page(40),
            {"items": [], "total_count": 10},
            {"items": [], "total_count": 10},
        ]
    )
    gh.get_earliest_commit_date = AsyncMock(return_value="2020-01-01T00:00:00Z")
    gh.list_issue_timeline = AsyncMock(return_value=[])
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    history = await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    assert history.search_truncated is True
    assert history.demand_recent_per_month is None
    assert history.demand_baseline_per_month is None
    rollup = assess_topic_hazard(
        bundle=_bundle(),
        profile=None,
        path="torch/masked",
        topic_first_seen=history.topic_first_seen,
        now=datetime(2026, 8, 24, tzinfo=UTC),
        assignment_precision_measured=True,
        n_resolved_items=40,
        demand_recent_per_month=history.demand_recent_per_month,
        demand_baseline_per_month=history.demand_baseline_per_month,
        realized_r1_rate_recent=0.3,
        realized_r1_rate_baseline=0.5,
    )
    assert rollup.rollup_refused is not None
    assert "demand axis" in rollup.rollup_refused


@pytest.mark.asyncio
async def test_r1_rate_excludes_unresolved_pr_paths():
    closed = "2026-07-01T00:00:00Z"
    issue = _closed_issue(1)
    timeline = [
        {
            "event": "merged",
            "created_at": closed,
            "pull_request": {"number": 99},
        }
    ]
    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(
        side_effect=[
            _inflow_page(5),
            _inflow_page(10),
            {"items": [issue], "total_count": 1},
            {"items": [], "total_count": 0},
        ]
    )
    gh.list_issue_timeline = AsyncMock(return_value=timeline)
    gh.list_pr_files = AsyncMock(return_value=[])  # unfetchable / empty paths
    gh.get_earliest_commit_date = AsyncMock(return_value="2020-01-01T00:00:00Z")
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    history = await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    assert history.r1_excluded_n == 1
    # Single sampled item excluded → rate None (no denominator), not R2.
    assert history.realized_r1_rate_recent is None


@pytest.mark.asyncio
async def test_r1_rate_refuses_when_exclusions_exceed_quarter():
    issues = [_closed_issue(i, closed_by={"login": "bot"}, created_at="2026-01-01T00:00:00Z") for i in range(1, 51)]
    timeline_ok = []
    timeline_merged = [
        {
            "event": "merged",
            "created_at": "2026-07-01T00:00:00Z",
            "pull_request": {"number": 900},
        }
    ]

    async def timeline_for(_o, _r, number, **_kw):
        # First 13 issues have unresolvable PR paths → excluded
        if number <= 13:
            return timeline_merged
        return timeline_ok

    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(
        side_effect=[
            _inflow_page(5),
            _inflow_page(10),
            {"items": issues, "total_count": 50},
            {"items": [], "total_count": 0},
        ]
    )
    gh.list_issue_timeline = AsyncMock(side_effect=timeline_for)
    gh.list_pr_files = AsyncMock(return_value=[])
    gh.get_earliest_commit_date = AsyncMock(return_value="2020-01-01T00:00:00Z")
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    history = await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    assert history.r1_excluded_n >= 13
    assert history.realized_r1_rate_recent is None
    result = assess_topic_hazard(
        bundle=_bundle(),
        profile=None,
        path="torch/masked",
        topic_first_seen=history.topic_first_seen,
        now=datetime(2026, 8, 24, tzinfo=UTC),
        assignment_precision_measured=True,
        n_resolved_items=50,
        demand_recent_per_month=3.0,
        demand_baseline_per_month=1.0,
        realized_r1_rate_recent=history.realized_r1_rate_recent,
        realized_r1_rate_baseline=0.5,
    )
    assert result.rollup_refused is not None
    assert "supply axis" in result.rollup_refused


@pytest.mark.asyncio
async def test_r1_sample_is_created_sorted():
    calls: list[dict] = []

    async def search(query, *, sort="created", order="desc", per_page=30, **_kw):
        calls.append({"query": query, "sort": sort, "order": order, "per_page": per_page})
        return {"items": [], "total_count": 0}

    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(side_effect=search)
    gh.get_earliest_commit_date = AsyncMock(return_value="2020-01-01T00:00:00Z")
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    created = [c["query"] for c in calls if "created:" in c["query"]]
    closed = [c["query"] for c in calls if "closed:" in c["query"]]
    assert any("created:2026-02-25..2026-08-24" in q for q in created)
    assert any("created:2024-08-24..2026-02-24" in q for q in created)
    assert any("closed:2026-02-25..2026-08-24" in q for q in closed)
    assert any("closed:2024-08-24..2026-02-24" in q for q in closed)
    closed_calls = [c for c in calls if "is:closed" in c["query"]]
    assert closed_calls
    assert all(c["sort"] == "created" for c in closed_calls)
    assert all(c["sort"] != "best-match" for c in closed_calls)


@pytest.mark.asyncio
async def test_r1_refuses_when_window_exceeds_sample_cap():
    issue = _closed_issue(1)
    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(
        side_effect=[
            _inflow_page(5),
            _inflow_page(10),
            {"items": [issue] * 50, "total_count": 51},
            {"items": [], "total_count": 0},
        ]
    )
    gh.get_earliest_commit_date = AsyncMock(return_value="2020-01-01T00:00:00Z")
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    history = await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    assert history.r1_truncated is True
    assert history.realized_r1_rate_recent is None
    assert history.realized_r1_rate_baseline is None
    assert any("truncated" in e for e in history.fetch_errors)


@pytest.mark.asyncio
async def test_empty_maintainer_set_does_not_call_for_snapshot(monkeypatch):
    def boom(self, at):
        raise AssertionError("for_snapshot must not run when logins are empty")

    monkeypatch.setattr(MaintainerSet, "for_snapshot", boom)
    issue = _closed_issue(1)
    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(
        side_effect=[
            _inflow_page(5),
            _inflow_page(10),
            {"items": [issue], "total_count": 1},
            {"items": [], "total_count": 0},
        ]
    )
    gh.list_issue_timeline = AsyncMock(return_value=[])
    gh.get_earliest_commit_date = AsyncMock(return_value="2020-01-01T00:00:00Z")
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    history = await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    assert any("R1-by-answer disabled" in e for e in history.fetch_errors)


@pytest.mark.asyncio
async def test_inflow_counts_only_assigned_issues():
    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(
        side_effect=[
            {
                "items": [
                    {"number": 1, "title": "torch/masked sum", "body": ""},
                    {"number": 2, "title": "boolean mask indexing", "body": ""},
                ],
                "total_count": 2,
            },
            _inflow_page(8),
            {"items": [], "total_count": 0},
            {"items": [], "total_count": 0},
        ]
    )
    gh.get_earliest_commit_date = AsyncMock(return_value="2020-01-01T00:00:00Z")
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    history = await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    assert history.inflow_recent_total == 1
    assert history.inflow_baseline_total == 8


@pytest.mark.asyncio
async def test_topic_first_seen_from_link_last_page(httpx_mock, tmp_path):
    from casefile.clients.github import GitHubClient
    from casefile.config import Settings

    settings = Settings(
        github_token="t",
        cache_dir=tmp_path / "cache",
        github_api_base="https://api.github.com",
    )
    link = (
        '<https://api.github.com/repos/o/r/commits?path=torch%2Fmasked&per_page=1&page=42>; '
        'rel="last"'
    )
    httpx_mock.add_response(
        url="https://api.github.com/repos/o/r/commits?path=torch%2Fmasked&per_page=1",
        json=[{"commit": {"author": {"date": "2024-01-01T00:00:00Z"}}}],
        headers={"Link": link},
    )
    httpx_mock.add_response(
        url="https://api.github.com/repos/o/r/commits?path=torch%2Fmasked&per_page=1&page=42",
        json=[
            {"commit": {"author": {"date": "2019-06-15T12:00:00Z"}}},
        ],
    )
    import httpx

    async with httpx.AsyncClient() as client:
        gh = GitHubClient(settings, client)
        date_str = await gh.get_earliest_commit_date("o", "r", path="torch/masked")
    assert date_str == "2019-06-15T12:00:00Z"


def test_assignment_precision_below_floor_refuses():
    profile = EcosystemProfile(
        id="pytorch",
        display_name="PyTorch",
        last_verified=date(2026, 8, 1),
        max_age_days=90,
        assignment_precision=AssignmentPrecision(
            measured_at=date(2026, 8, 1),
            n=100,
            precision=0.62,
            recall=0.55,
        ),
    )
    ok, reason = profile.assignment_precision_status(
        as_of=datetime(2026, 8, 24, tzinfo=UTC)
    )
    assert ok is False
    assert reason is not None
    assert "0.62" in reason
    assert "n=100" in reason
    result = assess_topic_hazard(
        bundle=_bundle(),
        profile=profile,
        path="torch/masked",
        topic_first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        assignment_precision_measured=False,
        assignment_precision_refusal=reason,
        n_resolved_items=40,
        demand_recent_per_month=3.0,
        demand_baseline_per_month=1.0,
        realized_r1_rate_recent=0.2,
        realized_r1_rate_baseline=0.5,
    )
    assert result.rollup_refused is not None
    assert "0.62" in result.rollup_refused
    assert "n=100" in result.rollup_refused


def test_assignment_precision_stale_refuses():
    profile = EcosystemProfile(
        id="pytorch",
        display_name="PyTorch",
        last_verified=date(2026, 8, 1),
        max_age_days=90,
        assignment_precision=AssignmentPrecision(
            measured_at=date(2025, 1, 1),
            n=100,
            precision=0.90,
            recall=0.80,
        ),
    )
    ok, reason = profile.assignment_precision_status(
        as_of=datetime(2026, 8, 24, tzinfo=UTC)
    )
    assert ok is False
    assert reason is not None
    assert "stale" in reason


def test_assignment_recall_below_floor_refuses():
    profile = EcosystemProfile(
        id="apertus",
        display_name="Apertus",
        last_verified=date(2026, 8, 1),
        max_age_days=90,
        assignment_precision=AssignmentPrecision(
            measured_at=date(2026, 8, 28),
            n=81,
            precision=0.88,
            recall=0.64,
        ),
    )
    ok, reason = profile.assignment_precision_status(
        as_of=datetime(2026, 8, 28, tzinfo=UTC)
    )
    assert ok is False
    assert reason is not None
    assert "recall 0.64" in reason


def test_assignment_recall_none_does_not_refuse():
    profile = EcosystemProfile(
        id="pytorch",
        display_name="PyTorch",
        last_verified=date(2026, 5, 30),
        max_age_days=90,
        assignment_precision=AssignmentPrecision(
            measured_at=date(2026, 8, 28),
            n=95,
            precision=0.91,
            recall=None,
        ),
    )
    ok, reason = profile.assignment_precision_status(
        as_of=datetime(2026, 8, 28, tzinfo=UTC)
    )
    assert ok is True
    assert reason is None


@pytest.mark.asyncio
async def test_fetch_error_never_imputes_zero():
    gh = AsyncMock()
    gh.search_issues_page_sorted = AsyncMock(side_effect=RuntimeError("rate limited"))
    gh.get_earliest_commit_date = AsyncMock(return_value=None)
    maintainers = MaintainerSet(frozenset(), as_of=datetime(2026, 8, 24, tzinfo=UTC))
    history = await compute_topic_history(
        gh,
        repo="pytorch/pytorch",
        path="torch/masked",
        topic_paths=("torch/masked",),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        maintainers=maintainers,
    )
    assert history.demand_recent_per_month is None
    assert history.demand_baseline_per_month is None
    assert history.inflow_recent_total is None
    assert history.fetch_errors
    assert all("0.0" not in e for e in history.fetch_errors)


def test_quadrant_renders_with_provenance_line():
    result = assess_topic_hazard(
        bundle=_bundle(),
        profile=None,
        path="torch/masked",
        topic_first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        now=datetime(2026, 8, 24, tzinfo=UTC),
        assignment_precision_measured=True,
        n_resolved_items=40,
        demand_recent_per_month=3.7,
        demand_baseline_per_month=1.9,
        realized_r1_rate_recent=0.31,
        realized_r1_rate_baseline=0.68,
        trained_artifact_available=False,
        observation_window_days=None,
    )
    assert result.rollup_refused is None
    assert result.quadrant == "gap"
    payload = topic_hazard_to_dict(
        result,
        provenance=build_topic_provenance(
            demand_recent_per_month=3.7,
            demand_baseline_per_month=1.9,
            realized_r1_rate_recent=0.31,
            realized_r1_rate_baseline=0.68,
            r1_sampled_n_recent=44,
            r1_sampled_n_baseline=50,
            r1_excluded_n=6,
            assignment_precision=0.86,
            assignment_precision_n=100,
            assignment_precision_measured_at="2026-09-01",
        ),
    )
    report = AssessmentReport(
        request=AssessmentRequest(
            question="masked tensor path health",
            repo="pytorch/pytorch",
            path="torch/masked",
            synthesize=False,
        ),
        evidence=_bundle(),
        topic_forecast=payload,
    )
    md = render_markdown(report)
    assert "## Topic trajectory" in md
    assert "quadrant **gap**" in md
    assert "Inflow 3.7/mo recent vs 1.9/mo baseline" in md
    assert "R1 rate 0.31 (n=44, 6 excluded) vs 0.68 (n=50)" in md
    assert "assignment precision 0.86 (applied to counted issues, n=100, 2026-09-01)" in md
    assert "_Hazard score refused:_" in md
    assert "## Activity forecast" not in md


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        items=[
            EvidenceItem(
                id="vs",
                kind=EvidenceKind.VITAL_SIGNS,
                title="Vitals",
                url="https://github.com/pytorch/pytorch",
                snippet="",
                source_retriever="vital_signs",
                metadata={"closed_issues_total": 40},
            ),
            EvidenceItem(
                id="issue-1",
                kind=EvidenceKind.ISSUE,
                title="Masked",
                url="https://github.com/pytorch/pytorch/issues/1",
                snippet="please",
                source_retriever="github_issues",
            ),
        ]
    )
