"""Topic-hazard labels, person-period, extractor pin, quadrant, refuse path."""

from datetime import UTC, datetime, timedelta
import json
import math

import pytest

import casefile.predict as predict_pkg
from casefile.models.evidence import EvidenceBundle, EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile, HuggingFaceConfig, HuggingFaceHubRepo
from casefile.predict.extractor import (
    parse_extracted_json,
    pin_extractor_version,
    validate_evidence_spans,
)
from casefile.predict.features import engagement_at_t
from casefile.predict.labels import (
    LinkedMergedPR,
    LinkedMergedPRs,
    Outcome,
    label_issue_outcome,
    labels_at_time,
    linked_merged_prs_le,
)
from casefile.predict.person_period import (
    cumulative_incidence_r1,
    expand_person_period,
    kaplan_meier_complement,
    to_training_row,
)
from casefile.predict.quadrant import (
    DemandTrend,
    Quadrant,
    SupplyTrend,
    build_topic_rollup,
    classify_quadrant,
)
from casefile.predict.time import MaintainerSet, parse_dt
from casefile.predict.topic_hazard import (
    assess_topic_hazard,
    rollup_refusals,
    score_refusals,
)


def _merged_prs(numbers: list[int], *, paths: tuple[str, ...] = ()) -> LinkedMergedPRs:
    return LinkedMergedPRs(
        prs=tuple(
            LinkedMergedPR(number=n, merged_at=None, paths_touched=paths) for n in numbers
        )
    )


def test_label_r1_linked_pr_requires_path_touch():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    closed = datetime(2024, 1, 20, tzinfo=UTC)
    merged = _merged_prs([99], paths=("torch/masked/",))
    out = label_issue_outcome(
        state="closed",
        closed_at=closed,
        closed_by_login="alice",
        labels_at_close=[],
        linked_merged=merged,
        topic_paths=("torch/masked/",),
        maintainer_answered=False,
        observation_end=datetime(2024, 6, 1, tzinfo=UTC),
        created_at=created,
    )
    assert out.outcome == Outcome.R1
    assert out.linked_pr_numbers == (99,)


def test_label_linked_pr_without_path_is_not_r1():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    closed = datetime(2024, 1, 20, tzinfo=UTC)
    merged = _merged_prs([99], paths=("dependabot/",))
    out = label_issue_outcome(
        state="closed",
        closed_at=closed,
        closed_by_login="alice",
        labels_at_close=[],
        linked_merged=merged,
        topic_paths=("torch/masked/",),
        maintainer_answered=False,
        observation_end=datetime(2024, 6, 1, tzinfo=UTC),
        created_at=created,
    )
    assert out.outcome == Outcome.R2


def test_unlabeled_closed_as_r2_flag():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    closed = datetime(2024, 4, 1, tzinfo=UTC)
    out_default = label_issue_outcome(
        state="closed",
        closed_at=closed,
        closed_by_login="alice",
        labels_at_close=[],
        linked_merged=LinkedMergedPRs(()),
        maintainer_answered=False,
        observation_end=datetime(2024, 6, 1, tzinfo=UTC),
        created_at=created,
    )
    assert out_default.outcome == Outcome.R2

    out_flag_off = label_issue_outcome(
        state="closed",
        closed_at=closed,
        closed_by_login="alice",
        labels_at_close=[],
        linked_merged=LinkedMergedPRs(()),
        maintainer_answered=False,
        observation_end=datetime(2024, 6, 1, tzinfo=UTC),
        created_at=created,
        unlabeled_closed_as_r2=False,
    )
    assert out_flag_off.outcome == Outcome.R3


def test_label_r2_stale_bot():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    closed = datetime(2024, 4, 1, tzinfo=UTC)
    out = label_issue_outcome(
        state="closed",
        closed_at=closed,
        closed_by_login="stale[bot]",
        labels_at_close=["stale"],
        linked_merged=LinkedMergedPRs(()),
        maintainer_answered=False,
        observation_end=datetime(2024, 6, 1, tzinfo=UTC),
        created_at=created,
    )
    assert out.outcome == Outcome.R2


def test_label_r3_still_open():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    out = label_issue_outcome(
        state="open",
        closed_at=None,
        closed_by_login=None,
        labels_at_close=[],
        linked_merged=LinkedMergedPRs(()),
        maintainer_answered=False,
        observation_end=datetime(2024, 6, 1, tzinfo=UTC),
        created_at=created,
    )
    assert out.outcome == Outcome.R3


def test_person_period_stops_at_r1():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    closed = created + timedelta(days=10)
    merged = _merged_prs([1], paths=("torch/masked/",))
    outcome = label_issue_outcome(
        state="closed",
        closed_at=closed,
        closed_by_login="alice",
        labels_at_close=[],
        linked_merged=merged,
        topic_paths=("torch/masked/",),
        maintainer_answered=False,
        observation_end=created + timedelta(days=180),
        created_at=created,
    )
    rows = expand_person_period(
        item_id="issue-1",
        created_at=created,
        outcome=outcome,
        observation_end=created + timedelta(days=180),
    )
    assert rows
    assert sum(r.y_r1 for r in rows) == 1
    assert rows[-1].y_r1 == 1
    assert len(rows) <= 3


def test_person_period_partial_final_week_exposure():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    observation_end = created + timedelta(days=10)
    outcome = label_issue_outcome(
        state="open",
        closed_at=None,
        closed_by_login=None,
        labels_at_close=[],
        linked_merged=LinkedMergedPRs(()),
        maintainer_answered=False,
        observation_end=observation_end,
        created_at=created,
    )
    rows = expand_person_period(
        item_id="issue-1",
        created_at=created,
        outcome=outcome,
        observation_end=observation_end,
    )
    assert len(rows) == 2
    assert rows[-1].exposure_days == pytest.approx(3.0)
    assert rows[-1].y_r1 == 0
    assert rows[-1].censored_after == 1


def test_cumulative_incidence_r1_competing_r2():
    assert cumulative_incidence_r1([0.0, 0.0], [0.0, 0.0]) == [0.0, 0.0]
    assert cumulative_incidence_r1([1.0], [0.0]) == [1.0]
    km_style = 1.0 - (1.0 - 0.3) * (1.0 - 0.3)
    cif = cumulative_incidence_r1([0.3, 0.3], [0.3, 0.3])
    assert cif[-1] == pytest.approx(0.42)
    assert cif[-1] < km_style


def test_cumulative_incidence_requires_h2():
    with pytest.raises(TypeError):
        cumulative_incidence_r1([0.1])  # type: ignore[call-arg]


def test_km_complement_exceeds_cif_when_r2_present():
    h1 = [0.1, 0.1, 0.1, 0.1]
    h2 = [0.2, 0.2, 0.2, 0.2]
    assert kaplan_meier_complement(h1)[-1] > cumulative_incidence_r1(h1, h2)[-1]


def test_survival_from_hazards_gone():
    assert not hasattr(predict_pkg, "survival_from_hazards")


def test_to_training_row_carries_exposure_offset():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    observation_end = created + timedelta(days=10)
    outcome = label_issue_outcome(
        state="open",
        closed_at=None,
        closed_by_login=None,
        labels_at_close=[],
        linked_merged=LinkedMergedPRs(()),
        maintainer_answered=False,
        observation_end=observation_end,
        created_at=created,
    )
    rows = expand_person_period(
        item_id="issue-1",
        created_at=created,
        outcome=outcome,
        observation_end=observation_end,
    )
    partial = to_training_row(rows[-1])
    assert partial["exposure_days"] == pytest.approx(3.0)
    assert partial["log_exposure_offset"] == pytest.approx(math.log(3 / 7))

    full_end = created + timedelta(days=14)
    full_outcome = label_issue_outcome(
        state="open",
        closed_at=None,
        closed_by_login=None,
        labels_at_close=[],
        linked_merged=LinkedMergedPRs(()),
        maintainer_answered=False,
        observation_end=full_end,
        created_at=created,
    )
    full_rows = expand_person_period(
        item_id="issue-2",
        created_at=created,
        outcome=full_outcome,
        observation_end=full_end,
    )
    assert to_training_row(full_rows[0])["log_exposure_offset"] == pytest.approx(0.0)

def test_linked_merged_prs_le_ignores_closed_issue_cross_ref():
    at = datetime(2024, 6, 1, tzinfo=UTC)
    timeline = [
        {
            "event": "cross-referenced",
            "created_at": "2024-01-10T00:00:00Z",
            "source": {"issue": {"number": 9, "state": "closed", "pull_request": {"url": "x"}}},
        }
    ]
    assert linked_merged_prs_le(timeline, at).numbers() == ()


def test_linked_merged_prs_le_requires_merged_event():
    at = datetime(2024, 6, 1, tzinfo=UTC)
    timeline = [
        {
            "event": "cross-referenced",
            "created_at": "2024-01-10T00:00:00Z",
            "source": {"issue": {"number": 9, "pull_request": {"url": "x"}}},
        },
        {
            "event": "merged",
            "created_at": "2024-01-15T00:00:00Z",
            "pull_request": {"number": 9},
        },
    ]
    assert linked_merged_prs_le(timeline, at).numbers() == (9,)


def test_parse_dt_naive_is_utc():
    dt = parse_dt("2024-01-15T12:00:00")
    assert dt is not None
    assert dt.tzinfo == UTC


def test_maintainer_set_rejects_future_snapshot():
    maintainers = MaintainerSet(
        logins=frozenset({"alice"}),
        as_of=datetime(2024, 6, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="after snapshot"):
        maintainers.for_snapshot(datetime(2024, 1, 1, tzinfo=UTC))


def test_gap_quadrant_is_demand_rising_supply_falling():
    assert classify_quadrant(DemandTrend.RISING, SupplyTrend.FALLING) == Quadrant.GAP
    rollup = build_topic_rollup(
        demand_recent_per_month=10.0,
        demand_baseline_per_month=4.0,
        realized_r1_rate_recent=0.2,
        realized_r1_rate_baseline=0.5,
        n_resolved_items=40,
    )
    assert rollup.quadrant == Quadrant.GAP
    assert "contribution" in rollup.note.lower()


def test_binomial_inflow_constant_rate_is_flat():
    from casefile.predict.quadrant import demand_from_inflow

    # Equal rates: 6 in 6mo vs 18 in 18mo. Marginal Poisson CIs overlap;
    # the conditional binomial test also stays FLAT.
    assert (
        demand_from_inflow(
            recent_per_month=1.0,
            baseline_per_month=1.0,
            recent_count=6,
            baseline_count=18,
        )
        == DemandTrend.FLAT
    )


def test_binomial_inflow_rate_ratio_is_rising():
    from casefile.predict.quadrant import demand_from_inflow

    assert (
        demand_from_inflow(
            recent_per_month=4.0,
            baseline_per_month=1.0,
            recent_count=24,
            baseline_count=18,
        )
        == DemandTrend.RISING
    )


def test_inflow_below_min_count_is_unknown():
    from casefile.predict.quadrant import demand_from_inflow

    assert (
        demand_from_inflow(
            recent_per_month=2.0,
            baseline_per_month=0.2,
            recent_count=2,
            baseline_count=4,
        )
        == DemandTrend.UNKNOWN
    )


def test_extractor_rejects_floating_alias():
    with pytest.raises(ValueError, match="concrete pin"):
        pin_extractor_version(model_id="openai/gpt-oss-120b:fastest")


def test_extractor_pin_and_span_check():
    version = pin_extractor_version(model_id="openai/gpt-oss-120b@rev-abc")
    assert "block6-v3" in version
    assert ":fastest" not in version
    quote = "add masked fill support or use NestedTensor instead"
    raw = json.dumps(
        {
            "intent": "feature_request",
            "specificity": 2,
            "proposed_solution_present": True,
            "patch_offered": False,
            "blocking_severity": 1,
            "affect": 0,
            "maintainer_stance": "acknowledged",
            "scope": "contained",
            "names_alternative": True,
            "names_alternative_text": "NestedTensor",
            "evidence_span": {
                "intent": quote,
                "proposed_solution_present": quote,
                "patch_offered": quote,
                "maintainer_stance": quote,
                "names_alternative": quote,
                "names_alternative_text": quote,
            },
            "evidence_anchor": {
                "specificity": "steps_or_code",
                "blocking_severity": "workaround_exists",
                "affect": "neutral_report",
                "scope": "one_module",
            },
        }
    )
    fields = parse_extracted_json(raw, extractor_version=version)
    hay = "Please add masked fill support or use NestedTensor instead for batching."
    assert validate_evidence_spans(fields, hay) == []
    assert validate_evidence_spans(fields, "unrelated text")


def test_extractor_rejects_out_of_schema_category():
    raw = json.dumps({"intent": "bug report", "evidence_span": {}})
    with pytest.raises(ValueError, match="intent must be one of"):
        parse_extracted_json(raw, extractor_version="test")


def test_extractor_requires_span_for_every_non_null_field():
    raw = json.dumps({"intent": "bug", "specificity": None, "evidence_span": {}})
    fields = parse_extracted_json(raw, extractor_version="test")
    assert validate_evidence_spans(fields, "bug report") == [
        "evidence_span[intent] missing for non-null field"
    ]


def test_extractor_absence_does_not_need_a_quote():
    raw = json.dumps(
        {
            "intent": "bug",
            "proposed_solution_present": False,
            "patch_offered": False,
            "maintainer_stance": "none",
            "names_alternative": False,
            "evidence_span": {"intent": "bug in masked fill"},
        }
    )
    fields = parse_extracted_json(raw, extractor_version="test")
    assert validate_evidence_spans(fields, "bug in masked fill on cpu") == []


def test_extractor_true_and_real_stance_still_need_quotes():
    raw = json.dumps(
        {
            "patch_offered": True,
            "maintainer_stance": "acknowledged",
            "evidence_span": {},
        }
    )
    fields = parse_extracted_json(raw, extractor_version="test")
    errors = validate_evidence_spans(fields, "please add a patch")
    assert "evidence_span[patch_offered] missing for non-null field" in errors
    assert "evidence_span[maintainer_stance] missing for non-null field" in errors


def test_extractor_judged_field_uses_rubric_not_quote():
    raw = json.dumps(
        {
            "specificity": 2,
            "evidence_span": {},
            "evidence_anchor": {"specificity": "steps_or_code"},
        }
    )
    fields = parse_extracted_json(raw, extractor_version="test")
    assert validate_evidence_spans(fields, "title with no reproducer") == []


def test_extractor_rejects_rubric_cell_from_the_wrong_level():
    raw = json.dumps(
        {
            "specificity": 2,
            "evidence_span": {},
            "evidence_anchor": {"specificity": "title_only"},
        }
    )
    fields = parse_extracted_json(raw, extractor_version="test")
    errors = validate_evidence_spans(fields, "anything")
    assert any("title_only" in err and "specificity=2" in err for err in errors)


def test_extractor_requires_anchor_for_judged_field():
    raw = json.dumps({"specificity": 3, "evidence_span": {}, "evidence_anchor": {}})
    fields = parse_extracted_json(raw, extractor_version="test")
    assert validate_evidence_spans(fields, "minimal reproducer in body") == [
        "evidence_anchor[specificity] missing for non-null field"
    ]


def test_labels_at_time_replays_timeline():
    at = datetime(2024, 2, 1, tzinfo=UTC)
    timeline = [
        {
            "event": "labeled",
            "created_at": "2024-01-10T00:00:00Z",
            "label": {"name": "bug"},
        },
        {
            "event": "labeled",
            "created_at": "2024-01-15T00:00:00Z",
            "label": {"name": "stale"},
        },
        {
            "event": "unlabeled",
            "created_at": "2024-01-20T00:00:00Z",
            "label": {"name": "stale"},
        },
        {
            "event": "labeled",
            "created_at": "2024-03-01T00:00:00Z",
            "label": {"name": "wontfix"},
        },
    ]
    assert labels_at_time(timeline, at) == ["bug"]


def test_engagement_uses_reactions_created_at():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    at = datetime(2024, 2, 1, tzinfo=UTC)
    timeline = [
        {
            "event": "commented",
            "created_at": "2024-01-05T00:00:00Z",
            "actor": {"login": "maintainer1"},
        }
    ]
    reactions = [
        {"content": "+1", "created_at": "2024-01-10T00:00:00Z"},
        {"content": "+1", "created_at": "2024-03-01T00:00:00Z"},
        {"content": "heart", "created_at": "2024-01-12T00:00:00Z"},
    ]
    maintainers = MaintainerSet(
        logins=frozenset({"maintainer1"}),
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
    )
    eng = engagement_at_t(
        created_at=created,
        timeline=timeline,
        reactions=reactions,
        at=at,
        maintainers=maintainers,
    )
    assert eng.n_comments_le_t == 1
    assert eng.maintainer_replied_le_t is True
    assert eng.reactions_plus1_le_t == 1
    assert eng.reactions_heart_le_t == 1


def test_engagement_counts_ghost_user_comments():
    created = datetime(2024, 1, 1, tzinfo=UTC)
    at = datetime(2024, 2, 1, tzinfo=UTC)
    timeline = [
        {
            "event": "commented",
            "created_at": "2024-01-05T00:00:00Z",
            "actor": {"id": 999},
        }
    ]
    maintainers = MaintainerSet(
        logins=frozenset({"maintainer1"}),
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
    )
    eng = engagement_at_t(
        created_at=created,
        timeline=timeline,
        reactions=[],
        at=at,
        maintainers=maintainers,
    )
    assert eng.n_comments_le_t == 1
    assert eng.n_participants_le_t == 1


def _path_scoped_bundle() -> EvidenceBundle:
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
                title="Masked tensor request",
                url="https://github.com/pytorch/pytorch/issues/1",
                snippet="please add",
                source_retriever="github_issues",
            ),
        ]
    )


def _assess_path_scoped(**overrides):
    kwargs = {
        "bundle": _path_scoped_bundle(),
        "profile": None,
        "path": "torch/masked/",
        "topic_first_seen": datetime(2020, 1, 1, tzinfo=UTC),
        "now": datetime(2026, 1, 1, tzinfo=UTC),
        "assignment_precision_measured": True,
        "n_resolved_items": 40,
        "demand_recent_per_month": 10.0,
        "demand_baseline_per_month": 4.0,
        "realized_r1_rate_recent": 0.2,
        "realized_r1_rate_baseline": 0.5,
        "extractor_version_ok": True,
        "trained_artifact_available": False,
        "observation_window_days": 400.0,
    }
    kwargs.update(overrides)
    return assess_topic_hazard(**kwargs)


def test_default_deny_refuses_hub_only_without_path():
    profile = EcosystemProfile(
        id="demo",
        display_name="Demo",
        last_verified=datetime(2026, 8, 23, tzinfo=UTC).date(),
        huggingface=HuggingFaceConfig(
            hub_repos=[HuggingFaceHubRepo(repo_id="org/model")],
        ),
    )
    bundle = EvidenceBundle(
        items=[
            EvidenceItem(
                id=f"hf-{i}",
                kind=EvidenceKind.HF_DISCUSSION,
                title=f"Thread {i}",
                url=f"https://huggingface.co/org/model/discussions/{i}",
                snippet="discussion",
                source_retriever="huggingface_discussions",
            )
            for i in range(1, 25)
        ]
        + [
            EvidenceItem(
                id="issue-1",
                kind=EvidenceKind.ISSUE,
                title="Feature request",
                url="https://github.com/org/repo/issues/1",
                snippet="please add",
                source_retriever="github_issues",
            )
        ]
    )
    reasons = rollup_refusals(
        bundle=bundle,
        profile=profile,
        path=None,
        topic_first_seen=None,
        now=datetime(2026, 8, 23, tzinfo=UTC),
        assignment_precision_measured=True,
        n_resolved_items=50,
        demand_recent_per_month=10.0,
        demand_baseline_per_month=5.0,
        realized_r1_rate_recent=0.3,
        realized_r1_rate_baseline=0.4,
    )
    assert any("path" in r.lower() for r in reasons)
    score = score_refusals(
        shared_refusals=reasons[:1],  # at least path is shared
        trained_artifact_available=False,
        extractor_version_ok=True,
        observation_window_days=400.0,
    )
    assert any("artifact" in r.lower() for r in score)


def test_apertus_like_bundle_refuses_topic_hazard():
    profile = EcosystemProfile(
        id="apertus",
        display_name="Apertus",
        last_verified=datetime(2026, 8, 23, tzinfo=UTC).date(),
        huggingface=HuggingFaceConfig(
            hub_repos=[HuggingFaceHubRepo(repo_id="swiss-ai/Apertus-v1.5-8B")],
        ),
    )
    bundle = EvidenceBundle(
        items=[
            EvidenceItem(
                id=f"hf-{i}",
                kind=EvidenceKind.HF_DISCUSSION,
                title=f"Thread {i}",
                url=f"https://huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/{i}",
                snippet="discussion",
                source_retriever="huggingface_discussions",
            )
            for i in range(1, 8)
        ]
    )
    result = assess_topic_hazard(
        bundle=bundle,
        profile=profile,
        path=None,
        now=datetime(2026, 8, 23, tzinfo=UTC),
        assignment_precision_measured=True,
    )
    assert result.rollup_refused
    assert result.score_refused
    assert "path" in (result.rollup_refused or "").lower()
    assert "Hub items" in (result.rollup_refused or "")


def test_resolved_floor_uses_search_total_not_bundle():
    bundle = EvidenceBundle(
        items=[
            EvidenceItem(
                id="vs",
                kind=EvidenceKind.VITAL_SIGNS,
                title="Vitals",
                url="https://github.com/org/repo",
                snippet="",
                source_retriever="vital_signs",
                metadata={"closed_issues_total": 3},
            ),
            EvidenceItem(
                id="closed-1",
                kind=EvidenceKind.ISSUE,
                title="Closed",
                url="https://github.com/org/repo/issues/1",
                snippet="",
                source_retriever="github_issues",
            ),
        ]
    )
    reasons = rollup_refusals(
        bundle=bundle,
        profile=None,
        path="torch/masked/",
        topic_first_seen=datetime(2020, 1, 1, tzinfo=UTC),
        now=datetime(2026, 1, 1, tzinfo=UTC),
        assignment_precision_measured=True,
        n_resolved_items=None,
        demand_recent_per_month=10.0,
        demand_baseline_per_month=5.0,
        realized_r1_rate_recent=0.3,
        realized_r1_rate_baseline=0.4,
    )
    assert any("n_resolved_items=3" in r for r in reasons)


def test_rollup_renders_when_score_refuses():
    result = _assess_path_scoped(trained_artifact_available=False)
    assert result.rollup_refused is None
    assert result.quadrant == "gap"
    assert result.score_refused is not None
    assert "trained topic-hazard artifact" in result.score_refused


def test_shared_refusal_blocks_both():
    profile = EcosystemProfile(
        id="apertus",
        display_name="Apertus",
        last_verified=datetime(2026, 8, 23, tzinfo=UTC).date(),
        huggingface=HuggingFaceConfig(
            hub_repos=[HuggingFaceHubRepo(repo_id="swiss-ai/Apertus-v1.5-8B")],
        ),
    )
    bundle = EvidenceBundle(
        items=[
            EvidenceItem(
                id=f"hf-{i}",
                kind=EvidenceKind.HF_DISCUSSION,
                title=f"Thread {i}",
                url=f"https://huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/{i}",
                snippet="discussion",
                source_retriever="huggingface_discussions",
            )
            for i in range(1, 8)
        ]
    )
    result = assess_topic_hazard(
        bundle=bundle,
        profile=profile,
        path=None,
        now=datetime(2026, 8, 23, tzinfo=UTC),
        assignment_precision_measured=True,
        n_resolved_items=50,
        demand_recent_per_month=10.0,
        demand_baseline_per_month=5.0,
        realized_r1_rate_recent=0.3,
        realized_r1_rate_baseline=0.4,
        observation_window_days=400.0,
    )
    assert result.rollup_refused is not None
    assert result.score_refused is not None
    for fragment in ("path", "Hub items"):
        assert fragment in result.rollup_refused
        assert fragment in result.score_refused


def test_extractor_mismatch_blocks_score_only():
    result = _assess_path_scoped(extractor_version_ok=False)
    assert result.rollup_refused is None
    assert result.quadrant == "gap"
    assert result.score_refused is not None
    assert "extractor_version" in result.score_refused


def test_rollup_refuses_on_unknown_inflow():
    result = _assess_path_scoped(demand_recent_per_month=None)
    assert result.rollup_refused is not None
    assert "demand axis" in result.rollup_refused


def test_rollup_refuses_on_unknown_r1_rates():
    result = _assess_path_scoped(realized_r1_rate_recent=None)
    assert result.rollup_refused is not None
    assert "supply axis" in result.rollup_refused


def test_quadrant_unknown_never_rendered():
    # FLAT × FLAT via equal monthly rates and equal R1 rates
    result = _assess_path_scoped(
        demand_recent_per_month=10.0,
        demand_baseline_per_month=10.0,
        realized_r1_rate_recent=0.4,
        realized_r1_rate_baseline=0.4,
    )
    assert result.quadrant != "unknown"
    assert result.rollup_refused is not None
    assert "both flat" in result.rollup_refused

    # Rising demand × flat supply must also refuse rather than print unknown
    result2 = _assess_path_scoped(
        demand_recent_per_month=20.0,
        demand_baseline_per_month=10.0,
        realized_r1_rate_recent=0.4,
        realized_r1_rate_baseline=0.4,
    )
    assert result2.quadrant != "unknown"
    assert result2.rollup_refused is not None


def test_observation_window_blocks_score():
    result = _assess_path_scoped(observation_window_days=200)
    assert result.rollup_refused is None
    assert result.quadrant == "gap"
    assert result.score_refused is not None
    assert "observation window 200d" in result.score_refused
