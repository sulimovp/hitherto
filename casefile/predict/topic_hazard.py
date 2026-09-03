"""Topic-hazard forecast entry — default-deny; refuses when preconditions fail.

Two independent gates: rollup (descriptive demand×supply) and score (predictive
CIF). The score inherits every rollup refusal. See casefile/docs/PREDICT.md §§7–9.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from casefile.models.evidence import EvidenceBundle, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.predict.quadrant import (
    DemandTrend,
    Quadrant,
    SupplyTrend,
    build_topic_rollup,
)
from casefile.predict.time import ensure_utc

_MIN_RESOLVED = 12
_MIN_TOPIC_AGE_DAYS = 365
_MIN_OBSERVATION_WINDOW_DAYS = 360  # 2 × 180d horizon
_MODEL_ID = "topic-hazard-v0-refuse"
_DISCLAIMER = (
    "Demand×supply rollup is descriptive, from realized outcomes. "
    "Resolution-hazard score is a forecast over ~180 days — not a recommendation."
)


@dataclass(frozen=True)
class TopicHazardResult:
    model_id: str
    rollup_refused: str | None  # descriptive quadrant blocked, or None
    score_refused: str | None  # predictive hazard blocked, or None
    quadrant: str | None = None
    demand: str | None = None
    supply: str | None = None
    rollup_note: str | None = None
    cif_180d: float | None = None  # reserved; None until a trained artifact ships
    disclaimer: str = _DISCLAIMER


def assess_topic_hazard(
    *,
    bundle: EvidenceBundle,
    profile: EcosystemProfile | None,
    path: str | None,
    topic_first_seen: datetime | None = None,
    now: datetime | None = None,
    assignment_precision_measured: bool = False,
    assignment_precision_refusal: str | None = None,
    n_resolved_items: int | None = None,
    demand_recent_per_month: float | None = None,
    demand_baseline_per_month: float | None = None,
    realized_r1_rate_recent: float | None = None,
    realized_r1_rate_baseline: float | None = None,
    inflow_recent_total: int | None = None,
    inflow_baseline_total: int | None = None,
    r1_sampled_n_recent: int | None = None,
    r1_sampled_n_baseline: int | None = None,
    extractor_version_ok: bool = True,
    trained_artifact_available: bool = False,
    observation_window_days: float | None = None,
) -> TopicHazardResult:
    """Default-deny: rollup and score refuse independently for printed reasons."""
    shared = _shared_refusals(
        bundle=bundle,
        path=path,
        topic_first_seen=topic_first_seen,
        now=now,
        assignment_precision_measured=assignment_precision_measured,
        assignment_precision_refusal=assignment_precision_refusal,
        n_resolved_items=n_resolved_items,
    )
    rollup_reasons = rollup_refusals(
        bundle=bundle,
        profile=profile,
        path=path,
        topic_first_seen=topic_first_seen,
        now=now,
        assignment_precision_measured=assignment_precision_measured,
        assignment_precision_refusal=assignment_precision_refusal,
        n_resolved_items=n_resolved_items,
        demand_recent_per_month=demand_recent_per_month,
        demand_baseline_per_month=demand_baseline_per_month,
        realized_r1_rate_recent=realized_r1_rate_recent,
        realized_r1_rate_baseline=realized_r1_rate_baseline,
    )
    score_reasons = score_refusals(
        shared_refusals=shared,
        trained_artifact_available=trained_artifact_available,
        extractor_version_ok=extractor_version_ok,
        observation_window_days=observation_window_days,
    )
    rollup_refused = "; ".join(rollup_reasons) if rollup_reasons else None
    score_refused = "; ".join(score_reasons) if score_reasons else None

    if rollup_refused is not None:
        return TopicHazardResult(
            model_id=_MODEL_ID,
            rollup_refused=rollup_refused,
            score_refused=score_refused,
        )

    rollup = build_topic_rollup(
        demand_recent_per_month=float(demand_recent_per_month),
        demand_baseline_per_month=float(demand_baseline_per_month),
        realized_r1_rate_recent=realized_r1_rate_recent,
        realized_r1_rate_baseline=realized_r1_rate_baseline,
        n_resolved_items=int(n_resolved_items or 0),
        inflow_recent_total=inflow_recent_total,
        inflow_baseline_total=inflow_baseline_total,
        r1_sampled_n_recent=r1_sampled_n_recent,
        r1_sampled_n_baseline=r1_sampled_n_baseline,
    )
    flat_refusal = _unknown_quadrant_refusal(rollup)
    if flat_refusal is not None:
        return TopicHazardResult(
            model_id=_MODEL_ID,
            rollup_refused=flat_refusal,
            score_refused=score_refused,
        )

    return TopicHazardResult(
        model_id=_MODEL_ID,
        rollup_refused=None,
        score_refused=score_refused,
        quadrant=rollup.quadrant.value,
        demand=rollup.demand.value,
        supply=rollup.supply.value,
        rollup_note=rollup.note,
        cif_180d=None,
    )


def rollup_refusals(
    *,
    bundle: EvidenceBundle,
    profile: EcosystemProfile | None,
    path: str | None,
    topic_first_seen: datetime | None,
    now: datetime | None,
    assignment_precision_measured: bool,
    n_resolved_items: int | None,
    demand_recent_per_month: float | None,
    demand_baseline_per_month: float | None,
    realized_r1_rate_recent: float | None,
    realized_r1_rate_baseline: float | None,
    assignment_precision_refusal: str | None = None,
) -> list[str]:
    """Shared preconditions plus rollup-only inflow / R1-rate checks."""
    del profile  # retained in signature for call-site stability; hub path covered by S2
    reasons = _shared_refusals(
        bundle=bundle,
        path=path,
        topic_first_seen=topic_first_seen,
        now=now,
        assignment_precision_measured=assignment_precision_measured,
        assignment_precision_refusal=assignment_precision_refusal,
        n_resolved_items=n_resolved_items,
    )
    if demand_recent_per_month is None or demand_baseline_per_month is None:
        reasons.append(
            "inflow rates unknown — cannot place topic on the demand axis"
        )
    if realized_r1_rate_recent is None or realized_r1_rate_baseline is None:
        reasons.append(
            "realized R1 rates unknown — cannot place topic on the supply axis"
        )
    return reasons


def score_refusals(
    *,
    shared_refusals: list[str],
    trained_artifact_available: bool,
    extractor_version_ok: bool,
    observation_window_days: float | None,
) -> list[str]:
    """Inherit every shared refusal, then add score-only gates."""
    reasons = list(shared_refusals)
    if not trained_artifact_available:
        reasons.append(
            "trained topic-hazard artifact not shipped (topic-hazard-v0-refuse)"
        )
    if not extractor_version_ok:
        reasons.append(
            "extractor_version differs from trained artifact — Block 6 disabled"
        )
    if observation_window_days is None:
        reasons.append(
            "observation window unknown — need ≥360d (2× the 180d horizon)"
        )
    elif observation_window_days < _MIN_OBSERVATION_WINDOW_DAYS:
        n = int(observation_window_days) if observation_window_days == int(
            observation_window_days
        ) else observation_window_days
        reasons.append(f"observation window {n}d < 2× the 180d horizon")
    return reasons


def _shared_refusals(
    *,
    bundle: EvidenceBundle,
    path: str | None,
    topic_first_seen: datetime | None,
    now: datetime | None,
    assignment_precision_measured: bool,
    n_resolved_items: int | None,
    assignment_precision_refusal: str | None = None,
) -> list[str]:
    """S1–S8: block both rollup and score."""
    reasons: list[str] = []

    for item in bundle.items:
        if item.kind == EvidenceKind.VITAL_SIGNS and item.metadata.get("fetch_errors"):
            reasons.append(
                "vital-signs retrieval failed for one or more sources; not scoring"
            )
            break

    if path is None:
        reasons.append("no path-scoped topic (--path required for topic rollup)")

    if not assignment_precision_measured:
        reasons.append(
            assignment_precision_refusal
            or "item→topic assignment precision unmeasured for this profile"
        )

    resolved = n_resolved_items
    if resolved is None:
        resolved = _closed_total_from_search(bundle)
    if resolved is None:
        reasons.append(
            "n_resolved_items unknown — need closed_issues_total from issue search API"
        )
    elif resolved < _MIN_RESOLVED:
        reasons.append(
            f"n_resolved_items={resolved} < {_MIN_RESOLVED}; topic rollup refused"
        )

    hub_n = sum(1 for item in bundle.items if item.kind == EvidenceKind.HF_DISCUSSION)
    issue_n = sum(1 for item in bundle.items if item.kind == EvidenceKind.ISSUE)
    if hub_n > 0 and issue_n == 0:
        reasons.append(
            "Hub items are inference-only in v1; model was not trained on them"
        )

    if path is not None:
        if topic_first_seen is None:
            reasons.append("topic_first_seen unknown for path-scoped topic")
        elif now is not None:
            age_days = (ensure_utc(now) - ensure_utc(topic_first_seen)).days
            if age_days < _MIN_TOPIC_AGE_DAYS:
                reasons.append(
                    f"topic first appeared {age_days} days before snapshot "
                    f"(<{_MIN_TOPIC_AGE_DAYS}); path history may be incomplete"
                )

    return reasons


def _unknown_quadrant_refusal(rollup: TopicRollup) -> str | None:
    """Quadrant.UNKNOWN must never render with rollup_refused=None."""
    if rollup.quadrant != Quadrant.UNKNOWN:
        return None
    if (
        rollup.demand == DemandTrend.FLAT
        and rollup.supply == SupplyTrend.FLAT
    ):
        return "demand and supply both flat — no trajectory to report"
    return "demand or supply flat — no trajectory to report"


def _closed_total_from_search(bundle: EvidenceBundle) -> int | None:
    """Use search API total_count — never a relevance-ranked bundle sample."""
    for item in bundle.items:
        if item.kind != EvidenceKind.VITAL_SIGNS:
            continue
        total = item.metadata.get("closed_issues_total")
        if total is not None:
            return int(total)
    return None


def topic_hazard_to_dict(
    result: TopicHazardResult,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_id": result.model_id,
        "rollup_refused": result.rollup_refused,
        "score_refused": result.score_refused,
        "quadrant": result.quadrant,
        "demand": result.demand,
        "supply": result.supply,
        "rollup_note": result.rollup_note,
        "cif_180d": result.cif_180d,
        "disclaimer": result.disclaimer,
    }
    if provenance:
        payload["provenance"] = provenance
    return payload


def build_topic_provenance(
    *,
    demand_recent_per_month: float | None,
    demand_baseline_per_month: float | None,
    realized_r1_rate_recent: float | None,
    realized_r1_rate_baseline: float | None,
    r1_sampled_n_recent: int = 0,
    r1_sampled_n_baseline: int = 0,
    r1_excluded_n: int = 0,
    assignment_precision: float | None = None,
    assignment_precision_n: int | None = None,
    assignment_precision_measured_at: str | None = None,
) -> dict[str, Any]:
    """Numbers printed under the quadrant — citation discipline for the rollup."""
    return {
        "demand_recent_per_month": demand_recent_per_month,
        "demand_baseline_per_month": demand_baseline_per_month,
        "realized_r1_rate_recent": realized_r1_rate_recent,
        "realized_r1_rate_baseline": realized_r1_rate_baseline,
        "r1_sampled_n_recent": r1_sampled_n_recent,
        "r1_sampled_n_baseline": r1_sampled_n_baseline,
        "r1_excluded_n": r1_excluded_n,
        "assignment_precision": assignment_precision,
        "assignment_precision_n": assignment_precision_n,
        "assignment_precision_measured_at": assignment_precision_measured_at,
    }
