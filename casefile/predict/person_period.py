"""Discrete-time person-period expansion for weekly competing-risks hazards.

Each item becomes one row per week alive until R1, R2, or censoring (cap 26 weeks).
Train separate h1 (R1) and h2 (R2) models; compose with cumulative_incidence_r1.
See PREDICT.md §4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from casefile.predict.labels import ItemOutcome, Outcome
from casefile.predict.time import ensure_utc

_MAX_WEEKS = 26
_WEEK = timedelta(days=7)


@dataclass(frozen=True)
class PersonPeriodRow:
    item_id: str
    week_index: int  # 0-based week since creation
    week_start: datetime
    exposure_days: float  # ≤7; partial final weeks use clipped exposure
    y_r1: int  # 1 if R1 in this week (training target for h1)
    y_r2: int  # 1 if R2 in this week (training target for h2)
    censored_after: int  # 1 if last row and outcome is R3


def expand_person_period(
    *,
    item_id: str,
    created_at: datetime,
    outcome: ItemOutcome,
    observation_end: datetime,
    max_weeks: int = _MAX_WEEKS,
) -> list[PersonPeriodRow]:
    """Expand one item into weekly rows. Empty if created_at is after observation_end."""
    created_at = ensure_utc(created_at)
    observation_end = ensure_utc(observation_end)

    if created_at > observation_end:
        return []

    event_end = observation_end
    if outcome.outcome in {Outcome.R1, Outcome.R2} and outcome.closed_at is not None:
        event_end = min(event_end, ensure_utc(outcome.closed_at))

    rows: list[PersonPeriodRow] = []
    for week in range(max_weeks):
        week_start = created_at + week * _WEEK
        if week_start >= observation_end:
            break

        natural_week_end = week_start + _WEEK
        week_end = min(natural_week_end, observation_end, event_end)
        exposure_days = (week_end - week_start).total_seconds() / 86400.0
        if exposure_days <= 0:
            break

        y_r1 = 0
        y_r2 = 0
        if outcome.closed_at is not None:
            closed_at = ensure_utc(outcome.closed_at)
            if week_start <= closed_at < natural_week_end:
                if outcome.outcome == Outcome.R1:
                    y_r1 = 1
                elif outcome.outcome == Outcome.R2:
                    y_r2 = 1

        censored_after = 0
        if y_r1 == 0 and y_r2 == 0 and outcome.outcome == Outcome.R3:
            if week_end >= observation_end or week == max_weeks - 1:
                censored_after = 1

        rows.append(
            PersonPeriodRow(
                item_id=item_id,
                week_index=week,
                week_start=week_start,
                exposure_days=round(exposure_days, 4),
                y_r1=y_r1,
                y_r2=y_r2,
                censored_after=censored_after,
            )
        )

        if y_r1 or y_r2:
            break
        if week_end >= observation_end:
            break

    return rows


def to_training_row(row: PersonPeriodRow) -> dict[str, float | int | str]:
    """Canonical training row. `log_exposure_offset` MUST be passed to the
    model as an offset (LightGBM `init_score`) or the row weighted by
    exposure_days / 7. Training a clipped week as a full week reintroduces
    the downward hazard bias that clipping exists to remove.
    """
    return {
        "item_id": row.item_id,
        "week_index": row.week_index,
        "exposure_days": row.exposure_days,
        "log_exposure_offset": math.log(row.exposure_days / 7.0),
        "y_r1": row.y_r1,
        "y_r2": row.y_r2,
        "censored_after": row.censored_after,
    }


def cumulative_incidence_r1(h1: list[float], h2: list[float]) -> list[float]:
    """CIF_1(k) = Σ_{j≤k} h1_j · Π_{i<j}(1 − h1_i − h2_i).

    R2 is a competing event, not censoring — do not use 1 − Π(1 − h1) alone.
    """
    if len(h1) != len(h2):
        raise ValueError("h1 and h2 must have the same length")
    cif: list[float] = []
    free_of_either = 1.0
    running = 0.0
    for h1_j, h2_j in zip(h1, h2, strict=True):
        h1_c = _clamp_prob(h1_j)
        h2_c = _clamp_prob(h2_j)
        if h1_c + h2_c > 1.0:
            total = h1_c + h2_c
            h1_c /= total
            h2_c /= total
        running += h1_c * free_of_either
        cif.append(running)
        free_of_either *= 1.0 - h1_c - h2_c
    return cif


def kaplan_meier_complement(h1: list[float]) -> list[float]:
    """1 − Π(1 − h1_j). WRONG for R1 under competing risks — it treats
    administrative closure as censoring and overestimates resolution,
    most on dying topics. Exists only as a baseline for the CIF-vs-KM
    divergence diagnostic. Never render this in a report.
    """
    free = 1.0
    out: list[float] = []
    for h1_j in h1:
        free *= 1.0 - _clamp_prob(h1_j)
        out.append(1.0 - free)
    return out


def _clamp_prob(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
