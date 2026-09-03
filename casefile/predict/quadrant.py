"""Demand × supply topic rollup — the product cell Casefile exists to find.

Demand rising + supply falling = gap (contribute here). That is torch/masked.
A single "dying" scalar would say avoid it. See PREDICT.md §2.3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from casefile.predict.assignment_score import wilson_interval


class DemandTrend(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    FLAT = "flat"
    UNKNOWN = "unknown"


class SupplyTrend(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    FLAT = "flat"
    UNKNOWN = "unknown"


class Quadrant(str, Enum):
    THRIVING = "thriving"
    GAP = "gap"
    MATURING = "maturing"
    DYING = "dying"
    UNKNOWN = "unknown"


# Non-overlapping: recent 6 months vs the preceding 18 (24-month lookback).
RECENT_MONTHS = 6.0
BASELINE_MONTHS = 18.0
_MIN_INFLOW_RECENT = 5
_MIN_INFLOW_BASELINE = 8
_MIN_R1_N = 8


@dataclass(frozen=True)
class TopicRollup:
    demand: DemandTrend
    supply: SupplyTrend
    quadrant: Quadrant
    demand_recent_per_month: float
    demand_baseline_per_month: float
    realized_r1_rate_recent: float | None
    realized_r1_rate_baseline: float | None
    n_resolved_items: int
    note: str


def classify_quadrant(demand: DemandTrend, supply: SupplyTrend) -> Quadrant:
    if demand == DemandTrend.UNKNOWN or supply == SupplyTrend.UNKNOWN:
        return Quadrant.UNKNOWN
    if demand == DemandTrend.RISING and supply == SupplyTrend.RISING:
        return Quadrant.THRIVING
    if demand == DemandTrend.RISING and supply == SupplyTrend.FALLING:
        return Quadrant.GAP
    if demand == DemandTrend.FALLING and supply == SupplyTrend.RISING:
        return Quadrant.MATURING
    if demand == DemandTrend.FALLING and supply == SupplyTrend.FALLING:
        return Quadrant.DYING
    return Quadrant.UNKNOWN


def poisson_rate_ci(count: int, months: float, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% interval for a Poisson rate (events per month). Display only."""
    if months <= 0:
        return (0.0, 0.0)
    if count <= 0:
        return (0.0, 3.689 / months)
    root = math.sqrt(count)
    return (max(0.0, count - z * root) / months, (count + z * root) / months)


def _binom_cdf_le(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). n is small (inflow windows)."""
    if n <= 0:
        return 1.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    q = 1.0 - p
    if q <= 0.0:
        return 1.0 if k >= n else 0.0
    if p <= 0.0:
        return 1.0
    term = q**n
    total = term
    for i in range(1, k + 1):
        term *= (n - i + 1) / i * p / q
        total += term
    return min(1.0, total)


def demand_from_inflow(
    *,
    recent_per_month: float,
    baseline_per_month: float,
    recent_count: int | None = None,
    baseline_count: int | None = None,
) -> DemandTrend:
    """Compare recent 6mo rate to baseline 18mo rate (non-overlapping).

    When integer window counts are supplied, classify with a conditional
    binomial test (equal-rate null: K ~ Bin(n_r+n_b, t_r/(t_r+t_b))), not
    by whether two marginal Poisson intervals miss each other. Rate-only
    callers keep the 1.25 / 0.8 ratio.
    """
    if recent_count is not None and baseline_count is not None:
        if recent_count < _MIN_INFLOW_RECENT or baseline_count < _MIN_INFLOW_BASELINE:
            return DemandTrend.UNKNOWN
        n = recent_count + baseline_count
        p0 = RECENT_MONTHS / (RECENT_MONTHS + BASELINE_MONTHS)
        upper = 1.0 - _binom_cdf_le(recent_count - 1, n, p0)
        lower = _binom_cdf_le(recent_count, n, p0)
        if upper < 0.05:
            return DemandTrend.RISING
        if lower < 0.05:
            return DemandTrend.FALLING
        return DemandTrend.FLAT

    if baseline_per_month <= 0 and recent_per_month <= 0:
        return DemandTrend.UNKNOWN
    if baseline_per_month <= 0:
        return DemandTrend.RISING if recent_per_month > 0 else DemandTrend.UNKNOWN
    ratio = recent_per_month / baseline_per_month
    if ratio >= 1.25:
        return DemandTrend.RISING
    if ratio <= 0.8:
        return DemandTrend.FALLING
    return DemandTrend.FLAT


def supply_from_r1_rates(
    *,
    recent_r1_rate: float | None,
    prior_r1_rate: float | None,
    recent_n: int | None = None,
    baseline_n: int | None = None,
) -> SupplyTrend:
    if recent_r1_rate is None or prior_r1_rate is None:
        return SupplyTrend.UNKNOWN
    if (
        recent_n is not None
        and baseline_n is not None
        and recent_n >= _MIN_R1_N
        and baseline_n >= _MIN_R1_N
    ):
        r_s = int(round(recent_r1_rate * recent_n))
        b_s = int(round(prior_r1_rate * baseline_n))
        r_lo, r_hi = wilson_interval(r_s, recent_n)
        b_lo, b_hi = wilson_interval(b_s, baseline_n)
        if r_lo > b_hi:
            return SupplyTrend.RISING
        if r_hi < b_lo:
            return SupplyTrend.FALLING
        return SupplyTrend.FLAT
    if recent_n is not None and recent_n < _MIN_R1_N:
        return SupplyTrend.UNKNOWN
    if baseline_n is not None and baseline_n < _MIN_R1_N:
        return SupplyTrend.UNKNOWN
    if prior_r1_rate <= 0 and recent_r1_rate <= 0:
        return SupplyTrend.UNKNOWN
    if prior_r1_rate <= 0:
        return SupplyTrend.RISING if recent_r1_rate > 0 else SupplyTrend.UNKNOWN
    ratio = recent_r1_rate / prior_r1_rate
    if ratio >= 1.25:
        return SupplyTrend.RISING
    if ratio <= 0.8:
        return SupplyTrend.FALLING
    return SupplyTrend.FLAT


def build_topic_rollup(
    *,
    demand_recent_per_month: float,
    demand_baseline_per_month: float,
    realized_r1_rate_recent: float | None,
    realized_r1_rate_baseline: float | None,
    n_resolved_items: int,
    inflow_recent_total: int | None = None,
    inflow_baseline_total: int | None = None,
    r1_sampled_n_recent: int | None = None,
    r1_sampled_n_baseline: int | None = None,
) -> TopicRollup:
    demand = demand_from_inflow(
        recent_per_month=demand_recent_per_month,
        baseline_per_month=demand_baseline_per_month,
        recent_count=inflow_recent_total,
        baseline_count=inflow_baseline_total,
    )
    supply = supply_from_r1_rates(
        recent_r1_rate=realized_r1_rate_recent,
        prior_r1_rate=realized_r1_rate_baseline,
        recent_n=r1_sampled_n_recent,
        baseline_n=r1_sampled_n_baseline,
    )
    quadrant = classify_quadrant(demand, supply)
    notes = {
        Quadrant.GAP: "Demand rising, resolution capacity falling — contribution target.",
        Quadrant.THRIVING: "Upstream is absorbing demand.",
        Quadrant.MATURING: "Demand falling while resolution capacity holds.",
        Quadrant.DYING: "Both demand and resolution capacity falling.",
        Quadrant.UNKNOWN: "Insufficient history to place the topic on the demand×supply grid.",
    }
    return TopicRollup(
        demand=demand,
        supply=supply,
        quadrant=quadrant,
        demand_recent_per_month=demand_recent_per_month,
        demand_baseline_per_month=demand_baseline_per_month,
        realized_r1_rate_recent=realized_r1_rate_recent,
        realized_r1_rate_baseline=realized_r1_rate_baseline,
        n_resolved_items=n_resolved_items,
        note=notes[quadrant],
    )
