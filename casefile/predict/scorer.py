"""Activity forecast — lightweight scorer on vital-signs features.

v0 is a transparent logistic over hand-set weights (not backtested). Export real
tree coefficients here later; keep the package free of lightgbm/onnxruntime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Soft ceiling so hand weights cannot print a fake certainty of 1.0.
_PROB_CAP = 0.95
_MIN_COMMITS_TO_SCORE = 3


@dataclass(frozen=True)
class ForecastResult:
    probability: float
    label: str
    top_features: list[tuple[str, float]]
    refused: str | None = None
    model_id: str = "vitals-logistic-v0"
    disclaimer: str = (
        "Hand-set weights, not backtested — activity forecast, not a recommendation."
    )


# Weights assume *scaled* features in roughly [0, 1] (see _featurize).
_WEIGHTS: dict[str, float] = {
    "bias": -0.4,
    "commits_3m_scaled": 0.9,
    "commits_12m_scaled": 0.5,
    "trend_rising": 0.5,
    "trend_falling": -0.6,
    "distinct_committers_scaled": 0.4,
    "top_active_on_path": 0.35,
    "closure_rate": 0.3,
    "has_codeowners": 0.2,
    "prototype_stale_scaled": -0.4,
}


def forecast_from_vital_metadata(meta: dict[str, Any]) -> ForecastResult:
    if meta.get("refused"):
        return ForecastResult(0.0, "refused", [], refused=str(meta["refused"]))

    errors = meta.get("fetch_errors") or []
    if errors:
        return ForecastResult(
            0.0,
            "refused",
            [],
            refused="Retrieval failed for one or more vital-signs sources; not scoring. "
            + "; ".join(str(e) for e in errors[:3]),
        )

    if not meta.get("issue_search_ok", False):
        return ForecastResult(
            0.0,
            "refused",
            [],
            refused="Issue search did not succeed; refusing to treat missing totals as zero.",
        )

    commits_12 = int(meta.get("commits_12m") or 0)
    open_total = meta.get("open_issues_total")
    open_n = int(open_total if open_total is not None else 0)

    if commits_12 < _MIN_COMMITS_TO_SCORE and open_n == 0:
        return ForecastResult(
            0.0,
            "refused",
            [],
            refused=(
                f"Too little path activity to score "
                f"(commits_12m={commits_12} < {_MIN_COMMITS_TO_SCORE} and no open issues)."
            ),
        )

    features = _featurize(meta)
    logit = _WEIGHTS["bias"]
    contribs: list[tuple[str, float]] = []
    for name, value in features.items():
        w = _WEIGHTS.get(name)
        if w is None:
            continue
        part = w * value
        logit += part
        contribs.append((name, part))

    raw = 1.0 / (1.0 + math.exp(-logit))
    prob = min(raw, _PROB_CAP)
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    if prob >= 0.6:
        label = "likely_active"
    elif prob >= 0.35:
        label = "uncertain"
    else:
        label = "likely_stale"
    return ForecastResult(round(prob, 3), label, contribs[:5])


def _featurize(meta: dict[str, Any]) -> dict[str, float]:
    trend = str(meta.get("trend") or "unknown")
    commits_3 = float(meta.get("commits_3m") or 0)
    commits_12 = float(meta.get("commits_12m") or 0)
    committers = float(meta.get("distinct_committers_12m") or 0)
    stale_hits = float(meta.get("prototype_or_stale_label_hits") or 0)
    closure = meta.get("closure_rate")
    return {
        "commits_3m_scaled": min(commits_3, 15.0) / 15.0,
        "commits_12m_scaled": min(commits_12, 40.0) / 40.0,
        "trend_rising": 1.0 if trend == "rising" else 0.0,
        "trend_falling": 1.0 if trend == "falling" else 0.0,
        "distinct_committers_scaled": min(committers, 10.0) / 10.0,
        "top_active_on_path": 1.0 if meta.get("top_committer_active_on_path_6m") else 0.0,
        "closure_rate": float(closure) if closure is not None else 0.0,
        "has_codeowners": 1.0 if meta.get("has_codeowners") else 0.0,
        "prototype_stale_scaled": min(stale_hits, 5.0) / 5.0,
    }
