"""Vital signs + activity forecast scorer."""

from casefile.predict.scorer import forecast_from_vital_metadata
from casefile.retrievers.vital_signs import VitalSigns, _to_evidence


def test_forecast_active_path():
    meta = {
        "commits_3m": 12,
        "commits_6m": 20,
        "commits_12m": 30,
        "trend": "rising",
        "distinct_committers_12m": 4,
        "top_committer_active_on_path_6m": True,
        "closure_rate": 0.6,
        "has_codeowners": True,
        "prototype_or_stale_label_hits": 0,
        "open_issues_total": 5,
        "issue_search_ok": True,
        "fetch_errors": [],
    }
    result = forecast_from_vital_metadata(meta)
    assert result.refused is None
    assert 0.0 < result.probability <= 0.95
    assert result.probability != 1.0
    assert result.label in {"likely_active", "uncertain", "likely_stale"}
    assert result.top_features
    assert "not backtested" in result.disclaimer


def test_forecast_refuses_fetch_errors():
    result = forecast_from_vital_metadata(
        {
            "commits_12m": 50,
            "open_issues_total": 0,
            "issue_search_ok": True,
            "fetch_errors": ["issue search failed: 403"],
        }
    )
    assert result.refused
    assert result.label == "refused"
    assert result.probability == 0.0


def test_forecast_refuses_missing_issue_search():
    result = forecast_from_vital_metadata(
        {
            "commits_12m": 50,
            "open_issues_total": 0,
            "issue_search_ok": False,
            "fetch_errors": [],
        }
    )
    assert result.refused
    assert "Issue search" in (result.refused or "")


def test_forecast_refuses_one_typo_commit():
    result = forecast_from_vital_metadata(
        {
            "commits_12m": 1,
            "open_issues_total": 0,
            "issue_search_ok": True,
            "fetch_errors": [],
        }
    )
    assert result.refused
    assert result.label == "refused"


def test_forecast_saturates_below_one():
    meta = {
        "commits_3m": 400,
        "commits_6m": 800,
        "commits_12m": 2000,
        "trend": "rising",
        "distinct_committers_12m": 50,
        "top_committer_active_on_path_6m": True,
        "closure_rate": 1.0,
        "has_codeowners": True,
        "prototype_or_stale_label_hits": 0,
        "open_issues_total": 1,
        "issue_search_ok": True,
        "fetch_errors": [],
    }
    result = forecast_from_vital_metadata(meta)
    assert result.refused is None
    assert result.probability <= 0.95


def test_vital_signs_evidence_shape():
    signs = VitalSigns(
        path="torch/masked",
        commits_3m=1,
        commits_6m=2,
        commits_12m=5,
        trend="falling",
        evidence_urls={"commits": "https://github.com/pytorch/pytorch/commits/torch/masked"},
    )
    item = _to_evidence(signs, "pytorch/pytorch")
    assert item.kind.value == "vital_signs"
    assert item.metadata["commits_12m"] == 5
    assert "top_committer_active_on_path_6m" in item.metadata
