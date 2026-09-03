"""Ecosystem profile loading and path-specific pins."""

import pytest

from casefile.engine.planner import build_plan
from casefile.models.assessment import AssessmentRequest
from casefile.profiles import list_profiles, load_profile


@pytest.fixture
def profiles_dir():
    from casefile.config import get_settings

    return get_settings().resolved_profiles_dir()


def test_profiles_load_without_stale_override(profiles_dir):
    for name in ("pytorch", "numpy", "sklearn", "apertus"):
        profile = load_profile(profiles_dir, name, allow_stale=False)
        assert profile.id == name


def test_all_profiles_load(profiles_dir):
    names = list_profiles(profiles_dir)
    assert "pytorch" in names
    assert "numpy" in names
    assert "sklearn" in names
    assert "apertus" in names
    for name in names:
        if name.startswith("_"):
            continue
        profile = load_profile(profiles_dir, name, allow_stale=True)
        assert profile.id == name
        assert profile.last_verified


def test_apertus_profile_has_hub_config(profiles_dir):
    profile = load_profile(profiles_dir, "apertus", allow_stale=True)
    assert profile.huggingface is not None
    ids = [r.repo_id for r in profile.huggingface.hub_repos]
    assert "swiss-ai/Apertus-v1.5-8B" in ids
    assert profile.assignment_precision is not None
    assert profile.assignment_precision.n == 81
    assert profile.assignment_precision.precision == 0.88


def test_pytorch_nested_path_pins(profiles_dir):
    profile = load_profile(profiles_dir, "pytorch", allow_stale=True)
    assert profile.pinned_issues_for_path("torch/nested") == [112398, 82534, 118580]
    assert 89734 in profile.pinned_issues_for_path("torch/masked")

    request = AssessmentRequest(
        question="torch.nested jagged variable length",
        repo="pytorch/pytorch",
        path="torch/nested",
        ecosystem="pytorch",
    )
    plan = build_plan(request, profile)
    issue_spec = next(s for s in plan.specs if s.retriever == "github_issues")
    joined = " ".join(issue_spec.queries)
    assert "is:issue 112398" in joined
    assert "is:issue 89734" not in joined


def test_pytorch_assignment_recall_is_unmeasured(profiles_dir):
    profile = load_profile(profiles_dir, "pytorch", allow_stale=True)
    assert profile.assignment_precision is not None
    assert profile.assignment_precision.precision == 0.91
    assert profile.assignment_precision.recall is None


def test_numpy_profile_has_ma_pins(profiles_dir):
    profile = load_profile(profiles_dir, "numpy", allow_stale=True)
    assert 22338 in profile.pinned_issue_numbers
    assert profile.pinned_issues_for_path(None) == profile.pinned_issue_numbers
