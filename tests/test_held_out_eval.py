"""Held-out eval cases from eval/sample_cases.yaml (offline plan checks)."""

import pytest

from casefile.engine.planner import build_plan
from casefile.eval.sample_cases import load_held_out_cases
from casefile.models.assessment import AssessmentRequest

_RETRIEVER_FOR_KIND = {
    "issue": "github_issues",
    "repo_file": "repo_files",
    "discourse_thread": "discourse",
    "adjacent_project": "adjacent_projects",
    "merged_pr": "github_prs",
    "git_activity": "git_activity",
}


@pytest.mark.parametrize("case", load_held_out_cases(), ids=lambda c: c.id)
def test_held_out_plan_includes_expected_retrievers(case, profiles_dir):
    from casefile.profiles import load_profile

    profile = load_profile(profiles_dir, case.ecosystem, allow_stale=True)
    request = AssessmentRequest(
        question=case.question,
        repo=case.repo,
        path=case.path or None,
        ecosystem=case.ecosystem,
        tier=case.tier,
    )
    plan = build_plan(request, profile)
    retrievers = {s.retriever for s in plan.specs}
    for kind in case.expect_evidence_kinds:
        name = _RETRIEVER_FOR_KIND.get(kind)
        assert name is not None, f"unknown kind {kind}"
        assert name in retrievers, f"{case.id}: missing {name} for {kind}"


@pytest.mark.parametrize("case", load_held_out_cases(), ids=lambda c: c.id)
def test_held_out_pinned_issues_in_plan(case, profiles_dir):
    if not case.expect_issue_numbers:
        return
    from casefile.profiles import load_profile

    profile = load_profile(profiles_dir, case.ecosystem, allow_stale=True)
    request = AssessmentRequest(
        question=case.question,
        repo=case.repo,
        path=case.path or None,
        ecosystem=case.ecosystem,
        tier=case.tier,
    )
    plan = build_plan(request, profile)
    issue_spec = next(s for s in plan.specs if s.retriever == "github_issues")
    for n in case.expect_issue_numbers:
        assert any(
            str(n) in q for q in issue_spec.queries
        ), f"{case.id}: issue #{n} not in issue search plan"
