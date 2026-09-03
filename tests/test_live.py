"""Optional live API smoke test — set CASEFILE_RUN_LIVE=1 and configure casefile/.env."""

import os
from pathlib import Path

import pytest

from casefile.config import Settings, get_settings
from casefile.engine.orchestrator import AssessmentEngine
from casefile.models.assessment import AssessmentRequest
from casefile.profiles import load_profile
from casefile.render.markdown import render_markdown
from tests.report_contract import (
    assert_golden_issues_present,
    assert_markdown_report_contract,
    assert_open_questions_include,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CASEFILE_RUN_LIVE") != "1",
    reason="Set CASEFILE_RUN_LIVE=1 to run live GitHub/LLM smoke tests",
)


@pytest.fixture
def profiles_dir():
    from casefile.config import get_settings

    return get_settings().resolved_profiles_dir()


@pytest.fixture
def live_settings():
    settings = get_settings()
    if not settings.github_token:
        pytest.skip("CASEFILE_GITHUB_TOKEN not set")
    return settings


@pytest.mark.asyncio
async def test_live_torch_masked_assessment(live_settings, profiles_dir):
    if not live_settings.openai_api_key and not live_settings.anthropic_api_key:
        pytest.skip("No LLM API key for synthesis")

    profile = load_profile(profiles_dir, "pytorch", allow_stale=True)
    engine = AssessmentEngine(live_settings)
    request = AssessmentRequest(
        question="Is reviving torch.masked / MaskedTensor worth an upstream contribution?",
        repo="pytorch/pytorch",
        path="torch/masked",
        ecosystem="pytorch",
        tier=2,
        synthesize=True,
        max_evidence=40,
    )

    report = await engine.run(request, profile)
    md = render_markdown(report)

    ids = {item.id for item in report.evidence.items}
    golden_found = sum(1 for n in (89734, 89320, 124964) if f"issue-{n}" in ids)
    assert golden_found >= 2, (
        f"Expected at least 2/3 golden tracking issues in evidence; found {golden_found}. "
        f"ids sample: {sorted(ids)[:15]}"
    )

    if report.summary is None:
        pytest.fail(
            "Live run produced no summary. "
            f"validation_errors={report.validation_errors!r} "
            f"synthesize={request.synthesize}"
        )
    assert_markdown_report_contract(md, expect_summary=True)
    assert_open_questions_include(report, "adjacent")
    assert len(report.evidence.items) >= 5

    out = Path(__file__).resolve().parents[1] / "report-live-test.md"
    out.write_text(md, encoding="utf-8")
