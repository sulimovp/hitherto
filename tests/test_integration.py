"""End-to-end assessment with mocked HTTP (no API keys required)."""

import json

import httpx
import pytest

from casefile.config import Settings
from casefile.engine.orchestrator import AssessmentEngine
from casefile.engine.planner import build_plan
from casefile.models.assessment import AssessmentRequest
from casefile.models.profile import DiscourseConfig, EcosystemProfile
from casefile.profiles import load_profile
from casefile.render.markdown import render_markdown
from tests.report_contract import (
    assert_evidence_kinds_present,
    assert_golden_issues_present,
    assert_markdown_report_contract,
    assert_open_questions_include,
)

_GOLDEN_ISSUES = {
    89734: {
        "number": 89734,
        "title": "Masked Tensor documentation is missing",
        "html_url": "https://github.com/pytorch/pytorch/issues/89734",
        "state": "open",
        "body": "Prototype module torch.masked documentation tracking",
        "labels": [{"name": "module: masked operators"}],
        "updated_at": "2026-01-01T00:00:00Z",
    },
    89320: {
        "number": 89320,
        "title": "MaskedTensor ops gap",
        "html_url": "https://github.com/pytorch/pytorch/issues/89320",
        "state": "open",
        "body": "Missing operators for MaskedTensor",
        "labels": [],
        "updated_at": "2026-01-01T00:00:00Z",
    },
    124964: {
        "number": 124964,
        "title": "MaskedTensor gradients feature",
        "html_url": "https://github.com/pytorch/pytorch/issues/124964",
        "state": "open",
        "body": "Gradients back to Tensor from MaskedTensor",
        "labels": [],
        "updated_at": "2026-01-01T00:00:00Z",
    },
}


@pytest.fixture
def pytorch_profile(profiles_dir):
    return load_profile(profiles_dir, "pytorch", allow_stale=True)


@pytest.fixture
def profiles_dir():
    from casefile.config import get_settings

    return get_settings().resolved_profiles_dir()


def _issues_for_query(q: str) -> list[dict]:
    items: list[dict] = []
    for number, issue in _GOLDEN_ISSUES.items():
        if str(number) in q:
            items.append(issue)
    if not items and ("masked" in q.lower() or "MaskedTensor" in q):
        items.append(
            {
                "number": 150186,
                "title": "MaskedTensor 10x slower softmax",
                "html_url": "https://github.com/pytorch/pytorch/issues/150186",
                "state": "open",
                "body": "performance MaskedTensor",
                "labels": [],
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
    return items


def _github_route(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/rate_limit":
        return httpx.Response(200, json={"resources": {"core": {"remaining": 100}}})
    if path == "/search/issues":
        q = request.url.params.get("q", "")
        if "is:pr" in q:
            return httpx.Response(
                200,
                json={
                    "total_count": 1,
                    "items": [
                        {
                            "number": 99001,
                            "title": "Improve torch.masked docs",
                            "html_url": "https://github.com/pytorch/pytorch/pull/99001",
                            "state": "closed",
                            "closed_at": "2026-01-01T00:00:00Z",
                        }
                    ],
                },
            )
        items = _issues_for_query(q)
        return httpx.Response(200, json={"total_count": len(items), "items": items})
    if "/contents/" in path:
        return httpx.Response(
            200,
            json={"encoding": "base64", "content": "IyBNb2NrIGRvY3M=\n"},
        )
    if path.endswith("/commits"):
        return httpx.Response(
            200,
            json=[
                {
                    "sha": "abc1234",
                    "html_url": "https://github.com/pytorch/pytorch/commit/abc1234",
                    "commit": {"author": {"date": "2026-03-01T00:00:00Z"}},
                    "author": {"login": "dev1"},
                }
            ],
        )
    if "/timeline" in path:
        return httpx.Response(200, json=[])
    if "/pulls/" in path and path.endswith("/files"):
        return httpx.Response(200, json=[{"filename": "torch/masked/__init__.py"}])
    if "/reactions" in path:
        return httpx.Response(200, json=[])
    return httpx.Response(404, json={"message": "not found"})


def _external_route(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "dev-discuss.pytorch.org/search" in url:
        return httpx.Response(
            200,
            text=(
                '<a href="https://dev-discuss.pytorch.org/t/state-of-pytorch-core-september-2021-edition/332">'
                "State of PyTorch core</a>"
            ),
        )
    if "dev-discuss" in url:
        title = "PyTorch dev-discuss thread"
        if "/332" in url:
            title = "State of PyTorch core: September 2021 edition"
        elif "/557" in url:
            title = "What (and Why) is __torch_dispatch__?"
        elif "/477" in url:
            title = "Torch.nn H2 2021 Lookback and H1 2022 Lookahead"
        return httpx.Response(
            200,
            text=(
                f"<html><head><title>{title}</title></head>"
                "<body>MaskedTensor and tensor subclass maintainer context.</body></html>"
            ),
        )
    if "pytorch.org" in url or "github.com/pytorch" in url:
        return httpx.Response(200)
    return httpx.Response(404)


async def _route_request(request: httpx.Request) -> httpx.Response:
    if "api.github.com" in str(request.url):
        return _github_route(request)
    return _external_route(request)


@pytest.fixture
def masked_request():
    return AssessmentRequest(
        question="Is reviving torch.masked / MaskedTensor worth an upstream contribution?",
        repo="pytorch/pytorch",
        path="torch/masked",
        ecosystem="pytorch",
        tier=2,
        synthesize=False,
    )


@pytest.fixture
def engine(tmp_path):
    return AssessmentEngine(
        Settings(
            github_token="test-token",
            openai_api_key="test",
            llm_provider="openai",
            cache_dir=tmp_path,
        )
    )


@pytest.mark.asyncio
async def test_assess_end_to_end_mocked(pytorch_profile, httpx_mock, masked_request, engine):
    httpx_mock.add_callback(_route_request, is_reusable=True)

    report = await engine.run(masked_request, pytorch_profile)

    assert_golden_issues_present(report)
    assert_evidence_kinds_present(report, "issue", "adjacent_project", "commit", "pull_request")
    assert_open_questions_include(report, "adjacent")
    assert "MaskedTensor" not in " ".join(report.evidence.open_questions)
    assert_evidence_kinds_present(report, "discourse_thread")

    md = render_markdown(report)
    assert_markdown_report_contract(md, expect_summary=False)
    for number in (89734, 89320, 124964):
        assert f"/issues/{number}" in md


@pytest.mark.asyncio
async def test_assess_with_synthesis_mocked(pytorch_profile, httpx_mock, masked_request, engine, monkeypatch):
    httpx_mock.add_callback(_route_request, is_reusable=True)
    masked_request.synthesize = True

    async def _fake_complete(self, system, user, *, max_tokens=1024):
        return json.dumps(
            {
                "paragraphs": (
                    'Documentation gaps for MaskedTensor remain open '
                    '"Masked Tensor documentation is missing" [1]. '
                    "Adjacent NestedTensor may cover batch masking."
                ),
                "citations": {"1": "issue-89734"},
                "open_questions": ["What is the maintainer roadmap for torch.masked?"],
            }
        )

    monkeypatch.setattr(
        "casefile.clients.llm.LlmClient.complete",
        _fake_complete,
    )

    report = await engine.run(masked_request, pytorch_profile)

    assert report.summary is not None
    assert "[1]" in report.summary
    assert report.validation_errors == []
    assert_open_questions_include(report, "roadmap", "adjacent")

    md = render_markdown(report)
    assert_markdown_report_contract(md, expect_summary=True)
    assert "Documentation gaps" in md
    assert "What is the maintainer roadmap" in md


@pytest.mark.asyncio
async def test_discourse_in_plan_when_pinned_threads(profiles_dir, httpx_mock, engine):
    from datetime import date

    profile = EcosystemProfile(
        id="disc-test",
        display_name="Disc Test",
        last_verified=date(2026, 5, 30),
        discourse=DiscourseConfig(
            base_url="https://dev-discuss.example.com",
            pinned_threads=["https://dev-discuss.example.com/t/masked-tensor-rfc/42"],
        ),
        pinned_issue_numbers=[89734],
    )
    httpx_mock.add_callback(_route_request, is_reusable=True)

    request = AssessmentRequest(
        question="torch.masked MaskedTensor contribution",
        repo="pytorch/pytorch",
        path="torch/masked",
        tier=2,
        synthesize=False,
    )
    plan = build_plan(request, profile)
    assert any(s.retriever == "discourse" for s in plan.specs)

    report = await engine.run(request, profile)
    assert any(i.kind.value == "discourse_thread" for i in report.evidence.items)

    md = render_markdown(report)
    assert "### Dev-discuss and forums" in md
    assert "dev-discuss.example.com" in md
