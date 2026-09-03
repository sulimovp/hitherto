"""Mocked end-to-end assess for numpy and sklearn profiles."""

import httpx
import pytest

from casefile.config import Settings
from casefile.engine.orchestrator import AssessmentEngine
from casefile.models.assessment import AssessmentRequest
from casefile.profiles import load_profile
from casefile.render.markdown import render_markdown
from tests.report_contract import assert_markdown_report_contract


@pytest.fixture
def profiles_dir():
    from casefile.config import get_settings

    return get_settings().resolved_profiles_dir()


def _issues_response(items: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"total_count": len(items), "items": items})


def _github_route(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/rate_limit":
        return httpx.Response(200, json={"resources": {"core": {"remaining": 100}}})
    if path == "/search/issues":
        q = request.url.params.get("q", "")
        if "numpy/numpy" in q:
            items = []
            for num, title in (
                (22338, "numpy.ma __array_function__"),
                (18675, "mask dropped by np.asarray"),
            ):
                if str(num) in q or "numpy.ma" in q:
                    items.append(
                        {
                            "number": num,
                            "title": title,
                            "html_url": f"https://github.com/numpy/numpy/issues/{num}",
                            "state": "open",
                            "body": "numpy.ma masked array",
                            "labels": [{"name": "component: numpy.ma"}],
                            "updated_at": "2026-01-01T00:00:00Z",
                        }
                    )
            return _issues_response(items or [_numpy_fallback_issue()])
        if "scikit-learn" in q:
            items = []
            for num, title in (
                (22893, "SLEP006 Metadata Routing task list"),
                (18936, "SLEP006 debugging tool sample props"),
            ):
                if str(num) in q or "SLEP" in q or "metadata" in q.lower():
                    items.append(
                        {
                            "number": num,
                            "title": title,
                            "html_url": f"https://github.com/scikit-learn/scikit-learn/issues/{num}",
                            "state": "open",
                            "body": "metadata routing SLEP006",
                            "labels": [{"name": "Enhancement"}],
                            "updated_at": "2026-01-01T00:00:00Z",
                        }
                    )
            return _issues_response(items or [_sklearn_fallback_issue()])
        return _issues_response([])
    if "/contents/" in path:
        return httpx.Response(
            200,
            json={"encoding": "base64", "content": "IyBNb2Nr\n"},
        )
    return httpx.Response(404, json={"message": "not found"})


def _numpy_fallback_issue() -> dict:
    return {
        "number": 22338,
        "title": "numpy.ma enhancement",
        "html_url": "https://github.com/numpy/numpy/issues/22338",
        "state": "open",
        "body": "MaskedArray",
        "labels": [],
        "updated_at": "2026-01-01T00:00:00Z",
    }


def _sklearn_fallback_issue() -> dict:
    return {
        "number": 22893,
        "title": "SLEP006 task list",
        "html_url": "https://github.com/scikit-learn/scikit-learn/issues/22893",
        "state": "open",
        "body": "metadata routing",
        "labels": [],
        "updated_at": "2026-01-01T00:00:00Z",
    }


def _external_route(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if any(
        host in url
        for host in (
            "pandas.pydata.org",
            "docs.xarray.dev",
            "jax.readthedocs.io",
            "imbalanced-learn.org",
            "skorch.readthedocs.io",
            "xgboost.readthedocs.io",
            "numpy.org",
            "scikit-learn.org",
            "readthedocs.io",
        )
    ):
        return httpx.Response(200, text="<html><head><title>doc</title></head><body>ok</body></html>")
    return httpx.Response(404)


async def _route(request: httpx.Request) -> httpx.Response:
    if "api.github.com" in str(request.url):
        return _github_route(request)
    return _external_route(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ecosystem,repo,question,expected_issue",
    [
        (
            "numpy",
            "numpy/numpy",
            "Is improving numpy.ma worth contributing?",
            "issue-22338",
        ),
        (
            "sklearn",
            "scikit-learn/scikit-learn",
            "Is sklearn metadata routing SLEP006 worth contributing?",
            "issue-22893",
        ),
    ],
)
async def test_ecosystem_assess_mocked(
    profiles_dir, httpx_mock, ecosystem, repo, question, expected_issue
):
    httpx_mock.add_callback(_route, is_reusable=True)
    profile = load_profile(profiles_dir, ecosystem, allow_stale=True)
    engine = AssessmentEngine(Settings(github_token="test"))
    request = AssessmentRequest(
        question=question,
        repo=repo,
        ecosystem=ecosystem,
        tier=2,
        synthesize=False,
    )
    report = await engine.run(request, profile)
    assert expected_issue in {i.id for i in report.evidence.items}
    joined_oq = " ".join(report.evidence.open_questions)
    assert "MaskedTensor" not in joined_oq
    md = render_markdown(report)
    assert_markdown_report_contract(md, expect_summary=False)
    assert ecosystem in md
