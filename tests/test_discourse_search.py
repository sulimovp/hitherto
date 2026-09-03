"""Discourse search URL planning and retrieval."""

import httpx
import pytest

from casefile.clients import build_clients
from casefile.config import Settings
from casefile.engine.planner import build_plan
from casefile.models.assessment import AssessmentRequest
from casefile.models.profile import DiscourseConfig, EcosystemProfile
from casefile.retrievers.discourse import DiscourseRetriever
from casefile.retrievers.base import RetrievalSpec
from datetime import date


@pytest.fixture
def pytorch_like_profile():
    return EcosystemProfile(
        id="test",
        display_name="Test",
        last_verified=date(2026, 5, 30),
        discourse=DiscourseConfig(
            base_url="https://dev-discuss.pytorch.org",
            search_path="/search?q=",
            pinned_threads=["https://dev-discuss.pytorch.org/t/pinned/1"],
        ),
        synonyms={"masked tensor": ["MaskedTensor", "torch.masked"]},
    )


def test_planner_adds_discourse_search_urls(pytorch_like_profile):
    request = AssessmentRequest(
        question="Is reviving torch.masked worth it?",
        repo="pytorch/pytorch",
        path="torch/masked",
        tier=2,
    )
    plan = build_plan(request, pytorch_like_profile)
    spec = next(s for s in plan.specs if s.retriever == "discourse")
    assert spec.metadata.get("search_urls")
    assert any("MaskedTensor" in u or "masked" in u for u in spec.metadata["search_urls"])


@pytest.mark.asyncio
async def test_discourse_search_fetches_topics_from_search_page(pytorch_like_profile, httpx_mock):
    search_url = "https://dev-discuss.pytorch.org/search?q=MaskedTensor"
    topic_url = "https://dev-discuss.pytorch.org/t/found-via-search/99"
    httpx_mock.add_response(url=search_url, method="GET", text=f'<a href="{topic_url}">Hit</a>')
    httpx_mock.add_response(url=topic_url, method="HEAD", status_code=200)
    httpx_mock.add_response(
        url=topic_url,
        method="GET",
        text="<html><head><title>Found via search</title></head><body>content</body></html>",
    )
    httpx_mock.add_response(
        url="https://dev-discuss.pytorch.org/t/pinned/1",
        method="HEAD",
        status_code=200,
    )
    httpx_mock.add_response(
        url="https://dev-discuss.pytorch.org/t/pinned/1",
        method="GET",
        text="<html><head><title>Pinned</title></head><body>pinned</body></html>",
    )

    async with httpx.AsyncClient() as client:
        clients = build_clients(Settings(), client)
        spec = RetrievalSpec(
            retriever="discourse",
            urls=["https://dev-discuss.pytorch.org/t/pinned/1"],
            metadata={
                "search_urls": [search_url],
                "discourse_base": "https://dev-discuss.pytorch.org",
            },
        )
        request = AssessmentRequest(question="test", repo="o/r")
        items = await DiscourseRetriever().fetch(spec, request, pytorch_like_profile, clients)

    ids = {i.id for i in items}
    assert "discourse-1" in ids
    assert "discourse-99" in ids
    search_items = [i for i in items if i.metadata.get("discourse_search")]
    assert len(search_items) >= 1
