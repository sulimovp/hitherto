"""Discourse retriever and HTTP page summary."""

from datetime import date

import httpx
import pytest

from casefile.clients import build_clients
from casefile.clients.http import HttpClient
from casefile.config import Settings
from casefile.models.assessment import AssessmentRequest
from casefile.models.profile import DiscourseConfig, EcosystemProfile
from casefile.retrievers.base import RetrievalSpec
from casefile.retrievers.discourse import DiscourseRetriever

_THREAD_URL = "https://dev-discuss.example.com/t/masked-tensor-rfc/42"
_THREAD_HTML = (
    "<html><head><title>MaskedTensor RFC</title></head>"
    "<body><p>Prototype status discussed here.</p></body></html>"
)


@pytest.fixture
def discourse_profile():
    return EcosystemProfile(
        id="test",
        display_name="Test",
        last_verified=date(2026, 5, 30),
        discourse=DiscourseConfig(
            base_url="https://dev-discuss.example.com",
            pinned_threads=[_THREAD_URL],
        ),
    )


@pytest.mark.asyncio
async def test_fetch_page_summary_extracts_title(httpx_mock):
    httpx_mock.add_response(url=_THREAD_URL, text=_THREAD_HTML)
    async with httpx.AsyncClient() as client:
        http = HttpClient(client)
        title, snippet = await http.fetch_page_summary(_THREAD_URL)
    assert "MaskedTensor RFC" in title
    assert "Prototype" in snippet


@pytest.mark.asyncio
async def test_discourse_retriever_emits_thread_evidence(discourse_profile, httpx_mock):
    httpx_mock.add_response(url=_THREAD_URL, method="HEAD", status_code=200)
    httpx_mock.add_response(url=_THREAD_URL, method="GET", text=_THREAD_HTML)

    async with httpx.AsyncClient() as client:
        clients = build_clients(Settings(), client)
        retriever = DiscourseRetriever()
        spec = RetrievalSpec(retriever="discourse", urls=[_THREAD_URL], metadata={"labels": [_THREAD_URL]})
        request = AssessmentRequest(question="test", repo="o/r")
        items = await retriever.fetch(spec, request, discourse_profile, clients)

    assert len(items) == 1
    assert items[0].kind.value == "discourse_thread"
    assert items[0].id == "discourse-42"
    assert items[0].snippet
