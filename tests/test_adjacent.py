"""Adjacent-project retriever fetches page text, not curator relevance."""

from datetime import date

import httpx
import pytest

from casefile.clients import build_clients
from casefile.config import Settings
from casefile.models.assessment import AssessmentRequest
from casefile.models.profile import AdjacentProject, EcosystemProfile
from casefile.retrievers.adjacent import AdjacentProjectsRetriever
from casefile.retrievers.base import RetrievalSpec

_URL = "https://huggingface.co/swiss-ai/Apertus-v1.5-8B"
_HTML = (
    "<html><head><title>Apertus 1.5 8B</title></head>"
    "<body><p>Native tool calling is documented on this model card.</p></body></html>"
)
_CURATOR = "Primary 1.5 weights; documents native tool use and optional thinking mode."


@pytest.mark.asyncio
async def test_adjacent_snippet_is_fetched_page_not_curator_note(httpx_mock):
    httpx_mock.add_response(url=_URL, method="HEAD", status_code=200)
    httpx_mock.add_response(url=_URL, method="GET", text=_HTML)
    profile = EcosystemProfile(
        id="apertus",
        display_name="Apertus",
        last_verified=date(2026, 8, 1),
        adjacent_projects=[
            AdjacentProject(name="Apertus-v1.5-8B", url=_URL, relevance=_CURATOR)
        ],
    )
    settings = Settings()
    async with httpx.AsyncClient() as client:
        clients = build_clients(settings, client)
        items = await AdjacentProjectsRetriever().fetch(
            RetrievalSpec(retriever="adjacent_projects"),
            AssessmentRequest(question="q", repo="swiss-ai/apertus-format"),
            profile,
            clients,
        )
    assert len(items) == 1
    assert "Native tool calling is documented" in items[0].snippet
    assert _CURATOR not in items[0].snippet
    assert items[0].metadata["curator_note"] == _CURATOR
    assert items[0].metadata["snippet_source"] == "fetched_page"


@pytest.mark.asyncio
async def test_unfetched_adjacent_is_excluded_not_dropped(httpx_mock):
    from casefile.engine.validator import validate_evidence

    httpx_mock.add_response(url=_URL, method="HEAD", status_code=403)
    profile = EcosystemProfile(
        id="apertus",
        display_name="Apertus",
        last_verified=date(2026, 8, 1),
        adjacent_projects=[
            AdjacentProject(name="Apertus-v1.5-8B", url=_URL, relevance=_CURATOR)
        ],
    )
    request = AssessmentRequest(question="q", repo="swiss-ai/apertus-format")
    settings = Settings()
    async with httpx.AsyncClient() as client:
        clients = build_clients(settings, client)
        items = await AdjacentProjectsRetriever().fetch(
            RetrievalSpec(retriever="adjacent_projects"),
            request,
            profile,
            clients,
        )
    assert len(items) == 1
    assert items[0].metadata["exclusion_reason"] == "adjacent URL did not respond"
    assert _CURATOR not in items[0].snippet
    kept, excluded = validate_evidence(items, profile, request)
    assert kept == []
    assert len(excluded) == 1
    assert excluded[0].title == "Apertus-v1.5-8B"
    assert "fetch" in excluded[0].metadata["exclusion_reason"] or "respond" in excluded[0].metadata["exclusion_reason"]
