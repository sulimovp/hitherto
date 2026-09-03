"""Hugging Face discussions retriever."""

from datetime import date

import httpx
import pytest

from casefile.clients import build_clients
from casefile.config import Settings
from casefile.engine.planner import build_plan
from casefile.models.assessment import AssessmentRequest
from casefile.models.profile import EcosystemProfile, HuggingFaceConfig, HuggingFaceHubRepo
from casefile.retrievers.base import RetrievalSpec
from casefile.retrievers.huggingface_discussions import (
    HuggingFaceDiscussionsRetriever,
    _relevance_score,
)


@pytest.fixture
def apertus_like_profile():
    return EcosystemProfile(
        id="apertus",
        display_name="Apertus",
        last_verified=date(2026, 8, 23),
        huggingface=HuggingFaceConfig(
            hub_repos=[HuggingFaceHubRepo(repo_id="swiss-ai/Apertus-v1.5-8B", repo_type="model")],
            max_discussions_per_repo=5,
        ),
    )


@pytest.mark.asyncio
async def test_hf_discussions_retriever(httpx_mock, apertus_like_profile):
    payload = {
        "discussions": [
            {
                "num": 5,
                "title": "Changes For Upstreaming to transformers",
                "status": "open",
                "isPullRequest": False,
                "numComments": 3,
                "createdAt": "2026-08-01T00:00:00.000Z",
            }
        ]
    }
    httpx_mock.add_response(
        url="https://huggingface.co/api/models/swiss-ai/Apertus-v1.5-8B/discussions?limit=5",
        json=payload,
    )

    async with httpx.AsyncClient() as client:
        clients = build_clients(Settings(), client)
        retriever = HuggingFaceDiscussionsRetriever()
        spec = RetrievalSpec(
            retriever="huggingface_discussions",
            metadata={
                "hub_repos": [{"repo_id": "swiss-ai/Apertus-v1.5-8B", "repo_type": "model"}],
                "max_discussions_per_repo": 5,
            },
        )
        request = AssessmentRequest(question="apertus-format", repo="swiss-ai/apertus-format", tier=2)
        items = await retriever.fetch(spec, request, apertus_like_profile, clients)

    assert len(items) == 1
    assert items[0].kind.value == "hf_discussion"
    assert items[0].id.endswith("-5")
    assert "huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/5" in str(items[0].url)


def test_planner_adds_hf_spec(apertus_like_profile):
    request = AssessmentRequest(
        question="Does apertus-format still earn its place?",
        repo="swiss-ai/apertus-format",
        ecosystem="apertus",
        tier=2,
    )
    plan = build_plan(request, apertus_like_profile)
    names = [s.retriever for s in plan.specs]
    assert "huggingface_discussions" in names


def test_hub_relevance_ranks_decision_threads_above_noise(apertus_like_profile):
    question = "Does apertus-format still earn its place after native tool calling?"
    noise = _relevance_score(
        title="Thank you",
        comments=1,
        is_pr=False,
        status="open",
        rank=0,
        question=question,
        profile=apertus_like_profile,
    )
    draft_pr = _relevance_score(
        title="Changes For Upstreaming to transformers",
        comments=4,
        is_pr=True,
        status="draft",
        rank=8,
        question=question,
        profile=apertus_like_profile,
    )
    tool_notes = _relevance_score(
        title="Apertus Tool Calling: Practical Notes",
        comments=6,
        is_pr=False,
        status="open",
        rank=11,
        question=question,
        profile=apertus_like_profile,
    )
    tool_parser = _relevance_score(
        title="Apertus tool parser",
        comments=14,
        is_pr=False,
        status="open",
        rank=25,
        question=question,
        profile=apertus_like_profile,
    )
    assert draft_pr > noise
    assert tool_notes > noise
    assert tool_parser > noise
    assert tool_parser >= draft_pr or tool_notes >= noise
