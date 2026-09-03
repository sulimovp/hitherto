"""Retrieve Hugging Face Hub discussions for ecosystem profiles."""

from __future__ import annotations

from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec

_DECISION_KEYWORDS = (
    "upstream",
    "transformers",
    "tool",
    "thinking",
    "format",
    "chat",
    "parser",
    "template",
    "native",
)


class HuggingFaceDiscussionsRetriever:
    name = "huggingface_discussions"
    tier = 2

    def plan(
        self,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        plan: RetrievalPlan,
    ) -> RetrievalSpec | None:
        for spec in plan.specs:
            if spec.retriever == self.name:
                return spec
        return None

    async def fetch(
        self,
        spec: RetrievalSpec,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        clients: ClientBundle,
    ) -> list[EvidenceItem]:
        by_url: dict[str, EvidenceItem] = {}

        for idx, url in enumerate(spec.urls):
            item = self._pinned_item(url, rank=idx)
            by_url[str(item.url)] = item

        repos = spec.metadata.get("hub_repos", [])
        limit = int(spec.metadata.get("max_discussions_per_repo", 15))
        errors: list[str] = []
        for repo in repos:
            repo_id = repo.get("repo_id")
            repo_type = repo.get("repo_type", "model")
            if not repo_id:
                continue
            result = await clients.huggingface.list_discussions(
                repo_id, repo_type=repo_type, limit=limit
            )
            if result.error:
                errors.append(result.error)
            for rank, disc in enumerate(result.items):
                item = self._from_api(
                    disc,
                    repo_id=repo_id,
                    rank=rank,
                    request=request,
                    profile=profile,
                )
                if item is None:
                    continue
                key = str(item.url)
                existing = by_url.get(key)
                if existing is None or item.relevance_score > existing.relevance_score:
                    by_url[key] = item

        if errors and not by_url:
            raise RuntimeError("; ".join(errors))
        if errors:
            first = next(iter(by_url.values()), None)
            if first is not None:
                first.metadata["hub_fetch_warnings"] = errors

        return sorted(by_url.values(), key=lambda i: i.relevance_score, reverse=True)

    def _pinned_item(self, url: str, *, rank: int) -> EvidenceItem:
        num = url.rstrip("/").rsplit("/", 1)[-1]
        return EvidenceItem(
            id=f"hf-disc-pinned-{num}",
            kind=EvidenceKind.HF_DISCUSSION,
            title=f"Pinned Hub discussion #{num}",
            url=url,
            snippet="Pinned Hugging Face discussion from the ecosystem profile.",
            source_retriever=self.name,
            relevance_score=0.9 - min(rank, 5) * 0.02,
            metadata={"pinned": True, "hub_url": url},
        )

    def _from_api(
        self,
        disc: dict,
        *,
        repo_id: str,
        rank: int,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
    ) -> EvidenceItem | None:
        num = disc.get("num")
        title = disc.get("title") or f"Discussion #{num}"
        if num is None:
            return None
        url = f"https://huggingface.co/{repo_id}/discussions/{num}"
        status = disc.get("status", "unknown")
        is_pr = bool(disc.get("isPullRequest"))
        comments = int(disc.get("numComments") or 0)
        kind_label = "PR" if is_pr else "discussion"
        snippet = (
            f"Hub {kind_label} #{num} ({status}) on `{repo_id}` — "
            f"{comments} comment(s). Created {disc.get('createdAt', 'unknown')}."
        )
        score = _relevance_score(
            title=str(title),
            comments=comments,
            is_pr=is_pr,
            status=str(status),
            rank=rank,
            question=request.question,
            profile=profile,
        )
        return EvidenceItem(
            id=f"hf-disc-{repo_id.replace('/', '-')}-{num}",
            kind=EvidenceKind.HF_DISCUSSION,
            title=str(title)[:200],
            url=url,
            snippet=snippet[:500],
            source_retriever=self.name,
            relevance_score=score,
            metadata={
                "repo_id": repo_id,
                "num": num,
                "status": status,
                "is_pull_request": is_pr,
                "num_comments": comments,
            },
        )


def _relevance_score(
    *,
    title: str,
    comments: int,
    is_pr: bool,
    status: str,
    rank: int,
    question: str,
    profile: EcosystemProfile | None,
) -> float:
    """Rank by question/synonym overlap and engagement — not API recency."""
    score = 0.35
    title_l = title.lower()
    question_l = question.lower()

    if profile is not None:
        for token in profile.topic_match_tokens(question):
            if len(token) < 4:
                continue
            if token in title_l or token.replace(" ", "") in title_l.replace(" ", ""):
                score += 0.12

    for word in question_l.replace("/", " ").split():
        cleaned = word.strip(".,?!\"'():").lower()
        if len(cleaned) >= 5 and cleaned in title_l:
            score += 0.08

    for kw in _DECISION_KEYWORDS:
        if kw in title_l:
            score += 0.1

    score += min(0.3, comments * 0.025)
    if is_pr:
        score += 0.12
    if status == "draft":
        score += 0.08  # draft upstream PRs are often the decision pivot
    if status == "open":
        score += 0.03

    # Tiny recency tie-break only.
    score += max(0.0, 0.05 - rank * 0.002)
    return round(min(1.0, score), 3)
