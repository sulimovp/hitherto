from urllib.parse import urlparse

from casefile.clients import ClientBundle
from casefile.clients.discourse_html import extract_discourse_topic_urls
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec


class DiscourseRetriever:
    name = "discourse"
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
            item = await self._fetch_thread(
                clients, url, label=url, pinned=True, rank=idx
            )
            if item is not None:
                by_url[str(item.url)] = item

        search_urls = spec.metadata.get("search_urls", [])
        base = str(spec.metadata.get("discourse_base", ""))
        for search_url in search_urls:
            html = await clients.http.fetch_html(str(search_url))
            if not html or not base:
                continue
            for rank, topic_url in enumerate(extract_discourse_topic_urls(html, base, limit=3)):
                item = await self._fetch_thread(
                    clients,
                    topic_url,
                    label=f"Search hit: {topic_url}",
                    pinned=False,
                    rank=rank,
                    from_search=True,
                )
                if item is not None:
                    existing = by_url.get(str(item.url))
                    if existing is None or item.relevance_score > existing.relevance_score:
                        by_url[str(item.url)] = item

        return sorted(by_url.values(), key=lambda i: i.relevance_score, reverse=True)

    async def _fetch_thread(
        self,
        clients: ClientBundle,
        url: str,
        *,
        label: str,
        pinned: bool,
        rank: int,
        from_search: bool = False,
    ) -> EvidenceItem | None:
        if not await clients.http.url_exists(url):
            return None
        title, snippet = await clients.http.fetch_page_summary(url)
        slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
        score = 0.88 if pinned else max(0.72, 0.82 - rank * 0.03)
        return EvidenceItem(
            id=f"discourse-{slug or rank}",
            kind=EvidenceKind.DISCOURSE_THREAD,
            title=title if title != url else str(label)[:200],
            url=url,
            snippet=snippet[:500],
            source_retriever=self.name,
            relevance_score=score,
            metadata={"pinned": pinned, "discourse_search": from_search},
        )
