from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec


class ProcessDocsRetriever:
    name = "process_docs"
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
        labels = spec.metadata.get("labels", [])
        items: list[EvidenceItem] = []
        for idx, url in enumerate(spec.urls):
            if not await clients.http.url_exists(url):
                continue
            label = labels[idx] if idx < len(labels) else url
            items.append(
                EvidenceItem(
                    id=f"process-doc-{idx}",
                    kind=EvidenceKind.PROCESS_DOC,
                    title=str(label),
                    url=url,
                    snippet=f"Design / RFC index: {label}",
                    source_retriever=self.name,
                    relevance_score=0.55,
                )
            )
        return items
