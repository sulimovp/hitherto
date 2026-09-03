from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec


class RepoFilesRetriever:
    name = "repo_files"
    tier = 1

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
        items: list[EvidenceItem] = []
        for path in spec.paths:
            content = await clients.github.get_file_content(request.owner, request.name, path)
            if content is None:
                continue
            kind = EvidenceKind.PROCESS_DOC if "CONTRIBUTING" in path or "CODEOWNERS" in path else EvidenceKind.FILE
            preview = content.strip().splitlines()
            snippet = "\n".join(preview[:8])[:500]
            items.append(
                EvidenceItem(
                    id=f"file-{path.replace('/', '-').replace('.', '-')}",
                    kind=kind,
                    title=path,
                    url=f"https://github.com/{request.repo}/blob/HEAD/{path}",
                    snippet=snippet,
                    source_retriever=self.name,
                    relevance_score=0.6 if kind == EvidenceKind.PROCESS_DOC else 0.45,
                    metadata={"path": path, "line_count": len(preview)},
                )
            )
        return items
