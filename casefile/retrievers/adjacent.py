from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import AdjacentProject, EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec

_FETCH_FAILED_SNIPPET = "Page could not be fetched."


class AdjacentProjectsRetriever:
    name = "adjacent_projects"
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
        raw_projects = spec.metadata.get("projects", [])
        projects: list[AdjacentProject] = []
        for entry in raw_projects:
            if isinstance(entry, dict):
                projects.append(AdjacentProject.model_validate(entry))
        if profile is not None and not projects:
            projects = profile.adjacent_projects

        items: list[EvidenceItem] = []
        for project in projects:
            url = str(project.url)
            exists = await clients.http.url_exists(url)
            page_text = ""
            if exists:
                _page_title, page_text = await clients.http.fetch_page_summary(
                    url, max_len=480
                )
            reason = None
            if not exists:
                reason = "adjacent URL did not respond"
            elif not page_text or page_text == _FETCH_FAILED_SNIPPET:
                reason = "adjacent page fetch failed"
            items.append(
                _adjacent_item(
                    project,
                    snippet=page_text if reason is None else _FETCH_FAILED_SNIPPET,
                    exclusion_reason=reason,
                )
            )
        return items


def _adjacent_item(
    project: AdjacentProject,
    *,
    snippet: str,
    exclusion_reason: str | None,
) -> EvidenceItem:
    slug = project.name.lower().replace(" ", "-")
    metadata: dict = {
        "snippet_source": "fetched_page" if exclusion_reason is None else "unfetched",
        "curator_note": project.relevance,
    }
    if exclusion_reason is None:
        metadata["validated"] = True
    else:
        metadata["exclusion_reason"] = exclusion_reason
    return EvidenceItem(
        id=f"adjacent-{slug}",
        kind=EvidenceKind.ADJACENT_PROJECT,
        title=project.name,
        url=project.url,
        snippet=snippet[:500],
        source_retriever="adjacent_projects",
        relevance_score=0.7,
        metadata=metadata,
    )
