from dataclasses import dataclass, field
from typing import Protocol

from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem
from casefile.models.profile import EcosystemProfile


@dataclass
class RetrievalSpec:
    retriever: str
    queries: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    specs: list[RetrievalSpec] = field(default_factory=list)
    search_terms: list[str] = field(default_factory=list)
    resolved_path: str | None = None


class Retriever(Protocol):
    name: str
    tier: int

    def plan(
        self,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        plan: RetrievalPlan,
    ) -> RetrievalSpec | None: ...

    async def fetch(
        self,
        spec: RetrievalSpec,
        request: AssessmentRequest,
        profile: EcosystemProfile | None,
        clients: ClientBundle,
    ) -> list[EvidenceItem]: ...
