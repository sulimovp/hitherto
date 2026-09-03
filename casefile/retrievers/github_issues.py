import asyncio
import re

from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec


class GitHubIssuesRetriever:
    name = "github_issues"
    tier = 1
    _PINNED_QUERY = re.compile(r"is:issue\s+(\d+)\s*$")

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
        by_number: dict[int, EvidenceItem] = {}
        for query in spec.queries:
            await asyncio.sleep(0.4)
            results = await clients.github.search_issues(query, per_page=20)
            is_pinned_query = bool(self._PINNED_QUERY.search(query))
            is_label_query = "label:" in query
            is_catch_all = bool(re.fullmatch(r"repo:\S+ is:issue", query.strip()))
            for rank, issue in enumerate(results):
                number = issue.get("number")
                if number is None:
                    continue
                labels = [lbl.get("name", "") for lbl in issue.get("labels", []) if isinstance(lbl, dict)]
                if is_pinned_query:
                    score = 0.95
                elif is_label_query:
                    score = max(0.85, 1.0 - rank * 0.03)
                elif is_catch_all:
                    score = max(0.05, 0.25 - rank * 0.02)
                else:
                    score = max(0.1, 1.0 - rank * 0.04)
                if profile is not None:
                    for label, boost in profile.label_boosts.items():
                        if label in labels:
                            score = min(1.0, score + boost)
                body = issue.get("body") or ""
                snippet = (issue.get("title") or "")[:200]
                if body:
                    snippet = f"{snippet} — {str(body)[:280]}"
                item = EvidenceItem(
                    id=f"issue-{number}",
                    kind=EvidenceKind.ISSUE,
                    title=str(issue.get("title") or f"Issue #{number}"),
                    url=issue.get("html_url", f"https://github.com/{request.repo}/issues/{number}"),
                    snippet=snippet[:500],
                    source_retriever=self.name,
                    relevance_score=score,
                    metadata={
                        "state": issue.get("state"),
                        "labels": labels,
                        "updated_at": issue.get("updated_at"),
                        "search_query": query,
                        "pinned": is_pinned_query,
                        "label_search": is_label_query,
                    },
                )
                prev = by_number.get(number)
                if prev is None or item.relevance_score > prev.relevance_score:
                    by_number[number] = item
        return sorted(by_number.values(), key=lambda i: i.relevance_score, reverse=True)
