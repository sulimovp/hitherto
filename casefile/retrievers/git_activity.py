from casefile.clients import ClientBundle
from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec


class GitActivityRetriever:
    name = "git_activity"
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
            commits, _truncated = await clients.github.list_commits(
                request.owner, request.name, path=path, per_page=30
            )
            if not commits:
                continue
            latest = commits[0]
            commit = latest.get("commit", {}) if isinstance(latest.get("commit"), dict) else {}
            date_str = commit.get("author", {}).get("date") if isinstance(commit.get("author"), dict) else None
            sha = latest.get("sha", "")[:7]
            authors: dict[str, int] = {}
            for entry in commits:
                author = entry.get("author") or entry.get("commit", {}).get("author", {})
                login = None
                if isinstance(entry.get("author"), dict):
                    login = entry["author"].get("login")
                if login:
                    authors[login] = authors.get(login, 0) + 1
            top = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:3]
            snippet = (
                f"Path `{path}`: {len(commits)} commits fetched (page cap 30). "
                f"Latest: {sha} on {date_str}. Top committers: {', '.join(a for a, _ in top) or 'unknown'}."
            )
            html_url = latest.get("html_url") or f"https://github.com/{request.repo}/commits/{path}"
            items.append(
                EvidenceItem(
                    id=f"activity-{path.replace('/', '-')}",
                    kind=EvidenceKind.COMMIT,
                    title=f"Recent activity on {path}",
                    url=html_url,
                    snippet=snippet[:500],
                    source_retriever=self.name,
                    relevance_score=0.75,
                    metadata={
                        "path": path,
                        "commit_count": len(commits),
                        "last_commit_date": date_str,
                        "top_committers": top,
                    },
                )
            )
        return items
