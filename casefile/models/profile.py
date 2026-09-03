from datetime import date, datetime

from pydantic import BaseModel, Field, HttpUrl


class AdjacentProject(BaseModel):
    name: str
    url: HttpUrl
    relevance: str


class DiscourseConfig(BaseModel):
    base_url: HttpUrl
    search_path: str = "/search?q="
    pinned_threads: list[HttpUrl] = Field(default_factory=list)


class RfcLink(BaseModel):
    label: str
    url: HttpUrl


class HuggingFaceHubRepo(BaseModel):
    repo_id: str
    repo_type: str = Field(default="model", pattern="^(model|dataset|space)$")


class HuggingFaceConfig(BaseModel):
    """Hub discussion sources — used when evidence lives outside GitHub."""

    hub_repos: list[HuggingFaceHubRepo] = Field(default_factory=list)
    pinned_discussions: list[HttpUrl] = Field(default_factory=list)
    max_discussions_per_repo: int = Field(default=15, ge=1, le=50)


class AssignmentPrecision(BaseModel):
    measured_at: date
    n: int = Field(ge=1)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)


_MIN_ASSIGNMENT_PRECISION = 0.80
_MIN_ASSIGNMENT_RECALL = 0.80


class EcosystemProfile(BaseModel):
    id: str
    display_name: str
    version: int = 1
    last_verified: date
    max_age_days: int = Field(default=90, ge=1)
    default_repos: list[str] = Field(default_factory=list)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    label_boosts: dict[str, float] = Field(default_factory=dict)
    adjacent_projects: list[AdjacentProject] = Field(default_factory=list)
    scope_files: list[str] = Field(default_factory=list)
    discourse: DiscourseConfig | None = None
    huggingface: HuggingFaceConfig | None = None
    rfc: list[RfcLink] = Field(default_factory=list)
    sibling_repos: list[str] = Field(default_factory=list)
    path_hints: dict[str, str] = Field(default_factory=dict)
    pinned_issue_numbers: list[int] = Field(default_factory=list)
    issue_search_queries: list[str] = Field(
        default_factory=list,
        description="Extra GitHub issue search qualifiers, e.g. label:\"module: masked\"",
    )
    path_pinned_issue_numbers: dict[str, list[int]] = Field(default_factory=dict)
    path_issue_search_queries: dict[str, list[str]] = Field(default_factory=dict)
    assignment_precision: AssignmentPrecision | None = None

    def assignment_precision_status(
        self, as_of: datetime | None = None
    ) -> tuple[bool, str | None]:
        """Return (measured_ok, refusal_reason_or_none)."""
        ap = self.assignment_precision
        if ap is None:
            return False, "item→topic assignment precision unmeasured for this profile"
        now = (as_of or datetime.now()).date()
        age = (now - ap.measured_at).days
        if age > self.max_age_days:
            return False, (
                f"item→topic assignment precision stale "
                f"(measured {ap.measured_at.isoformat()}, {age}d ago > {self.max_age_days}d)"
            )
        if ap.precision < _MIN_ASSIGNMENT_PRECISION:
            return False, (
                f"item→topic assignment precision {ap.precision:.2f} < "
                f"{_MIN_ASSIGNMENT_PRECISION} (n={ap.n}, measured {ap.measured_at.isoformat()})"
            )
        if ap.recall is not None and ap.recall < _MIN_ASSIGNMENT_RECALL:
            return False, (
                f"item→topic assignment recall {ap.recall:.2f} < "
                f"{_MIN_ASSIGNMENT_RECALL} (n={ap.n}, measured {ap.measured_at.isoformat()})"
            )
        return True, None

    def is_stale(self, as_of: datetime | None = None) -> bool:
        now = (as_of or datetime.now()).date()
        age = (now - self.last_verified).days
        return age > self.max_age_days

    def expanded_terms(self, question: str) -> list[str]:
        terms = [question]
        lower = question.lower()
        for phrase, aliases in self.synonyms.items():
            if phrase.lower() in lower:
                terms.extend(aliases)
        return list(dict.fromkeys(terms))

    def pinned_issues_for_path(self, path: str | None) -> list[int]:
        if path and path in self.path_pinned_issue_numbers:
            return list(self.path_pinned_issue_numbers[path])
        return list(self.pinned_issue_numbers)

    def issue_queries_for_path(self, path: str | None) -> list[str]:
        queries = list(self.issue_search_queries)
        if path:
            queries.extend(self.path_issue_search_queries.get(path, []))
        return queries

    def topic_match_tokens(self, question: str, path: str | None = None) -> set[str]:
        """Tokens that must appear (substring) in issue title/snippet to count as on-topic."""
        tokens: set[str] = set()
        for term in self.expanded_terms(question):
            for word in term.replace("/", ".").replace("-", " ").split():
                cleaned = word.strip(".,?!\"'():/").lower()
                if len(cleaned) >= 4:
                    tokens.add(cleaned)
        for aliases in self.synonyms.values():
            for alias in aliases:
                tokens.add(alias.lower())
                compact = alias.lower().replace(" ", "")
                if len(compact) >= 5:
                    tokens.add(compact)
        if path:
            tokens.add(path.lower())
            tokens.add(path.rsplit("/", 1)[-1].lower())
            tokens.add(path.replace("/", ".").lower())
        return tokens
