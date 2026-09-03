from urllib.parse import quote_plus

from casefile.models.assessment import AssessmentRequest
from casefile.models.profile import EcosystemProfile
from casefile.retrievers.base import RetrievalPlan, RetrievalSpec

DEFAULT_SCOPE_FILES = ("README.md", "CONTRIBUTING.md", ".github/CODEOWNERS")

# Question filler — not useful as GitHub search terms (AND semantics make long queries miss).
_STOPWORDS = frozenset(
    {
        "about",
        "all",
        "and",
        "any",
        "are",
        "for",
        "from",
        "how",
        "into",
        "is",
        "its",
        "our",
        "should",
        "the",
        "this",
        "that",
        "their",
        "there",
        "these",
        "those",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "worth",
        "revive",
        "reviving",
        "upstream",
        "contribution",
        "contributions",
        "invest",
        "investment",
        "work",
        "doing",
        "make",
        "need",
        "open",
        "source",
    }
)

MAX_PHRASE_ISSUE_QUERIES = 3
MAX_TOTAL_ISSUE_QUERIES = 8


def build_plan(
    request: AssessmentRequest,
    profile: EcosystemProfile | None,
) -> RetrievalPlan:
    terms = [request.question]
    if profile is not None:
        terms = profile.expanded_terms(request.question)

    resolved_path = request.path
    if resolved_path is None and profile is not None:
        lower = request.question.lower()
        for hint, path in profile.path_hints.items():
            if hint.lower() in lower:
                resolved_path = path
                break

    plan = RetrievalPlan(search_terms=terms, resolved_path=resolved_path)

    issue_queries = _issue_search_queries(request.repo, request.question, profile, resolved_path)
    plan.specs.append(RetrievalSpec(retriever="github_issues", queries=issue_queries))

    file_paths = list(DEFAULT_SCOPE_FILES)
    if profile is not None:
        file_paths.extend(profile.scope_files)
    plan.specs.append(
        RetrievalSpec(retriever="repo_files", paths=list(dict.fromkeys(file_paths)))
    )

    if resolved_path:
        plan.specs.append(
            RetrievalSpec(retriever="git_activity", paths=[resolved_path])
        )
        plan.specs.append(
            RetrievalSpec(retriever="vital_signs", paths=[resolved_path])
        )

    if profile is not None and profile.adjacent_projects:
        plan.specs.append(
            RetrievalSpec(
                retriever="adjacent_projects",
                urls=[str(p.url) for p in profile.adjacent_projects],
                metadata={"projects": [p.model_dump(mode="json") for p in profile.adjacent_projects]},
            )
        )

    if request.tier >= 2 and resolved_path:
        pr_queries = _pr_search_queries(request.repo, resolved_path, profile)
        plan.specs.append(RetrievalSpec(retriever="github_prs", queries=pr_queries))

    if request.tier >= 2 and profile is not None and profile.rfc:
        plan.specs.append(
            RetrievalSpec(
                retriever="process_docs",
                urls=[str(r.url) for r in profile.rfc],
                metadata={"labels": [r.label for r in profile.rfc]},
            )
        )

    if request.tier >= 2 and profile is not None and profile.discourse:
        disc = profile.discourse
        pinned = [str(u) for u in disc.pinned_threads]
        search_urls: list[str] = []
        base = str(disc.base_url).rstrip("/")
        path = disc.search_path if disc.search_path.startswith("/") else f"/{disc.search_path}"
        for phrase in _search_phrases(request.question, profile, resolved_path)[:2]:
            search_urls.append(f"{base}{path}{quote_plus(phrase)}")
        plan.specs.append(
            RetrievalSpec(
                retriever="discourse",
                urls=pinned,
                metadata={
                    "pinned_labels": pinned,
                    "search_urls": search_urls,
                    "discourse_base": base,
                },
            )
        )

    if request.tier >= 2 and profile is not None and profile.huggingface:
        hf = profile.huggingface
        plan.specs.append(
            RetrievalSpec(
                retriever="huggingface_discussions",
                urls=[str(u) for u in hf.pinned_discussions],
                metadata={
                    "hub_repos": [r.model_dump() for r in hf.hub_repos],
                    "max_discussions_per_repo": hf.max_discussions_per_repo,
                },
            )
        )

    return plan


def _issue_search_queries(
    repo: str,
    question: str,
    profile: EcosystemProfile | None,
    path: str | None,
) -> list[str]:
    queries: list[str] = []

    if profile is not None:
        for number in profile.pinned_issues_for_path(path):
            queries.append(f"repo:{repo} is:issue {number}")
        for qualifier in profile.issue_queries_for_path(path):
            queries.append(f"repo:{repo} is:issue {qualifier}")

    phrases = _search_phrases(question, profile, path)
    if not phrases:
        phrases = ["help wanted"]
    for phrase in phrases[:MAX_PHRASE_ISSUE_QUERIES]:
        queries.append(f"repo:{repo} is:issue {phrase}")

    # Catch-all so small repos still surface issues topical queries miss
    # (validator excludes them with a reason instead of looking empty).
    catch_all = f"repo:{repo} is:issue"
    deduped = list(dict.fromkeys(queries))
    if catch_all in deduped:
        deduped.remove(catch_all)
    reserved = deduped[: MAX_TOTAL_ISSUE_QUERIES - 1]
    reserved.append(catch_all)
    return reserved


def _pr_search_queries(
    repo: str,
    path: str,
    profile: EcosystemProfile | None,
) -> list[str]:
    phrases = _search_phrases("", profile, path)
    queries = [f"repo:{repo} is:pr is:merged path:{path}"]
    for phrase in phrases[:2]:
        queries.append(f"repo:{repo} is:pr is:merged {phrase}")
    return list(dict.fromkeys(queries))


def _search_phrases(
    question: str,
    profile: EcosystemProfile | None,
    path: str | None,
) -> list[str]:
    """Domain phrases for GitHub search — short, from profile synonyms and path."""
    seen: set[str] = set()
    phrases: list[str] = []

    def add(phrase: str) -> None:
        normalized = phrase.strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        phrases.append(normalized)

    if profile is not None:
        lower_q = question.lower()
        for key, aliases in profile.synonyms.items():
            key_l = key.lower()
            matched = key_l in lower_q
            if not matched:
                for alias in aliases:
                    if alias.lower() in lower_q:
                        add(alias)
                        matched = True
            if matched or (
                path
                and (
                    key_l.replace(" ", "") in path.replace("/", "").replace("_", "").lower()
                    or any(a.lower() in path.lower() for a in aliases)
                )
            ):
                add(key)
                for alias in aliases:
                    add(alias)

    if path:
        # torch/masked -> torch.masked as search hint
        add(path.replace("/", "."))
        segment = path.rsplit("/", 1)[-1]
        if len(segment) >= 3:
            add(segment)

    for word in _significant_words(question):
        add(word)

    return phrases


def _significant_words(text: str) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for word in text.replace("/", " ").replace("-", " ").split():
        cleaned = word.strip(".,?!\"'():/").lower()
        if len(cleaned) < 4 or cleaned in _STOPWORDS or cleaned in seen:
            continue
        seen.add(cleaned)
        words.append(cleaned)
    return words
