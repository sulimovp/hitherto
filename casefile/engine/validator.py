from __future__ import annotations

import re

from casefile.models.assessment import AssessmentRequest
from casefile.models.evidence import EvidenceItem, EvidenceKind
from casefile.models.profile import EcosystemProfile

# Filler words that appear in questions but must not keep an issue on-topic alone.
_TOPIC_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "does",
        "from",
        "into",
        "mode",  # matches truncated "model" too easily
        "more",
        "only",
        "over",
        "place",
        "still",
        "than",
        "that",
        "them",
        "then",
        "they",
        "this",
        "with",
        "your",
    }
)


def validate_evidence(
    items: list[EvidenceItem],
    profile: EcosystemProfile | None,
    request: AssessmentRequest | None = None,
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """Filter evidence. Returns (kept, excluded) — excluded items carry exclusion_reason."""
    if profile is not None and profile.is_stale():
        pass

    path = request.path if request else None
    question = request.question if request else ""
    repo = request.repo if request else None

    seen: dict[str, EvidenceItem] = {}
    excluded: list[EvidenceItem] = []

    for item in items:
        if item.metadata.get("exclusion_reason"):
            excluded.append(item.model_copy(deep=True))
            continue
        if item.kind == EvidenceKind.ISSUE and profile is not None and request is not None:
            if not _issue_on_topic(item, profile, question, path, repo):
                dropped = item.model_copy(deep=True)
                dropped.metadata = {
                    **dropped.metadata,
                    "exclusion_reason": (
                        "off-topic: title/snippet matched none of the profile/question tokens"
                    ),
                }
                excluded.append(dropped)
                continue

        key = str(item.url)
        if key in seen:
            existing = seen[key]
            if item.relevance_score > existing.relevance_score:
                seen[key] = item
            continue
        if not item.snippet.strip():
            dropped = item.model_copy(deep=True)
            dropped.metadata = {**dropped.metadata, "exclusion_reason": "empty snippet"}
            excluded.append(dropped)
            continue
        seen[key] = item

    kept = sorted(seen.values(), key=lambda i: i.relevance_score, reverse=True)
    return kept, excluded


def _issue_on_topic(
    item: EvidenceItem,
    profile: EcosystemProfile,
    question: str,
    path: str | None,
    repo: str | None,
) -> bool:
    if (
        item.metadata.get("pinned")
        or item.metadata.get("label_search")
        or item.metadata.get("discourse_search")
    ):
        return True

    hay = f"{item.title} {item.snippet}".lower()
    hay_compact = hay.replace(" ", "").replace("_", "")
    generic = _generic_project_tokens(profile, repo)
    specific_hits = 0
    for token in profile.topic_match_tokens(question, path):
        if len(token) < 4 or token in _TOPIC_STOPWORDS:
            continue
        if not _token_matches(token, hay, hay_compact):
            continue
        if token in generic or any(token in g or g in token for g in generic if len(g) >= 4):
            continue
        specific_hits += 1
    # Mentions of the project name alone are not on-topic for the question.
    return specific_hits > 0


def _token_matches(token: str, hay: str, hay_compact: str) -> bool:
    compact = token.replace(" ", "")
    if len(compact) >= 6 and compact in hay_compact:
        return True
    # Word-boundary match for short tokens so "mode" does not hit "model".
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", hay))


def _generic_project_tokens(profile: EcosystemProfile, repo: str | None) -> set[str]:
    tokens: set[str] = {profile.id.lower()}
    for part in profile.display_name.lower().replace("(", " ").replace(")", " ").split():
        if len(part) >= 4:
            tokens.add(part)
    for default in profile.default_repos:
        for piece in default.lower().replace("/", " ").replace("-", " ").split():
            if len(piece) >= 4:
                tokens.add(piece)
        tokens.add(default.lower().rsplit("/", 1)[-1])
    if repo:
        for piece in repo.lower().replace("/", " ").replace("-", " ").split():
            if len(piece) >= 4:
                tokens.add(piece)
        tokens.add(repo.lower().rsplit("/", 1)[-1])
    return tokens
