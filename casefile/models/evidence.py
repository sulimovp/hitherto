from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class EvidenceKind(StrEnum):
    ISSUE = "issue"
    ISSUE_COMMENT = "issue_comment"
    PULL_REQUEST = "pull_request"
    COMMIT = "commit"
    FILE = "file"
    DISCOURSE_THREAD = "discourse_thread"
    HF_DISCUSSION = "hf_discussion"
    RELEASE = "release"
    ADJACENT_PROJECT = "adjacent_project"
    PROCESS_DOC = "process_doc"
    VITAL_SIGNS = "vital_signs"


class EvidenceItem(BaseModel):
    id: str
    kind: EvidenceKind
    title: str
    url: HttpUrl
    snippet: str = Field(max_length=500)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_retriever: str
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    excluded: list[EvidenceItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    freshness: datetime | None = None
    retrieval_stats: dict[str, Any] = Field(default_factory=dict)

    def dedupe_by_url(self) -> None:
        seen: dict[str, EvidenceItem] = {}
        for item in self.items:
            key = str(item.url)
            existing = seen.get(key)
            if existing is None or item.relevance_score > existing.relevance_score:
                seen[key] = item
        self.items = sorted(seen.values(), key=lambda i: i.relevance_score, reverse=True)

    def evidence_ids(self) -> set[str]:
        return {item.id for item in self.items}
