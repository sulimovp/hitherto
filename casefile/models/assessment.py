from pydantic import BaseModel, Field, field_validator

from casefile.models.evidence import EvidenceBundle


class AssessmentRequest(BaseModel):
    question: str = Field(min_length=1)
    repo: str = Field(min_length=3)
    path: str | None = None
    ecosystem: str | None = None
    tier: int = Field(default=1, ge=1, le=3)
    max_evidence: int = Field(default=40, ge=1, le=200)
    synthesize: bool = True

    @field_validator("repo")
    @classmethod
    def repo_must_be_owner_name(cls, value: str) -> str:
        if value.count("/") != 1 or value.startswith("/") or value.endswith("/"):
            raise ValueError("repo must be owner/name")
        return value

    @property
    def owner(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.repo.split("/", 1)[1]


class AssessmentReport(BaseModel):
    request: AssessmentRequest
    evidence: EvidenceBundle
    summary: str | None = None
    citation_map: dict[int, str] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    activity_forecast: dict | None = None
    topic_forecast: dict | None = None
