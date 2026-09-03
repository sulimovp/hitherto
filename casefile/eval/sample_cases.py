"""Load curated sample cases from eval/sample_cases.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "eval" / "sample_cases.yaml"


@dataclass(frozen=True)
class SampleCase:
    id: str
    label: str
    blurb: str
    question: str
    repo: str
    path: str
    ecosystem: str
    tier: int
    synthesize_default: bool
    held_out: bool
    expect_issue_numbers: tuple[int, ...]
    expect_evidence_kinds: tuple[str, ...]

    def preset_dict(self) -> dict[str, str | int | bool]:
        return {
            "label": self.label,
            "blurb": self.blurb,
            "question": self.question,
            "repo": self.repo,
            "path": self.path,
            "ecosystem": self.ecosystem,
            "tier": self.tier,
            "synthesize_default": self.synthesize_default,
        }


def _parse_case(raw: dict[str, Any]) -> SampleCase:
    return SampleCase(
        id=str(raw["id"]),
        label=str(raw["label"]),
        blurb=str(raw.get("blurb", "")).strip(),
        question=str(raw["question"]),
        repo=str(raw["repo"]),
        path=str(raw.get("path") or ""),
        ecosystem=str(raw.get("ecosystem") or ""),
        tier=int(raw.get("tier", 2)),
        synthesize_default=bool(raw.get("synthesize_default", True)),
        held_out=bool(raw.get("held_out", False)),
        expect_issue_numbers=tuple(raw.get("expect_issue_numbers") or []),
        expect_evidence_kinds=tuple(raw.get("expect_evidence_kinds") or []),
    )


def load_sample_cases(path: Path | None = None) -> dict[str, SampleCase]:
    yaml_path = path or _DEFAULT_PATH
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cases = [_parse_case(c) for c in data.get("cases", [])]
    return {c.id: c for c in cases}


def load_held_out_cases(path: Path | None = None) -> list[SampleCase]:
    return [c for c in load_sample_cases(path).values() if c.held_out]
