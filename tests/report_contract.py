"""Shared assertions for report content and markdown format."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from casefile.models.assessment import AssessmentReport

GOLDEN_PYTORCH_MASKED = (89734, 89320, 124964)

REQUIRED_MARKDOWN_SECTIONS = (
    "## Evidence",
    "## Open questions",
    "## Sources index",
)

# Optional when path-scoped assessments emit vital signs / topic forecasts
OPTIONAL_MARKDOWN_SECTIONS = (
    "## Activity forecast",
    "## Topic trajectory",
)


EVIDENCE_LINE_RE = re.compile(
    r"^\[\d+\] .+ — https?://\S+ — .+$",
    re.MULTILINE,
)

SOURCES_LINE_RE = re.compile(
    r"^\[\d+\] [\w-]+ — https?://\S+$",
    re.MULTILINE,
)


def assert_markdown_report_contract(md: str, *, expect_summary: bool) -> None:
    """Validate stable report structure (format contract)."""
    lines = md.splitlines()
    assert lines[0].startswith("# Assessment:"), "Report must begin with H1 question title"
    assert any(line.startswith("Generated:") for line in lines[:6]), "Missing metadata line"
    assert "Repo:" in md, "Missing repo metadata line"

    for section in REQUIRED_MARKDOWN_SECTIONS:
        assert section in md, f"Missing required section: {section}"

    if expect_summary:
        assert "## Summary (model synthesis" in md, "Expected synthesis summary section"
        summary_block = md.split("## Summary", 1)[1].split("## Evidence", 1)[0]
        assert summary_block.strip(), "Summary section is empty"
        assert "{" not in summary_block or '"paragraphs"' not in summary_block, (
            "Raw JSON must not appear in summary"
        )
    else:
        assert "## Summary" not in md or "_Synthesis skipped" in md

    open_q = md.split("## Open questions", 1)[1].split("##", 1)[0]
    assert open_q.strip(), "Open questions section is empty"
    assert re.search(r"^- .+", open_q, re.MULTILINE), "Open questions must use bullet list"

    evidence_lines = EVIDENCE_LINE_RE.findall(md)
    assert evidence_lines, "Expected at least one formatted evidence line [n] title — url — snippet"

    source_lines = SOURCES_LINE_RE.findall(md)
    assert source_lines, "Expected at least one sources index line"
    assert len(source_lines) == len(set(source_lines)), "Duplicate sources index entries"


def assert_golden_issues_present(report: AssessmentReport, numbers: tuple[int, ...] = GOLDEN_PYTORCH_MASKED) -> None:
    ids = {item.id for item in report.evidence.items}
    for number in numbers:
        assert f"issue-{number}" in ids, f"Golden issue #{number} missing from evidence"


def assert_evidence_kinds_present(report: AssessmentReport, *kinds: str) -> None:
    found = {item.kind.value for item in report.evidence.items}
    for kind in kinds:
        assert kind in found, f"Expected evidence kind {kind!r}, got {sorted(found)}"


def assert_open_questions_include(report: AssessmentReport, *fragments: str) -> None:
    joined = " ".join(report.evidence.open_questions).lower()
    for fragment in fragments:
        assert fragment.lower() in joined, (
            f"Expected open question containing {fragment!r}; got: {report.evidence.open_questions}"
        )
