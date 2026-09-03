"""Citation integrity checks for synthesis summaries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from casefile.models.evidence import EvidenceItem

_REF_RE = re.compile(r"\[(\d+)\]")
# Quote of at most 15 words immediately before one or more citation markers.
# `"span" [1][2]` attaches the same span to both numbers.
_QUOTE_THEN_CITES_RE = re.compile(
    r'"([^"]{1,240})"\s*((?:\[\d+\])+)'
)
_CITE_NUM_RE = re.compile(r"\[(\d+)\]")


def check_citations(
    summary: str | None,
    citation_map: dict[int, str],
    valid_ids: set[str],
    *,
    id_by_num: dict[int, str] | None = None,
    items_by_id: dict[str, EvidenceItem] | None = None,
) -> list[str]:
    if not summary:
        return []

    errors: list[str] = []
    refs_in_text = {int(m.group(1)) for m in _REF_RE.finditer(summary)}

    for num in sorted(refs_in_text):
        evidence_id = citation_map.get(num)
        if evidence_id is None:
            errors.append(
                f"Citation [{num}] appears in prose but has no citations map entry "
                "(unsourced bracket — not filled from list position)."
            )
            continue
        if evidence_id not in valid_ids:
            errors.append(f"Citation [{num}] maps to unknown evidence id {evidence_id!r}")
        if id_by_num is not None:
            expected = id_by_num.get(num)
            if expected is not None and evidence_id != expected:
                errors.append(
                    f"Citation [{num}] maps to {evidence_id!r} but evidence slot [{num}] "
                    f"is {expected!r}"
                )

    for num, evidence_id in citation_map.items():
        if evidence_id not in valid_ids:
            errors.append(f"citation_map[{num}] references unknown evidence id {evidence_id!r}")

    if items_by_id is not None:
        quote_hits = _quotes_for_citations(summary)
        for num in sorted(refs_in_text):
            evidence_id = citation_map.get(num)
            if evidence_id is None or evidence_id not in items_by_id:
                continue
            quote = quote_hits.get(num)
            if not quote:
                errors.append(
                    f"Citation [{num}] is missing a short verbatim quote (≤15 words) "
                    "from the cited evidence immediately before the marker."
                )
                continue
            if len(quote.split()) > 15:
                errors.append(f"Citation [{num}] quote exceeds 15 words: {quote!r}")
                continue
            item = items_by_id[evidence_id]
            hay = quote_haystack(item)
            if _normalize_span(quote) not in _normalize_span(hay):
                errors.append(
                    f"Citation [{num}] quote {quote!r} not found in cited item {evidence_id!r}"
                )

    return errors


def _quotes_for_citations(summary: str) -> dict[int, str]:
    hits: dict[int, str] = {}
    for match in _QUOTE_THEN_CITES_RE.finditer(summary):
        quote = match.group(1).strip()
        for num_s in _CITE_NUM_RE.findall(match.group(2)):
            hits[int(num_s)] = quote
    return hits


def quote_haystack(item: EvidenceItem) -> str:
    """Title + fetched snippet only. Curator `relevance` notes are not source text."""
    return f"{item.title} {item.snippet}"


def _normalize_span(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
