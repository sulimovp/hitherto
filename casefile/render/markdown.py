from datetime import UTC, datetime

from casefile.models.assessment import AssessmentReport
from casefile.models.evidence import EvidenceItem, EvidenceKind

_KIND_HEADINGS: dict[EvidenceKind, str] = {
    EvidenceKind.ISSUE: "### Issues and discussions",
    EvidenceKind.ISSUE_COMMENT: "### Issue comments",
    EvidenceKind.PULL_REQUEST: "### Merged work",
    EvidenceKind.COMMIT: "### Module activity",
    EvidenceKind.VITAL_SIGNS: "### Module vital signs",
    EvidenceKind.ADJACENT_PROJECT: "### Adjacent projects (validated)",
    EvidenceKind.PROCESS_DOC: "### Maintainer / process",
    EvidenceKind.FILE: "### Repository files",
    EvidenceKind.DISCOURSE_THREAD: "### Dev-discuss and forums",
    EvidenceKind.HF_DISCUSSION: "### Hugging Face discussions",
}


def render_markdown(report: AssessmentReport) -> str:
    req = report.request
    ev = report.evidence
    freshness = ev.freshness.isoformat() if ev.freshness else "unknown"
    lines = [
        f"# Assessment: {req.question}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()} | Repo: `{req.repo}` | "
        f"Profile: {req.ecosystem or 'none'} | Evidence freshness: {freshness}",
        "",
    ]

    if report.summary:
        lines.extend(
            [
                "## Summary (model synthesis — verify citations below)",
                "",
                report.summary,
                "",
            ]
        )
    elif req.synthesize:
        lines.extend(
            [
                "## Summary",
                "",
                "_Synthesis skipped (no LLM key, validation failure, or empty evidence)._",
                "",
            ]
        )

    topic = report.topic_forecast
    if topic:
        lines.extend(["## Topic trajectory", ""])
        if topic.get("rollup_refused"):
            lines.append(f"_Topic rollup refused:_ {topic['rollup_refused']}")
        else:
            lines.append(
                f"Demand **{topic.get('demand')}**, supply **{topic.get('supply')}** "
                f"→ quadrant **{topic.get('quadrant')}**."
            )
            if topic.get("rollup_note"):
                lines.append("")
                lines.append(topic["rollup_note"])
            provenance_line = _topic_provenance_line(topic.get("provenance") or {})
            if provenance_line:
                lines.append("")
                lines.append(provenance_line)
        if topic.get("score_refused"):
            lines.append("")
            lines.append(f"_Hazard score refused:_ {topic['score_refused']}")
        lines.append("")
        disclaimer = topic.get("disclaimer")
        if disclaimer:
            lines.append(f"_{disclaimer}_")
            lines.append("")

    lines.extend(["## Evidence", ""])
    by_kind: dict[EvidenceKind, list[EvidenceItem]] = {}
    for item in ev.items:
        by_kind.setdefault(item.kind, []).append(item)

    for kind in EvidenceKind:
        items = by_kind.get(kind)
        if not items:
            continue
        heading = _KIND_HEADINGS.get(kind, f"### {kind.value}")
        lines.append(heading)
        lines.append("")
        for item in items:
            num = ev.items.index(item) + 1
            if kind == EvidenceKind.VITAL_SIGNS:
                lines.append(f"[{num}] {item.title} — {item.url} — {item.snippet[:120]}")
                lines.append("")
                lines.extend(_vital_signs_bullets(item))
            else:
                lines.append(f"[{num}] {item.title} — {item.url} — {item.snippet[:120]}")
                note = item.metadata.get("curator_note")
                if kind == EvidenceKind.ADJACENT_PROJECT and note:
                    lines.append(f"  _Curator note, not source text:_ {note}")
        lines.append("")

    if ev.excluded:
        lines.extend(["## Retrieved but excluded", ""])
        for item in ev.excluded:
            reason = item.metadata.get("exclusion_reason", "excluded")
            lines.append(f"- {item.title} — {item.url} — _{reason}_")
        lines.append("")

    lines.extend(["## Open questions", ""])
    if ev.open_questions:
        lines.extend(f"- {q}" for q in ev.open_questions)
    else:
        lines.append("- _None recorded._")
    lines.append("")

    if report.validation_errors:
        lines.extend(["## Validation errors", ""])
        lines.extend(f"- {e}" for e in report.validation_errors)
        lines.append("")

    lines.extend(["## Sources index", ""])
    for idx, item in enumerate(ev.items, start=1):
        lines.append(f"[{idx}] {item.id} — {item.url}")
    lines.append("")

    return "\n".join(lines)


def _topic_provenance_line(prov: dict) -> str | None:
    """Citation line under the quadrant — every number must be traceable."""
    recent = prov.get("demand_recent_per_month")
    baseline = prov.get("demand_baseline_per_month")
    r1_recent = prov.get("realized_r1_rate_recent")
    r1_baseline = prov.get("realized_r1_rate_baseline")
    if recent is None and baseline is None and r1_recent is None and r1_baseline is None:
        return None

    parts: list[str] = []
    if recent is not None and baseline is not None:
        parts.append(f"Inflow {recent:.1f}/mo recent vs {baseline:.1f}/mo baseline")

    n_recent = int(prov.get("r1_sampled_n_recent") or 0)
    n_baseline = int(prov.get("r1_sampled_n_baseline") or 0)
    excluded = int(prov.get("r1_excluded_n") or 0)
    if r1_recent is not None and r1_baseline is not None:
        recent_bit = f"R1 rate {r1_recent:.2f} (n={n_recent}"
        if excluded:
            recent_bit += f", {excluded} excluded"
        recent_bit += f") vs {r1_baseline:.2f} (n={n_baseline})"
        parts.append(recent_bit)

    precision = prov.get("assignment_precision")
    if precision is not None:
        n = prov.get("assignment_precision_n")
        measured = prov.get("assignment_precision_measured_at") or "?"
        n_bit = f"n={n}, " if n is not None else ""
        parts.append(
            f"assignment precision {float(precision):.2f} "
            f"(applied to counted issues, {n_bit}{measured})"
        )

    if not parts:
        return None
    return " · ".join(parts)


def _vital_signs_bullets(item: EvidenceItem) -> list[str]:
    meta = item.metadata
    urls = meta.get("evidence_urls") or {}
    trunc_note = (
        f" (sample truncated at {meta.get('commits_fetched')} commits — trend marked unknown)"
        if meta.get("commits_truncated")
        else ""
    )
    bullets = [
        f"- Commits 3 / 6 / 12 months: {meta.get('commits_3m')} / {meta.get('commits_6m')} / "
        f"{meta.get('commits_12m')} (trend: {meta.get('trend')}){trunc_note}"
        + (f" — {urls['commits']}" if urls.get("commits") else ""),
        f"- Distinct committers (12mo sample): {meta.get('distinct_committers_12m')}; "
        f"top={meta.get('top_committer')}; "
        f"active on this path in 6mo={meta.get('top_committer_active_on_path_6m')}",
        f"- Open issues (API total_count): {meta.get('open_issues_total')}; "
        f"closed total={meta.get('closed_issues_total')}; "
        f"median open age days={meta.get('median_open_issue_age_days')}; "
        f"closure_rate={meta.get('closure_rate')}"
        + (f" — {urls['issues']}" if urls.get("issues") else ""),
        f"- CODEOWNERS present={meta.get('has_codeowners')}; "
        f"mentions path={meta.get('codeowners_mentions_path')}"
        + (f" — {urls['codeowners']}" if urls.get("codeowners") else ""),
        f"- Prototype/stale label hits in open sample: {meta.get('prototype_or_stale_label_hits')}",
    ]
    errors = meta.get("fetch_errors") or []
    if errors:
        bullets.append("- Fetch errors: " + "; ".join(str(e) for e in errors))
    return bullets
