"""Minimal web product shell: form → assess → cited report."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from markupsafe import Markup
from markdown import markdown

from casefile.config import get_settings
from casefile.engine.orchestrator import AssessmentEngine
from casefile.eval.sample_cases import load_sample_cases
from casefile.models.assessment import AssessmentRequest
from casefile.profiles import ProfileNotFoundError, ProfileStaleError, list_profiles, load_profile
from casefile.render.markdown import render_markdown
from casefile.web.jobs import get_job, start_async_assessment


def _presets() -> dict[str, dict[str, str | int | bool]]:
    return {case_id: case.preset_dict() for case_id, case in load_sample_cases().items()}


def create_app() -> Flask:
    root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(root / "templates"),
        static_folder=str(root / "static"),
    )
    app.secret_key = os.environ.get("CASEFILE_FLASK_SECRET", "dev-only-change-in-production")

    @app.context_processor
    def inject_globals():
        settings = get_settings()
        return {
            "profiles": _list_profile_names(),
            "presets": _presets(),
            "github_configured": bool(settings.github_token),
            "llm_configured": _llm_configured(settings),
            "llm_provider": settings.llm_provider,
        }

    @app.get("/")
    def index():
        presets = _presets()
        preset_id = request.args.get("preset", "")
        preset = presets.get(preset_id, {})
        return render_template(
            "index.html",
            form={
                "question": preset.get("question", ""),
                "repo": preset.get("repo", ""),
                "path": preset.get("path", ""),
                "ecosystem": preset.get("ecosystem", ""),
                "tier": preset.get("tier", 2),
                "synthesize": preset.get("synthesize_default", True),
                "allow_stale_profile": False,
            },
            preset_id=preset_id,
            preset_blurb=preset.get("blurb", ""),
        )

    @app.get("/health")
    def health():
        return render_template("health.html", status=_connectivity_status())

    @app.post("/assess")
    def assess_submit():
        question = (request.form.get("question") or "").strip()
        repo = (request.form.get("repo") or "").strip()
        path = (request.form.get("path") or "").strip() or None
        ecosystem = (request.form.get("ecosystem") or "").strip() or None
        tier = int(request.form.get("tier") or 2)
        synthesize = request.form.get("synthesize") == "on"
        allow_stale = request.form.get("allow_stale_profile") == "on"

        if not question or not repo:
            flash("Question and repository are required.", "danger")
            return redirect(url_for("index"))

        settings = get_settings()
        if not settings.github_token:
            flash("Set CASEFILE_GITHUB_TOKEN in casefile/.env before running assessments.", "warning")

        profile = None
        if ecosystem:
            try:
                profile = load_profile(
                    settings.resolved_profiles_dir(),
                    ecosystem,
                    allow_stale=allow_stale,
                )
            except ProfileNotFoundError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("index"))
            except ProfileStaleError as exc:
                flash(f"{exc} Enable “Allow stale profile” or refresh the YAML.", "warning")
                return redirect(url_for("index"))

        req = AssessmentRequest(
            question=question,
            repo=repo,
            path=path,
            ecosystem=ecosystem,
            tier=max(1, min(3, tier)),
            synthesize=synthesize,
        )

        job_id = start_async_assessment(AssessmentEngine(settings).run(req, profile))
        return redirect(url_for("assess_status", job_id=job_id))

    @app.get("/assess/status/<job_id>")
    def assess_status(job_id: str):
        record = get_job(job_id)
        if record is None:
            flash("Unknown or expired assessment job.", "danger")
            return redirect(url_for("index"))
        if record.status == "pending":
            return render_template("status.html", job_id=job_id)
        if record.status == "error":
            flash(f"Assessment failed: {record.error}", "danger")
            return redirect(url_for("index"))
        assert record.report is not None
        return _render_result(record.report)

    return app


def _render_result(report):
    md = render_markdown(report)
    summary_html = None
    if report.summary:
        summary_html = Markup(markdown(report.summary, extensions=["extra"]))
    return render_template(
        "result.html",
        report=report,
        markdown_raw=md,
        summary_html=summary_html,
        evidence_by_kind=_group_evidence(report),
    )


def _list_profile_names() -> list[str]:
    try:
        return list_profiles(get_settings().resolved_profiles_dir())
    except Exception:  # noqa: BLE001
        return []


def _llm_configured(settings) -> bool:
    if settings.llm_provider == "openai":
        return bool(settings.openai_api_key)
    return bool(settings.anthropic_api_key)


def _connectivity_status() -> dict[str, str]:
    settings = get_settings()
    out: dict[str, str] = {}
    if not settings.github_token:
        out["github"] = "not configured"
    else:
        try:
            asyncio.run(_ping_github(settings))
            out["github"] = "ok"
        except Exception as exc:  # noqa: BLE001
            out["github"] = f"error: {exc}"

    if not _llm_configured(settings):
        out["llm"] = "not configured"
    else:
        try:
            asyncio.run(_ping_llm(settings))
            out["llm"] = "ok"
        except Exception as exc:  # noqa: BLE001
            out["llm"] = f"error: {exc}"
    return out


async def _ping_github(settings) -> None:
    import httpx

    from casefile.clients import build_clients

    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        await build_clients(settings, client).github.ping()


async def _ping_llm(settings) -> None:
    import httpx

    from casefile.clients import build_clients

    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        await build_clients(settings, client).llm.ping()


def _group_evidence(report) -> list[tuple[str, list]]:
    from casefile.models.evidence import EvidenceKind

    headings = {
        EvidenceKind.ISSUE: "Issues",
        EvidenceKind.PULL_REQUEST: "Merged PRs",
        EvidenceKind.COMMIT: "Module activity",
        EvidenceKind.FILE: "Repository files",
        EvidenceKind.DISCOURSE_THREAD: "Dev-discuss",
        EvidenceKind.ADJACENT_PROJECT: "Adjacent projects",
        EvidenceKind.PROCESS_DOC: "Process / RFC",
    }
    buckets: dict[str, list] = {}
    for item in report.evidence.items:
        label = headings.get(item.kind, item.kind.value)
        buckets.setdefault(label, []).append(item)
    return list(buckets.items())


def main() -> None:
    host = os.environ.get("CASEFILE_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CASEFILE_WEB_PORT", "5050"))
    debug = os.environ.get("CASEFILE_WEB_DEBUG", "").lower() in ("1", "true", "yes")
    create_app().run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
