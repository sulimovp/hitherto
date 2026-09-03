import asyncio
from pathlib import Path
from typing import Optional

import httpx
import typer

from casefile.clients import build_clients
from casefile.config import get_settings
from casefile.engine.orchestrator import AssessmentEngine
from casefile.models.assessment import AssessmentRequest
from casefile.profiles import ProfileNotFoundError, ProfileStaleError, list_profiles, load_profile
from casefile.render.markdown import render_markdown

app = typer.Typer(no_args_is_help=True, help="Evidence-first OSS contribution assessment.")


@app.command()
def ping() -> None:
    """Verify GitHub API and LLM connectivity."""
    asyncio.run(_ping())


@app.command()
def assess(
    question: str = typer.Option(..., "--question", "-q", help="Feature or contribution hypothesis."),
    repo: str = typer.Option(..., "--repo", "-r", help="Target repository owner/name."),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Module path inside the repo."),
    ecosystem: Optional[str] = typer.Option(None, "--ecosystem", "-e", help="Ecosystem profile name."),
    tier: int = typer.Option(1, "--tier", "-t", min=1, max=3, help="Retriever tier cap."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write markdown report to file."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON report to stdout."),
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="Skip LLM synthesis."),
    max_evidence: int = typer.Option(60, "--max-evidence", min=1, max=200, help="Cap on evidence items."),
    allow_stale_profile: bool = typer.Option(
        False, "--allow-stale-profile", help="Use ecosystem profile past max_age_days."
    ),
) -> None:
    """Run an evidence assessment and print or save a report."""
    asyncio.run(
        _assess(
            question=question,
            repo=repo,
            path=path,
            ecosystem=ecosystem,
            tier=tier,
            output=output,
            json_output=json_output,
            no_synthesis=no_synthesis,
            max_evidence=max_evidence,
            allow_stale_profile=allow_stale_profile,
        )
    )


@app.command("list-profiles")
def list_profiles_cmd() -> None:
    """List available ecosystem profiles."""
    settings = get_settings()
    for name in list_profiles(settings.resolved_profiles_dir()):
        typer.echo(name)


async def _ping() -> None:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        bundle = build_clients(settings, client)
        if not settings.github_token:
            typer.echo("GitHub: skipped (CASEFILE_GITHUB_TOKEN not set)", err=True)
        else:
            data = await bundle.github.ping()
            core = data.get("resources", {}).get("core", {}) if isinstance(data, dict) else {}
            remaining = core.get("remaining", "?")
            typer.echo(f"GitHub: ok (core rate limit remaining: {remaining})")

        if not bundle.llm.available:
            typer.echo("LLM: skipped (no API key for configured provider)", err=True)
        else:
            reply = await bundle.llm.ping()
            typer.echo(f"LLM ({settings.llm_provider}): ok ({reply.strip()[:40]})")


async def _assess(
    *,
    question: str,
    repo: str,
    path: str | None,
    ecosystem: str | None,
    tier: int,
    output: Path | None,
    json_output: bool,
    no_synthesis: bool,
    max_evidence: int,
    allow_stale_profile: bool,
) -> None:
    settings = get_settings()
    profile = None
    if ecosystem:
        try:
            profile = load_profile(
                settings.resolved_profiles_dir(),
                ecosystem,
                allow_stale=allow_stale_profile,
            )
        except ProfileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except ProfileStaleError as exc:
            raise typer.BadParameter(f"{exc}. Pass --allow-stale-profile to override.") from exc

    request = AssessmentRequest(
        question=question,
        repo=repo,
        path=path,
        ecosystem=ecosystem,
        tier=tier,
        synthesize=not no_synthesis,
        max_evidence=max_evidence,
    )
    engine = AssessmentEngine(settings)
    report = await engine.run(request, profile)

    if json_output and output is None:
        typer.echo(report.model_dump_json(indent=2))
        return

    md = render_markdown(report)
    if output:
        if str(output).endswith(".json"):
            output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            typer.echo(f"Wrote {output}")
        else:
            output.write_text(md, encoding="utf-8")
            typer.echo(f"Wrote {output}")
    elif json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(md)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
