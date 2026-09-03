"""Fetch Apertus assignment sample: Hub discussions + GitHub issues.

Writes eval/topic_assignment/apertus_raw.json. Gold labels applied afterwards.
Topic under test: apertus format (profile synonym cluster).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

from casefile.clients.github import GitHubClient
from casefile.clients.huggingface import HuggingFaceClient
from casefile.config import get_settings
from casefile.predict.assignment_score import automatic_assign

TOPIC = "apertus format"
SYNONYMS = (
    "apertus-format",
    "chat template",
    "tool call parser",
)
HUB_REPOS = (
    ("swiss-ai/Apertus-v1.5-8B", "model"),
    ("swiss-ai/Apertus-8B-Instruct-2509", "model"),
    ("swiss-ai/Apertus-v1.5-70B", "model"),
)
GITHUB_QUERIES = (
    "repo:swiss-ai/apertus-format is:issue",
    "repo:swiss-ai/apertus-format is:pr",
    "apertus-format in:title",
    "repo:huggingface/transformers Apertus in:title",
    "repo:vllm-project/vllm Apertus in:title",
    'repo:vllm-project/vllm "Apertus Tool Parser"',
    "repo:vllm-project/vllm Apertus --tool-call-parser",
    "repo:ggml-org/llama.cpp Apertus in:title",
)

OUT = Path(__file__).resolve().parents[1] / "eval" / "topic_assignment" / "apertus_raw.json"


def _assigned(title: str, body: str, *, repo: str = "") -> bool:
    return automatic_assign(title, body, TOPIC, synonyms=SYNONYMS, repo=repo)


def _mentions_apertus(title: str, body: str, repo: str) -> bool:
    if "apertus-format" in (repo or "").lower():
        return True
    hay = f"{title}\n{body}".lower()
    return "apertus" in hay


def _hub_body(detail: dict | None) -> str:
    if not isinstance(detail, dict):
        return ""
    texts: list[str] = []
    for event in detail.get("events") or []:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "comment":
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        latest = data.get("latest") if isinstance(data.get("latest"), dict) else {}
        raw = latest.get("raw") or data.get("raw") or ""
        if raw:
            texts.append(str(raw))
        if len(texts) >= 3:
            break
    return "\n\n".join(texts)[:4000]


async def _hub_rows(hf: HuggingFaceClient) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for repo_id, repo_type in HUB_REPOS:
        listed = await hf.list_discussions(repo_id, repo_type=repo_type, limit=100)
        for disc in listed.items:
            num = disc.get("num")
            if num is None:
                continue
            key = (repo_id, int(num))
            if key in seen:
                continue
            seen.add(key)
            detail = await hf.get_discussion(repo_id, int(num), repo_type=repo_type)
            title = str((detail or disc).get("title") or disc.get("title") or "")
            body = _hub_body(detail)
            rows.append(
                {
                    "source": "hub",
                    "repo": repo_id,
                    "number": int(num),
                    "url": f"https://huggingface.co/{repo_id}/discussions/{num}",
                    "title": title,
                    "body": body,
                    "assigner": _assigned(title, body, repo=repo_id),
                    "is_pull_request": bool(disc.get("isPullRequest")),
                    "status": disc.get("status"),
                }
            )
    return rows


async def _github_rows(gh: GitHubClient) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for query in GITHUB_QUERIES:
        for page_n in (1,):
            try:
                page = await gh.search_issues_page_sorted(
                    query, sort="created", order="desc", per_page=100, page=page_n
                )
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 403:
                    break
                raise
            for item in page.get("items") or []:
                url = str(item.get("html_url") or "")
                number = item.get("number")
                if number is None:
                    continue
                repo = "/".join(url.split("/")[3:5]) if "github.com" in url else "unknown"
                key = (repo, int(number))
                if key in seen:
                    continue
                seen.add(key)
                title = str(item.get("title") or "")
                body = str(item.get("body") or "")[:4000]
                if not _mentions_apertus(title, body, repo):
                    continue
                rows.append(
                    {
                        "source": "github",
                        "repo": repo,
                        "number": int(number),
                        "url": url,
                        "title": title,
                        "body": body,
                        "assigner": _assigned(title, body, repo=repo),
                        "query": query,
                    }
                )
    return rows


async def main() -> int:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        hf = HuggingFaceClient(settings, client)
        gh = GitHubClient(settings, client)
        hub = await _hub_rows(hf)
        github = await _github_rows(gh)
    payload = {
        "topic": TOPIC,
        "synonyms": list(SYNONYMS),
        "hub": hub,
        "github": github,
        "n_hub": len(hub),
        "n_github": len(github),
        "n_assigner_true": sum(1 for r in hub + github if r["assigner"]),
        "n_assigner_false": sum(1 for r in hub + github if not r["assigner"]),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {OUT} hub={len(hub)} github={len(github)} "
        f"assigner_true={payload['n_assigner_true']} "
        f"assigner_false={payload['n_assigner_false']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
