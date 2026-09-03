"""Fetch the 50+50 assignment-precision sample from GitHub search.

Writes eval/topic_assignment/_raw.json — gold labels are applied afterwards.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

from casefile.clients.github import GitHubClient
from casefile.config import get_settings
from casefile.predict.assignment_score import automatic_assign

REPO = "pytorch/pytorch"
PATH = "torch/masked"
OUT = Path(__file__).resolve().parents[1] / "eval" / "topic_assignment" / "_raw.json"

POSITIVE_QUERIES = [
    f'repo:{REPO} is:issue "torch/masked"',
    f'repo:{REPO} is:issue "torch.masked"',
    f'repo:{REPO} is:issue MaskedTensor',
    f'repo:{REPO} is:issue "masked tensor"',
    f'repo:{REPO} is:issue label:"module: masked operators"',
]

ADJACENT_QUERIES = [
    f'repo:{REPO} is:issue NestedTensor in:title',
    f'repo:{REPO} is:issue "torch.nested" in:title',
    f'repo:{REPO} is:issue "nested tensor" in:title',
    f'repo:{REPO} is:issue "nested_tensor_from_jagged"',
    f'repo:{REPO} is:issue FlexAttention in:title',
    f'repo:{REPO} is:issue "flex_attention" in:title',
]


def _row(item: dict, query: str) -> dict:
    title = str(item.get("title") or "")
    body = str(item.get("body") or "")
    number = int(item["number"])
    return {
        "repo": REPO,
        "number": number,
        "url": item.get("html_url"),
        "title": title,
        "body": body[:4000],
        "query": query,
        "assigner": automatic_assign(title, body, PATH),
        "state": item.get("state"),
        "labels": [
            str(lab.get("name"))
            for lab in (item.get("labels") or [])
            if isinstance(lab, dict)
        ],
    }


def _skip_clone(title: str, seen_disabled: list[bool]) -> bool:
    """Keep one DISABLED-test clone; drop the rest — they are one leak class."""
    if title.startswith("DISABLED test_"):
        if seen_disabled[0]:
            return True
        seen_disabled[0] = True
    return False


async def _collect(gh: GitHubClient, queries: list[str], *, want_assigner: bool, n: int) -> list[dict]:
    seen: set[int] = set()
    out: list[dict] = []
    seen_disabled = [False]
    for query in queries:
        if len(out) >= n:
            break
        for order in ("desc", "asc"):
            if len(out) >= n:
                break
            for page_n in (1, 2):
                if len(out) >= n:
                    break
                page = await gh.search_issues_page_sorted(
                    query, sort="created", order=order, per_page=100, page=page_n
                )
            for item in page.get("items") or []:
                if item.get("pull_request"):
                    continue
                number = item.get("number")
                if number is None or int(number) in seen:
                    continue
                row = _row(item, query)
                if row["assigner"] != want_assigner:
                    continue
                if want_assigner and _skip_clone(row["title"], seen_disabled):
                    continue
                seen.add(int(number))
                out.append(row)
                if len(out) >= n:
                    break
    return out


async def main() -> int:
    settings = get_settings()
    if not settings.github_token:
        print("CASEFILE_GITHUB_TOKEN missing", file=sys.stderr)
        return 1
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        gh = GitHubClient(settings, client)
        positives = await _collect(gh, POSITIVE_QUERIES, want_assigner=True, n=50)
        adjacent = await _collect(gh, ADJACENT_QUERIES, want_assigner=False, n=50)
    payload = {
        "positives": positives,
        "adjacent": adjacent,
        "n_positives": len(positives),
        "n_adjacent": len(adjacent),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} positives={len(positives)} adjacent={len(adjacent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
