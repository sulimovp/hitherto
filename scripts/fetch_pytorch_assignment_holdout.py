"""Fetch MaskedTensor assignment items disjoint from eval/topic_assignment/pytorch.yaml."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import yaml

from casefile.clients.github import GitHubClient
from casefile.config import get_settings
from casefile.predict.assignment_score import automatic_assign

REPO = "pytorch/pytorch"
PATH = "torch/masked"
LABELLED = Path(__file__).resolve().parents[1] / "eval" / "topic_assignment" / "pytorch.yaml"
OUT = Path(__file__).resolve().parents[1] / "eval" / "topic_assignment" / "_raw_holdout.json"

POSITIVE_QUERIES = [
    f'repo:{REPO} is:issue "torch/masked"',
    f'repo:{REPO} is:issue "torch.masked"',
    f'repo:{REPO} is:issue MaskedTensor',
    f'repo:{REPO} is:pr MaskedTensor',
    f'repo:{REPO} is:pr "torch.masked"',
]

ADJACENT_QUERIES = [
    f'repo:{REPO} is:issue masked_select in:title',
    f'repo:{REPO} is:issue masked_fill in:title',
    f'repo:{REPO} is:issue "boolean mask" in:title',
    f'repo:{REPO} is:issue torch.sparse in:title',
    f'repo:{REPO} is:issue DTensor in:title',
]


def _known_numbers() -> set[int]:
    data = yaml.safe_load(LABELLED.read_text(encoding="utf-8"))
    return {int(i["number"]) for i in data.get("items") or []}


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
        "is_pull_request": bool(item.get("pull_request")),
        "state": item.get("state"),
    }


async def _collect(
    gh: GitHubClient,
    queries: list[str],
    *,
    want_assigner: bool,
    n: int,
    exclude: set[int],
) -> list[dict]:
    seen: set[int] = set(exclude)
    out: list[dict] = []
    for query in queries:
        if len(out) >= n:
            break
        for order in ("desc", "asc"):
            if len(out) >= n:
                break
            for page_n in (1, 2):
                if len(out) >= n:
                    break
                page = await gh.search_issues_page(
                    query, per_page=100
                )
                for item in page.get("items") or []:
                    number = item.get("number")
                    if number is None or int(number) in seen:
                        continue
                    row = _row(item, query)
                    if row["assigner"] != want_assigner:
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
    exclude = _known_numbers()
    async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
        gh = GitHubClient(settings, client)
        positives = await _collect(
            gh, POSITIVE_QUERIES, want_assigner=True, n=50, exclude=exclude
        )
        adjacent = await _collect(
            gh, ADJACENT_QUERIES, want_assigner=False, n=50, exclude=exclude
        )
    payload = {
        "positives": positives,
        "adjacent": adjacent,
        "n_positives": len(positives),
        "n_adjacent": len(adjacent),
        "excluded_n": len(exclude),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {OUT} positives={len(positives)} adjacent={len(adjacent)} "
        f"excluded={len(exclude)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
