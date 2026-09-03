"""GitHub client retry on rate-limit responses."""

import httpx
import pytest

from casefile.clients.github import GitHubClient
from casefile.config import Settings


@pytest.mark.asyncio
async def test_search_issues_retries_on_403(httpx_mock, tmp_path):
    httpx_mock.add_response(status_code=403)
    httpx_mock.add_response(
        status_code=200,
        json={
            "total_count": 1,
            "items": [
                {
                    "number": 1,
                    "title": "ok",
                    "html_url": "https://github.com/o/r/issues/1",
                }
            ],
        },
    )
    query = "repo:o/r is:issue retry-403-unique"
    async with httpx.AsyncClient() as client:
        gh = GitHubClient(
            Settings(github_token="t", cache_dir=tmp_path), client
        )
        items = await gh.search_issues(query)
    assert len(items) == 1
