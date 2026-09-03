import asyncio
import time

import httpx

from casefile.cache.github_search import GitHubSearchCache
from casefile.config import Settings

_RETRY_STATUS = frozenset({403, 429, 502, 503})
_MAX_RETRIES = 4
_GITHUB_CONCURRENCY = 2


class GitHubClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self._semaphore = asyncio.Semaphore(_GITHUB_CONCURRENCY)
        self._search_cache = GitHubSearchCache(settings.cache_dir)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._settings.github_token:
            headers["Authorization"] = f"Bearer {self._settings.github_token}"
        return headers

    async def get_json(self, path: str, *, params: dict | None = None) -> dict | list:
        url = f"{self._settings.github_api_base.rstrip('/')}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            async with self._semaphore:
                response = await self._client.get(
                    url, headers=self._headers(), params=params
                )
            if response.status_code not in _RETRY_STATUS:
                response.raise_for_status()
                return response.json()
            last_exc = httpx.HTTPStatusError(
                f"GitHub {response.status_code}",
                request=response.request,
                response=response,
            )
            wait = min(2**attempt, 8)
            retry_after = response.headers.get("Retry-After") or response.headers.get(
                "retry-after"
            )
            if retry_after and str(retry_after).isdigit():
                wait = min(int(retry_after), 20)
            else:
                reset = response.headers.get("x-ratelimit-reset") or response.headers.get(
                    "X-RateLimit-Reset"
                )
                if reset and str(reset).isdigit():
                    wait = min(max(int(reset) - int(time.time()), wait), 20)
            await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc

    async def ping(self) -> dict:
        return await self.get_json("/rate_limit")  # type: ignore[return-value]

    async def search_issues(self, query: str, *, per_page: int = 20) -> list[dict]:
        page = await self.search_issues_page(query, per_page=per_page)
        return page["items"]

    async def search_issues_page(self, query: str, *, per_page: int = 20) -> dict:
        """Return ``{"items": [...], "total_count": int}`` from the search API."""
        cached = self._search_cache.get(query)
        if cached is not None:
            if isinstance(cached, dict) and "items" in cached:
                return cached
            # Legacy cache entries were bare lists — treat total as unknown/len.
            items = cached if isinstance(cached, list) else []
            return {"items": items, "total_count": len(items)}

        data = await self.get_json(
            "/search/issues",
            params={"q": query, "sort": "updated", "order": "desc", "per_page": per_page},
        )
        assert isinstance(data, dict)
        items = data.get("items", [])
        result = {
            "items": items if isinstance(items, list) else [],
            "total_count": int(data.get("total_count") or 0),
        }
        self._search_cache.set(query, result)
        return result

    async def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        try:
            data = await self.get_json(f"/repos/{owner}/{repo}/contents/{path}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        if not isinstance(data, dict):
            return None
        import base64

        encoding = data.get("encoding")
        content = data.get("content")
        if encoding == "base64" and isinstance(content, str):
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return None

    async def list_commits(
        self,
        owner: str,
        repo: str,
        *,
        path: str,
        per_page: int = 100,
        since: str | None = None,
        max_pages: int = 5,
    ) -> tuple[list[dict], bool]:
        """Return (commits, truncated). truncated=True when max_pages was exhausted."""
        all_commits: list[dict] = []
        truncated = False
        for page in range(1, max_pages + 1):
            params: dict = {"path": path, "per_page": per_page, "page": page}
            if since:
                params["since"] = since
            data = await self.get_json(
                f"/repos/{owner}/{repo}/commits",
                params=params,
            )
            batch = data if isinstance(data, list) else []
            all_commits.extend(batch)
            if len(batch) < per_page:
                break
            if page == max_pages:
                truncated = True
        return all_commits, truncated

    async def search_issues_page_sorted(
        self,
        query: str,
        *,
        sort: str = "created",
        order: str = "desc",
        per_page: int = 30,
        page: int = 1,
    ) -> dict:
        """Like search_issues_page but with explicit sort/order. Cached by query+sort+page."""
        cache_key = f"{query}|{sort}|{order}|{per_page}|{page}"
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            if isinstance(cached, dict) and "items" in cached:
                return cached
            items = cached if isinstance(cached, list) else []
            return {"items": items, "total_count": len(items)}
        data = await self.get_json(
            "/search/issues",
            params={
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": per_page,
                "page": page,
            },
        )
        assert isinstance(data, dict)
        items = data.get("items", [])
        result = {
            "items": items if isinstance(items, list) else [],
            "total_count": int(data.get("total_count") or 0),
        }
        self._search_cache.set(cache_key, result)
        return result

    async def list_pr_files(
        self, owner: str, repo: str, number: int, *, per_page: int = 100
    ) -> list[str]:
        """Return file paths touched by a PR. Single page — PRs with >100 files are unusual."""
        try:
            data = await self.get_json(
                f"/repos/{owner}/{repo}/pulls/{number}/files",
                params={"per_page": per_page},
            )
        except httpx.HTTPStatusError:
            return []
        if not isinstance(data, list):
            return []
        return [f.get("filename", "") for f in data if isinstance(f, dict)]

    async def get_earliest_commit_date(
        self, owner: str, repo: str, *, path: str
    ) -> str | None:
        """Earliest commit date for a path, via Link:last page trick. Returns ISO string or None."""
        url = f"{self._settings.github_api_base.rstrip('/')}/repos/{owner}/{repo}/commits"
        params: dict = {"path": path, "per_page": 1}
        try:
            async with self._semaphore:
                response = await self._client.get(
                    url, headers=self._headers(), params=params
                )
            response.raise_for_status()
        except Exception:
            return None

        link = response.headers.get("Link", "")
        last_page = _parse_last_page(link)
        if last_page is not None and last_page > 1:
            params["page"] = last_page
            try:
                async with self._semaphore:
                    response = await self._client.get(
                        url, headers=self._headers(), params=params
                    )
                response.raise_for_status()
            except Exception:
                return None

        data = response.json()
        if not isinstance(data, list) or not data:
            return None
        last_commit = data[-1]
        commit = last_commit.get("commit") if isinstance(last_commit, dict) else {}
        if not isinstance(commit, dict):
            return None
        author = commit.get("author") if isinstance(commit.get("author"), dict) else {}
        return author.get("date") if isinstance(author, dict) else None

    async def list_issue_timeline(
        self,
        owner: str,
        repo: str,
        number: int,
        *,
        per_page: int = 100,
        max_pages: int = 10,
    ) -> list[dict]:
        """Issue timeline events (each carries its own created_at). Point-in-time by filter."""
        headers = {
            **self._headers(),
            # Timeline is a preview media type on older API versions; keep Accept explicit.
            "Accept": "application/vnd.github.mockingbird-preview+json",
        }
        all_events: list[dict] = []
        for page in range(1, max_pages + 1):
            url = (
                f"{self._settings.github_api_base.rstrip('/')}"
                f"/repos/{owner}/{repo}/issues/{number}/timeline"
            )
            async with self._semaphore:
                response = await self._client.get(
                    url,
                    headers=headers,
                    params={"per_page": per_page, "page": page},
                )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                break
            all_events.extend(batch)
            if len(batch) < per_page:
                break
        return all_events

    async def list_issue_reactions(
        self,
        owner: str,
        repo: str,
        number: int,
        *,
        per_page: int = 100,
        max_pages: int = 5,
    ) -> list[dict]:
        """Per-reaction objects with created_at — not the aggregate reactions summary."""
        headers = {
            **self._headers(),
            "Accept": "application/vnd.github+json, application/vnd.github.squirrel-girl-preview+json",
        }
        all_reactions: list[dict] = []
        for page in range(1, max_pages + 1):
            url = (
                f"{self._settings.github_api_base.rstrip('/')}"
                f"/repos/{owner}/{repo}/issues/{number}/reactions"
            )
            async with self._semaphore:
                response = await self._client.get(
                    url,
                    headers=headers,
                    params={"per_page": per_page, "page": page},
                )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                break
            all_reactions.extend(batch)
            if len(batch) < per_page:
                break
        return all_reactions


import re as _re

def _parse_last_page(link_header: str) -> int | None:
    match = _re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link_header)
    return int(match.group(1)) if match else None
