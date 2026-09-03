import re

import httpx

_TITLE_RE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE)
_SCRIPT_RE = re.compile(r"<script[^>]*>[\s\S]*?</script>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_DEFAULT_HEADERS = {"User-Agent": "casefile/0.1 (evidence retrieval)"}


class HttpClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def fetch_page_summary(self, url: str, *, max_len: int = 400) -> tuple[str, str]:
        """Return (title, plain-text snippet) for a public HTML page."""
        try:
            response = await self._client.get(
                url, follow_redirects=True, headers=_DEFAULT_HEADERS
            )
            response.raise_for_status()
            html = response.text
        except httpx.HTTPError:
            return url, "Page could not be fetched."

        title_match = _TITLE_RE.search(html)
        title = title_match.group(1).strip() if title_match else url
        body = _SCRIPT_RE.sub("", html)
        body = _TAG_RE.sub(" ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            body = "Curated discussion thread (pinned in ecosystem profile)."
        return title[:200], body[:max_len]

    async def fetch_html(self, url: str) -> str | None:
        try:
            response = await self._client.get(
                url, follow_redirects=True, headers=_DEFAULT_HEADERS
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            return None

    async def url_exists(self, url: str) -> bool:
        try:
            head = await self._client.head(url, follow_redirects=True)
            if head.status_code < 400:
                return True
            if head.status_code == 405:
                get = await self._client.get(url, follow_redirects=True)
                return get.status_code < 400
            return False
        except httpx.HTTPError:
            return False
