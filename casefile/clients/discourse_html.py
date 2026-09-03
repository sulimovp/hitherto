"""Parse Discourse search result pages for topic URLs."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

_TOPIC_HREF_RE = re.compile(
    r'href="((?:https?://[^"/]+)?/t/[^"?#]+/\d+)"',
    re.IGNORECASE,
)


def extract_discourse_topic_urls(html: str, base_url: str, *, limit: int = 5) -> list[str]:
    """Return absolute topic URLs from a Discourse /search HTML page."""
    base = base_url.rstrip("/")
    seen: set[str] = set()
    urls: list[str] = []
    for match in _TOPIC_HREF_RE.finditer(html):
        href = match.group(1)
        absolute = href if href.startswith("http") else urljoin(f"{base}/", href.lstrip("/"))
        parsed = urlparse(absolute)
        if "/t/" not in parsed.path:
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        if "/search" in normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
        if len(urls) >= limit:
            break
    return urls
