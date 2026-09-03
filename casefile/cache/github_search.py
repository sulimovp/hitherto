"""Disk cache for GitHub issue search results (reduces rate-limit pressure)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class GitHubSearchCache:
    def __init__(self, cache_dir: Path, *, ttl_seconds: int = 3600) -> None:
        self._dir = cache_dir / "github_search"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds

    def get(self, query: str) -> dict[str, Any] | list[dict] | None:
        path = self._path(query)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if time.time() - float(payload.get("ts", 0)) > self._ttl:
            return None
        if "total_count" in payload and "items" in payload:
            items = payload.get("items")
            if isinstance(items, list):
                return {"items": items, "total_count": int(payload.get("total_count") or 0)}
        items = payload.get("items")
        return items if isinstance(items, list) else None

    def set(self, query: str, page: dict[str, Any] | list[dict]) -> None:
        path = self._path(query)
        if isinstance(page, list):
            body = {"ts": time.time(), "items": page, "total_count": len(page)}
        else:
            body = {
                "ts": time.time(),
                "items": page.get("items", []),
                "total_count": int(page.get("total_count") or 0),
            }
        path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")

    def _path(self, query: str) -> Path:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]
        return self._dir / f"{digest}.json"
