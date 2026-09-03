"""Thin Hugging Face Hub client — discussions (and optional commits) via REST."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from casefile.config import Settings

_HF_API = "https://huggingface.co/api"
_HEADERS = {"User-Agent": "casefile/0.1 (evidence retrieval)"}


@dataclass
class HubListResult:
    items: list[dict]
    error: str | None = None


class HuggingFaceClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = dict(_HEADERS)
        token = getattr(self._settings, "hf_token", None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _api_prefix(repo_type: str) -> str:
        mapping = {"model": "models", "dataset": "datasets", "space": "spaces"}
        return mapping.get(repo_type, "models")

    async def list_discussions(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        limit: int = 20,
    ) -> HubListResult:
        prefix = self._api_prefix(repo_type)
        url = f"{_HF_API}/{prefix}/{repo_id}/discussions"
        response = await self._client.get(
            url,
            headers=self._headers(),
            params={"limit": limit},
        )
        if response.status_code in {401, 404}:
            return HubListResult([], error=f"Hub discussions unavailable ({response.status_code}) for {repo_id}")
        if response.status_code == 403:
            body = (response.text or "").lower()
            if "discussions are disabled" in body:
                return HubListResult([], error=None)  # known empty, not a failure
            return HubListResult([], error=f"Hub discussions forbidden for {repo_id}")
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            items = data.get("discussions", [])
            return HubListResult(items if isinstance(items, list) else [])
        return HubListResult([])

    async def list_commits(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        repo_type: str = "model",
        limit: int = 20,
    ) -> HubListResult:
        prefix = self._api_prefix(repo_type)
        url = f"{_HF_API}/{prefix}/{repo_id}/commits/{revision}"
        response = await self._client.get(
            url,
            headers=self._headers(),
            params={"limit": limit},
        )
        if response.status_code in {401, 403, 404}:
            return HubListResult(
                [],
                error=f"Hub commits unavailable ({response.status_code}) for {repo_id}",
            )
        response.raise_for_status()
        data = response.json()
        return HubListResult(data if isinstance(data, list) else [])

    async def get_discussion(
        self,
        repo_id: str,
        num: int,
        *,
        repo_type: str = "model",
    ) -> dict | None:
        """One Hub discussion including the opening post, or None on 404."""
        prefix = self._api_prefix(repo_type)
        url = f"{_HF_API}/{prefix}/{repo_id}/discussions/{num}"
        response = await self._client.get(url, headers=self._headers())
        if response.status_code in {401, 403, 404}:
            return None
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
