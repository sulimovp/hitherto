from dataclasses import dataclass, field

import httpx

from casefile.clients.github import GitHubClient
from casefile.clients.http import HttpClient
from casefile.clients.huggingface import HuggingFaceClient
from casefile.clients.llm import LlmClient
from casefile.config import Settings


@dataclass
class ClientBundle:
    github: GitHubClient
    http: HttpClient
    llm: LlmClient
    huggingface: HuggingFaceClient
    stats: dict[str, int] = field(
        default_factory=lambda: {"github_api": 0, "http": 0, "llm": 0, "huggingface": 0}
    )


def build_clients(settings: Settings, httpx_client: httpx.AsyncClient | None = None) -> ClientBundle:
    client = httpx_client or httpx.AsyncClient(timeout=settings.http_timeout)
    return ClientBundle(
        github=GitHubClient(settings, client),
        http=HttpClient(client),
        llm=LlmClient(settings, client),
        huggingface=HuggingFaceClient(settings, client),
    )
