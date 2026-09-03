from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CASEFILE_", env_file=".env", extra="ignore")

    github_token: str | None = None
    hf_token: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_provider: str = "anthropic"
    llm_model: str | None = None
    cache_dir: Path = Path.home() / ".cache" / "casefile"
    profiles_dir: Path | None = None
    github_api_base: str = "https://api.github.com"
    http_timeout: float = 30.0

    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        if self.llm_provider == "openai":
            return "gpt-4o-mini"
        if self.llm_provider == "huggingface":
            return "openai/gpt-oss-120b:groq"
        return "claude-3-5-haiku-20241022"

    def resolved_profiles_dir(self) -> Path:
        if self.profiles_dir is not None:
            return self.profiles_dir
        return Path(__file__).resolve().parent.parent / "profiles"


def get_settings() -> Settings:
    return Settings()
