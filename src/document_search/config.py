"""Application settings, read from environment variables and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """The source checkout root (holds pyproject.toml); the working directory when installed."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").exists() else Path.cwd()


# .env and relative paths resolve against the project root, not the working directory.
PROJECT_ROOT = _project_root()
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-opus-5-5"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # Storage
    data_dir: Path = Path("data")

    # Ingestion
    max_upload_mb: int = 10
    max_documents: int = 50
    chunk_max_chars: int = 500
    chunk_overlap_chars: int = 100

    # Retrieval
    index_backend: Literal["numpy", "faiss"] = "faiss"
    embedding_model: str = "intfloat/multilingual-e5-small"
    top_k: int = 5
    # Chunks scoring below this never reach the LLM. Calibrated for multilingual-e5-small on
    # data/samples: answerable questions scored >= 0.78, off-topic ones 0.68-0.70. On-topic
    # questions the documents don't answer still score 0.76-0.83, so this only drops clear
    # misses; the citation rule in generation.py catches the rest. Re-tune for other models.
    min_score: float = 0.75

    # Generation (no provider or no key -> retrieval-only mode)
    llm_provider: Literal["openai", "anthropic", "none"] = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_model: str = ""  # empty -> provider default from DEFAULT_MODELS
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_max_tokens: int = 4000
    llm_timeout_s: float = 60.0

    # Auth (empty APP_PASSWORD disables login, for local development only)
    app_password: SecretStr | None = None
    session_secret: SecretStr | None = None
    session_max_age_s: int = 8 * 3600
    cookie_secure: Literal["auto", "true", "false"] = "auto"

    # Rate limits (requests per minute)
    login_rate_per_min: int = 5
    ask_rate_per_min: int = 20
    upload_rate_per_min: int = 30

    log_level: str = "INFO"

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index" / self.index_backend

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider, "")

    def llm_api_key(self) -> str | None:
        """The key for the selected provider, or None when it is missing or blank."""
        secret = {"openai": self.openai_api_key, "anthropic": self.anthropic_api_key}.get(
            self.llm_provider
        )
        value = secret.get_secret_value().strip() if secret else ""
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
