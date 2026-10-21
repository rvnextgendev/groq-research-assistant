"""Validated configuration with credentials excluded from repr."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from research_assistant.errors import ConfigurationError

# Resolve relative to this module so startup does not depend on the shell directory.
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings with conservative document and context limits."""
    # Exclude keys from repr to avoid accidental disclosure when debugging settings.
    groq_api_key: str = field(repr=False)
    serper_api_key: str = field(repr=False)
    model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_upload_mb: int = 10
    max_pages: int = 200
    max_text_chars: int = 1_000_000
    chunk_size: int = 1000
    # Overlap preserves context around chunk boundaries during retrieval.
    chunk_overlap: int = 200
    retrieval_k: int = 4
    max_question_chars: int = 4000
    max_history_chars: int = 12000

    def __post_init__(self):
        """Fail before loading models or making paid requests if settings are invalid."""
        if not self.groq_api_key.strip() or not self.serper_api_key.strip():
            raise ConfigurationError("Configure GROQ_API_KEY and SERPER_API_KEY before starting.")
        if not self.model.strip():
            raise ConfigurationError("LLM_MODEL must not be empty.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigurationError("Chunk overlap must be smaller than chunk size.")
        limits = (self.max_upload_mb, self.max_pages, self.max_text_chars,
                  self.retrieval_k, self.max_question_chars, self.max_history_chars)
        if any(value <= 0 for value in limits):
            raise ConfigurationError("Resource limits must be positive.")

    @classmethod
    def load(cls, secrets: Mapping | None = None) -> "Settings":
        """Read local .env, environment, then optional Streamlit secrets."""
        # dotenv leaves existing process environment variables unchanged by default.
        load_dotenv(ROOT / ".env")
        # Deployment secrets take precedence when the same key exists in both sources.
        values = {**os.environ, **dict(secrets or {})}
        if str(values.get("LLM_PROVIDER", "groq")).lower() != "groq":
            raise ConfigurationError("LLM_PROVIDER must be groq.")
        return cls(groq_api_key=str(values.get("GROQ_API_KEY", "")),
                   serper_api_key=str(values.get("SERPER_API_KEY", "")),
                   model=str(values.get("LLM_MODEL", "llama-3.3-70b-versatile")))
