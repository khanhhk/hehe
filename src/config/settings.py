import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    These settings configure server behavior, LLM parameters,
    vector storage (Chroma), caching, observability (Langfuse),
    and guardrails configuration. Default values can be overridden
    by environment variables or a .env file.
    """

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/v1"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # LiteLLM Configuration
    LITELLM_BASE_URL: str = "http://localhost:4000"
    LITELLM_API_KEY: str = "sk-llmops"
    LITELLM_MODEL: str = os.getenv(
        "LITELLM_MODEL", "groq"
    )  # Default LLM model to use via LiteLLM

    # LLM Generation Parameters
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    LLM_STREAMING: bool = False

    # RAG Configuration
    DATASET_NAME: str = os.getenv("DATASET_NAME", "environment_battery")
    CHROMA_COLLECTION_NAME: str = f"rag-pipeline-{DATASET_NAME}"
    CHROMA_PERSIST_DIR: str = str(
        PROJECT_ROOT / "infrastructure" / "storage" / "chromadb"
    )

    # Performance & Caching
    CACHE_TTL: int = 3600
    MAX_RESPONSE_LENGTH: int = 2048
    REDIS_URI: str = "localhost:6378"

    # Langfuse Observability
    LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST: Optional[str] = os.getenv("LANGFUSE_HOST")

    # Guardrails Configuration
    GUARDRAILS_RESTAPI_PATH: str = "guardrails/config_restapi"
    GUARDRAILS_SSE_PATH: str = "guardrails/config_sse"

    @property
    def llm_config(self) -> Dict[str, Any]:
        """
        Generate a dictionary of parameters for initializing LLM clients.

        Returns:
            Dict[str, Any]: A dictionary including temperature, streaming mode,
            token limits, base URL, API key, and model name.
        """
        return {
            "temperature": self.LLM_TEMPERATURE,
            "streaming": self.LLM_STREAMING,
            "max_tokens": self.LLM_MAX_TOKENS,
            "base_url": self.LITELLM_BASE_URL,
            "api_key": self.LITELLM_API_KEY,
            "model": self.LITELLM_MODEL,
        }


# Global settings instance (imported throughout the app)
SETTINGS: Settings = Settings()

# FastAPI configuration for metadata display
APP_CONFIGS: Dict[str, Any] = {
    "title": "RAG Ops - Production Architecture",
    "description": (
        "Architecture RAG system with multi-LLM provider support via LiteLLM"
    ),
    "version": "1.0.0",
    "debug": SETTINGS.DEBUG,
}
