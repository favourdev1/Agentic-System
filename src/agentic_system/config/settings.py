from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(
        default="agentic-system-local", alias="LANGSMITH_PROJECT"
    )

    default_api_timeout_seconds: int = Field(
        default=20, alias="DEFAULT_API_TIMEOUT_SECONDS"
    )

    # API Configuration
    api_base_url: str = Field(default="", alias="API_BASE_URL")

    # File-based session persistence
    session_store_dir: str = Field(
        default=".agentic_sessions",
        alias="SESSION_STORE_DIR",
    )
    session_store_backend: str = Field(
        default="file",
        alias="SESSION_STORE_BACKEND",
    )

    # Database configuration
    database_url: str = Field(
        default="sqlite:///./agentic_system.db",
        alias="DATABASE_URL",
    )
    database_echo: bool = Field(
        default=False,
        alias="DATABASE_ECHO",
    )
    database_auto_migrate: bool = Field(
        default=True,
        alias="DATABASE_AUTO_MIGRATE",
    )

    # Prompt governance
    prompt_config_dir: str = Field(
        default="prompts",
        alias="PROMPT_CONFIG_DIR",
    )
    prompt_version: str = Field(
        default="",
        alias="PROMPT_VERSION",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
