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

    # Bank API Configuration
    bank_api_base_url: str = Field(
        default="http://localhost:8000/api/v2/bank-account/request",
        alias="BANK_API_BASE_URL",
    )
    bank_api_auth_token: str = Field(default="", alias="BANK_API_AUTH_TOKEN")
    bank_api_session_cookie: str = Field(default="", alias="BANK_API_SESSION_COOKIE")


@lru_cache
def get_settings() -> Settings:
    return Settings()
