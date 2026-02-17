from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from agentic_system.config.settings import get_settings


class LLMFactory:
    @staticmethod
    def create_chat_model(streaming: bool = False) -> BaseChatModel:
        settings = get_settings()
        provider = settings.llm_provider.strip().lower()

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                streaming=streaming,
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                streaming=streaming,
            )

        raise ValueError("Unsupported LLM_PROVIDER. Use 'gemini' or 'openai'.")
