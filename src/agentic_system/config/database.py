from __future__ import annotations

from dataclasses import dataclass

from agentic_system.config.settings import get_settings


@dataclass(frozen=True)
class DatabaseConfig:
    """Resolved database configuration from environment settings."""

    url: str
    echo: bool
    auto_migrate: bool


def get_database_config() -> DatabaseConfig:
    settings = get_settings()
    return DatabaseConfig(
        url=settings.database_url,
        echo=settings.database_echo,
        auto_migrate=settings.database_auto_migrate,
    )
