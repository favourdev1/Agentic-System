from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from agentic_system.config.database import get_database_config


@lru_cache
def get_engine() -> Engine:
    """Build and cache a shared SQLAlchemy engine."""
    config = get_database_config()
    kwargs: dict[str, object] = {
        "echo": config.echo,
        "future": True,
        "pool_pre_ping": True,
    }

    # SQLite needs this for multithreaded app servers.
    if config.url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}

    return create_engine(config.url, **kwargs)
