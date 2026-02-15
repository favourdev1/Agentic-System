"""Pluggable session persistence backends for orchestrator runs."""

from agentic_system.config.settings import get_settings
from agentic_system.session_store.db_store import DbSessionStore
from agentic_system.session_store.file_store import FileSessionStore
from agentic_system.session_store.interface import SessionStore


def build_session_store() -> SessionStore:
    settings = get_settings()
    backend = settings.session_store_backend.strip().lower()
    if backend == "file":
        return FileSessionStore(settings.session_store_dir)
    if backend == "db":
        return DbSessionStore(auto_init=settings.database_auto_migrate)
    raise ValueError(
        f"Unsupported SESSION_STORE_BACKEND={settings.session_store_backend!r}. "
        "Use 'file' or 'db'."
    )


__all__ = ["SessionStore", "FileSessionStore", "DbSessionStore", "build_session_store"]
