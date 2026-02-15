from __future__ import annotations

from typing import Any, Protocol


class SessionStore(Protocol):
    """Shared contract for pluggable session persistence backends."""

    def get_or_create(self, session_id: str | None = None) -> dict[str, Any]:
        ...

    def load(self, session_id: str) -> dict[str, Any] | None:
        ...

    def save(self, record: dict[str, Any]) -> None:
        ...

    def build_context(self, record: dict[str, Any]) -> str:
        ...

    def upsert_plan(self, record: dict[str, Any], objective: str, steps: list[dict[str, Any]]) -> None:
        ...

    def apply_step_results(self, record: dict[str, Any], step_results: list[dict[str, Any]]) -> None:
        ...

    def set_last_run(
        self,
        record: dict[str, Any],
        *,
        user_input: str,
        response: str,
        selected_agent: str,
        execution_mode: str,
        route_reason: str,
        execution_reason: str,
        prompt_version: str | None = None,
    ) -> None:
        ...
