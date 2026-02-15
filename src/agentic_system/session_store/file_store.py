from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from . import record_ops


class FileSessionStore:
    """Lightweight file-based session store with atomic writes.

    Files are isolated under a dedicated directory so this layer can be removed
    cleanly in the future.
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("..", "_")
        return self._base / f"{safe}.json"

    def get_or_create(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            existing = self.load(session_id)
            if existing is not None:
                return existing

        sid = session_id or uuid.uuid4().hex
        record = record_ops.default_record(sid)
        self.save(record)
        return record

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def save(self, record: dict[str, Any]) -> None:
        session_id = record["session_id"]
        record["updated_at"] = record_ops.now_iso()
        path = self._path(session_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        tmp_path.replace(path)

    def build_context(self, record: dict[str, Any]) -> str:
        return record_ops.build_context(record)

    def upsert_plan(
        self, record: dict[str, Any], objective: str, steps: list[dict[str, Any]]
    ) -> None:
        record_ops.upsert_plan(record, objective, steps)

    def apply_step_results(
        self, record: dict[str, Any], step_results: list[dict[str, Any]]
    ) -> None:
        record_ops.apply_step_results(record, step_results)

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
        record_ops.set_last_run(
            record,
            user_input=user_input,
            response=response,
            selected_agent=selected_agent,
            execution_mode=execution_mode,
            route_reason=route_reason,
            execution_reason=execution_reason,
            prompt_version=prompt_version,
        )
