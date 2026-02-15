from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from agentic_system.database.init_db import init_database
from agentic_system.database.session import session_scope
from agentic_system.models.session_record import SessionRecord
from . import record_ops


class DbSessionStore:
    """SQL-backed session persistence with the same contract as FileSessionStore."""

    def __init__(self, *, auto_init: bool = True) -> None:
        if auto_init:
            init_database()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _parse_iso(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)

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
        with session_scope() as db:
            row = db.scalar(
                select(SessionRecord).where(SessionRecord.session_id == session_id)
            )
            if row is None:
                return None
            try:
                record = json.loads(row.payload)
                if isinstance(record, dict):
                    record.setdefault("session_id", session_id)
                    return record
            except Exception:  # noqa: BLE001
                return None
        return None

    def save(self, record: dict[str, Any]) -> None:
        session_id = str(record["session_id"])
        record["updated_at"] = self._now_iso()

        payload = json.dumps(record, ensure_ascii=True)
        created_at = self._parse_iso(record.get("created_at"))
        updated_at = self._parse_iso(record.get("updated_at"))

        with session_scope() as db:
            existing = db.scalar(
                select(SessionRecord).where(SessionRecord.session_id == session_id)
            )
            if existing is None:
                db.add(
                    SessionRecord(
                        session_id=session_id,
                        payload=payload,
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                )
            else:
                existing.payload = payload
                existing.updated_at = updated_at

    def build_context(self, record: dict[str, Any]) -> str:
        return record_ops.build_context(record)

    def upsert_plan(self, record: dict[str, Any], objective: str, steps: list[dict[str, Any]]) -> None:
        record_ops.upsert_plan(record, objective, steps)

    def apply_step_results(self, record: dict[str, Any], step_results: list[dict[str, Any]]) -> None:
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
