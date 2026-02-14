from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FileSessionStore:
    """Lightweight file-based session store with atomic writes.

    Files are isolated under a dedicated directory so this layer can be removed
    cleanly in the future.
    """

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _path(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("..", "_")
        return self._base / f"{safe}.json"

    def get_or_create(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            existing = self.load(session_id)
            if existing is not None:
                return existing

        sid = session_id or uuid.uuid4().hex
        record = {
            "session_id": sid,
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "plan": None,
            "last_run": None,
            "run_history": [],
        }
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
        record["updated_at"] = self._now_iso()
        path = self._path(session_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def build_context(self, record: dict[str, Any]) -> str:
        plan = record.get("plan") or {}
        objective = plan.get("objective", "")
        steps = plan.get("steps", []) or []

        done = [s.get("title", "") for s in steps if s.get("status") == "completed"]
        pending = [s.get("title", "") for s in steps if s.get("status") == "pending"]
        failed = [s.get("title", "") for s in steps if s.get("status") == "failed"]

        last_run = record.get("last_run") or {}
        previous_input = last_run.get("user_input", "")
        previous_response = last_run.get("response", "")

        return (
            f"Session ID: {record.get('session_id', '')}\n"
            f"Previous input: {previous_input or 'None'}\n"
            f"Previous response summary: {previous_response[:500] if previous_response else 'None'}\n"
            f"Plan objective: {objective or 'None'}\n"
            f"Completed steps: {', '.join(done) if done else 'None'}\n"
            f"Pending steps: {', '.join(pending) if pending else 'None'}\n"
            f"Failed steps: {', '.join(failed) if failed else 'None'}"
        )

    def upsert_plan(self, record: dict[str, Any], objective: str, steps: list[dict[str, Any]]) -> None:
        previous = (record.get("plan") or {}).get("steps", []) or []
        previous_by_title = {s.get("title", ""): s for s in previous}

        normalized: list[dict[str, Any]] = []
        for step in steps:
            title = step.get("title", "")
            old = previous_by_title.get(title, {})
            normalized.append(
                {
                    "title": title,
                    "instruction": step.get("instruction", ""),
                    "success_criteria": step.get("success_criteria", ""),
                    "status": old.get("status", "pending"),
                    "result": old.get("result", ""),
                }
            )

        record["plan"] = {
            "objective": objective,
            "steps": normalized,
        }

    def apply_step_results(self, record: dict[str, Any], step_results: list[dict[str, Any]]) -> None:
        plan = record.get("plan")
        if not isinstance(plan, dict):
            return
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return

        by_title = {s.get("title", ""): s for s in steps}
        for result in step_results:
            title = result.get("title", "")
            target = by_title.get(title)
            if not target:
                continue
            target["status"] = result.get("status", target.get("status", "pending"))
            target["result"] = result.get("result", target.get("result", ""))

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
    ) -> None:
        run = {
            "timestamp": self._now_iso(),
            "user_input": user_input,
            "response": response,
            "selected_agent": selected_agent,
            "execution_mode": execution_mode,
            "route_reason": route_reason,
            "execution_reason": execution_reason,
        }
        record["last_run"] = run
        history = record.get("run_history")
        if not isinstance(history, list):
            history = []
            record["run_history"] = history
        history.append(run)
        if len(history) > 20:
            record["run_history"] = history[-20:]
