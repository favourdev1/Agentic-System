from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_record(session_id: str) -> dict[str, Any]:
    now = now_iso()
    return {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "plan": None,
        "last_run": None,
        "run_history": [],
    }


def build_context(record: dict[str, Any]) -> str:
    plan = record.get("plan") or {}
    objective = plan.get("objective", "")
    steps = plan.get("steps", []) or []

    done = [s.get("title", "") for s in steps if s.get("status") == "completed"]
    pending = [s.get("title", "") for s in steps if s.get("status") == "pending"]
    failed = [s.get("title", "") for s in steps if s.get("status") == "failed"]

    last_run = record.get("last_run") or {}
    previous_input = last_run.get("user_input", "")
    previous_response = last_run.get("response", "")

    history = record.get("run_history") or []
    recent = history[-3:] if isinstance(history, list) else []
    recent_lines: list[str] = []
    for idx, run in enumerate(recent, start=1):
        if not isinstance(run, dict):
            continue
        recent_lines.append(
            f"{idx}. user={str(run.get('user_input', ''))[:140]} | "
            f"assistant={str(run.get('response', ''))[:200]}"
        )
    recent_turns = "\n".join(recent_lines) if recent_lines else "None"

    return (
        f"Session ID: {record.get('session_id', '')}\n"
        f"Previous input: {previous_input or 'None'}\n"
        f"Previous response summary: {previous_response[:500] if previous_response else 'None'}\n"
        f"Recent turns:\n{recent_turns}\n"
        f"Plan objective: {objective or 'None'}\n"
        f"Completed steps: {', '.join(done) if done else 'None'}\n"
        f"Pending steps: {', '.join(pending) if pending else 'None'}\n"
        f"Failed steps: {', '.join(failed) if failed else 'None'}"
    )


def upsert_plan(record: dict[str, Any], objective: str, steps: list[dict[str, Any]]) -> None:
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


def apply_step_results(record: dict[str, Any], step_results: list[dict[str, Any]]) -> None:
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
    run = {
        "timestamp": now_iso(),
        "user_input": user_input,
        "response": response,
        "selected_agent": selected_agent,
        "execution_mode": execution_mode,
        "route_reason": route_reason,
        "execution_reason": execution_reason,
        "prompt_version": prompt_version or "",
    }
    record["last_run"] = run
    history = record.get("run_history")
    if not isinstance(history, list):
        history = []
        record["run_history"] = history
    history.append(run)
    if len(history) > 20:
        record["run_history"] = history[-20:]
