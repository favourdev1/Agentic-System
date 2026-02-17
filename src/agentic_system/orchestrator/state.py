from typing import Any, TypedDict


class OrchestratorState(TypedDict, total=False):
    user_input: str
    session_id: str
    session_context: str
    target_agent: str
    plan_step_budget: int
    selected_agent: str
    route_reason: str
    execution_mode: str
    execution_reason: str
    plan_objective: str
    plan_steps: list[dict[str, str]]
    step_results: list[dict[str, str]]
    response: str
    raw_agent_output: Any
    # True Hierarchical Flow fields
    subtasks: list[dict[str, Any]]
    recursion_depth: int
