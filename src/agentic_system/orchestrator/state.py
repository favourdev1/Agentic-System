from typing import Any, TypedDict


class OrchestratorState(TypedDict, total=False):
    user_input: str
    target_agent: str
    selected_agent: str
    route_reason: str
    response: str
    raw_agent_output: Any
