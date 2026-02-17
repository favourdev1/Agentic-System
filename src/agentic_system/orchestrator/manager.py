from __future__ import annotations

from typing import Any, Type, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class AgentDelegateInput(BaseModel):
    agent_id: str = Field(description="The ID of the specialized agent to invoke.")
    objective: str = Field(
        description="Detailed objective or instruction for the agent."
    )
    expected_output: str = Field(
        description="Clear definition of what success looks like for this sub-task."
    )
    task_context: Optional[str] = Field(
        default=None,
        description="Additional context or data the worker needs to complete the task.",
    )


class AgentDelegateTool(BaseTool):
    """Tool that allows a manager to delegate a formal task to another agent via sub-orchestration."""

    name: str = "delegate_task"
    description: str = (
        "Delegate a specific task to a specialized agent. This triggers a full "
        "sub-orchestration process (planning/direct execution) for the worker."
    )
    args_schema: Type[BaseModel] = AgentDelegateInput
    orchestrator: Any = None

    def _run(
        self,
        agent_id: str,
        objective: str,
        expected_output: str,
        task_context: Optional[str] = None,
    ) -> str:
        """Synchronous execution."""
        print(f"\n[Manager] Delegating Task to: {agent_id}")
        print(f"[Manager] Objective: {objective}")
        print(f"[Manager] Expected Output: {expected_output}")

        if not self.orchestrator:
            return "Configuration error: Orchestrator not linked."

        try:
            # Recursive call to the full orchestration pipeline for the sub-task
            full_objective = f"TASK: {objective}\nEXPECTED OUTPUT: {expected_output}"
            if task_context:
                full_objective += f"\nCONTEXT:\n{task_context}"

            result = self.orchestrator.invoke_subtask(agent_id, full_objective)
            return result
        except Exception as e:
            return f"Error delegating to agent {agent_id}: {str(e)}"

    async def _arun(
        self,
        agent_id: str,
        objective: str,
        expected_output: str,
        task_context: Optional[str] = None,
    ) -> str:
        """Asynchronous execution."""
        if not self.orchestrator:
            return "Configuration error: Orchestrator not linked."

        try:
            full_objective = f"TASK: {objective}\nEXPECTED OUTPUT: {expected_output}"
            if task_context:
                full_objective += f"\nCONTEXT:\n{task_context}"

            result = await self.orchestrator.ainvoke_subtask(agent_id, full_objective)
            return result
        except Exception as e:
            return f"Error delegating to agent {agent_id}: {str(e)}"
