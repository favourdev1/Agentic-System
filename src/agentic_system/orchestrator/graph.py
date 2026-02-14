from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agentic_system.agents.registry import AgentRegistry, AgentSpec
from agentic_system.orchestrator.llm_factory import LLMFactory
from agentic_system.orchestrator.state import OrchestratorState
from agentic_system.agents.tool_registry import ToolRegistry


class IntentResponse(BaseModel):
    """Structured response from the LLM router."""

    selected_agent: str = Field(
        description="The ID of the agent to route the request to"
    )
    reasoning: str = Field(
        description="Short explanation for why this agent was selected"
    )


class Orchestrator:
    def __init__(self) -> None:
        self._app = self._build_graph()

    def _llm_router(self, user_input: str) -> IntentResponse:
        """Uses an LLM to semantically determine the best agent for the task."""
        llm = LLMFactory.create_chat_model()

        # Get available agents and their descriptions to provide context to the LLM
        agents = AgentRegistry.descriptions()
        agent_list_str = "\n".join(
            [f"- {name}: {desc}" for name, desc in agents.items()]
        )

        system_prompt = (
            "You are an intelligent intent classifier for an agentic system. "
            "Your task is to analyze the user's input and select the most appropriate agent from the list below.\n\n"
            "Available Agents:\n"
            f"{agent_list_str}\n\n"
            "Return the selected agent ID and a brief reasoning."
        )

        # Use structured output to guarantee valid routing
        structured_llm = llm.with_structured_output(IntentResponse)
        response = structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]
        )
        return response

    def route_node(self, state: OrchestratorState) -> OrchestratorState:
        # If a target agent is explicitly provided, skip semantic routing
        if state.get("target_agent"):
           
            return {
                "selected_agent": state["target_agent"],
                "route_reason": f"Explicitly targeted: {state['target_agent']}",
            }

        user_input = state["user_input"]
        router_result = self._llm_router(user_input)
        print(f"routing to {router_result.selected_agent} explicitly")
        print(f"reason : {router_result.reasoning}")
        return {
            "selected_agent": router_result.selected_agent,
            "route_reason": f"LLM Routing: {router_result.reasoning}",
        }

    def _resolve_agent(self, user_input: str, agent_id: str | None) -> tuple[str, str]:
        if agent_id:
            return agent_id, f"Explicitly targeted: {agent_id}"
        router_result = self._llm_router(user_input)
        return router_result.selected_agent, f"LLM Routing: {router_result.reasoning}"

    @staticmethod
    def _build_worker(spec: AgentSpec):
        llm = LLMFactory.create_chat_model()
        tools = ToolRegistry.get_tools(spec.tool_names, spec.tool_groups)
        return create_react_agent(llm, tools)

    def agent_node(self, state: OrchestratorState) -> OrchestratorState:
        spec = AgentRegistry.get_agent(state["selected_agent"])
        worker = self._build_worker(spec)
        result = worker.invoke(
            {
                "messages": [
                    SystemMessage(content=spec.system_prompt),
                    HumanMessage(content=state["user_input"]),
                ]
            }
        )

        messages = result.get("messages", [])
        text_output = messages[-1].content if messages else ""
        return {"raw_agent_output": result, "response": str(text_output)}

    @staticmethod
    def finalize_node(state: OrchestratorState) -> OrchestratorState:
        response = state.get("response", "")
        selected = state.get("selected_agent", "unknown")
        reason = state.get("route_reason", "")
        formatted = f"[agent={selected}] {response}\n\n(router: {reason})"
        return {"response": formatted}

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("route", self.route_node)
        graph.add_node("run_agent", self.agent_node)
        graph.add_node("finalize", self.finalize_node)

        graph.add_edge(START, "route")
        graph.add_edge("route", "run_agent")
        graph.add_edge("run_agent", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def invoke(self, user_input: str, agent_id: str | None = None) -> str:
        input_data = {"user_input": user_input}
        if agent_id:
            input_data["target_agent"] = agent_id
        result = self._app.invoke(input_data)
        return result.get("response", "")

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    async def astream_response(
        self, user_input: str, agent_id: str | None = None
    ) -> AsyncIterator[str]:
        selected_agent, route_reason = self._resolve_agent(user_input, agent_id)
        spec = AgentRegistry.get_agent(selected_agent)
        worker = self._build_worker(spec)

        streamed_any = False
        async for event in worker.astream_events(
            {
                "messages": [
                    SystemMessage(content=spec.system_prompt),
                    HumanMessage(content=user_input),
                ]
            },
            version="v1",
        ):
            if event.get("event") != "on_chat_model_stream":
                continue
            chunk = event.get("data", {}).get("chunk")
            text = self._chunk_to_text(chunk)
            if text:
                streamed_any = True
                yield text

        if not streamed_any:
            result = worker.invoke(
                {
                    "messages": [
                        SystemMessage(content=spec.system_prompt),
                        HumanMessage(content=user_input),
                    ]
                }
            )
            messages = result.get("messages", [])
            text_output = messages[-1].content if messages else ""
            if text_output:
                yield str(text_output)

        yield f"\n\n(router: {route_reason}, agent: {selected_agent})"

    def mermaid(self) -> str:
        return self._app.get_graph().draw_mermaid()

    def ascii_graph(self) -> str:
        return self._app.get_graph().draw_ascii()


def invoke_orchestrator(user_input: str) -> str:
    return Orchestrator().invoke(user_input)


def list_registered_agents() -> dict[str, str]:
    return AgentRegistry.descriptions()
