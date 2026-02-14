from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agentic_system.agents.registry import AgentRegistry, AgentSpec
from agentic_system.orchestrator.llm_factory import LLMFactory
from agentic_system.orchestrator.state import OrchestratorState
from agentic_system.tools.registry import ToolRegistry


class Orchestrator:
    def __init__(self) -> None:
        self._app = self._build_graph()

    @staticmethod
    def _keyword_router(user_input: str) -> tuple[str, str]:
        lowered = user_input.lower()
        if any(
            token in lowered
            for token in ["analyze", "analysis", "compare", "tradeoff", "reason"]
        ):
            return "analysis_assistant", "Matched analysis-oriented keywords"
        return "general_assistant", "Default route"

    def route_node(self, state: OrchestratorState) -> OrchestratorState:
        # If a target agent is explicitly provided, skip keyword routing
        if state.get("target_agent"):
            return {
                "selected_agent": state["target_agent"],
                "route_reason": f"Explicitly targeted: {state['target_agent']}",
            }

        user_input = state["user_input"]
        selected_agent, reason = self._keyword_router(user_input)
        return {"selected_agent": selected_agent, "route_reason": reason}

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

    def mermaid(self) -> str:
        return self._app.get_graph().draw_mermaid()

    def ascii_graph(self) -> str:
        return self._app.get_graph().draw_ascii()


def invoke_orchestrator(user_input: str) -> str:
    return Orchestrator().invoke(user_input)


def list_registered_agents() -> dict[str, str]:
    return AgentRegistry.descriptions()
