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


class StreamProcessor:
    """Helper to transform raw LangGraph events into user-friendly SSE payloads."""

    def __init__(self, selected_agent: str, route_reason: str) -> None:
        self.selected_agent = selected_agent
        self.route_reason = route_reason
        self.streamed_any = False
        self.final_output_text = ""
        self._tool_map = {
            "calculator": "Consulting the calculator...",
            "bank_account_api": "Checking bank records...",
            "external_search_api": "Searching the web...",
        }

    @staticmethod
    def chunk_to_text(chunk: Any) -> str:
        """Extracts plain text from various chat model chunk formats."""
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

    def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Translates a raw LangGraph event into a structured payload."""
        event_type = event.get("event")

        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            text = self.chunk_to_text(chunk)
            if text:
                self.streamed_any = True
                return {"type": "token", "content": text}

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "tool")
            msg = self._tool_map.get(tool_name, f"Using {tool_name}...")
            return {"type": "status", "content": msg}

        elif event_type == "on_tool_end":
            tool_name = event.get("name", "tool")
            return {"type": "status", "content": f"Finished using {tool_name}."}

        elif event_type == "on_chain_end":
            output = event.get("data", {}).get("output")
            text = self._extract_output_text(output)
            if text:
                self.final_output_text = text

        return None

    @classmethod
    def _extract_output_text(cls, output: Any) -> str:
        if not isinstance(output, dict):
            return ""
        messages = output.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""
        last = messages[-1]
        return cls.chunk_to_text(last)

    def get_metadata(self) -> dict[str, Any]:
        """Returns the final metadata payload."""
        return {
            "type": "metadata",
            "route_reason": self.route_reason,
            "agent": self.selected_agent,
        }


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
        return {
            "selected_agent": router_result.selected_agent,
            "route_reason": f"LLM Routing: {router_result.reasoning}",
        }

    def _resolve_agent(self, user_input: str, agent_id: str | None) -> tuple[str, str]:
        if agent_id:
            return agent_id, f"Explicitly targeted: {agent_id}"
        router_result = self._llm_router(user_input)
        selected_agent = router_result.selected_agent
        if selected_agent not in AgentRegistry.descriptions():
            selected_agent = "general_assistant"
            return selected_agent, "LLM Routing fallback: invalid agent id from router"
        return selected_agent, f"LLM Routing: {router_result.reasoning}"

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

    async def astream_response(
        self, user_input: str, agent_id: str | None = None, trace_tools: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        """Streams the agent execution as a series of structured events."""
        selected_agent, route_reason = self._resolve_agent(user_input, agent_id)
        spec = AgentRegistry.get_agent(selected_agent)
        worker = self._build_worker(spec)

        processor = StreamProcessor(selected_agent, route_reason)

        # Signal initial routing
        yield {"type": "status", "content": f"Thinking... (Routing: {selected_agent})"}

        async for event in worker.astream_events(
            {
                "messages": [
                    SystemMessage(content=spec.system_prompt),
                    HumanMessage(content=user_input),
                ]
            },
            version="v1",
        ):
            payload = processor.process_event(event)
            if payload and payload.get("type") == "status" and not trace_tools:
                continue
            if payload:
                yield payload

        # Fallback for non-streaming models: use captured final output from stream events.
        if not processor.streamed_any:
            if processor.final_output_text:
                yield {"type": "token", "content": processor.final_output_text}
            else:
                yield {
                    "type": "status",
                    "content": "No token stream available from this model/provider for this run.",
                }

        # Output final metadata
        yield processor.get_metadata()

    def mermaid(self) -> str:
        return self._app.get_graph().draw_mermaid()

    def ascii_graph(self) -> str:
        return self._app.get_graph().draw_ascii()


def invoke_orchestrator(user_input: str) -> str:
    return Orchestrator().invoke(user_input)


def list_registered_agents() -> dict[str, str]:
    return AgentRegistry.descriptions()
