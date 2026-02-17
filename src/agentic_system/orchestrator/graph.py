from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from agentic_system.agents.registry import AgentRegistry, AgentSpec
from agentic_system.config.settings import get_settings
from agentic_system.orchestrator.llm_factory import LLMFactory
from agentic_system.orchestrator.state import OrchestratorState
from agentic_system.orchestrator.ui_models import UiSpec
from agentic_system.prompting import PromptManager
from agentic_system.session_store import build_session_store
from agentic_system.tools.registry import ToolRegistry


class IntentResponse(BaseModel):
    selected_agent: str = Field(description="Agent ID selected for this request")
    reasoning: str = Field(description="Short reason for agent selection")


class ExecutionDecision(BaseModel):
    mode: Literal["direct", "plan"] = Field(
        description="Execution strategy. Use direct for single-pass tasks, plan for multi-step tasks."
    )
    reason: str = Field(description="Brief reason for choosing the strategy")


class PlanStep(BaseModel):
    title: str = Field(description="Short step title")
    instruction: str = Field(
        description="Actionable instruction for the selected agent"
    )
    success_criteria: str = Field(description="Observable completion criteria")


class ExecutionPlan(BaseModel):
    objective: str = Field(description="Execution objective")
    steps: list[PlanStep] = Field(description="Ordered executable steps")


class StreamProcessor:
    """Manages the translation of LangGraph/LangChain events into user-facing stream updates.

    The processor enforces a formal Trace Lifecycle:
    1. 'status': Real-time operational updates (reasoning, tool starts/ends).
    2. 'token': Incremental content generation (partial model outputs).
    3. 'metadata': Final trace information and session context.
    4. 'plan' & 'step_result': Specialized events for multi-step execution.
    """

    def __init__(self) -> None:
        self.streamed_any = False
        self.final_output_text = ""

    @classmethod
    def chunk_to_text(cls, chunk: Any) -> str:
        content = getattr(chunk, "content", chunk)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)

        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text

        return ""

    @classmethod
    def _extract_output_text(cls, output: Any) -> str:
        if not isinstance(output, dict):
            return ""
        messages = output.get("messages")
        if not isinstance(messages, list) or not messages:
            return ""
        return cls.chunk_to_text(messages[-1])

    def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Formal event handler for LangChain 'astream_events'.

        Args:
            event: The raw event dict from the runtime.

        Returns:
            A normalized payload dict if the event matches the trace lifecycle, else None.
        """
        event_type = event.get("event")

        # Lifecycle Phase: Content Generation
        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            text = self.chunk_to_text(chunk)
            if text:
                self.streamed_any = True
                return {"type": "token", "content": text}

        # Lifecycle Phase: Tool Operations (Trace Triggers)
        if event_type == "on_tool_start":
            tool_name = event.get("name", "tool")
            msg = ToolRegistry.get_status_message(tool_name)
            return {"type": "status", "content": msg}

        if event_type == "on_tool_end":
            tool_name = event.get("name", "tool")
            return {"type": "status", "content": f"Finished using {tool_name}."}

        # Lifecycle Phase: Synthesis
        if event_type == "on_chain_end":
            output = event.get("data", {}).get("output")
            text = self._extract_output_text(output)
            if text:
                self.final_output_text = text

        return None


class Orchestrator:
    """The central intelligence hub of the Agentic System.

    The Orchestrator manages the end-to-end lifecycle of a user request:
    1. Governance: Loads versioned prompts and manages session persistence.
    2. Intent Routing: Selects the optimal agent via the LLM Router.
    3. Strategy Selection: Decides between 'Direct' or 'Plan' execution.
    4. Execution: Runs the selected agent or executes a multi-step plan.
    5. Post-processing: Optionally generates UI payloads.
    """

    def __init__(self) -> None:
        self._app = self._build_graph()
        settings = get_settings()
        self._store = build_session_store()
        self._prompts = PromptManager(
            settings.prompt_config_dir,
            version_override=settings.prompt_version or None,
        )

    @staticmethod
    def _safe_agent_id(candidate: str) -> str:
        return (
            candidate
            if candidate in AgentRegistry.descriptions()
            else "general_assistant"
        )

    def _prepare_session(
        self, session_id: str | None
    ) -> tuple[str, str, dict[str, Any]]:
        record = self._store.get_or_create(session_id)
        sid = record["session_id"]
        context = self._store.build_context(record)
        return sid, context, record

    def _llm_router(self, user_input: str, session_context: str = "") -> IntentResponse:
        llm = LLMFactory.create_chat_model()
        agents = AgentRegistry.descriptions()
        agent_list = "\n".join([f"- {name}: {desc}" for name, desc in agents.items()])
        system_prompt = self._prompts.get_prompt("router_system", agent_list=agent_list)
        user_prompt = self._prompts.get_prompt(
            "router_user",
            user_input=user_input,
            session_context=session_context or "None",
        )
        structured_llm = llm.with_structured_output(IntentResponse)
        result = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        result.selected_agent = self._safe_agent_id(result.selected_agent)
        return result

    def _decide_mode(
        self,
        user_input: str,
        selected_agent: str,
        target_agent: str | None,
        session_context: str,
    ) -> ExecutionDecision:
        if target_agent:
            return ExecutionDecision(
                mode="direct",
                reason="Explicit target agent supplied; bypass planning by design.",
            )

        llm = LLMFactory.create_chat_model()
        spec = AgentRegistry.get_agent(selected_agent)
        tools = ToolRegistry.resolve_tool_names(spec.tool_names, spec.tool_groups)
        system_prompt = self._prompts.get_prompt("mode_system")
        context_prompt = self._prompts.get_prompt(
            "mode_user",
            selected_agent=selected_agent,
            agent_description=spec.description,
            available_tools=", ".join(tools) if tools else "none",
            user_input=user_input,
            session_context=session_context or "None",
        )

        structured_llm = llm.with_structured_output(ExecutionDecision)
        return structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context_prompt),
            ]
        )

    def _build_plan(
        self, user_input: str, selected_agent: str, session_context: str
    ) -> ExecutionPlan:
        llm = LLMFactory.create_chat_model()
        spec = AgentRegistry.get_agent(selected_agent)
        tools = ToolRegistry.resolve_tool_names(spec.tool_names, spec.tool_groups)
        system_prompt = self._prompts.get_prompt("plan_system")
        context_prompt = self._prompts.get_prompt(
            "plan_user",
            selected_agent=selected_agent,
            agent_description=spec.description,
            available_tools=", ".join(tools) if tools else "none",
            user_input=user_input,
            session_context=session_context or "None",
        )

        structured_llm = llm.with_structured_output(ExecutionPlan)
        plan = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=context_prompt)]
        )

        normalized_steps = plan.steps[:6]
        if len(normalized_steps) < 2:
            normalized_steps = [
                PlanStep(
                    title="Execute request",
                    instruction="Complete the user request directly with available tools.",
                    success_criteria="A complete and accurate response is produced.",
                )
            ]
        return ExecutionPlan(
            objective=plan.objective or user_input, steps=normalized_steps
        )

    def _build_ui_spec(self, user_input: str, response_text: str) -> UiSpec | None:
        # UI generation is optional and runs as a post-processing pass over the final text response.
        if not response_text.strip():
            return None

        llm = LLMFactory.create_chat_model()
        system_prompt = self._prompts.get_prompt("ui_system")
        user_prompt = self._prompts.get_prompt(
            "ui_user",
            user_input=user_input,
            response_text=response_text,
        )
        structured_llm = llm.with_structured_output(UiSpec)
        ui = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        if ui.layout == "none" and not ui.elements:
            return None
        return ui

    @staticmethod
    def _build_worker(spec: AgentSpec, streaming: bool = False):
        llm = LLMFactory.create_chat_model(streaming=streaming)
        tools = ToolRegistry.get_tools(spec.tool_names, spec.tool_groups)
        return create_react_agent(llm, tools)

    @staticmethod
    def _extract_result_text(result: dict[str, Any]) -> str:
        messages = result.get("messages", [])
        if not messages:
            return ""
        content = getattr(messages[-1], "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return str(content)

    def route_node(self, state: OrchestratorState) -> OrchestratorState:
        if state.get("target_agent"):
            selected = self._safe_agent_id(state["target_agent"])
            return {
                "selected_agent": selected,
                "route_reason": f"Explicitly targeted: {selected}",
            }

        router_result = self._llm_router(
            state["user_input"],
            session_context=state.get("session_context", ""),
        )
        return {
            "selected_agent": router_result.selected_agent,
            "route_reason": f"LLM Routing: {router_result.reasoning}",
        }

    def decide_mode_node(self, state: OrchestratorState) -> OrchestratorState:
        decision = self._decide_mode(
            user_input=state["user_input"],
            selected_agent=state["selected_agent"],
            target_agent=state.get("target_agent"),
            session_context=state.get("session_context", ""),
        )
        return {
            "execution_mode": decision.mode,
            "execution_reason": decision.reason,
        }

    def plan_node(self, state: OrchestratorState) -> OrchestratorState:
        plan = self._build_plan(
            state["user_input"],
            state["selected_agent"],
            session_context=state.get("session_context", ""),
        )
        steps = [
            {
                "title": step.title,
                "instruction": step.instruction,
                "success_criteria": step.success_criteria,
            }
            for step in plan.steps
        ]
        return {"plan_objective": plan.objective, "plan_steps": steps}

    def agent_node(self, state: OrchestratorState) -> OrchestratorState:
        spec = AgentRegistry.get_agent(state["selected_agent"])
        worker = self._build_worker(spec, streaming=False)
        result = worker.invoke(
            {
                "messages": [
                    SystemMessage(content=spec.runtime_system_prompt()),
                    HumanMessage(content=state["user_input"]),
                ]
            }
        )
        return {
            "raw_agent_output": result,
            "response": self._extract_result_text(result),
            "step_results": [],
        }

    def execute_plan_node(self, state: OrchestratorState) -> OrchestratorState:
        spec = AgentRegistry.get_agent(state["selected_agent"])
        worker = self._build_worker(spec, streaming=False)

        plan_steps = state.get("plan_steps", [])
        budget = max(1, int(state.get("plan_step_budget") or len(plan_steps)))

        step_results: list[dict[str, str]] = [
            {
                "title": step.get("title", ""),
                "status": "pending",
                "result": "",
            }
            for step in plan_steps
        ]
        completed: list[dict[str, str]] = []
        stopped_early = False

        for index, step in enumerate(plan_steps, start=1):
            if index > budget:
                stopped_early = True
                break

            previous = "\n".join(
                [
                    f"{i+1}. {item['title']}: {item['result']}"
                    for i, item in enumerate(completed)
                ]
            )
            step_prompt = self._prompts.get_prompt(
                "step_user",
                user_input=state["user_input"],
                plan_objective=state.get("plan_objective", state["user_input"]),
                step_index=index,
                step_count=len(plan_steps),
                step_title=step["title"],
                step_instruction=step["instruction"],
                step_success_criteria=step["success_criteria"],
                completed_context=previous if previous else "None yet",
            )

            try:
                step_result = worker.invoke(
                    {
                        "messages": [
                            SystemMessage(content=spec.runtime_system_prompt()),
                            HumanMessage(content=step_prompt),
                        ]
                    }
                )
                text = self._extract_result_text(step_result)
                step_results[index - 1]["status"] = "completed"
                step_results[index - 1]["result"] = text
                completed.append({"title": step["title"], "result": text})
            except Exception as exc:  # noqa: BLE001
                step_results[index - 1]["status"] = "failed"
                step_results[index - 1]["result"] = str(exc)
                stopped_early = True
                break

        all_completed = all(step["status"] == "completed" for step in step_results)

        if all_completed:
            synthesis_prompt = self._prompts.get_prompt(
                "synthesis_user",
                user_input=state["user_input"],
                plan_objective=state.get("plan_objective", state["user_input"]),
                completed_steps="\n".join(
                    [f"- {item['title']}: {item['result']}" for item in completed]
                ),
            )
            final_result = worker.invoke(
                {
                    "messages": [
                        SystemMessage(content=spec.runtime_system_prompt()),
                        HumanMessage(content=synthesis_prompt),
                    ]
                }
            )
            response = self._extract_result_text(final_result)
            return {
                "step_results": step_results,
                "raw_agent_output": final_result,
                "response": response,
            }

        done = [x["title"] for x in step_results if x["status"] == "completed"]
        pending = [x["title"] for x in step_results if x["status"] == "pending"]
        failed = [x["title"] for x in step_results if x["status"] == "failed"]

        response = (
            "Plan execution progress update:\n"
            f"- Completed: {', '.join(done) if done else 'None'}\n"
            f"- Pending: {', '.join(pending) if pending else 'None'}\n"
            f"- Failed: {', '.join(failed) if failed else 'None'}\n"
        )
        if stopped_early and budget < len(plan_steps):
            response += f"- Note: execution paused by step budget ({budget})."

        return {
            "step_results": step_results,
            "response": response,
            "raw_agent_output": {"messages": []},
        }

    @staticmethod
    def _mode_edge(state: OrchestratorState) -> str:
        return "plan" if state.get("execution_mode") == "plan" else "direct"

    @staticmethod
    def finalize_node(state: OrchestratorState) -> OrchestratorState:
        response = state.get("response", "")
        selected = state.get("selected_agent", "unknown")
        route_reason = state.get("route_reason", "")
        execution_mode = state.get("execution_mode", "direct")
        execution_reason = state.get("execution_reason", "")

        suffix = f"(router: {route_reason}; mode: {execution_mode}; reason: {execution_reason}; agent: {selected})"
        return {"response": f"{response}\n\n{suffix}"}

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("route", self.route_node)
        graph.add_node("decide_mode", self.decide_mode_node)
        graph.add_node("build_plan", self.plan_node)
        graph.add_node("run_direct", self.agent_node)
        graph.add_node("run_plan", self.execute_plan_node)
        graph.add_node("finalize", self.finalize_node)

        graph.add_edge(START, "route")
        graph.add_edge("route", "decide_mode")
        graph.add_conditional_edges(
            "decide_mode",
            self._mode_edge,
            {
                "direct": "run_direct",
                "plan": "build_plan",
            },
        )
        graph.add_edge("build_plan", "run_plan")
        graph.add_edge("run_direct", "finalize")
        graph.add_edge("run_plan", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _persist_session(
        self,
        *,
        session_id: str,
        user_input: str,
        response: str,
        selected_agent: str,
        execution_mode: str,
        route_reason: str,
        execution_reason: str,
        prompt_version: str,
        plan_objective: str | None,
        plan_steps: list[dict[str, Any]] | None,
        step_results: list[dict[str, Any]] | None,
    ) -> None:
        record = self._store.get_or_create(session_id)
        if plan_steps:
            self._store.upsert_plan(record, plan_objective or user_input, plan_steps)
        if step_results:
            self._store.apply_step_results(record, step_results)

        self._store.set_last_run(
            record,
            user_input=user_input,
            response=response,
            selected_agent=selected_agent,
            execution_mode=execution_mode,
            route_reason=route_reason,
            execution_reason=execution_reason,
            prompt_version=prompt_version,
        )
        self._store.save(record)

    def invoke_with_metadata(
        self,
        user_input: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        plan_step_budget: int | None = None,
        generate_ui: bool = False,
    ) -> dict[str, Any]:
        sid, session_context, _ = self._prepare_session(session_id)

        input_data: OrchestratorState = {
            "user_input": user_input,
            "session_id": sid,
            "session_context": session_context,
        }
        if agent_id:
            input_data["target_agent"] = agent_id
        if plan_step_budget:
            input_data["plan_step_budget"] = plan_step_budget

        result = self._app.invoke(input_data)

        response = result.get("response", "")
        selected_agent = result.get("selected_agent", "general_assistant")
        execution_mode = result.get("execution_mode", "direct")
        route_reason = result.get("route_reason", "")
        execution_reason = result.get("execution_reason", "")
        prompt_version = self._prompts.get_active_version()
        ui_spec: UiSpec | None = None
        if generate_ui:
            ui_spec = self._build_ui_spec(user_input=user_input, response_text=response)

        self._persist_session(
            session_id=sid,
            user_input=user_input,
            response=response,
            selected_agent=selected_agent,
            execution_mode=execution_mode,
            route_reason=route_reason,
            execution_reason=execution_reason,
            prompt_version=prompt_version,
            plan_objective=result.get("plan_objective"),
            plan_steps=result.get("plan_steps"),
            step_results=result.get("step_results"),
        )

        return {
            "response": response,
            "session_id": sid,
            "selected_agent": selected_agent,
            "execution_mode": execution_mode,
            "execution_reason": execution_reason,
            "route_reason": route_reason,
            "prompt_version": prompt_version,
            "ui_spec": ui_spec.model_dump() if ui_spec else None,
        }

    def invoke(
        self,
        user_input: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        plan_step_budget: int | None = None,
        generate_ui: bool = False,
    ) -> str:
        return self.invoke_with_metadata(
            user_input=user_input,
            agent_id=agent_id,
            session_id=session_id,
            plan_step_budget=plan_step_budget,
            generate_ui=generate_ui,
        )["response"]

    async def _stream_worker_events(
        self,
        worker: Any,
        system_prompt: str,
        user_prompt: str,
        trace_tools: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        processor = StreamProcessor()
        async for event in worker.astream_events(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            },
            version="v1",
        ):
            payload = processor.process_event(event)
            if payload and payload.get("type") == "status" and not trace_tools:
                continue
            if payload:
                yield payload

        if not processor.streamed_any and processor.final_output_text:
            yield {"type": "token", "content": processor.final_output_text}
        elif not processor.streamed_any:
            yield {
                "type": "status",
                "content": "No token stream available from this model/provider for this run.",
            }

    async def astream_response(
        self,
        user_input: str,
        agent_id: str | None = None,
        trace_tools: bool = False,
        session_id: str | None = None,
        plan_step_budget: int | None = None,
        generate_ui: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        sid, session_context, _ = self._prepare_session(session_id)

        if agent_id:
            selected_agent = self._safe_agent_id(agent_id)
            route_reason = f"Explicitly targeted: {selected_agent}"
        else:
            router_result = self._llm_router(
                user_input, session_context=session_context
            )
            selected_agent = router_result.selected_agent
            route_reason = f"LLM Routing: {router_result.reasoning}"

        decision = self._decide_mode(
            user_input,
            selected_agent,
            agent_id,
            session_context=session_context,
        )

        yield {
            "type": "metadata",
            "stage": "routing",
            "session_id": sid,
            "route_reason": route_reason,
            "agent": selected_agent,
            "execution_mode": decision.mode,
            "execution_reason": decision.reason,
            "prompt_version": self._prompts.get_active_version(),
        }

        spec = AgentRegistry.get_agent(selected_agent)
        worker = self._build_worker(spec, streaming=True)

        if decision.mode == "direct":
            streamed_text_parts: list[str] = []
            async for payload in self._stream_worker_events(
                worker=worker,
                system_prompt=spec.runtime_system_prompt(),
                user_prompt=user_input,
                trace_tools=trace_tools,
            ):
                if payload.get("type") == "token":
                    streamed_text_parts.append(str(payload.get("content", "")))
                yield payload

            final_response = "".join(streamed_text_parts)
            ui_spec: UiSpec | None = None
            if generate_ui:
                ui_spec = self._build_ui_spec(
                    user_input=user_input,
                    response_text=final_response,
                )
            self._persist_session(
                session_id=sid,
                user_input=user_input,
                response=final_response,
                selected_agent=selected_agent,
                execution_mode=decision.mode,
                route_reason=route_reason,
                execution_reason=decision.reason,
                prompt_version=self._prompts.get_active_version(),
                plan_objective=None,
                plan_steps=None,
                step_results=None,
            )

            yield {
                "type": "metadata",
                "stage": "done",
                "session_id": sid,
                "route_reason": route_reason,
                "agent": selected_agent,
                "execution_mode": decision.mode,
                "execution_reason": decision.reason,
                "prompt_version": self._prompts.get_active_version(),
            }
            if ui_spec:
                yield {"type": "ui", "payload": ui_spec.model_dump()}
            return

        plan = self._build_plan(
            user_input, selected_agent, session_context=session_context
        )
        plan_payload = {
            "type": "plan",
            "objective": plan.objective,
            "steps": [
                {
                    "title": step.title,
                    "instruction": step.instruction,
                    "success_criteria": step.success_criteria,
                }
                for step in plan.steps
            ],
        }
        yield plan_payload

        budget = max(1, int(plan_step_budget or len(plan.steps)))
        step_results: list[dict[str, str]] = [
            {"title": step.title, "status": "pending", "result": ""}
            for step in plan.steps
        ]

        completed: list[dict[str, str]] = []
        for index, step in enumerate(plan.steps, start=1):
            if index > budget:
                break

            yield {
                "type": "status",
                "content": f"Executing step {index}/{len(plan.steps)}: {step.title}",
            }

            completed_context = "\n".join(
                [
                    f"{i+1}. {item['title']}: {item['result']}"
                    for i, item in enumerate(completed)
                ]
            )
            step_prompt = self._prompts.get_prompt(
                "step_user",
                user_input=user_input,
                plan_objective=plan.objective,
                step_index=index,
                step_count=len(plan.steps),
                step_title=step.title,
                step_instruction=step.instruction,
                step_success_criteria=step.success_criteria,
                completed_context=(
                    completed_context if completed_context else "None yet"
                ),
            )

            try:
                result = worker.invoke(
                    {
                        "messages": [
                            SystemMessage(content=spec.runtime_system_prompt()),
                            HumanMessage(content=step_prompt),
                        ]
                    }
                )
                step_text = self._extract_result_text(result)
                step_results[index - 1]["status"] = "completed"
                step_results[index - 1]["result"] = step_text
                completed.append({"title": step.title, "result": step_text})
                yield {
                    "type": "step_result",
                    "step_index": index,
                    "step_title": step.title,
                    "content": step_text,
                }
            except Exception as exc:  # noqa: BLE001
                step_results[index - 1]["status"] = "failed"
                step_results[index - 1]["result"] = str(exc)
                yield {
                    "type": "status",
                    "content": f"Step failed: {step.title}",
                }
                break

        all_completed = all(s["status"] == "completed" for s in step_results)
        final_text = ""
        if all_completed:
            synthesis_prompt = self._prompts.get_prompt(
                "synthesis_user",
                user_input=user_input,
                plan_objective=plan.objective,
                completed_steps="\n".join(
                    [f"- {item['title']}: {item['result']}" for item in completed]
                ),
            )
            final_result = worker.invoke(
                {
                    "messages": [
                        SystemMessage(content=spec.runtime_system_prompt()),
                        HumanMessage(content=synthesis_prompt),
                    ]
                }
            )
            final_text = self._extract_result_text(final_result)
            if final_text:
                yield {"type": "token", "content": final_text}
        else:
            done = [x["title"] for x in step_results if x["status"] == "completed"]
            pending = [x["title"] for x in step_results if x["status"] == "pending"]
            failed = [x["title"] for x in step_results if x["status"] == "failed"]
            final_text = (
                "Plan execution progress update:\n"
                f"- Completed: {', '.join(done) if done else 'None'}\n"
                f"- Pending: {', '.join(pending) if pending else 'None'}\n"
                f"- Failed: {', '.join(failed) if failed else 'None'}"
            )
            yield {"type": "status", "content": final_text}

        self._persist_session(
            session_id=sid,
            user_input=user_input,
            response=final_text,
            selected_agent=selected_agent,
            execution_mode=decision.mode,
            route_reason=route_reason,
            execution_reason=decision.reason,
            prompt_version=self._prompts.get_active_version(),
            plan_objective=plan.objective,
            plan_steps=plan_payload["steps"],
            step_results=step_results,
        )

        yield {
            "type": "metadata",
            "stage": "done",
            "session_id": sid,
            "route_reason": route_reason,
            "agent": selected_agent,
            "execution_mode": decision.mode,
            "execution_reason": decision.reason,
            "plan_steps": len(plan.steps),
            "prompt_version": self._prompts.get_active_version(),
        }
        if generate_ui:
            ui_spec = self._build_ui_spec(
                user_input=user_input, response_text=final_text
            )
            if ui_spec:
                yield {"type": "ui", "payload": ui_spec.model_dump()}

    def current_prompt_version(self) -> str:
        return self._prompts.get_active_version()

    def list_prompt_versions(self) -> list[str]:
        return self._prompts.list_versions()

    def set_prompt_version(self, version: str) -> None:
        self._prompts.set_active_version(version)

    def mermaid(self) -> str:
        return self._app.get_graph().draw_mermaid()

    def ascii_graph(self) -> str:
        return self._app.get_graph().draw_ascii()


def invoke_orchestrator(user_input: str) -> str:
    return Orchestrator().invoke(user_input)


def list_registered_agents() -> dict[str, str]:
    return AgentRegistry.descriptions()
