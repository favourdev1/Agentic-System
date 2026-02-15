from __future__ import annotations

import argparse
import asyncio
import json
import os

from agentic_system.config.settings import get_settings
from agentic_system.prompting import PromptManager


def _configure_langsmith() -> None:
    # LangSmith relies on environment variables; this function keeps behavior explicit.
    os.environ.setdefault(
        "LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "true")
    )
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
    if os.getenv("LANGSMITH_PROJECT"):
        os.environ.setdefault(
            "LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "agentic-system-local")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph orchestrator")
    parser.add_argument("prompt", nargs="?", help="User request to process")
    parser.add_argument(
        "--list-agents", action="store_true", help="List available agents"
    )
    parser.add_argument(
        "--list-tool-groups", action="store_true", help="List available tool groups"
    )
    parser.add_argument(
        "--show-graph",
        choices=["mermaid", "ascii"],
        help="Print orchestrator graph in the selected format",
    )
    parser.add_argument("--save-graph", help="Save graph output to a file path")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream model output tokens instead of waiting for final response",
    )
    parser.add_argument(
        "--trace-tools",
        action="store_true",
        help="When streaming, include tool status events in output",
    )
    parser.add_argument("--server", action="store_true", help="Start the API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument("--reload", action="store_true", help="Enable hot reloading")
    parser.add_argument(
        "--session-id",
        help="Reuse an existing session id for persisted planning state",
    )
    parser.add_argument(
        "--plan-step-budget",
        type=int,
        help="Max plan steps to execute in this run (plan mode only)",
    )
    parser.add_argument(
        "--generate-ui",
        action="store_true",
        help="Generate a structured UI payload (cards/table/mixed) alongside text",
    )
    parser.add_argument(
        "--list-prompt-versions",
        action="store_true",
        help="List available governed prompt versions",
    )
    parser.add_argument(
        "--show-prompt-version",
        action="store_true",
        help="Show currently active prompt version",
    )
    parser.add_argument(
        "--set-prompt-version",
        help="Set active prompt version (supports rollback)",
    )
    args = parser.parse_args()

    _configure_langsmith()

    if args.list_agents:
        from agentic_system.agents.registry import AgentRegistry

        for name, description in AgentRegistry.descriptions().items():
            print(f"- {name}: {description}")
        return

    if args.server:
        import uvicorn

        print(
            f"Starting server on {args.host}:{args.port} (reload={'on' if args.reload else 'off'})"
        )
        if args.reload:
            # When reloading, pass the import string instead of the app object
            uvicorn.run(
                "agentic_system.api:app", host=args.host, port=args.port, reload=True
            )
        else:
            from agentic_system.api import app

            uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.list_tool_groups:
        from agentic_system.agents.tool_registry import ToolRegistry

        for group_name, tools in ToolRegistry.list_groups().items():
            print(f"- {group_name}: {', '.join(tools)}")
        return

    if args.list_prompt_versions or args.show_prompt_version or args.set_prompt_version:
        settings = get_settings()
        manager = PromptManager(settings.prompt_config_dir)
        if args.set_prompt_version:
            manager.set_active_version(args.set_prompt_version)
            print(f"Active prompt version set to: {manager.get_active_version()}")
            return
        if args.list_prompt_versions:
            for version in manager.list_versions():
                print(f"- {version}")
            return
        if args.show_prompt_version:
            print(manager.get_active_version())
            return

    if args.show_graph:
        from agentic_system.orchestrator.graph import Orchestrator

        orchestrator = Orchestrator()
        graph_output = (
            orchestrator.mermaid()
            if args.show_graph == "mermaid"
            else orchestrator.ascii_graph()
        )
        if args.save_graph:
            with open(args.save_graph, "w", encoding="utf-8") as file_handle:
                file_handle.write(graph_output)
            print(f"Saved {args.show_graph} graph to {args.save_graph}")
            return
        print(graph_output)
        return

    if not args.prompt:
        raise SystemExit(
            "Provide a prompt or use --list-agents / --list-tool-groups / --show-graph"
        )

    from agentic_system.orchestrator.graph import Orchestrator

    orchestrator = Orchestrator()
    if args.stream:

        async def _run_stream() -> None:
            async for payload in orchestrator.astream_response(
                args.prompt,
                trace_tools=args.trace_tools,
                session_id=args.session_id,
                plan_step_budget=args.plan_step_budget,
                generate_ui=args.generate_ui,
            ):
                event_type = payload.get("type")
                if event_type == "token":
                    print(payload.get("content", ""), end="", flush=True)
                elif event_type == "status":
                    print(f"\n[status] {payload.get('content', '')}", flush=True)
                elif event_type == "metadata":
                    route = payload.get("route_reason", "")
                    agent = payload.get("agent", "")
                    mode = payload.get("execution_mode", "")
                    reason = payload.get("execution_reason", "")
                    sid = payload.get("session_id", "")
                    prompt_version = payload.get("prompt_version", "")
                    print(
                        f"\n\n[router] {route} (agent={agent}, mode={mode}, session={sid})",
                        flush=True,
                    )
                    if prompt_version:
                        print(f"[prompts] version={prompt_version}", flush=True)
                    if reason:
                        print(f"[strategy] {reason}", flush=True)
                elif event_type == "plan":
                    objective = payload.get("objective", "")
                    print(f"\n[plan] {objective}", flush=True)
                    for idx, step in enumerate(payload.get("steps", []), start=1):
                        print(f"  {idx}. {step.get('title', '')}", flush=True)
                elif event_type == "step_result":
                    title = payload.get("step_title", "")
                    content = payload.get("content", "")
                    print(f"\n[step] {title}\n{content}\n", flush=True)
                elif event_type == "ui":
                    print("\n[ui_payload]", flush=True)
                    print(json.dumps(payload.get("payload", {}), indent=2), flush=True)
            print()

        asyncio.run(_run_stream())
        return

    result = orchestrator.invoke_with_metadata(
        args.prompt,
        session_id=args.session_id,
        plan_step_budget=args.plan_step_budget,
        generate_ui=args.generate_ui,
    )
    print(result["response"])
    print(f"\n[session_id] {result['session_id']}")
    print(f"[prompt_version] {result.get('prompt_version', '')}")
    if result.get("ui_spec"):
        print("[ui_payload]")
        print(json.dumps(result["ui_spec"], indent=2))


if __name__ == "__main__":
    main()
