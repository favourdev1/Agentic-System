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
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Group: Chat/Inference (Default if no command provided)
    chat_parser = subparsers.add_parser("chat", help="Start a chat session (default)")
    chat_parser.add_argument("prompt", help="User request to process")
    chat_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream model output tokens instead of waiting for final response",
    )
    chat_parser.add_argument(
        "--trace-tools",
        action="store_true",
        help="When streaming, include tool status events in output",
    )
    chat_parser.add_argument(
        "--session-id",
        help="Reuse an existing session id for persisted planning state",
    )
    chat_parser.add_argument(
        "--plan-step-budget",
        type=int,
        help="Max plan steps to execute in this run (plan mode only)",
    )
    chat_parser.add_argument(
        "--generate-ui",
        action="store_true",
        help="Generate a structured UI payload (cards/table/mixed) alongside text",
    )

    # Group: System Operations
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server to"
    )
    server_parser.add_argument(
        "--port", type=int, default=8888, help="Port to run the server on"
    )
    server_parser.add_argument(
        "--reload", action="store_true", help="Enable hot reloading"
    )

    list_agents_parser = subparsers.add_parser(
        "list:agents", help="List available agents"
    )
    list_tools_parser = subparsers.add_parser(
        "list:tools", help="List available tool groups"
    )

    graph_parser = subparsers.add_parser("show:graph", help="Show orchestrator graph")
    graph_parser.add_argument(
        "--format",
        choices=["mermaid", "ascii"],
        default="mermaid",
        help="Output format",
    )
    graph_parser.add_argument("--save", help="Save graph output to a file path")

    prompts_parser = subparsers.add_parser("prompts", help="Manage prompt versions")
    prompts_parser.add_argument(
        "--list", action="store_true", help="List available versions"
    )
    prompts_parser.add_argument(
        "--show", action="store_true", help="Show active version"
    )
    prompts_parser.add_argument("--set", help="Set active version")

    # Group: Generators (Laravel Style)
    make_agent_parser = subparsers.add_parser("make:agent", help="Generate a new agent")
    make_agent_parser.add_argument("name", help="Name of the new agent")
    make_agent_parser.add_argument("--role", help="Formal role definition")
    make_agent_parser.add_argument(
        "--backstory", help="Narrative context that shapes the agent behavior"
    )
    make_agent_parser.add_argument(
        "--goal",
        action="append",
        help="Agent goal. Repeat flag to add multiple goals.",
    )
    make_agent_parser.add_argument(
        "--boundary", help="Explicit operational constraints"
    )
    make_agent_parser.add_argument(
        "--description", help="High-level summary for routing"
    )

    # Legacy Shim: Map old-style flags to new subcommands for backward compatibility.
    import sys

    processed_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--server" and "server" not in sys.argv:
            processed_argv.append("server")
        elif arg == "--chat" and "chat" not in sys.argv:
            processed_argv.append("chat")
        else:
            processed_argv.append(arg)
        i += 1
    sys.argv = processed_argv

    # Compatibility: If no command is provided but there are arguments, assume 'chat'
    if (
        len(sys.argv) > 1
        and sys.argv[1] not in subparsers.choices
        and not sys.argv[1].startswith("-")
    ):
        sys.argv.insert(1, "chat")

    args = parser.parse_args()
    _configure_langsmith()

    if args.command == "make:agent":
        from agentic_system.commands.make_agent import run_make_agent

        run_make_agent(args)
        return

    if args.command == "make:tool":
        from agentic_system.commands.make_tool import run_make_tool

        run_make_tool(args)
        return

    if args.command == "list:agents":
        from agentic_system.agents.registry import AgentRegistry

        for name, description in AgentRegistry.descriptions().items():
            print(f"- {name}: {description}")
        return

    if args.command == "server":
        import uvicorn

        print(
            f"Starting server on {args.host}:{args.port} (reload={'on' if args.reload else 'off'})"
        )
        if args.reload:
            uvicorn.run(
                "agentic_system.api:app", host=args.host, port=args.port, reload=True
            )
        else:
            from agentic_system.api import app

            uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.command == "list:tools":
        from agentic_system.tools.registry import ToolRegistry

        groups = ToolRegistry.list_groups()
        all_tools = ToolRegistry.list_all_tools()
        print("Groups:")
        for name, tools in groups.items():
            print(f"- {name}: {', '.join(tools)}")
        print("\nAvailable Tools:")
        for name in sorted(all_tools):
            print(f"- {name}")
        return

    if args.command == "prompts":
        settings = get_settings()
        manager = PromptManager(settings.prompt_config_dir)
        if args.set:
            manager.set_active_version(args.set)
            print(f"Active prompt version set to: {manager.get_active_version()}")
            return
        if args.list:
            for version in manager.list_versions():
                print(f"- {version}")
            return
        if args.show:
            print(manager.get_active_version())
            return

    if args.command == "show:graph":
        from agentic_system.orchestrator.graph import Orchestrator

        orchestrator = Orchestrator()
        graph_output = (
            orchestrator.mermaid()
            if args.format == "mermaid"
            else orchestrator.ascii_graph()
        )
        if args.save:
            with open(args.save, "w", encoding="utf-8") as file_handle:
                file_handle.write(graph_output)
            print(f"Saved {args.format} graph to {args.save}")
            return
        print(graph_output)
        return

    if args.command == "chat":
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
                        sid = payload.get("session_id", "")
                        prompt_version = payload.get("prompt_version", "")
                        print(
                            f"\n\n[router] {route} (agent={agent}, mode={mode}, session={sid})",
                            flush=True,
                        )
                        if prompt_version:
                            print(f"[prompts] version={prompt_version}", flush=True)
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
                        print(
                            json.dumps(payload.get("payload", {}), indent=2), flush=True
                        )
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
        return

    parser.print_help()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
