from __future__ import annotations

import argparse
import asyncio
import os

from agentic_system.agents.registry import AgentRegistry
from agentic_system.agents.tool_registry import ToolRegistry


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
    parser.add_argument("--server", action="store_true", help="Start the API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument("--reload", action="store_true", help="Enable hot reloading")
    args = parser.parse_args()

    _configure_langsmith()

    if args.list_agents:
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
        for group_name, tools in ToolRegistry.list_groups().items():
            print(f"- {group_name}: {', '.join(tools)}")
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
            async for chunk in orchestrator.astream_response(args.prompt):
                print(chunk, end="", flush=True)
            print()

        asyncio.run(_run_stream())
        return

    print(orchestrator.invoke(args.prompt))


if __name__ == "__main__":
    main()
