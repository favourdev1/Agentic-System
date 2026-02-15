from __future__ import annotations

import argparse
from pathlib import Path


def run_make_agent(args: argparse.Namespace) -> None:
    """Generates a new AgentSpec file in the definitions directory."""
    name = args.name.lower().replace(" ", "_")
    description = (
        args.description or f"Assistant specialized in {name.replace('_', ' ')}"
    )
    role = args.role or f"Expert at {name.replace('_', ' ')}."
    boundary = args.boundary or "Avoid tasks outside of specialized domain."

    # Template for the new agent specification file
    file_content = f"""from agentic_system.agents.registry import AgentSpec

agent = AgentSpec(
    name="{name}",
    description="{description}",
    role="{role}",
    boundary="{boundary}",
    system_prompt=(
        "You are a specialized assistant."
    ),
    tool_groups=["core"],
)
"""

    definitions_dir = Path(__file__).parent.parent / "agents" / "definitions"
    if not definitions_dir.exists():
        definitions_dir.mkdir(parents=True, exist_ok=True)
        (definitions_dir / "__init__.py").write_text("# Agent definitions package.\n")

    agent_file = definitions_dir / f"{name}.py"

    if agent_file.exists():
        print(f"Error: Agent file '{agent_file.name}' already exists.")
        return

    agent_file.write_text(file_content, encoding="utf-8")

    print(f"✅ Successfully created agent file: {name}.py")
    print(f"📍 Location: {agent_file}")
