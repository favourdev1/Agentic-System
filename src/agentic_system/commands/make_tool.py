from __future__ import annotations

import argparse
from pathlib import Path


def run_make_tool(args: argparse.Namespace) -> None:
    """Generates a new ToolSpec file and basic implementation."""
    name = args.name.lower().replace(" ", "_")
    intent = args.intent or f"Execute {name.replace('_', ' ')} logic."
    schema_notes = args.schema_notes or "Define input/output patterns here."
    groups = args.groups or []

    # Format the groups list as a Python literal
    groups_repr = "[" + ", ".join(f'"{g}"' for g in groups) + "]"

    # Template for the tool implementation and definition
    file_content = f"""from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from agentic_system.tools.tool_models import ToolSpec

class {name.title().replace('_', '')}Input(BaseModel):
    query: str = Field(description="Search or action query")

def build_{name}_tool() -> StructuredTool:
    def _{name}(query: str) -> str:
        # TODO: Implement tool logic
        return f"Tool {name} executed with query: {{query}}"

    return StructuredTool.from_function(
        name="{name}",
        description="{intent}",
        func=_{name},
        args_schema={name.title().replace('_', '')}Input,
    )

def build_{name}():
    return build_{name}_tool()

tool = ToolSpec(
    name="{name}",
    builder=build_{name},
    intent="{intent}",
    schema_notes="{schema_notes}",
    groups={groups_repr},
)
"""

    definitions_dir = Path(__file__).parent.parent / "tools" / "definitions"
    if not definitions_dir.exists():
        definitions_dir.mkdir(parents=True, exist_ok=True)
        (definitions_dir / "__init__.py").write_text("# Tool definitions package.\n")

    tool_file = definitions_dir / f"{name}.py"

    if tool_file.exists():
        print(f"Error: Tool file '{tool_file.name}' already exists.")
        return

    tool_file.write_text(file_content, encoding="utf-8")

    print(f"✅ Successfully created tool file: {name}.py")
    print(f"📍 Location: {tool_file}")
