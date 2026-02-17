from __future__ import annotations

import argparse
from pathlib import Path


def _normalize_tool_path(path: str) -> Path:
    # Accept either `hotel/search` or `hotel.search`, then normalize to package path.
    raw = (path or "shared").strip().replace(".", "/").replace("\\", "/")
    parts = [part.strip().replace("-", "_") for part in raw.split("/") if part.strip()]
    if not parts:
        parts = ["shared"]
    return Path(*parts)


def _ensure_package_tree(root: Path, subpath: Path) -> Path:
    current = root
    init_file = current / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Tool definitions package.\n", encoding="utf-8")

    for part in subpath.parts:
        current = current / part
        current.mkdir(parents=True, exist_ok=True)
        init_file = current / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                f"# Tool definitions package: {part}\n", encoding="utf-8"
            )
    return current


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

#================================================================
# TOOL CONFIGURATION GUIDE
#================================================================
# intent:        Formal definition of what the tool does (for the LLM).
#                Example: "Useful for searching internal flight records."
#
# schema_notes:  Instructional notes for the LLM on input/output logic.
#                Example: "Always returns a list of JSON objects."
#================================================================

class {name.title().replace('_', '')}Input(BaseModel):
    query: str = Field(description="Search or action query")

def build_{name}_tool() -> StructuredTool:
    def __run(query: str) -> str:
        # TODO: Implement tool logic
        return f"Tool {name} executed with query: {{query}}"

    return StructuredTool.from_function(
        name="{name}",
        description="{intent}",
        func=__run,
        args_schema={name.title().replace('_', '')}Input,
    )

def build_{name}():
    return build_{name}_tool()

tool = ToolSpec(
    name         = "{name}",
    builder      = build_{name},
    intent       = "{intent}",
    schema_notes = "{schema_notes}",
)
"""

    definitions_dir = Path(__file__).parent.parent / "tools" / "definitions"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    target_subpath = _normalize_tool_path(args.path)
    target_dir = _ensure_package_tree(definitions_dir, target_subpath)

    tool_file = target_dir / f"{name}.py"

    if tool_file.exists():
        print(f"Error: Tool file '{tool_file.name}' already exists.")
        return

    tool_file.write_text(file_content, encoding="utf-8")

    print(f"✅ Successfully created tool file: {name}.py")
    print(f"📂 Tool path: tools/definitions/{target_subpath.as_posix()}")
    print(f"📍 Location: {tool_file}")
    print(
        f"\n💡 NEXT STEP: Manually register '{name}' in src/agentic_system/tools/groups.py"
    )
