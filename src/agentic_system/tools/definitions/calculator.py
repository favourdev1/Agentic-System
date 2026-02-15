from agentic_system.tools.tool_models import ToolSpec
from agentic_system.tools.builders.common.core import build_calculator

tool = ToolSpec(
    name="calculator",
    builder=build_calculator,
    intent="Execute mathematical computations for absolute precision.",
    schema_notes="Expects 'expression' string. Returns numerical result.",
    groups=["core", "analysis_plus_api"],
)
