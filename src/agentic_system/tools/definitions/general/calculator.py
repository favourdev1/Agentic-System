from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic_system.tools.tool_models import ToolSpec


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Simple python arithmetic expression, e.g. '(12+5)*3'"
    )


def build_calculator() -> StructuredTool:
    def _calculate(expression: str) -> str:
        allowed = set("0123456789+-*/(). ")
        if any(ch not in allowed for ch in expression):
            return "Invalid expression: only numbers and + - * / ( ) are allowed."
        try:
            # Restricted eval for simple math expressions.
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"Calculation error: {exc}"

    return StructuredTool.from_function(
        name="calculator",
        description="Evaluate a basic arithmetic expression",
        func=_calculate,
        args_schema=CalculatorInput,
    )


tool = ToolSpec(
    name="calculator",
    builder=build_calculator,
    intent="Execute mathematical computations for absolute precision.",
    schema_notes="Expects 'expression' string. Returns numerical result.",
)
