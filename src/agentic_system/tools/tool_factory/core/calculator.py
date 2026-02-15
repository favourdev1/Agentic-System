from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Simple python arithmetic expression, e.g. '(12+5)*3'"
    )


def build_calculator_tool() -> StructuredTool:
    def _calculate(expression: str) -> str:
        allowed = set("0123456789+-*/(). ")
        if any(ch not in allowed for ch in expression):
            return "Invalid expression: only numbers and + - * / ( ) are allowed."
        try:
            # Note: eval is used here with restricted globals/locals for simple math.
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
