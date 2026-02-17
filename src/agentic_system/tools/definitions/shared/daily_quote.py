import random
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from agentic_system.tools.tool_models import ToolSpec

# ================================================================
# TOOL CONFIGURATION GUIDE
# ================================================================
# intent:        Formal definition of what the tool does (for the LLM).
#                Example: "Useful for searching internal flight records."
#
# schema_notes:  Instructional notes for the LLM on input/output logic.
#                Example: "Always returns a list of JSON objects."
# ================================================================


class DailyQuoteInput(BaseModel):
    category: str = Field(
        default="random", description="Type of quote (e.g., 'motivational', 'funny')."
    )


def build_daily_quote():
    def run(category: str = "random"):
        """Returns a random inspirational or funny quote."""
        quotes = [
            "Believe you can and you're halfway there. - Theodore Roosevelt",
            "The only way to do great work is to love what you do. - Steve Jobs",
            "If you're going through hell, keep going. - Winston Churchill",
            "Your time is limited, don't waste it living someone else's life. - Steve Jobs",
            "Stay hungry, stay foolish. - Steve Jobs",
            "The best way to predict the future is to create it. - Peter Drucker",
            "Everything you've ever wanted is on the other side of fear. - George Addair",
            "Success is not final, failure is not fatal: it is the courage to continue that counts. - Winston Churchill",
            "Hardships often prepare ordinary people for an extraordinary destiny. - C.S. Lewis",
            "The only limit to our realization of tomorrow will be our doubts of today. - Franklin D. Roosevelt",
        ]
        return random.choice(quotes)

    return StructuredTool.from_function(
        name="daily_quote",
        description="Search for a random inspirational or funny quote.",
        func=run,
        args_schema=DailyQuoteInput,
    )


tool = ToolSpec(
    name="daily_quote",
    builder=build_daily_quote,
    intent="Generate inspirational or funny quotes to lift the mood.",
    status_message="Incubating some inspiration...",
    schema_notes="Returns a random quote as a string.",
)
