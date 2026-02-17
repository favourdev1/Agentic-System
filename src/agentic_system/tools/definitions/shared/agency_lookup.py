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
#
# groups:        Predefined tool sets this tool belongs to.
#                Example: ["research", "travel", "core"]
# ================================================================


class AgencyLookupInput(BaseModel):
    query: str = Field(description="Search or action query")


def build_agency_lookup_tool() -> StructuredTool:
    def _run(query: str) -> str:
        # TODO: Implement tool logic
        return f"Tool agency_lookup executed with query: {query}"

    return StructuredTool.from_function(
        name="agency_lookup",
        description="Execute agency lookup logic.",
        func=_run,
        args_schema=AgencyLookupInput,
    )


def build_agency_lookup():
    return build_agency_lookup_tool()


tool = ToolSpec(
    name="agency_lookup",
    builder=build_agency_lookup,
    intent="Retrieve official details and metadata for specific Nigerian agencies.",
    status_message="Consulting agency directory...",
    schema_notes="Define input/output patterns here.",
)
