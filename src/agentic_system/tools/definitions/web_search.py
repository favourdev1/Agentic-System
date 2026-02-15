from agentic_system.tools.tool_models import ToolSpec
from agentic_system.tools.builders.common.web import build_web_search

tool = ToolSpec(
    name="web_search",
    builder=build_web_search,
    intent="Discover relevant real-time information from the open web.",
    schema_notes="Takes 'query' string. Returns top-K snippets or URLs.",
    groups=["core", "analysis_plus_api"],
)
