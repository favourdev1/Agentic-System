from agentic_system.tools.tool_models import ToolSpec
from agentic_system.tools.builders.common.web import build_web_scrape

tool = ToolSpec(
    name="web_scrape",
    builder=build_web_scrape,
    intent="Extract deep-text context from specific URLs identified during search.",
    schema_notes="Takes 'url' string. Returns clean markdown-ready text.",
    groups=["core", "analysis_plus_api"],
)
