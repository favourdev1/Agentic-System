from agentic_system.tools.tool_factory.web.scraper import build_web_scrape_tool
from agentic_system.tools.tool_factory.web.search import build_web_search_tool


def build_web_search():
    """Build the web search tool instance."""
    return build_web_search_tool()


def build_web_scrape():
    """Build the web scraping tool instance."""
    return build_web_scrape_tool()
