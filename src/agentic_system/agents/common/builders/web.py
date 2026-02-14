from agentic_system.tools.web.search import build_web_search_tool
from agentic_system.tools.web.scraper import build_web_scrape_tool


def build_web_search():
    """Builds the Google Search tool."""
    return build_web_search_tool()


def build_web_scrape():
    """Builds the Web Scraper tool."""
    return build_web_scrape_tool()
