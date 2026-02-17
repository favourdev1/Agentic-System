import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic_system.tools.tool_models import ToolSpec


class WebScrapeInput(BaseModel):
    url: str = Field(description="The URL to scrape content from.")


def build_web_scrape() -> StructuredTool:
    def _scrape(url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = httpx.get(
                url, headers=headers, timeout=10.0, follow_redirects=True
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            return text[:4000] + ("..." if len(text) > 4000 else "")
        except Exception as exc:  # noqa: BLE001
            return f"Scraping failed: {exc}"

    return StructuredTool.from_function(
        name="web_scrape",
        description="Scrape text content from a specific URL",
        func=_scrape,
        args_schema=WebScrapeInput,
    )


tool = ToolSpec(
    name="web_scrape",
    builder=build_web_scrape,
    intent="Extract deep-text context from specific URLs identified during search.",
    schema_notes="Takes 'url' string. Returns clean markdown-ready text.",
    groups=["core", "analysis_plus_api"],
)
