import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WebScrapeInput(BaseModel):
    url: str = Field(description="The URL to scrape content from.")


def build_web_scrape_tool() -> StructuredTool:
    def _scrape(url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = httpx.get(
                url, headers=headers, timeout=10.0, follow_redirects=True
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()

            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            # Limit length to avoid context overflow (approx 4000 chars)
            return text[:4000] + ("..." if len(text) > 4000 else "")

        except Exception as e:
            return f"Scraping failed: {e}"

    return StructuredTool.from_function(
        name="web_scrape",
        description="Scrape text content from a specific URL",
        func=_scrape,
        args_schema=WebScrapeInput,
    )
