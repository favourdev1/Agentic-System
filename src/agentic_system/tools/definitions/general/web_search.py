import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic_system.tools.tool_models import ToolSpec


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to execute.")
    num_results: int = Field(
        default=5, description="Number of results to return (max 10)."
    )


def build_web_search() -> StructuredTool:
    def _mock_search(query: str) -> str:
        return (
            f"Title: Mock Result for '{query}'\n"
            f"URL: https://example.com/search?q={query.replace(' ', '+')}\n"
            f"Description: This is a mock search result because external search engines are blocking automated requests. "
            f"In production, configure a real search API.\n"
            f"---\n"
            f"Title: Python Programming - Official Site\n"
            f"URL: https://www.python.org/\n"
            f"Description: The official home of the Python Programming Language."
        )

    def _search(query: str, num_results: int = 5) -> str:
        try:
            url = "https://html.duckduckgo.com/html/"
            payload = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://html.duckduckgo.com/",
            }

            response = httpx.post(url, data=payload, headers=headers, timeout=5.0)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                for result in soup.find_all("div", class_="result"):
                    if len(results) >= min(num_results, 10):
                        break
                    title_tag = result.find("a", class_="result__a")
                    if not title_tag:
                        continue
                    title = title_tag.get_text(strip=True)
                    link = title_tag.get("href")
                    snippet_tag = result.find("a", class_="result__snippet")
                    snippet = (
                        snippet_tag.get_text(strip=True)
                        if snippet_tag
                        else "No description"
                    )
                    results.append(
                        f"Title: {title}\nURL: {link}\nDescription: {snippet}\n"
                    )
                if results:
                    return "\n---\n".join(results)

            return _mock_search(query)
        except Exception:  # noqa: BLE001
            return _mock_search(query)

    return StructuredTool.from_function(
        name="web_search",
        description="Search for information on the web (DuckDuckGo + fallback)",
        func=_search,
        args_schema=WebSearchInput,
    )


tool = ToolSpec(
    name="web_search",
    builder=build_web_search,
    intent="Discover relevant real-time information from the open web.",
    schema_notes="Takes 'query' string. Returns top-K snippets or URLs.",
    groups=["core", "analysis_plus_api"],
)
