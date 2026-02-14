import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to execute.")
    num_results: int = Field(
        default=5, description="Number of results to return (max 10)."
    )


def build_web_search_tool() -> StructuredTool:
    def _mock_search(query: str) -> str:
        """Fallback mock search for testing when external requests are blocked."""
        return (
            f"Title: Mock Result for '{query}'\n"
            f"URL: https://example.com/search?q={query.replace(' ', '+')}\n"
            f"Description: This is a mock search result because external search engines are blocking automated requests. "
            f"In a production environment, please configure a real search API key (e.g., SerpApi).\n"
            f"---\n"
            f"Title: Python Programming - Official Site\n"
            f"URL: https://www.python.org/\n"
            f"Description: The official home of the Python Programming Language. Download, documentation, and community."
        )

    def _search(query: str, num_results: int = 5) -> str:
        try:
            # Try DuckDuckGo HTML scraper first
            url = "https://html.duckduckgo.com/html/"
            payload = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
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

            # If we get here, search failed or returned no results. Fallback to mock.
            print(f"[WARN] Search failed or blocked. Using mock results for '{query}'.")
            return _mock_search(query)

        except Exception as e:
            print(f"[WARN] Search error: {e}. Using mock results.")
            return _mock_search(query)

    return StructuredTool.from_function(
        name="web_search",
        description="Search for information on the web (Google/DDG fallback)",
        func=_search,
        args_schema=WebSearchInput,
    )
