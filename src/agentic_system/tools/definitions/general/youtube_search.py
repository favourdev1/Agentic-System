import httpx
import re
import json
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from agentic_system.tools.tool_models import ToolSpec


class YouTubeSearchInput(BaseModel):
    query: str = Field(description="The search query for YouTube videos.")
    max_results: int = Field(
        default=5, description="Maximum number of results to return."
    )


def build_youtube_search_tool() -> StructuredTool:
    def run(query: str, max_results: int = 5) -> str:
        """Searches YouTube and returns a list of video titles and URLs."""
        try:
            url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            response = httpx.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()

            # YouTube search results are embedded in a JSON object in the HTML
            # We look for "ytInitialData =" to find the video metadata
            match = re.search(r"var ytInitialData = ({.*?});", response.text)
            if not match:
                return "Could not parse YouTube results. The page structure might have changed."

            data = json.loads(match.group(1))

            # Navigate the complex YouTube JSON structure to find video renders
            videos = []
            try:
                contents = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][
                    0
                ]["content"]["sectionListRenderer"]["contents"]
            except KeyError:
                try:
                    contents = data["contents"]["twoColumnSearchResultsRenderer"][
                        "primaryContents"
                    ]["sectionListRenderer"]["contents"]
                except KeyError:
                    return "No results found or YouTube structure changed."

            for content in contents:
                if "itemSectionRenderer" in content:
                    items = content["itemSectionRenderer"]["contents"]
                    for item in items:
                        if "videoRenderer" in item:
                            v = item["videoRenderer"]
                            title = v["title"]["runs"][0]["text"]
                            video_id = v["videoId"]
                            link = f"https://www.youtube.com/watch?v={video_id}"
                            videos.append(f"Title: {title}\nURL: {link}")
                            if len(videos) >= max_results:
                                break
                if len(videos) >= max_results:
                    break

            if not videos:
                return "No videos found for this query."

            return "\n---\n".join(videos)

        except Exception as e:
            return f"YouTube search failed: {str(e)}"

    return StructuredTool.from_function(
        name="youtube_search",
        description="Search for videos on YouTube. Returns titles and URLs.",
        func=run,
        args_schema=YouTubeSearchInput,
    )


def build_youtube_search():
    return build_youtube_search_tool()


tool = ToolSpec(
    name="youtube_search",
    builder=build_youtube_search,
    intent="Discover video content and tutorials on YouTube.",
    status_message="Searching YouTube...",
    schema_notes="Returns a formatted list of Titles and URLs.",
)
