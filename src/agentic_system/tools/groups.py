from typing import Dict, List

# Central registry of tool groups.
# Map group names to the internal names of the tools they contain.
TOOL_GROUPS: Dict[str, List[str]] = {
    "core": [
        "web_search",
        "calculator",
        "web_scrape",
        "youtube_search",
    ],
    "analysis_plus_api": [
        "web_search",
        "calculator",
        "web_scrape",
    ],
    "social": [
        "daily_quote",
        "youtube_search",
    ],
}
