from typing import Any, Dict, List

# Central registry of tool groups.
# Map group names to the internal names of the tools they contain (using a list of objects).
TOOL_GROUPS: List[Dict[str, Any]] = [
    {
        "group_name": "core",
        "tools": [
            "web_search",
            "calculator",
            "web_scrape",
            "youtube_search",
        ],
    },
    {
        "group_name": "analysis_plus_api",
        "tools": [
            "web_search",
            "calculator",
            "web_scrape",
        ],
    },
    {
        "group_name": "social",
        "tools": [
            "daily_quote",
            "youtube_search",
        ],
    },
]
