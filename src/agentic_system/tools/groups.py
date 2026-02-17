from typing import Dict, List

# Central registry of tool groups.
# Map group names to the internal names of the tools they contain.
TOOL_GROUPS: Dict[str, List[str]] = {
    "core": [
        "web_search",
        "calculator",
        "web_scrape",
    ],
    "analysis_plus_api": [
        "web_search",
        "calculator",
        "web_scrape",
        "bank_account_api",
        "agency_lookup",
        "baby"
    ],
}
