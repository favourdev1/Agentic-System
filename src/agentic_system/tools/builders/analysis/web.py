from agentic_system.config.settings import get_settings
from agentic_system.tools.tool_factory.web.http_get import build_http_get_tool

settings = get_settings()


def build_external_search_api():
    """Build an external search API tool placeholder."""
    return build_http_get_tool(
        name="external_search_api",
        description="Query external search endpoint. Pass search keywords in the 'q' parameter.",
        base_url="https://example.com/search",
        timeout_seconds=settings.default_api_timeout_seconds,
    )


def build_bank_account_api():
    """Build the bank account API tool with semantic usage guidance for the agent."""
    base_url = settings.bank_api_base_url
    headers = {}

    if settings.bank_api_auth_token:
        headers["Authorization"] = f"Bearer {settings.bank_api_auth_token}"

    if settings.bank_api_session_cookie:
        headers["Cookie"] = f"gopaddi_session={settings.bank_api_session_cookie}"

    description = (
        "Fetch bank account requests. Use this tool to search or filter bank transaction requests. "
        "IMPORTANT: You must provide a 'params' dictionary with any of these optional keys:\n"
        "- 'bank_name': Name of the bank (e.g., 'Access')\n"
        "- 'status': Request status (e.g., 'pending', 'approved')\n"
        "- 'search': General keyword to search for (e.g., 'baby')\n"
        "- 'date_from': Start date for filtering (YYYY-MM-DD)\n"
        "- 'start_date': Specific start date (YYYY-MM-DD)\n"
        "- 'end_date': Specific end date (YYYY-MM-DD)\n"
        "All fields are optional. The tool returns a list of matching records."
    )

    return build_http_get_tool(
        name="bank_account_api",
        description=description,
        base_url=base_url,
        headers=headers,
    )
