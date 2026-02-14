from agentic_system.config.settings import get_settings
from agentic_system.tools.web.http_get import build_http_get_tool

settings = get_settings()


def build_external_search_api():
    """Builds the external search API tool."""
    return build_http_get_tool(
        name="external_search_api",
        description="Query external search endpoint. Pass search keywords in the 'q' parameter.",
        base_url="https://example.com/search",
        timeout_seconds=settings.default_api_timeout_seconds,
    )


def build_bank_account_api():
    """Builds the bank account API tool with enhanced semantic descriptions for the agent."""
    # Auth configuration provided by user
    base_url = "http://localhost:8000/api/v2/bank-account/request"
    headers = {
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vMTI3LjAuMC4xOjgwMDAvYXBpL3YxL2FkbWluL2xvZ2luIiwiaWF0IjoxNzY5NDk0OTY2LCJleHAiOjIzNjk0OTQ5MDYsIm5iZiI6MTc2OTQ5NDk2NiwianRpIjoiS29rZHdydjZ1Q291YlE0WSIsInN1YiI6IjIyIiwicHJ2IjoiZGY4ODNkYjk3YmQwNWVmOGZmODUwODJkNjg2YzQ1ZTgzMmU1OTNhOSJ9.8YRts4fQGOCgru84Ui7j-m39geQ2a3PEojRDxPeNM8M",
        "Cookie": "gopaddi_session=giuVUe4DEWaaFV9UESgGGcS83Br6FPPTfzj67pM5",
    }

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
