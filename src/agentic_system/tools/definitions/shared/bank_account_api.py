from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic_system.config.settings import get_settings
from agentic_system.tools.tool_models import ToolSpec

settings = get_settings()


class BankAccountApiInput(BaseModel):
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Query parameters: bank_name, status, search, date_from, start_date, end_date",
    )


def build_bank_account_api() -> StructuredTool:
    base_url = f"{settings.api_base_url}/bank-account/request".rstrip("/")
    headers: dict[str, str] = {}

    if settings.bank_api_auth_token:
        headers["Authorization"] = f"Bearer {settings.bank_api_auth_token}"

    if settings.bank_api_session_cookie:
        headers["Cookie"] = f"gopaddi_session={settings.bank_api_session_cookie}"

    description = (
        "Fetch bank account requests. Provide optional params dict with keys: "
        "bank_name, status, search, date_from, start_date, end_date."
    )

    def _run(params: dict[str, Any]) -> str:
        try:
            with httpx.Client(
                timeout=settings.default_api_timeout_seconds, headers=headers
            ) as client:
                response = client.get(base_url, params=params)
                response.raise_for_status()
                payload: Any = response.json()
                return str(payload)
        except Exception as exc:  # noqa: BLE001
            return f"bank_account_api tool failed: {exc}"

    return StructuredTool.from_function(
        name="bank_account_api",
        description=description,
        func=_run,
        args_schema=BankAccountApiInput,
    )


tool = ToolSpec(
    name="bank_account_api",
    builder=build_bank_account_api,
    intent="Access and filter internal bank transaction records.",
    schema_notes="Requires optional 'params' dict (bank_name, status, search, date range).",
    groups=["analysis_plus_api"],
)
