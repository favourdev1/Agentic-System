from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class HttpGetInput(BaseModel):
    query: str = Field(description="Free text query string passed to the upstream API")


def build_http_get_tool(name: str, description: str, base_url: str, timeout_seconds: int = 20) -> StructuredTool:
    """Create a standardized GET tool wrapper for API-based integrations."""

    def _run(query: str) -> str:
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.get(base_url, params={"q": query})
                response.raise_for_status()
                payload: Any = response.json()
                return str(payload)
        except Exception as exc:  # noqa: BLE001
            return f"{name} tool failed: {exc}"

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_run,
        args_schema=HttpGetInput,
    )
