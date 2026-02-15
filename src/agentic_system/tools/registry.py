from __future__ import annotations

from typing import Any

from agentic_system.tools.builders.analysis.web import build_bank_account_api
from agentic_system.tools.builders.common.core import build_calculator
from agentic_system.tools.builders.common.web import (
    build_web_scrape,
    build_web_search,
)
from agentic_system.tools.tool_models import ToolSpec


class ToolRegistry:
    """Central tool registry and reusable tool groups."""

    _tools: dict[str, ToolSpec] = {
        "calculator": ToolSpec(
            name="calculator",
            builder=build_calculator,
            intent="Execute mathematical computations for absolute precision.",
            schema_notes="Expects 'expression' string. Returns numerical result.",
        ),
        "web_search": ToolSpec(
            name="web_search",
            builder=build_web_search,
            intent="Discover relevant real-time information from the open web.",
            schema_notes="Takes 'query' string. Returns top-K snippets or URLs.",
        ),
        "web_scrape": ToolSpec(
            name="web_scrape",
            builder=build_web_scrape,
            intent="Extract deep-text context from specific URLs identified during search.",
            schema_notes="Takes 'url' string. Returns clean markdown-ready text.",
        ),
        "bank_account_api": ToolSpec(
            name="bank_account_api",
            builder=build_bank_account_api,
            intent="Access and filter internal bank transaction records.",
            schema_notes="Requires optional 'params' dict (bank_name, status, search, date range).",
        ),
    }

    _groups: dict[str, list[str]] = {
        "core": ["calculator", "web_search", "web_scrape"],
        "analysis_plus_api": [
            "calculator",
            "web_search",
            "web_scrape",
            "bank_account_api",
        ],
    }

    @classmethod
    def resolve_tool_names(
        cls, tool_names: list[str], group_names: list[str]
    ) -> list[str]:
        merged: list[str] = []
        for group_name in group_names:
            if group_name not in cls._groups:
                raise ValueError(f"Unknown tool group: {group_name}")
            merged.extend(cls._groups[group_name])
        merged.extend(tool_names)
        # Keep deterministic order while de-duplicating.
        return list(dict.fromkeys(merged))

    @classmethod
    def get_tools(
        cls, tool_names: list[str], group_names: list[str] | None = None
    ) -> list[Any]:
        groups = group_names or []
        resolved = cls.resolve_tool_names(tool_names, groups)
        missing = [name for name in resolved if name not in cls._tools]
        if missing:
            raise ValueError(f"Unknown tool(s): {', '.join(missing)}")
        return [cls._tools[name].builder() for name in resolved]

    @classmethod
    def list_groups(cls) -> dict[str, list[str]]:
        return dict(cls._groups)
