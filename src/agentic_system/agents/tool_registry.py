from __future__ import annotations

from typing import Any

from agentic_system.agents.tool_models import ToolSpec
from agentic_system.agents.common.builders.core import build_calculator
from agentic_system.agents.analysis.builders.web import (
    build_external_search_api,
    build_bank_account_api,
)


class ToolRegistry:
    # Standardized tool registry. Add new tools here and reference by name from groups/agents.
    _tools: dict[str, ToolSpec] = {
        "calculator": ToolSpec(
            name="calculator",
            builder=build_calculator,
        ),
        "external_search_api": ToolSpec(
            name="external_search_api",
            builder=build_external_search_api,
        ),
        "bank_account_api": ToolSpec(
            name="bank_account_api",
            builder=build_bank_account_api,
        ),
    }

    # Standardized tool groups. Reuse these across agents.
    _groups: dict[str, list[str]] = {
        "core": ["calculator"],
        "analysis_plus_api": ["calculator", "external_search_api", "bank_account_api"],
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
