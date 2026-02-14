from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agentic_system.config.settings import get_settings


@dataclass(frozen=True)
class ToolSpec:
    name: str
    builder: Callable[[], Any]


settings = get_settings()


def _build_calculator():
    from agentic_system.tools.core_tools import build_calculator_tool

    return build_calculator_tool()


def _build_external_search_api():
    from agentic_system.tools.http_api import build_http_get_tool

    return build_http_get_tool(
        name="external_search_api",
        description="Query external search endpoint (placeholder URL until provided)",
        base_url="https://example.com/search",
        timeout_seconds=settings.default_api_timeout_seconds,
    )


class ToolRegistry:
    # Standardized tool registry. Add new tools here and reference by name from groups/agents.
    _tools: dict[str, ToolSpec] = {
        "calculator": ToolSpec(
            name="calculator",
            builder=_build_calculator,
        ),
        "external_search_api": ToolSpec(
            name="external_search_api",
            builder=_build_external_search_api,
        ),
    }

    # Standardized tool groups. Reuse these across agents.
    _groups: dict[str, list[str]] = {
        "core": ["calculator"],
        "analysis_plus_api": ["calculator", "external_search_api"],
    }

    @classmethod
    def resolve_tool_names(cls, tool_names: list[str], group_names: list[str]) -> list[str]:
        merged: list[str] = []
        for group_name in group_names:
            if group_name not in cls._groups:
                raise ValueError(f"Unknown tool group: {group_name}")
            merged.extend(cls._groups[group_name])
        merged.extend(tool_names)
        # Keep deterministic order while de-duplicating.
        return list(dict.fromkeys(merged))

    @classmethod
    def get_tools(cls, tool_names: list[str], group_names: list[str] | None = None) -> list[Any]:
        groups = group_names or []
        resolved = cls.resolve_tool_names(tool_names, groups)
        missing = [name for name in resolved if name not in cls._tools]
        if missing:
            raise ValueError(f"Unknown tool(s): {', '.join(missing)}")
        return [cls._tools[name].builder() for name in resolved]

    @classmethod
    def list_groups(cls) -> dict[str, list[str]]:
        return dict(cls._groups)
