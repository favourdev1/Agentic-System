import importlib
import pkgutil
from typing import Any

from agentic_system.tools import definitions
from agentic_system.tools.groups import TOOL_GROUPS
from agentic_system.tools.tool_models import ToolSpec


class ToolRegistry:
    """Central tool registry that dynamically discovers tools in the 'definitions' package."""

    _cached_tools: dict[str, ToolSpec] | None = None

    @classmethod
    def _discover_tools(cls) -> dict[str, ToolSpec]:
        if cls._cached_tools is not None:
            return cls._cached_tools

        tools: dict[str, ToolSpec] = {}
        # Walk recursively so tools can be organized by domain folders.
        for _, module_name, is_pkg in pkgutil.walk_packages(
            definitions.__path__, prefix="agentic_system.tools.definitions."
        ):
            if is_pkg:
                continue

            module = importlib.import_module(module_name)

            # Look for a 'tool' attribute that is a ToolSpec.
            tool_spec = getattr(module, "tool", None)
            if isinstance(tool_spec, ToolSpec):
                if tool_spec.name in tools:
                    raise ValueError(
                        f"Duplicate tool name detected: {tool_spec.name} "
                        f"(module {module_name})"
                    )
                tools[tool_spec.name] = tool_spec

        cls._cached_tools = tools
        return tools

    @classmethod
    def _get_dynamic_groups(cls) -> dict[str, list[str]]:
        """Returns tool groups defined in the central groups configuration."""
        return TOOL_GROUPS

    @classmethod
    def resolve_tool_names(
        cls, tool_names: list[str], group_names: list[str]
    ) -> list[str]:
        merged: list[str] = []
        dynamic_groups = cls._get_dynamic_groups()
        for group_name in group_names:
            if group_name not in dynamic_groups:
                raise ValueError(f"Unknown tool group: {group_name}")
            merged.extend(dynamic_groups[group_name])
        merged.extend(tool_names)
        # Keep deterministic order while de-duplicating.
        return list(dict.fromkeys(merged))

    @classmethod
    def get_tools(
        cls, tool_names: list[str], group_names: list[str] | None = None
    ) -> list[Any]:
        groups = group_names or []
        resolved = cls.resolve_tool_names(tool_names, groups)
        tools_map = cls._discover_tools()
        missing = [name for name in resolved if name not in tools_map]
        if missing:
            raise ValueError(f"Unknown tool(s): {', '.join(missing)}")
        return [tools_map[name].builder() for name in resolved]

    @classmethod
    def list_groups(cls) -> dict[str, list[str]]:
        return cls._get_dynamic_groups()

    @classmethod
    def list_all_tools(cls) -> list[str]:
        return list(cls._discover_tools().keys())
