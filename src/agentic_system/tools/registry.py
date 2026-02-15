import importlib
import pkgutil
from typing import Any
from agentic_system.tools import definitions
from agentic_system.tools.tool_models import ToolSpec


class ToolRegistry:
    """Central tool registry that dynamically discovers tools in the 'definitions' package."""

    _cached_tools: dict[str, ToolSpec] | None = None

    @classmethod
    def _discover_tools(cls) -> dict[str, ToolSpec]:
        if cls._cached_tools is not None:
            return cls._cached_tools

        tools = {}
        # Iterate through all modules in the 'definitions' package
        for _, name, is_pkg in pkgutil.iter_modules(definitions.__path__):
            if is_pkg:
                continue

            # Import the module dynamically
            module_name = f"agentic_system.tools.definitions.{name}"
            module = importlib.import_module(module_name)

            # Look for a 'tool' attribute that is a ToolSpec
            tool_spec = getattr(module, "tool", None)
            if isinstance(tool_spec, ToolSpec):
                tools[tool_spec.name] = tool_spec

        cls._cached_tools = tools
        return tools

    @classmethod
    def _get_dynamic_groups(cls) -> dict[str, list[str]]:
        """Builds tool groups dynamically from discovered tool specifications."""
        groups = {}
        for tool_spec in cls._discover_tools().values():
            for group_name in tool_spec.groups:
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(tool_spec.name)
        return groups

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
