from __future__ import annotations

import json
from pathlib import Path
from string import Formatter
from typing import Any


class PromptManager:
    """Loads versioned prompt packs and supports safe rollback by version switch."""

    def __init__(self, base_dir: str, version_override: str | None = None) -> None:
        self._base = Path(base_dir)
        self._versions_dir = self._base / "versions"
        self._active_file = self._base / "active_version.txt"
        self._cache: dict[str, dict[str, Any]] = {}
        self._version_override = version_override.strip() if version_override else None

    def _ensure_layout(self) -> None:
        if not self._versions_dir.exists():
            raise FileNotFoundError(f"Prompt versions directory not found: {self._versions_dir}")

    def list_versions(self) -> list[str]:
        self._ensure_layout()
        versions = [p.stem for p in self._versions_dir.glob("*.json") if p.is_file()]
        return sorted(versions)

    def get_active_version(self) -> str:
        if self._version_override:
            return self._version_override

        self._ensure_layout()
        if self._active_file.exists():
            version = self._active_file.read_text(encoding="utf-8").strip()
            if version:
                return version

        versions = self.list_versions()
        if not versions:
            raise ValueError("No prompt versions found.")
        default_version = versions[0]
        self._active_file.write_text(default_version, encoding="utf-8")
        return default_version

    def set_active_version(self, version: str) -> None:
        if version not in self.list_versions():
            raise ValueError(f"Unknown prompt version: {version}")
        self._active_file.write_text(version, encoding="utf-8")

    def _load_version(self, version: str) -> dict[str, Any]:
        if version in self._cache:
            return self._cache[version]

        path = self._versions_dir / f"{version}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt pack not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "prompts" not in data:
            raise ValueError(f"Invalid prompt pack format in: {path}")

        self._cache[version] = data
        return data

    @staticmethod
    def _safe_format(template: str, variables: dict[str, Any]) -> str:
        # Keep unresolved placeholders as-is to avoid runtime crashes from missing optional fields.
        formatter = Formatter()
        out: list[str] = []
        for literal_text, field_name, format_spec, conversion in formatter.parse(template):
            out.append(literal_text)
            if field_name is None:
                continue
            if field_name in variables:
                value = variables[field_name]
                out.append(format(value, format_spec) if format_spec else str(value))
            else:
                suffix = (":" + format_spec) if format_spec else ""
                conv = ("!" + conversion) if conversion else ""
                out.append("{" + field_name + conv + suffix + "}")
        return "".join(out)

    def get_prompt(self, key: str, **variables: Any) -> str:
        version = self.get_active_version()
        pack = self._load_version(version)
        prompts = pack.get("prompts", {})
        if key not in prompts:
            raise KeyError(f"Prompt key '{key}' not found in version '{version}'")
        template = prompts[key]
        if not isinstance(template, str):
            raise TypeError(f"Prompt key '{key}' in version '{version}' must be a string")
        return self._safe_format(template, variables)
