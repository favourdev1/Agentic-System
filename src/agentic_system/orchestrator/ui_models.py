from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UiCard(BaseModel):
    # Simple card primitive for KPI-like or highlight-like content.
    title: str = Field(default="")
    value: str = Field(default="")
    description: str = Field(default="")


class UiTable(BaseModel):
    # Tabular primitive for grid rendering in client-side UI.
    title: str = Field(default="")
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class UiSpec(BaseModel):
    # High-level UI payload your frontend can render directly.
    layout: Literal["none", "cards", "table", "mixed"] = "none"
    summary: str = Field(default="")
    cards: list[UiCard] = Field(default_factory=list)
    table: UiTable | None = None
    notes: list[str] = Field(default_factory=list)
