from __future__ import annotations

from typing import Literal, Union

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


class UiElement(BaseModel):
    # A single UI block that can be sequenced in any order.
    type: Literal["text", "table", "cards"]
    content: str | UiTable | list[UiCard]


class UiSpec(BaseModel):
    # Ordered list of UI elements for dynamic layout.
    elements: list[UiElement] = Field(default_factory=list)
    summary: str = Field(default="")
    layout: Literal["none", "cards", "table", "mixed", "blocks"] = "blocks"
    notes: list[str] = Field(default_factory=list)
