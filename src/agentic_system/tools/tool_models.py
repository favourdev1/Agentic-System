from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    """Static definition of a tool and its builder function."""

    name: str
    builder: Callable[[], Any]
