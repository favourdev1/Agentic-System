from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    """Static definition of a tool and its builder function.

    Attributes:
        name: The unique identifier for the tool.
        builder: A callable that returns the tool instance.
        intent: Formal semantic purpose of the tool for developer clarity.
        status_message: A human-readable message to display when the tool starts.
        schema_notes: Expected input/output patterns and semantic constraints.
    """

    name: str
    builder: Callable[[], Any]
    intent: str = ""
    status_message: str = ""
    schema_notes: str = ""
