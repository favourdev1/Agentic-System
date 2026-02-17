from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    """Formal specification for an AI agent.

    Attributes:
        name: Unique ID of the agent.
        description: High-level summary used for semantic routing.
        role: Formal operational definition of the agent's persona.
        backstory: Historical/context framing for behavior shaping.
        goals: Ordered objective list to bias execution priorities.
        boundary: Explicit constraints on what the agent should NOT do.
        system_prompt: The actual instruction set used by the LLM.
        tool_names: Explicit tool IDs assigned to this agent.
        tool_groups: Pre-defined group IDs assigned to this agent.
    """

    name: str
    description: str
    role: str
    boundary: str
    system_prompt: str
    backstory: str = ""
    goals: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    tool_groups: list[str] = field(default_factory=list)

    def runtime_system_prompt(self) -> str:
        """Build the final runtime prompt from structured identity + base prompt."""
        sections: list[str] = []
        if self.role:
            sections.append(f"** Role **: {self.role}")
        if self.backstory:
            sections.append(f"** Backstory **: {self.backstory}")
        if self.goals:
            goals_text = "\n".join([f"- {goal}" for goal in self.goals if goal.strip()])
            if goals_text:
                sections.append(f"** Goals **:\n{goals_text}")
        if self.boundary:
            sections.append(f"** Operating boundaries **: {self.boundary}")
        if self.system_prompt:
            sections.append(f"** Core instructions **: {self.system_prompt}")
        return "\n\n".join(sections).strip()


import importlib
import pkgutil
from agentic_system.agents import definitions


class AgentRegistry:
    """Central registry that dynamically discovers agents in the 'definitions' package."""

    _cached_agents: dict[str, AgentSpec] | None = None

    @classmethod
    def _discover_agents(cls) -> dict[str, AgentSpec]:
        if cls._cached_agents is not None:
            return cls._cached_agents

        agents = {}
        # Iterate through all modules in the 'definitions' package
        for _, name, is_pkg in pkgutil.iter_modules(definitions.__path__):
            if is_pkg:
                continue

            # Import the module dynamically
            module_name = f"agentic_system.agents.definitions.{name}"
            module = importlib.import_module(module_name)

            # Look for an 'agent' attribute that is an AgentSpec
            agent_spec = getattr(module, "agent", None)
            if isinstance(agent_spec, AgentSpec):
                agents[agent_spec.name] = agent_spec

        cls._cached_agents = agents
        return agents

    @classmethod
    def list_agents(cls) -> list[AgentSpec]:
        return list(cls._discover_agents().values())

    @classmethod
    def get_agent(cls, name: str) -> AgentSpec:
        agents = cls._discover_agents()
        if name not in agents:
            print(f"Unknown agent: {name}")
            raise ValueError(f"Unknown agent: {name}")
        return agents[name]

    @classmethod
    def descriptions(cls) -> dict[str, str]:
        return {name: spec.description for name, spec in cls._discover_agents().items()}
