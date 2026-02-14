from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    system_prompt: str
    tool_names: list[str] = field(default_factory=list)
    tool_groups: list[str] = field(default_factory=list)


class AgentRegistry:
    # Standardized agent registry. Add agents here.
    _agents: dict[str, AgentSpec] = {
        "general_assistant": AgentSpec(
            name="general_assistant",
            description="General-purpose assistant for broad tasks",
            system_prompt=(
                "You are a reliable general assistant. "
                "Use tools when they materially improve correctness. "
                "Be concise and actionable."
            ),
            tool_groups=["core"],
        ),
        "analysis_assistant": AgentSpec(
            name="analysis_assistant",
            description="Analytical assistant for structured reasoning and decomposition",
            system_prompt=(
                "You are an analytical assistant. "
                "Break tasks into steps, validate assumptions, and return clear conclusions."
            ),
            tool_groups=["analysis_plus_api"],
        ),
        "skill_enhancer": AgentSpec(
            name="skill_enhancer",
            description="Expert at expanding and refining AI skill instructions",
            system_prompt=(
                "You are an expert prompt engineer. Your task is to take a brief description of an AI skill "
                "and expand it into a comprehensive set of professional instructions. "
                "You should include: "
                "1. A clear personality description. "
                "2. The Do's: specific behaviors and styles to adopt. "
                "3. The Dont's: specific edge cases or behaviors to avoid. "
                "4. Step-by-step logic if applicable. "
                "Format the output as a clean, actionable professional instruction set."
            ),
            tool_groups=["core"],
        ),
    }

    @classmethod
    def list_agents(cls) -> list[AgentSpec]:
        return list(cls._agents.values())

    @classmethod
    def get_agent(cls, name: str) -> AgentSpec:
        if name not in cls._agents:
            raise ValueError(f"Unknown agent: {name}")
        return cls._agents[name]

    @classmethod
    def descriptions(cls) -> dict[str, str]:
        return {name: spec.description for name, spec in cls._agents.items()}
