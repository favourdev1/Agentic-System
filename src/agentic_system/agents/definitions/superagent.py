from agentic_system.agents.registry import AgentSpec

agent = AgentSpec(
    name="superagent",
    description="Assistant specialized in superagent",
    role="Expert at superagent.",
    boundary="Avoid tasks outside of specialized domain.",
    system_prompt=(
        "You are a specialized assistant."
    ),
    tool_groups=["core"],
)
