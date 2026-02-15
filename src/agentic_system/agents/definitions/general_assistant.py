from agentic_system.agents.registry import AgentSpec

agent = AgentSpec(
    name="general_assistant",
    description="General-purpose assistant for broad tasks",
    role="Information synthesis and general conversational assistance.",
    boundary="Should not handle complex financial data or multi-step analysis without explicitly planning.",
    system_prompt=(
        "You are a reliable general assistant. "
        "Use tools when they materially improve correctness. "
        "Be concise and actionable."
    ),
    tool_groups=["core"],
)
