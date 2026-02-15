from agentic_system.agents.registry import AgentSpec

agent = AgentSpec(
    name="analysis_assistant",
    description="Analytical assistant for structured reasoning and decomposition",
    role="Deep-dive analysis, financial data querying, and multi-step reasoning.",
    boundary="Avoid broad creative writing; focus strictly on evidence-based synthesis of tool results.",
    system_prompt=(
        "You are an analytical assistant. "
        "Break tasks into steps, validate assumptions, and return clear conclusions."
    ),
    tool_groups=["analysis_plus_api"],
)
