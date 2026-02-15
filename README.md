# Agentic System Scaffold (LangChain + LangGraph + LangSmith)

This project is a standardized Python baseline for a multi-agent system with:
- a central orchestrator
- plan-aware execution strategy (`direct` vs `plan`)
- class-based registries for agents and tools
- reusable tool groups
- LangSmith tracing
- API-tool placeholders (URL configs can be added later)

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set at least:
- `LLM_PROVIDER=gemini`
- `GOOGLE_API_KEY`
- `LANGSMITH_API_KEY` (if tracing enabled)

## 2) Run

```bash
agentic "Summarize current system design and suggest next steps"
agentic --stream "Summarize current system design and suggest next steps"
agentic --stream --trace-tools "Summarize current system design and suggest next steps"
agentic --generate-ui "Summarize current system design and suggest next steps"
```

Use default mode for blocking responses, or `--stream` for token streaming.
Use `--trace-tools` only if you want tool-status trace events during streaming.
Use `--session-id` to continue a persisted session and `--plan-step-budget` to run only part of a plan.
Use `--generate-ui` to attach structured render payloads (cards/table/mixed).

## 2.1) API modes

`POST /api/invoke` supports both modes:
- Blocking mode: `{ "prompt": "..." }`
- Streaming mode (SSE): `{ "prompt": "...", "stream": true }`
- Streaming with tool traces (SSE): `{ "prompt": "...", "stream": true, "trace_tools": true }`
- Optional explicit routing in both modes: `{ "prompt": "...", "agent_id": "skill_enhancer" }`
- Optional persistence controls: `{ "session_id": "existing-id", "plan_step_budget": 2 }`
- Optional generative UI payload: `{ "generate_ui": true }` (returns `ui_spec` in blocking mode and `type: \"ui\"` event in streaming mode)
- Responses include `prompt_version` for auditability.

Execution behavior:
- If `agent_id` is provided, orchestrator runs that agent directly (planning is bypassed).
- Otherwise, orchestrator decides whether to execute directly or generate/execute a multi-step plan.
- Session state (plan steps and completion status) is persisted in `SESSION_STORE_DIR` (default: `.agentic_sessions`).

## 2.2) Prompt Governance

Prompt packs are versioned on disk:
- `prompts/versions/*.json`
- `prompts/active_version.txt`
- `prompts/CHANGELOG.md`

CLI governance commands:
```bash
agentic --list-prompt-versions
agentic --show-prompt-version
agentic --set-prompt-version v1   # rollback example
agentic --set-prompt-version v2   # move forward again
```

Environment controls:
- `PROMPT_CONFIG_DIR` (default `prompts`)
- `PROMPT_VERSION` (optional override; if set, it forces that version at runtime)

## 3) Explore registry + graph

```bash
agentic --list-agents
agentic --list-tool-groups
agentic --show-graph mermaid
agentic --show-graph ascii
agentic --show-graph mermaid --save-graph graph.mmd
```

## 4) Architecture

- `src/agentic_system/orchestrator/graph.py`
  - `Orchestrator` class owns routing, strategy decision, planning, execution, and graph rendering
- `src/agentic_system/orchestrator/llm_factory.py`
  - `LLMFactory` class for provider-specific model initialization (`gemini` or `openai`)
- `src/agentic_system/agents/registry.py`
  - `AgentRegistry` class + `AgentSpec`
- `src/agentic_system/agents/tool_registry.py`
  - `ToolRegistry` class + `ToolSpec` + tool groups
- `src/agentic_system/tools/web/http_get.py`
  - reusable HTTP API tool adapter (URL-ready)

## 5) Add a new tool (standard way)

1. Implement tool builder in `src/agentic_system/agents/*/builders/` (or `src/agentic_system/tools/` if shared)
2. Register in `ToolRegistry._tools` in `src/agentic_system/agents/tool_registry.py`
3. Optionally include it in `ToolRegistry._groups`
4. Reference group/tool in an agent spec

## 6) Add a new agent (standard way)

1. Add `AgentSpec` in `AgentRegistry._agents` in `src/agentic_system/agents/registry.py`
2. Set `name`, `description`, `system_prompt`
3. Attach either `tool_groups=[...]`, `tool_names=[...]`, or both
4. If needed, tune semantic routing logic in `Orchestrator._llm_router`

## 7) API tools with URLs later

Wire endpoint URLs/env vars into your web tool builders and register them in:
- `src/agentic_system/agents/tool_registry.py`

No orchestrator changes are required.
