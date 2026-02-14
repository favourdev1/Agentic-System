# Agentic System Scaffold (LangChain + LangGraph + LangSmith)

This project is a standardized Python baseline for a multi-agent system with:
- a central orchestrator
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
```

Use default mode for blocking responses, or `--stream` for token streaming.
Use `--trace-tools` only if you want tool-status trace events during streaming.

## 2.1) API modes

`POST /api/invoke` supports both modes:
- Blocking mode: `{ "prompt": "..." }`
- Streaming mode (SSE): `{ "prompt": "...", "stream": true }`
- Streaming with tool traces (SSE): `{ "prompt": "...", "stream": true, "trace_tools": true }`
- Optional explicit routing in both modes: `{ "prompt": "...", "agent_id": "skill_enhancer" }`

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
  - `Orchestrator` class owns routing, graph construction, invocation, and graph rendering
- `src/agentic_system/orchestrator/llm_factory.py`
  - `LLMFactory` class for provider-specific model initialization (`gemini` or `openai`)
- `src/agentic_system/agents/registry.py`
  - `AgentRegistry` class + `AgentSpec`
- `src/agentic_system/tools/registry.py`
  - `ToolRegistry` class + `ToolSpec` + tool groups
- `src/agentic_system/tools/web/http_get.py`
  - reusable HTTP API tool adapter (URL-ready)

## 5) Add a new tool (standard way)

1. Implement tool in `src/agentic_system/tools/`
2. Register in `ToolRegistry._tools` in `src/agentic_system/tools/registry.py`
3. Optionally include it in `ToolRegistry._groups`
4. Reference group/tool in an agent spec

## 6) Add a new agent (standard way)

1. Add `AgentSpec` in `AgentRegistry._agents` in `src/agentic_system/agents/registry.py`
2. Set `name`, `description`, `system_prompt`
3. Attach either `tool_groups=[...]`, `tool_names=[...]`, or both
4. If needed, tune semantic routing logic in `Orchestrator._llm_router`

## 7) API tools with URLs later

Wire endpoint URLs/env vars into `build_http_get_tool(...)` in:
- `src/agentic_system/tools/registry.py`

No orchestrator changes are required.
