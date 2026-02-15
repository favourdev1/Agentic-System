# Agentic System

A modular agent orchestration system built with LangChain, LangGraph, FastAPI, and a pluggable session persistence layer (file or database).

This README is implementation-accurate for the current codebase and documents architecture, runtime behavior, extension patterns, and operational commands.

## Table of Contents
1. Overview
2. Core Features
3. Architecture
4. Folder Structure
5. Requirements
6. Installation
7. Environment Configuration
8. Running the System
9. CLI Reference
10. API Reference
11. Streaming Event Contract
12. Session and Memory Model
13. Planning and Execution Strategy
14. Agents
15. Tools and Tool Registry
16. Prompt Governance
17. Database and Migrations
18. Add a New Agent
19. Add a New Tool
20. Troubleshooting
21. Known Gaps and Next Improvements

## 1. Overview
The system processes user requests through an orchestrator that can:
- route to a target agent,
- decide direct execution or plan-based execution,
- execute with tools,
- stream events and tokens,
- optionally generate structured UI payloads,
- persist session state.

Primary entrypoints:
- CLI: `agentic`
- API server: FastAPI app in `src/agentic_system/api.py`
- Browser UI: served at `/` from `src/agentic_system/web/index.html`

## 2. Core Features
- Dynamic agent discovery from `src/agentic_system/agents/definitions`
- Dynamic tool discovery from `src/agentic_system/tools/definitions`
- ReAct-style worker execution (`langgraph.prebuilt.create_react_agent`)
- Strategy selector: `direct` or `plan`
- Stepwise planning with budget control (`plan_step_budget`)
- Streaming support (SSE via `/api/invoke`)
- Optional generative UI payload (`ui_spec`)
- Prompt version governance (`prompts/versions/*.json`)
- Session persistence backend switch:
  - file: JSON files
  - db: SQLAlchemy + Alembic-backed storage

## 3. Architecture
High-level pipeline:
1. Request enters orchestrator.
2. Session is prepared (`session_id` resolved/created, context built).
3. Router chooses agent unless `agent_id` is explicitly provided.
4. Strategy model decides `direct` vs `plan` (explicit `agent_id` forces `direct`).
5. Worker executes:
   - direct: single pass,
   - plan: step execution + synthesis.
6. Optional UI payload generation from final text.
7. Session data persisted (`last_run`, plan state, history).

Core orchestrator file:
- `src/agentic_system/orchestrator/graph.py`

## 4. Folder Structure
```text
src/agentic_system/
├── api.py
├── main.py
├── agents/
│   ├── registry.py
│   └── definitions/
│       ├── general_assistant.py
│       ├── analysis_assistant.py
│       ├── skill_enhancer.py
│       └── superagent.py
├── commands/
│   ├── make_agent.py
│   └── make_tool.py
├── config/
│   ├── settings.py
│   └── database.py
├── database/
│   ├── base.py
│   ├── engine.py
│   ├── session.py
│   └── init_db.py
├── models/
│   └── session_record.py
├── orchestrator/
│   ├── graph.py
│   ├── llm_factory.py
│   ├── state.py
│   └── ui_models.py
├── prompting/
│   └── manager.py
├── session_store/
│   ├── __init__.py
│   ├── interface.py
│   ├── record_ops.py
│   ├── file_store.py
│   └── db_store.py
├── tools/
│   ├── registry.py
│   ├── tool_models.py
│   ├── definitions/
│   │   ├── calculator.py
│   │   ├── web_search.py
│   │   ├── web_scrape.py
│   │   └── bank_account_api.py
│   ├── builders/
│   │   ├── common/
│   │   │   ├── core.py
│   │   │   └── web.py
│   │   └── analysis/
│   │       └── web.py
│   └── tool_factory/
│       ├── core/
│       │   └── calculator.py
│       └── web/
│           ├── search.py
│           ├── scraper.py
│           └── http_get.py
└── web/
    └── index.html

prompts/
├── active_version.txt
└── versions/
    ├── v1.json
    └── v2.json

migrations/
├── env.py
├── script.py.mako
└── versions/
    └── 0001_create_session_records.py
```

## 5. Requirements
- Python >= 3.11
- pip tooling for editable install
- API key for chosen LLM provider:
  - Gemini (`GOOGLE_API_KEY`) or
  - OpenAI (`OPENAI_API_KEY`)

## 6. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Alternative:
```bash
pip install -r requirements.txt
```

## 7. Environment Configuration
Use `.env` and keep secrets out of git.

Key variables:

LLM/provider:
- `LLM_PROVIDER=gemini|openai`
- `GOOGLE_API_KEY=`
- `GEMINI_MODEL=gemini-1.5-flash` (or other model id)
- `OPENAI_API_KEY=`
- `OPENAI_MODEL=gpt-4o-mini`

LangSmith:
- `LANGSMITH_API_KEY=`
- `LANGSMITH_TRACING=true`
- `LANGSMITH_PROJECT=agentic-system-local`

Tool/API defaults:
- `DEFAULT_API_TIMEOUT_SECONDS=20`
- `BANK_API_BASE_URL=http://localhost:8000/api/v2/bank-account/request`
- `BANK_API_AUTH_TOKEN=`
- `BANK_API_SESSION_COOKIE=`

Session persistence:
- `SESSION_STORE_BACKEND=file|db`
- `SESSION_STORE_DIR=.agentic_sessions` (used by file backend)

Database:
- `DATABASE_URL=sqlite:///./agentic_system.db`
- `DATABASE_ECHO=false`
- `DATABASE_AUTO_MIGRATE=true`

Prompt governance:
- `PROMPT_CONFIG_DIR=prompts`
- `PROMPT_VERSION=` (optional hard override)

## 8. Running the System
### Start API server
```bash
agentic server --host 0.0.0.0 --port 8888 --reload
```

### Open browser UI
- `http://127.0.0.1:8888`

### Basic CLI chat
```bash
agentic chat "Summarize the latest AI model families"
```

Compatibility mode also works:
```bash
agentic "Summarize the latest AI model families"
```

## 9. CLI Reference
Command groups are defined in `src/agentic_system/main.py`.

### `chat`
```bash
agentic chat "<prompt>" [--stream] [--trace-tools] [--session-id <id>] [--plan-step-budget <n>] [--generate-ui]
```

### `server`
```bash
agentic server [--host 0.0.0.0] [--port 8888] [--reload]
```

### `list:agents`
```bash
agentic list:agents
```

### `list:tools`
```bash
agentic list:tools
```
Prints dynamic groups and all discovered tools.

### `show:graph`
```bash
agentic show:graph --format mermaid
agentic show:graph --format ascii
agentic show:graph --format mermaid --save graph.mmd
```

### `prompts`
```bash
agentic prompts --list
agentic prompts --show
agentic prompts --set v2
```

### Scaffolding commands
```bash
agentic make:agent market_analyst --role "Financial analyst" --boundary "No legal advice" --description "Market intelligence specialist"
agentic make:tool fx_rates --intent "Retrieve exchange rates" --schema-notes "query:string -> json" --groups core analysis_plus_api
```

## 10. API Reference
Base: server root (e.g. `http://127.0.0.1:8888`)

### `GET /`
Serves web UI (`index.html`).

### `GET /api/health`
Health check.

### `GET /api/docs`
Swagger UI endpoint.

### `POST /api/invoke`
Primary inference endpoint.

Request body:
```json
{
  "prompt": "...",
  "stream": false,
  "trace_tools": false,
  "generate_ui": false,
  "agent_id": null,
  "session_id": null,
  "plan_step_budget": null
}
```

Non-stream response:
```json
{
  "response": "...",
  "session_id": "...",
  "execution_mode": "direct|plan",
  "selected_agent": "...",
  "prompt_version": "v1|v2|...",
  "ui_spec": { }
}
```

When `stream=true`, response is `text/event-stream` with SSE messages.

### `POST /api/enhance-skill`
Body:
```json
{
  "title": "...",
  "description": "..."
}
```
Routes request explicitly to `skill_enhancer`.

### `GET /api/get-models`
Pass-through provider model listing for currently selected provider.

## 11. Streaming Event Contract
SSE events from `/api/invoke` with `stream=true` include normalized payloads.

Event types produced by orchestrator stream:
- `metadata`
- `status`
- `token`
- `plan`
- `step_result`
- `ui`
- terminal server event: `{"type":"done"}`

Typical sequence:
1. `metadata` (routing info)
2. zero or more `status`/`token`
3. if plan mode: `plan`, then step updates
4. optional `ui`
5. final `metadata` (stage `done`)
6. `done`

Notes:
- `trace_tools=false` suppresses tool status lines from stream.
- Token streaming fallback exists: if no token stream but final text exists, a synthetic `token` is emitted.

## 12. Session and Memory Model
Session store contract in `src/agentic_system/session_store/interface.py`.

Supported backends:
- File backend: `FileSessionStore` stores JSON per session under `SESSION_STORE_DIR`.
- DB backend: `DbSessionStore` stores session JSON payload in `session_records` table.

Persisted fields include:
- `session_id`
- `plan` (objective + steps + statuses)
- `last_run`
- `run_history` (bounded to recent entries)

Context summary is built by `record_ops.build_context(...)` and contains:
- previous input and response summary,
- recent turns,
- plan status (completed/pending/failed).

Important behavior:
- A new session is created when `session_id` is not provided.
- For continuity, always send the same `session_id`.

## 13. Planning and Execution Strategy
Routing and strategy are model-driven with structured outputs:
- Router output model: `IntentResponse`
- Strategy output model: `ExecutionDecision`
- Plan output model: `ExecutionPlan`

Decision rules:
- explicit `agent_id` => forced `direct`
- otherwise model decides `direct` or `plan`

Plan mode:
- creates 2-6 steps (normalized)
- executes sequentially
- respects `plan_step_budget`
- synthesizes final response if all steps completed
- returns progress summary if incomplete/failed

## 14. Agents
Agents are dynamically discovered from `src/agentic_system/agents/definitions/*.py`.

Each file exports:
- `agent = AgentSpec(...)`

Current discovered agents include:
- `general_assistant`
- `analysis_assistant`
- `skill_enhancer`
- `superagent` (generated scaffold example)

Agent schema fields:
- `name`
- `description`
- `role`
- `boundary`
- `system_prompt`
- `tool_names`
- `tool_groups`

## 15. Tools and Tool Registry
Tools are discovered dynamically from `src/agentic_system/tools/definitions/*.py`.

Each tool definition exports:
- `tool = ToolSpec(...)`

Tool layers:
- `tool_factory/`: concrete tool implementations
- `builders/`: composition/wrappers that return `StructuredTool`
- `definitions/`: declarative registration using `ToolSpec`
- `registry.py`: dynamic discovery, group resolution, and instantiation

Groups are dynamic based on `ToolSpec.groups` values across discovered tools.

Current group usage examples:
- `core`
- `analysis_plus_api`

## 16. Prompt Governance
Prompt manager: `src/agentic_system/prompting/manager.py`

Prompt packs:
- `prompts/versions/v1.json`
- `prompts/versions/v2.json`

Active version:
- stored in `prompts/active_version.txt`, unless overridden by `PROMPT_VERSION` env.

Design details:
- safe formatter keeps unresolved placeholders intact (prevents hard crashes on missing optional values)
- version switching supports runtime prompt rollback

## 17. Database and Migrations
ORM stack:
- SQLAlchemy ORM model: `SessionRecord`
- Alembic migration tooling

Current table:
- `session_records(session_id, payload, created_at, updated_at)`

Switch to DB backend:
```bash
export SESSION_STORE_BACKEND=db
export DATABASE_URL=sqlite:///./agentic_system.db
```

Run migrations:
```bash
alembic upgrade head
```

Create migration:
```bash
alembic revision --autogenerate -m "add my table"
```

Rollback one migration:
```bash
alembic downgrade -1
```

## 18. Add a New Agent
Option A (generator):
```bash
agentic make:agent procurement_bot --role "Procurement analyst" --boundary "No legal decisions" --description "Vendor and pricing assistant"
```

Option B (manual):
1. Create file under `src/agentic_system/agents/definitions/`.
2. Export `agent = AgentSpec(...)`.
3. Run `agentic list:agents` to confirm discovery.

## 19. Add a New Tool
Option A (generator):
```bash
agentic make:tool weather_lookup --intent "Get weather data" --schema-notes "query:string" --groups core
```

Option B (manual, recommended for production tools):
1. Implement concrete logic in `tools/tool_factory/...`.
2. Add builder in `tools/builders/...`.
3. Add `tools/definitions/<name>.py` with `tool = ToolSpec(...)`.
4. Verify with `agentic list:tools`.

## 20. Troubleshooting
### `Unknown tool group`
Cause: group requested by agent is not present in discovered `ToolSpec.groups`.
Fix: confirm tool definitions and group names in `tools/definitions`.

### No memory continuity between turns
Cause: new session each call.
Fix: pass same `session_id` for follow-up requests.

### `GOOGLE_API_KEY is not set` or provider errors
Cause: provider env not configured.
Fix: set required API key and model in `.env`.

### DB backend not persisting
Cause: wrong backend/env or migration not run.
Fix:
1. set `SESSION_STORE_BACKEND=db`
2. set valid `DATABASE_URL`
3. run `alembic upgrade head`

### Streaming appears empty
Cause: provider/model may not emit incremental tokens for that run.
Fix: check final `metadata` and fallback token behavior.

## 21. Known Gaps and Next Improvements
- Session context is used in routing/mode/planning context; direct worker prompt memory can be further strengthened with richer conversation replay.
- DB schema currently stores session payload as JSON blob. A normalized relational schema (`sessions`, `runs`, `plans`, `plan_steps`) would improve analytics and queryability.
- Automated tests and CI checks should be added for orchestrator branches, SSE behavior, and migration integrity.
- Frontend UI can be split into maintainable components if migrated from single-file HTML/JS to a framework.
