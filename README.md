# 🤖 Repingme-Ai: Agentic System Scaffold

A high-performance, modular multi-agent system built with **LangChain**, **LangGraph**, and **FastAPI**. This scaffold provides a production-ready baseline for building intelligent agents with complex reasoning, automated planning, and rich generative UIs.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [Anaconda](https://www.anaconda.com/) (Recommended) or `venv`

### 2. Installation
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .

# Configure environment
cp .env.example .env
```

### 3. Environment Configuration
Edit your `.env` file and set the following critical variables:
- `LLM_PROVIDER`: `gemini` (default) or `openai`
- `GOOGLE_API_KEY`: Your Google AI Studio key
- `LANGSMITH_API_KEY`: (Optional) For detailed trace auditing
- `SESSION_STORE_DIR`: Where chat sessions are persisted (default: `.agentic_sessions`)

---

## 🏗️ Project Structure

The codebase follows a strictly modular "Brain" architecture:

```text
repingme-ai/
├── src/agentic_system/
│   ├── agents/             # Agent definitions & registration
│   │   ├── common/         # Shared agent patterns
│   │   └── registry.py     # Central Agent Registry [CRITICAL]
│   ├── tools/              # Tool implementations
│   │   ├── builders/       # Individual tool constructors
│   │   ├── registry.py     # Central Tool Registry [CRITICAL]
│   │   └── web/            # Scrapers and API adapters
│   ├── orchestrator/       # The "Brain" (LangGraph Logic)
│   │   ├── graph.py        # Logic Flow & Intent Routing
│   │   └── llm_factory.py  # Model provider management
│   ├── prompting/          # Versioned Prompt Management
│   └── web/                # High-fidelity Agent Console (HTML/JS)
├── prompts/                # Governance: Versioned JSON prompt packs
└── .agentic_sessions/      # Persisted planning & memory state
```

---

## 🎮 How to Run

### Command Line Interface (CLI)
```bash
# Simple usage
agentic "Help me plan a trip to Tokyo"

# Streaming mode with trace tools enabled
agentic --stream --trace-tools "Search for the latest NVIDIA stock price"

# Generate UI specs for frontend rendering
agentic --generate-ui "Give me a comparison table of 3 electric cars"
```

### Web Agent Console
1. **Start the server**:
   ```bash
   agentic --server --port 8888 --reload
   ```
2. **Open your browser**: [http://127.0.0.1:8888](http://127.0.0.1:8888)
3. **Features**:
   - **Real-time Thought Trace**: Watch the AI's internal planning process.
   - **Generative UI**: View cards, tables, and mixed media rendered by the AI.
   - **Developer Tabs**: Inspect raw events, JSON payloads, and session params.

---

## 🛠️ Extending the System

### 1. Adding a New Tool
1. **Create Builder**: Add a new file in `src/agentic_system/tools/builders/` that returns a `StructuredTool`.
2. **Register Tool**: Add your tool to the `_tools` map in `src/agentic_system/tools/registry.py`:
   ```python
   "my_tool": ToolSpec(name="my_tool", builder=build_my_tool)
   ```
3. **Group (Optional)**: Add it to a group in `_groups` to make it available to multiple agents.

### 2. Creating a New Agent
1. **Define Spec**: Open `src/agentic_system/agents/registry.py`.
2. **Register**: Add a new `AgentSpec` to the `_agents` dictionary:
   ```python
   "content_writer": AgentSpec(
       name="content_writer",
       description="Expert at SEO writing",
       system_prompt="You are an SEO expert...",
       tool_groups=["core"]
   )
   ```
3. **Routing**: The orchestrator will automatically pick up the new agent via semantic intent detection.

---

## 🧠 Core Concepts

### Direct vs. Planning Mode
- **Direct**: For simple queries, the system executes a single agent pass for speed.
- **Planning**: For complex requests, the system generates a multi-step checklist, executes them sequentially, and synthesizes a final answer.

### AI Thought Trace
The UI includes a **Thought Process** trace that exposes the AI's internal reasoning. It handles:
- `plan`: The interactive execution checklist.
- `status`: Live pulse indicators during tool usage.
- `step_result`: Updates on individual planning steps.

### Prompt Governance
Prompts are not hardcoded. They are versioned assets in `prompts/versions/`. Use the CLI to roll back or update prompt logic across all agents instantly:
```bash
agentic --list-prompt-versions
agentic --set-prompt-version v2
```

---

## 🔒 Security & Safety
- **State Persistence**: Sessions are saved to disk, allowing for long-running workflows to resume.
- **ORM Strictness**: All database-related tools (if added) MUST use Laravel-style ORM patterns to avoid SQL injection.
- **Prompt Safety**: Always use the system-governed prompt versions for production deployments.
