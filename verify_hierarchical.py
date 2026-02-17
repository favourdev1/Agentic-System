import os
import asyncio
from agentic_system.orchestrator.graph import Orchestrator
from agentic_system.config.settings import get_settings


def test_mode(mode, user_input):
    print(f"\n--- Testing Mode: {mode} ---")
    os.environ["PROCESS_MODE"] = mode
    # Clear lru_cache for settings
    get_settings.cache_clear()

    orchestrator = Orchestrator()
    print(f"User Input: {user_input}")

    # We use a non-streaming invocation for clean output in verification
    result = orchestrator.invoke_with_metadata(user_input)

    print(f"Selected Agent: {result.get('selected_agent')}")
    print(f"Execution Mode: {result.get('execution_mode')}")
    print(f"Execution Reason: {result.get('execution_reason')}")


def main():
    # 1. Test Autonomous Mode with a simple question (should be DIRECT)
    test_mode("autonomous", "What is 2+2?")

    # 2. Test Autonomous Mode with a complex question (should be PLAN or HIERARCHICAL)
    test_mode(
        "autonomous", "Research tech trends and then give me a pep talk as a guru."
    )

    # 3. Test Forced Sequential Mode
    test_mode("sequential", "Search for AI news and summarize them.")

    # 4. Test Forced Hierarchical Mode
    test_mode("hierarchical", "Search for AI news and summarize them.")


if __name__ == "__main__":
    main()
