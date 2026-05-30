"""
FlexLLM Coach — interactive CLI.

Run from the project root:
    python -m src.cli
"""
from langchain_core.messages import AIMessageChunk

from src.agent.coach_agent import build_coach_graph, get_athlete_context
from src.agent.memory import SummaryStore, save_session_summary, maybe_refresh_weekly_summary
from src.config import config

THREAD_ID = "default"

_AGENT_NODES = {"trainer", "physiotherapist", "recovery_coach", "dietitian"}


def _persist_session_summaries(graph, run_config: dict, initial_message_count: int) -> None:
    """Generate and store daily summaries for any domain active this session.

    Reads messages added since session start from the graph checkpoint, groups
    them by the domain that was active (via active_agent on the state), and
    generates a compact bullet-point summary for each domain touched.
    """
    try:
        state = graph.get_state(run_config)
        all_messages = list(state.values.get("messages", []))
        new_messages = all_messages[initial_message_count:]
        if len(new_messages) < 4:
            return

        active_agent = state.values.get("active_agent") or "trainer"
        store = SummaryStore(config.DB_PATH)
        save_session_summary(new_messages, active_agent, store, config.MODEL_ID)
        maybe_refresh_weekly_summary(store, config.MODEL_ID)
    except Exception:
        pass  # Summaries are best-effort; never disrupt the exit flow.


def main() -> None:
    print("Loading FlexLLM Coach (first run loads the embedding model, ~5s)...")
    athlete_ctx = get_athlete_context()
    run_config = {"configurable": {"thread_id": THREAD_ID}}

    with build_coach_graph() as graph:
        # Snapshot message count so we can extract only this session's messages later.
        try:
            init_state = graph.get_state(run_config)
            initial_message_count = len(list(init_state.values.get("messages", [])))
        except Exception:
            initial_message_count = 0

        print("\nFlexLLM Coach ready. Type 'quit' to exit.\n")
        print(athlete_ctx)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                _persist_session_summaries(graph, run_config, initial_message_count)
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                _persist_session_summaries(graph, run_config, initial_message_count)
                break

            current_agent = None
            for chunk, metadata in graph.stream(
                {"messages": [("human", user_input)], "athlete_context": athlete_ctx},
                config=run_config,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node")
                if node in _AGENT_NODES and isinstance(chunk, AIMessageChunk):
                    if node != current_agent:
                        if current_agent is not None:
                            print()
                        print(f"[{node.replace('_', ' ').title()}]: ", end="", flush=True)
                        current_agent = node
                    if isinstance(chunk.content, str) and chunk.content:
                        print(chunk.content, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    main()
