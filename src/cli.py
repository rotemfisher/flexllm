"""
FlexLLM Coach — interactive CLI.

Run from the project root:
    python -m src.cli
"""
from langchain_core.messages import AIMessageChunk

from src.agent.coach_agent import build_coach_graph, get_athlete_context

THREAD_ID = "default"

_AGENT_NODES = {"trainer", "physiotherapist", "recovery_coach", "dietitian"}


def main() -> None:
    print("Loading FlexLLM Coach (first run loads the embedding model, ~5s)...")
    athlete_ctx = get_athlete_context()
    run_config = {"configurable": {"thread_id": THREAD_ID}}

    with build_coach_graph() as graph:
        print("\nFlexLLM Coach ready. Type 'quit' to exit.\n")
        print(athlete_ctx)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
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
