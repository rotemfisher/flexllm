"""
FlexLLM Coach — interactive CLI.

Run from the project root:
    python -m src.cli
"""
import uuid

from src.agent.coach_agent import build_coach_graph, get_athlete_context


def main() -> None:
    print("Loading FlexLLM Coach (first run loads the embedding model, ~5s)...")
    graph = build_coach_graph()
    athlete_ctx = get_athlete_context()
    session_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": session_id}}

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

        result = graph.invoke(
            {"messages": [("human", user_input)], "athlete_context": athlete_ctx},
            config=run_config,
        )

        response = result["messages"][-1].content
        print(f"\nCoach: {response}\n")


if __name__ == "__main__":
    main()
