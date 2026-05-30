import sqlite3
from contextlib import contextmanager

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import config
from src.models.agent_state import CoachState
from src.promts.system_promt import build_system_prompt
from src.tools import (
    search_coaching_books,
    query_running_database,
    get_vdot_paces,
    get_daily_readiness,
    get_active_injuries,
    get_injury_recovery_trend,
    resolve_injury,
    log_injury,
    log_injury_checkin,
    get_recent_workouts,
    log_workout_rpe_and_notes,
    save_workout_plan,
    get_current_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    get_nutrition_profile,
    get_progress_report,
    log_strength_sets,
    get_recent_strength_sets,
    get_onboarding_status,
    log_fitness_assessment,
    get_fitness_assessments,
    update_athlete_profile,
)

TOOLS = [
    # Onboarding & assessment
    get_onboarding_status,
    log_fitness_assessment,
    get_fitness_assessments,
    # Readiness & health
    get_daily_readiness,
    get_vdot_paces,
    # Injury
    get_active_injuries,
    get_injury_recovery_trend,
    resolve_injury,
    log_injury,
    log_injury_checkin,
    # Training history & feedback
    get_recent_workouts,
    log_workout_rpe_and_notes,
    # Strength tracking
    get_recent_strength_sets,
    log_strength_sets,
    # Planning
    get_current_workout_plan,
    save_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    # Nutrition & progress
    get_nutrition_profile,
    get_progress_report,
    # Profile management
    update_athlete_profile,
    # Research
    query_running_database,
    search_coaching_books,
]


def get_athlete_context() -> str:
    """Return static profile info to seed the system prompt."""
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        profile = con.execute(
            "SELECT fitness_level, current_goal, secondary_goal FROM athlete_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        con.close()
    except Exception as exc:
        return f"(Unable to load athlete profile: {exc})"

    if not profile:
        return "(No athlete profile found — call get_onboarding_status to set up.)"

    lines = ["=== ATHLETE PROFILE ==="]
    lines.append(f"Fitness level: {profile['fitness_level']}")
    if profile["current_goal"]:
        lines.append(f"Primary goal: {profile['current_goal']}")
    if profile["secondary_goal"]:
        lines.append(f"Secondary goal: {profile['secondary_goal']}")
    return "\n".join(lines)


@contextmanager
def build_coach_graph():
    """
    Build and compile the LangGraph ReAct coaching agent.
    Yields the compiled graph inside a SqliteSaver context so conversation
    history persists across CLI restarts.
    """
    llm = ChatOllama(model=config.MODEL_ID, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def call_model(state: CoachState) -> dict:
        system_prompt = build_system_prompt(state.get("athlete_context", ""))
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: CoachState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(CoachState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    with SqliteSaver.from_conn_string(config.DB_PATH) as checkpointer:
        yield graph.compile(checkpointer=checkpointer)
