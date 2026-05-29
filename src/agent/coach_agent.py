import sqlite3

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
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
    log_injury,
    log_injury_checkin,
    get_recent_workouts,
    log_workout_rpe_and_notes,
    save_workout_plan,
    get_current_workout_plan,
    update_planned_workout_status,
    get_nutrition_profile,
    get_progress_report,
    log_strength_sets,
    get_recent_strength_sets,
    get_onboarding_status,
    log_fitness_assessment,
    get_fitness_assessments,
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
    update_planned_workout_status,
    # Nutrition & progress
    get_nutrition_profile,
    get_progress_report,
    # Research
    query_running_database,
    search_coaching_books,
]

def get_athlete_context() -> str:
    """
    connect to the athlete's database (read only mode) and build a context string summarizing their recent training and current status.
    the summary includes today's training load (ATL, CTL, TSB), resting heart rate, HRV, sleep data, and the last 5 workouts with key metrics.
    returns a formatted string that can be included in the coach agent's system prompt to provide personalized advice based on the athlete's current 
    training condition and recent training history.
    """
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True) # open in read-only mode
        con.row_factory = sqlite3.Row # access columns by name
        
        # Get today's status (most recent daily health record)
        today = con.execute("""
            SELECT date, atl, ctl, tsb,
                   resting_heart_rate_bpm, hrv_sdnn_ms,
                   sleep_total_min, sleep_deep_min, sleep_rem_min
            FROM daily_health
            ORDER BY date DESC
            LIMIT 1
        """).fetchone()
        
        # Get last 5 workouts
        recent = con.execute("""
            SELECT start_date, activity_type, distance_km,
                   duration_min, avg_heart_rate_bpm, training_stress_score
            FROM workouts
            ORDER BY start_date DESC
            LIMIT 5
        """).fetchall()

        con.close()
    except Exception as exc:
        return f"(Unable to load athlete context: {exc})"

    lines = ["=== CURRENT ATHLETE STATUS ==="]

    if today:
        lines.append(f"Date: {today['date']}")
        if today["atl"] is not None:
            tsb = today["tsb"]
            readiness = (
                "very fresh" if tsb > 25
                else "race-ready" if tsb > 5
                else "normal training" if tsb > -10
                else "building / some fatigue" if tsb > -25
                else "HIGH FATIGUE — consider rest"
            )
            lines.append(
                f"Training load: ATL={today['atl']:.1f}  CTL={today['ctl']:.1f}  "
                f"TSB={tsb:.1f} ({readiness})"
            )
        if today["resting_heart_rate_bpm"]:
            lines.append(f"Resting HR: {today['resting_heart_rate_bpm']:.0f} bpm")
        if today["hrv_sdnn_ms"]:
            lines.append(f"HRV: {today['hrv_sdnn_ms']:.0f} ms")
        if today["sleep_total_min"]:
            lines.append(
                f"Last night: {today['sleep_total_min']:.0f} min sleep  "
                f"({today['sleep_deep_min'] or 0:.0f} deep  "
                f"{today['sleep_rem_min'] or 0:.0f} REM)"
            )

    if recent:
        lines.append("\nRecent workouts:")
        for w in recent:
            parts = [w["start_date"][:10], w["activity_type"]]
            if w["distance_km"]:
                parts.append(f"{w['distance_km']:.1f} km")
            if w["duration_min"]:
                parts.append(f"{w['duration_min']:.0f} min")
            if w["training_stress_score"]:
                parts.append(f"TSS={w['training_stress_score']:.0f}")
            lines.append("  " + " | ".join(parts))

    return "\n".join(lines)


def build_coach_graph():
    """
    Build and compile the LangGraph ReAct coaching agent.
    The agent will use the athlete context from the database and can call tools for additional information or actions.
    """
    llm = ChatOllama(model=config.MODEL_ID, temperature=0) # use temperature=0 for deterministic responses in coaching context
    llm_with_tools = llm.bind_tools(TOOLS) # bind the tools to the LLM so it can call them when needed
    tool_node = ToolNode(TOOLS) # create a tool node that can execute any of the defined tools

    def call_model(state: CoachState) -> dict:
        """
        this function is called at the "agent" node. 
        it builds the system prompt by combining the static persona with the dynamic athlete context,
        and then calls the LLM with the conversation history (including the system prompt) to get the model's response.
        input: the current state, which includes the conversation history and athlete context
        output: a new state with the model's response appended to the messages
        """
        system_prompt = build_system_prompt(state.get("athlete_context", ""))
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"]) # prepend the system prompt to the conversation history
        response = llm_with_tools.invoke(messages) # get the model's response, which may include tool calls
        return {"messages": [response]}

    def should_continue(state: CoachState) -> str:
        """
        check the last message from the model to see if it included any tool calls. 
        if it did, we need to go to the "tools" node to execute them and get the results before we can call the model again.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(CoachState) # create a state graph with the CoachState as the state type
    graph.add_node("agent", call_model) # add the main agent node that calls the model
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent") # set the entry point to the agent node
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())
