from typing import Annotated, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

# Every handoff lands on the target's gather node so startup data is always
# pre-loaded before the agent's first LLM call, regardless of activation path.
_GATHER_NODE: dict[str, str] = {
    "trainer":         "gather_trainer_context",
    "physiotherapist": "gather_physio_context",
    "recovery_coach":  "gather_recovery_context",
    "dietitian":       "gather_dietitian_context",
    "psychologist":    "gather_psychologist_context",
}


@tool
def trainer_transfer(
    target: Literal["physiotherapist", "recovery_coach", "dietitian", "psychologist"],
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    HANDOFF — transfer control from the Trainer to another specialist.

    CRITICAL: Call this as the LAST tool in your response.
    EXCEPTION: In the NEW PLAN CREATION PROTOCOL you MUST call save_workout_plan
    AND this tool together in the same response — that is the only allowed pairing.
    NEVER combine this with read-only domain tools (get_recent_workouts,
    get_vdot_paces, get_current_workout_plan, etc.) in the same turn.
    Complete all data-gathering and analysis first, THEN call this tool
    (optionally paired with save_workout_plan if saving a new plan).

    target — choose exactly one string:
      "physiotherapist"  : athlete reports pain, injury, or movement limitation
      "recovery_coach"   : TSB < -20, HRV critically low, sleep < 5h, or fatigue-only topic
      "dietitian"        : nutrition, meal planning, macros, weight management, fuelling
      "psychologist"     : mental blocks, anxiety, motivation loss, confidence issues, performance slump

    reason — concise handoff note for the receiving agent (key values, clinical context).
    """
    return Command(
        goto=_GATHER_NODE[target],
        update={
            "active_agent": target,
            "handoff_reason": reason,
            "pending_agents": [],
            "messages": [ToolMessage(f"Transferred to {target}. Reason: {reason}", tool_call_id=tool_call_id)],
        },
    )


@tool
def physio_transfer(
    target: Literal["trainer", "recovery_coach", "dietitian", "psychologist"],
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    HANDOFF — transfer control from the Physiotherapist to another specialist.

    CRITICAL: Call this as the LAST and ONLY tool in your response.
    NEVER combine this with a domain tool (get_active_injuries, log_injury_checkin,
    save_workout_plan, etc.) in the same turn.
    Complete all injury assessment work first, THEN call this tool alone.

    target — choose exactly one string:
      "trainer"          : injury resolved, athlete cleared — include return-to-train protocol in reason
      "recovery_coach"   : accumulated fatigue is the root cause of the injury
      "dietitian"        : dietary support needed (collagen synthesis, anti-inflammatory nutrition)
      "psychologist"     : fear of re-injury, confidence loss, or chronic pain affecting mental state

    reason — handoff note with return-to-train restrictions or clinical context for the receiving agent.
    """
    return Command(
        goto=_GATHER_NODE[target],
        update={
            "active_agent": target,
            "handoff_reason": reason,
            "pending_agents": [],
            "messages": [ToolMessage(f"Transferred to {target}. Reason: {reason}", tool_call_id=tool_call_id)],
        },
    )


@tool
def recovery_transfer(
    target: Literal["trainer", "physiotherapist", "dietitian", "psychologist"],
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    HANDOFF — transfer control from the Recovery Coach to another specialist.

    CRITICAL: Call this as the LAST and ONLY tool in your response.
    NEVER combine this with a domain tool (get_daily_readiness, update_planned_workout_status,
    get_current_workout_plan, etc.) in the same turn.
    Complete readiness assessment first, THEN call this tool alone.

    target — choose exactly one string:
      "trainer"          : readiness assessed, session modification determined, back to training
      "physiotherapist"  : fatigue symptoms may indicate an underlying injury
      "dietitian"        : under-fuelling or caloric deficit is driving poor recovery
      "psychologist"     : psychological stress or burnout is the primary recovery barrier

    reason — handoff note with TSB, HRV, and sleep values plus clinical rationale.
    """
    return Command(
        goto=_GATHER_NODE[target],
        update={
            "active_agent": target,
            "handoff_reason": reason,
            "pending_agents": [],
            "messages": [ToolMessage(f"Transferred to {target}. Reason: {reason}", tool_call_id=tool_call_id)],
        },
    )


@tool
def dietitian_transfer(
    target: Literal["trainer", "physiotherapist", "recovery_coach", "psychologist"],
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    HANDOFF — transfer control from the Dietitian to another specialist.

    CRITICAL: Call this as the LAST and ONLY tool in your response.
    NEVER combine this with a domain tool (get_nutrition_profile, update_athlete_profile,
    get_recent_workouts, etc.) in the same turn.
    Complete all nutritional assessment first, THEN call this tool alone.

    target — choose exactly one string:
      "trainer"          : nutrition plan set, athlete has training or workout questions
      "physiotherapist"  : dietary topic intersects with injury (bone stress, tendon, inflammation)
      "recovery_coach"   : nutrition question relates to sleep quality, HRV, or recovery capacity
      "psychologist"     : disordered eating pattern, body image concern, or emotional eating

    reason — handoff note with nutritional context and recommendations already provided.
    """
    return Command(
        goto=_GATHER_NODE[target],
        update={
            "active_agent": target,
            "handoff_reason": reason,
            "pending_agents": [],
            "messages": [ToolMessage(f"Transferred to {target}. Reason: {reason}", tool_call_id=tool_call_id)],
        },
    )


@tool
def psychologist_transfer(
    target: Literal["trainer", "physiotherapist", "recovery_coach", "dietitian"],
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    HANDOFF — transfer control from the Psychologist to another specialist.

    CRITICAL: Call this as the LAST and ONLY tool in your response.
    NEVER combine this with a domain tool (get_recent_workouts, get_daily_readiness,
    search_knowledge_base, etc.) in the same turn.
    Complete all psychological assessment and intervention first, THEN call this tool alone.

    target — choose exactly one string:
      "trainer"          : mental skills addressed, athlete ready to resume training focus
      "physiotherapist"  : psychological issue is linked to physical pain or injury fear
      "recovery_coach"   : burnout or overtraining stress needs physical recovery management
      "dietitian"        : eating concern (restriction, body image) requires nutritional support

    reason — handoff note with psychological context and interventions already provided.
    """
    return Command(
        goto=_GATHER_NODE[target],
        update={
            "active_agent": target,
            "handoff_reason": reason,
            "pending_agents": [],
            "messages": [ToolMessage(f"Transferred to {target}. Reason: {reason}", tool_call_id=tool_call_id)],
        },
    )
