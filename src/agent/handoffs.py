from typing import Literal

from langchain_core.tools import tool
from langgraph.types import Command


@tool
def trainer_transfer(
    target: Literal["physiotherapist", "recovery_coach", "dietitian"],
    reason: str,
) -> Command:
    """
    HANDOFF — transfer control from the Trainer to another specialist.

    CRITICAL: Call this as the LAST and ONLY tool in your response.
    NEVER combine this with a domain tool (get_recent_workouts, save_workout_plan,
    get_vdot_paces, etc.) in the same turn.
    Complete all data-gathering and analysis first, THEN call this tool alone.

    target — choose exactly one string:
      "physiotherapist"  : athlete reports pain, injury, or movement limitation
      "recovery_coach"   : TSB < -20, HRV critically low, sleep < 5h, or fatigue-only topic
      "dietitian"        : nutrition, meal planning, macros, weight management, fuelling

    reason — concise handoff note for the receiving agent (key values, clinical context).
    """
    return Command(goto=target, update={"active_agent": target, "handoff_reason": reason})



@tool
def physio_transfer(
    target: Literal["trainer", "recovery_coach", "dietitian"],
    reason: str,
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

    reason — handoff note with return-to-train restrictions or clinical context for the receiving agent.
    """
    return Command(goto=target, update={"active_agent": target, "handoff_reason": reason})


@tool
def recovery_transfer(
    target: Literal["trainer", "physiotherapist", "dietitian"],
    reason: str,
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

    reason — handoff note with TSB, HRV, and sleep values plus clinical rationale.
    """
    return Command(goto=target, update={"active_agent": target, "handoff_reason": reason})


@tool
def dietitian_transfer(
    target: Literal["trainer", "physiotherapist", "recovery_coach"],
    reason: str,
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

    reason — handoff note with nutritional context and recommendations already provided.
    """
    return Command(goto=target, update={"active_agent": target, "handoff_reason": reason})
