from src.tools import (
    get_active_injuries,
    get_injury_recovery_trend,
    log_injury,
    log_injury_checkin,
    resolve_injury,
    get_recent_workouts,
    save_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    search_knowledge_base,
)
from src.agent.handoffs import physio_transfer

PHYSIO_TOOLS = [
    get_active_injuries,
    get_injury_recovery_trend,
    log_injury,
    log_injury_checkin,
    resolve_injury,
    get_recent_workouts,
    save_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    # Qdrant knowledge base — source of truth for all domain knowledge
    search_knowledge_base,
    physio_transfer,
]
