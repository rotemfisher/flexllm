from src.tools import (
    get_daily_readiness,
    get_recent_workouts,
    get_current_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    get_progress_report,
    search_knowledge_base,
    check_upcoming_race_or_test,
    check_training_anomaly,
)
from src.agent.handoffs import recovery_transfer

RECOVERY_TOOLS = [
    check_upcoming_race_or_test,
    check_training_anomaly,
    get_daily_readiness,
    get_recent_workouts,
    get_current_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    get_progress_report,
    # Qdrant knowledge base — source of truth for all domain knowledge
    search_knowledge_base,
    recovery_transfer,
]
