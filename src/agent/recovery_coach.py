from src.tools import (
    get_daily_readiness,
    get_recent_workouts,
    get_current_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    get_progress_report,
    search_coaching_books,
)
from src.agent.handoffs import (
    recovery_transfer_to_trainer,
    recovery_transfer_to_physiotherapist,
    recovery_transfer_to_dietitian,
)

RECOVERY_TOOLS = [
    get_daily_readiness,
    get_recent_workouts,
    get_current_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    get_progress_report,
    search_coaching_books,
    recovery_transfer_to_trainer,
    recovery_transfer_to_physiotherapist,
    recovery_transfer_to_dietitian,
]
