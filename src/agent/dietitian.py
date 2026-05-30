from src.tools import (
    get_nutrition_profile,
    get_daily_readiness,
    get_recent_workouts,
    update_athlete_profile,
    search_coaching_books,
    query_running_database,
)
from src.agent.handoffs import (
    dietitian_transfer_to_trainer,
    dietitian_transfer_to_physiotherapist,
    dietitian_transfer_to_recovery_coach,
)

DIETITIAN_TOOLS = [
    get_nutrition_profile,
    get_daily_readiness,
    get_recent_workouts,
    update_athlete_profile,
    search_coaching_books,
    query_running_database,
    dietitian_transfer_to_trainer,
    dietitian_transfer_to_physiotherapist,
    dietitian_transfer_to_recovery_coach,
]
