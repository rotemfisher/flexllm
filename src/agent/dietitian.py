from src.tools import (
    get_nutrition_profile,
    get_daily_readiness,
    get_recent_workouts,
    update_athlete_profile,
    search_coaching_books,
    query_running_database,
)
from src.agent.handoffs import dietitian_transfer

DIETITIAN_TOOLS = [
    get_nutrition_profile,
    get_daily_readiness,
    get_recent_workouts,
    update_athlete_profile,
    search_coaching_books,
    query_running_database,
    dietitian_transfer,
]
