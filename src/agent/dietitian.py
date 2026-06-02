from src.tools import (
    get_nutrition_profile,
    get_daily_readiness,
    get_recent_workouts,
    update_athlete_profile,
    search_knowledge_base,
    query_running_database,
)
from src.agent.handoffs import dietitian_transfer

DIETITIAN_TOOLS = [
    get_nutrition_profile,
    get_daily_readiness,
    get_recent_workouts,
    update_athlete_profile,
    # Qdrant knowledge base — source of truth for all domain knowledge
    search_knowledge_base,
    query_running_database,
    dietitian_transfer,
]
