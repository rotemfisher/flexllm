from src.tools import (
    get_recent_workouts,
    get_daily_readiness,
    update_athlete_profile,
    search_knowledge_base,
    query_running_database,
    get_situational_psych_tips,
    search_psychology_books,
)
from src.agent.handoffs import psychologist_transfer

PSYCHOLOGIST_TOOLS = [
    get_situational_psych_tips,
    search_psychology_books,
    get_recent_workouts,
    get_daily_readiness,
    update_athlete_profile,
    # Qdrant knowledge base — source of truth for all domain knowledge
    search_knowledge_base,
    query_running_database,
    psychologist_transfer,
]
