from src.tools import (
    get_onboarding_status,
    get_daily_readiness,
    log_fitness_assessment,
    get_fitness_assessments,
    get_vdot_paces,
    get_recent_workouts,
    log_workout_rpe_and_notes,
    get_recent_strength_sets,
    log_strength_sets,
    save_workout_plan,
    get_current_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    get_progress_report,
    update_athlete_profile,
    search_knowledge_base,
    query_running_database,
    check_upcoming_race_or_test,
    check_training_anomaly,
)
from src.agent.handoffs import trainer_transfer

TRAINER_TOOLS = [
    # Proactive detection — called first at session start
    check_upcoming_race_or_test,
    check_training_anomaly,
    # Core training tools
    get_daily_readiness,
    get_onboarding_status,
    log_fitness_assessment,
    get_fitness_assessments,
    get_vdot_paces,
    get_recent_workouts,
    log_workout_rpe_and_notes,
    get_recent_strength_sets,
    log_strength_sets,
    save_workout_plan,
    get_current_workout_plan,
    replace_day_in_plan,
    update_planned_workout_status,
    get_progress_report,
    update_athlete_profile,
    # Qdrant knowledge base — source of truth for all domain knowledge
    search_knowledge_base,
    query_running_database,
    trainer_transfer,
]
