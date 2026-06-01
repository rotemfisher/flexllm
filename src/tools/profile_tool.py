import logging

from langchain_core.tools import tool

from src.tools._utils import db_rw

logger = logging.getLogger(__name__)

_ALLOWED_FIELDS = {
    "current_goal",
    "secondary_goal",
    "fitness_level",
    "target_weight_kg",
    "dietary_pref",
    "onboarding_complete",
}

# Pre-build UPDATE SQL for each allowed field at import time so there is no
# f-string interpolation at runtime (even though _ALLOWED_FIELDS is controlled,
# building SQL strings dynamically in a hot path is an avoidable smell).
_UPDATE_SQL: dict[str, str] = {
    field: (
        f"UPDATE athlete_profile SET {field} = ?, updated_at = CURRENT_TIMESTAMP"
        " WHERE id = (SELECT MAX(id) FROM athlete_profile)"
    )
    for field in _ALLOWED_FIELDS
}


@tool
def update_athlete_profile(field: str, value: str) -> str:
    """
    Update a single field in the athlete's profile.
    Use this to mark onboarding complete, change goals, update fitness level, etc.

    Args:
        field: Field to update. Allowed values:
               current_goal, secondary_goal, fitness_level,
               target_weight_kg, dietary_pref, onboarding_complete.
        value: New value as a string (e.g. '1' for onboarding_complete, 'intermediate' for fitness_level).
    """
    if field not in _ALLOWED_FIELDS:
        return (
            f"Error: '{field}' is not updatable. "
            f"Allowed fields: {', '.join(sorted(_ALLOWED_FIELDS))}"
        )

    try:
        with db_rw() as con:
            con.execute(_UPDATE_SQL[field], (value,))
            con.commit()
            if con.total_changes == 0:
                return "No athlete profile found. Create a profile first."
        return f"Updated athlete_profile.{field} = '{value}'"
    except Exception as exc:
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"
