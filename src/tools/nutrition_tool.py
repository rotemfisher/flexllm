from langchain_core.tools import tool

from src.tools._utils import db_ro


@tool
def get_nutrition_profile() -> str:
    """
    Get the athlete's physiological profile, current goals, and recent energy expenditure.
    The Dietitian role MUST call this tool before generating any meal plans, macro recommendations,
    or diet advice, to ensure the plan is personalized to their height, weight, sex, and activity level.
    """
    try:
        with db_ro() as con:
            profile = con.execute(
                "SELECT date_of_birth, biological_sex, height_cm, current_goal, target_weight_kg, dietary_pref FROM athlete_profile ORDER BY id DESC LIMIT 1"
            ).fetchone()

            if not profile:
                return "Athlete profile is missing. Ask the athlete for their age, height, sex, and nutrition goals."

            health = con.execute(
                "SELECT body_mass_kg FROM daily_health WHERE body_mass_kg IS NOT NULL ORDER BY date DESC LIMIT 1"
            ).fetchone()
            current_weight = health["body_mass_kg"] if health else "Unknown"

            activity = con.execute(
                """
                SELECT AVG(active_calories) as avg_active_cals
                FROM daily_health
                WHERE date >= date('now', '-7 days') AND active_calories IS NOT NULL
                """
            ).fetchone()
            avg_active_cals = round(activity["avg_active_cals"] or 0)

        return (
            f"--- Athlete Nutrition Profile ---\n"
            f"Demographics:\n"
            f"  - DOB: {profile['date_of_birth']}\n"
            f"  - Sex: {profile['biological_sex'] or 'Not specified'}\n"
            f"  - Height: {profile['height_cm'] or 'Unknown'} cm\n"
            f"  - Current Weight: {current_weight} kg\n\n"
            f"Goals & Preferences:\n"
            f"  - Primary Goal: {profile['current_goal'] or 'Not specified'}\n"
            f"  - Target Weight: {profile['target_weight_kg'] or 'Not specified'} kg\n"
            f"  - Diet Type: {profile['dietary_pref'] or 'No restrictions'}\n\n"
            f"Energy Expenditure (Last 7 Days):\n"
            f"  - Avg Active Calories Burned/Day: {avg_active_cals} kcal\n\n"
            f"Dietitian Directive: Use this data (Age, Height, Weight, Sex) to calculate their BMR (e.g., Mifflin-St Jeor equation). "
            f"Add the Avg Active Calories to find their TDEE. Then, generate a diet plan aligned with their 'Primary Goal', "
            f"sourcing specific nutrient timing and macro guidelines from the coaching_books database (e.g., 'clinical_sports_nutrition')."
        )

    except Exception as exc:
        return f"Database error: {exc}"
