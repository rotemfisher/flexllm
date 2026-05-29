import sqlite3
from datetime import datetime
from langchain_core.tools import tool
from src.config import config


def _epley_1rm(weight_kg: float, reps: int) -> float:
    return weight_kg if reps == 1 else weight_kg * (1 + reps / 30)


def _pace_to_vdot(pace_sec_per_km: float, con: sqlite3.Connection) -> float | None:
    """Find the closest VDOT where T-pace matches the given pace (reverse lookup)."""
    row = con.execute(
        """
        SELECT vdot FROM vdot_paces
        ORDER BY ABS(t_pace_sec - ?) ASC
        LIMIT 1
        """,
        (pace_sec_per_km,),
    ).fetchone()
    return row[0] if row else None


@tool
def get_onboarding_status() -> str:
    """
    Check whether the athlete needs a physical assessment before starting training.
    Call this at the very start of a first session or when the athlete profile is fresh.
    Returns the onboarding protocol if assessment is needed, or clears the athlete to train.
    """
    con = None
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        profile = con.execute(
            "SELECT fitness_level, onboarding_complete, current_goal, secondary_goal FROM athlete_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if not profile:
            return (
                "No athlete profile found. Ask the athlete for: age, sex, height, weight, "
                "fitness_level (beginner/intermediate/advanced), primary goal, and any dietary preferences."
            )

        if profile["onboarding_complete"]:
            return (
                f"Onboarding complete. Athlete is '{profile['fitness_level']}' level. "
                f"Cleared to train — proceed with daily readiness check."
            )

        level = profile["fitness_level"]
        goal = profile["current_goal"] or "general fitness"

        # Check what assessments already exist
        has_run_assessment = con.execute(
            "SELECT 1 FROM fitness_assessments WHERE assessment_type IN ('onboarding_run','time_trial') LIMIT 1"
        ).fetchone()
        has_strength_assessment = con.execute(
            "SELECT 1 FROM fitness_assessments WHERE assessment_type IN ('onboarding_strength','strength_1rm') LIMIT 1"
        ).fetchone()

        lines = [f"ONBOARDING REQUIRED — Fitness level: {level} | Goal: {goal}\n"]

        if level == "beginner":
            lines.append(
                "This athlete is a beginner. Build a 2-day physical assessment plan before any training:\n"
                "\nDAY 1 — Running Assessment (save as workout_type='assessment', is_assessment=1):\n"
                "  1. 5-min walk warm-up\n"
                "  2. Easy jog for 10 min — note if they can maintain conversation pace\n"
                "  3. If yes: 1km time trial at 'comfortably hard' effort (8/10 RPE)\n"
                "  4. Record time → call log_fitness_assessment with assessment_type='onboarding_run'\n"
                "\nDAY 3 — Strength Assessment (save as workout_type='assessment', is_assessment=1):\n"
                "  1. Bodyweight squat: 3 sets of max reps → find technical failure point\n"
                "  2. Push-ups: 3 sets of max reps\n"
                "  3. If any barbell/dumbbell available: goblet squat with light weight, 10 reps → assess form\n"
                "  4. Record results → call log_fitness_assessment with assessment_type='onboarding_strength'\n"
                "\nAfter both assessments are logged, the system will auto-calculate VDOT and strength baseline."
            )
        else:
            missing = []
            if not has_run_assessment:
                missing.append("Running: schedule a 2km time trial (assessment_type='onboarding_run')")
            if not has_strength_assessment:
                missing.append("Strength: schedule a 3RM test for main lifts (assessment_type='onboarding_strength')")

            if missing:
                lines.append("Missing baseline assessments:\n" + "\n".join(f"  - {m}" for m in missing))
            else:
                lines.append("All baseline assessments recorded. Mark onboarding_complete=1 in athlete_profile.")

        return "\n".join(lines)

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()


@tool
def log_fitness_assessment(
    assessment_type: str,
    metric_name: str,
    metric_value: float,
    exercise_name: str | None = None,
    notes: str | None = None,
) -> str:
    """
    Record the result of a physical assessment or progress test.
    Call this after the athlete completes any timed test, time trial, strength test, or assessment session.
    Automatically computes estimated VDOT (for running) or estimated 1RM (for strength).

    Args:
        assessment_type:
            'onboarding_run'      → beginner first run assessment
            'onboarding_strength' → beginner first strength assessment
            'time_trial'          → periodic running progress test (use every 4 weeks)
            'strength_1rm'        → periodic strength progress test (use every 6 weeks)
            'cooper_test'         → 12-min run for distance
            'body_composition'    → weight/body-fat snapshot
        metric_name:
            Running: 'time_sec' (time for a fixed distance) | 'pace_min_per_km' | 'distance_m' (for cooper)
            Strength: 'weight_kg' + pair with reps via notes | 'reps' (for bodyweight max)
        metric_value: the raw number (seconds, kg, metres, etc.)
        exercise_name: required for strength assessments (e.g. 'squat', 'bench_press', 'pull_up')
        notes: any context (e.g. 'distance was 1km', 'reps at 80kg', 'felt tired')
    """
    today = datetime.now().strftime("%Y-%m-%d")
    estimated_vdot = None
    estimated_1rm_kg = None

    con = None
    try:
        con = sqlite3.connect(config.DB_PATH)
        con.row_factory = sqlite3.Row

        # Auto-compute derived values
        if assessment_type in ("onboarding_run", "time_trial"):
            if metric_name == "time_sec" and notes:
                # Try to extract distance from notes (e.g. "distance was 1km" → 1000m)
                import re
                km_match = re.search(r"(\d+(?:\.\d+)?)\s*km", notes or "", re.IGNORECASE)
                m_match  = re.search(r"(\d+(?:\.\d+)?)\s*m\b", notes or "", re.IGNORECASE)
                if km_match:
                    dist_m = float(km_match.group(1)) * 1000
                    pace_sec_per_km = metric_value / (dist_m / 1000)
                    estimated_vdot = _pace_to_vdot(pace_sec_per_km, con)
                elif m_match:
                    dist_m = float(m_match.group(1))
                    pace_sec_per_km = metric_value / (dist_m / 1000)
                    estimated_vdot = _pace_to_vdot(pace_sec_per_km, con)
            elif metric_name == "pace_min_per_km":
                pace_sec = metric_value * 60
                estimated_vdot = _pace_to_vdot(pace_sec, con)

        elif assessment_type == "cooper_test" and metric_name == "distance_m":
            # Cooper: VO2max ≈ (distance_m - 504.9) / 44.73, then VDOT ≈ VO2max
            vo2max = (metric_value - 504.9) / 44.73
            row = con.execute(
                "SELECT vdot FROM vdot_paces ORDER BY ABS(vdot - ?) ASC LIMIT 1", (vo2max,)
            ).fetchone()
            estimated_vdot = row[0] if row else None

        elif assessment_type in ("onboarding_strength", "strength_1rm") and metric_name == "weight_kg":
            import re
            reps_match = re.search(r"(\d+)\s*reps?", notes or "", re.IGNORECASE)
            if reps_match:
                estimated_1rm_kg = round(_epley_1rm(metric_value, int(reps_match.group(1))), 1)

        con.row_factory = None
        con.execute(
            """
            INSERT INTO fitness_assessments
                (assessment_date, assessment_type, exercise_name, metric_name,
                 metric_value, estimated_vdot, estimated_1rm_kg, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (today, assessment_type, exercise_name, metric_name,
             metric_value, estimated_vdot, estimated_1rm_kg, notes),
        )
        con.commit()

        result = f"Assessment logged ({assessment_type}) on {today}: {metric_name} = {metric_value}"
        if estimated_vdot:
            result += f"\n  → Estimated VDOT: {estimated_vdot}"
            result += f"\n  → Use get_vdot_paces({estimated_vdot}) to get training paces."
        if estimated_1rm_kg:
            result += f"\n  → Estimated 1RM ({exercise_name}): {estimated_1rm_kg} kg"
        return result

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()


@tool
def get_fitness_assessments(assessment_type: str | None = None, limit: int = 6) -> str:
    """
    Retrieve the history of fitness assessments to track progress toward goals.
    Use this during progress reviews or before building a new training block.

    Args:
        assessment_type: filter by type (e.g. 'time_trial', 'strength_1rm') — omit to see all.
        limit: number of records to return (default 6).
    """
    limit = min(max(limit, 1), 20)
    con = None
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        if assessment_type:
            rows = con.execute(
                """
                SELECT assessment_date, assessment_type, exercise_name, metric_name,
                       metric_value, estimated_vdot, estimated_1rm_kg, notes
                FROM fitness_assessments
                WHERE assessment_type = ?
                ORDER BY assessment_date DESC LIMIT ?
                """,
                (assessment_type, limit),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT assessment_date, assessment_type, exercise_name, metric_name,
                       metric_value, estimated_vdot, estimated_1rm_kg, notes
                FROM fitness_assessments
                ORDER BY assessment_date DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

        if not rows:
            return "No fitness assessments recorded yet."

        lines = ["--- Fitness Assessment History ---\n"]
        for r in rows:
            label = r["assessment_type"]
            if r["exercise_name"]:
                label += f" ({r['exercise_name']})"
            value_str = f"{r['metric_name']} = {r['metric_value']}"
            derived = ""
            if r["estimated_vdot"]:
                derived = f"  → VDOT {r['estimated_vdot']}"
            elif r["estimated_1rm_kg"]:
                derived = f"  → est. 1RM {r['estimated_1rm_kg']} kg"
            lines.append(
                f"{r['assessment_date']}  [{label}]  {value_str}{derived}"
                + (f"  | {r['notes']}" if r["notes"] else "")
            )

        # Trend for VDOT
        vdot_rows = [r for r in rows if r["estimated_vdot"]]
        if len(vdot_rows) >= 2:
            delta = vdot_rows[0]["estimated_vdot"] - vdot_rows[-1]["estimated_vdot"]
            direction = "improved" if delta > 0 else "declined"
            lines.append(f"\nRunning trend: VDOT {direction} by {abs(delta):.0f} points over this period.")

        # Trend for a specific exercise 1RM
        strength_rows = [r for r in rows if r["estimated_1rm_kg"]]
        if len(strength_rows) >= 2:
            delta = strength_rows[0]["estimated_1rm_kg"] - strength_rows[-1]["estimated_1rm_kg"]
            direction = "increased" if delta > 0 else "decreased"
            ex = strength_rows[0]["exercise_name"] or "strength"
            lines.append(f"{ex.title()} trend: est. 1RM {direction} by {abs(delta):.1f} kg.")

        return "\n".join(lines)

    except Exception as exc:
        return f"Database error: {exc}"
    finally:
        if con:
            con.close()
