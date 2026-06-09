import json
import logging

from langchain_core.tools import tool

from src.tools._utils import db_ro, db_rw, epley_1rm

logger = logging.getLogger(__name__)

# Sanity bounds for LLM-generated strength data.
# World-record raw squat is ~335 kg; 500 gives generous headroom for assisted/machine work.
_MAX_WEIGHT_KG = 500.0
_MAX_REPS      = 100   # >100 reps per set is physiologically implausible for weighted lifts
_MAX_SET_NUM   = 50    # >50 sets of one exercise in a session is an error


@tool
def log_strength_sets(workout_id: int, sets_json: str) -> str:
    """
    Log all sets performed in a strength session. Call this after the athlete
    reports what they lifted during a strength workout.
    Use get_recent_workouts(activity_type='strength') first to find the workout_id.

    Args:
        workout_id: ID of the strength workout (from get_recent_workouts).
        sets_json: JSON array string. Each object must have:
            - exercise_name (str): e.g. 'squat', 'bench_press', 'deadlift', 'pull_up', 'plank'
            - set_number (int): 1-based set index
            - reps (int): reps completed (omit for timed sets)
            Optional: weight_kg (float), duration_sec (float for timed sets), rpe (int 1-10), notes (str)

    Example:
        '[{"exercise_name":"squat","set_number":1,"weight_kg":80,"reps":5,"rpe":7},
          {"exercise_name":"squat","set_number":2,"weight_kg":80,"reps":5,"rpe":8},
          {"exercise_name":"bench_press","set_number":1,"weight_kg":60,"reps":8,"rpe":7}]'
    """
    try:
        sets = json.loads(sets_json)
    except json.JSONDecodeError as exc:
        return f"Error: sets_json must be a valid JSON array. {exc}"

    for i, s in enumerate(sets):
        if "exercise_name" not in s or "set_number" not in s:
            return f"Error: set {i} missing 'exercise_name' or 'set_number'."

        set_num = s["set_number"]
        if not (1 <= set_num <= _MAX_SET_NUM):
            return f"Error: set {i} has set_number={set_num} outside valid range [1, {_MAX_SET_NUM}]."

        weight = s.get("weight_kg")
        if weight is not None and not (0 < weight <= _MAX_WEIGHT_KG):
            return (
                f"Error: set {i} has weight_kg={weight} outside valid range "
                f"(0, {_MAX_WEIGHT_KG}]. Check the value and retry."
            )

        reps = s.get("reps")
        if reps is not None and not (1 <= reps <= _MAX_REPS):
            return (
                f"Error: set {i} has reps={reps} outside valid range [1, {_MAX_REPS}]. "
                f"Check the value and retry."
            )

    try:
        with db_rw() as con:
            if not con.execute("SELECT id FROM workouts WHERE id = %s", (workout_id,)).fetchone():
                return f"Error: workout ID {workout_id} not found."

            # Clear existing sets for this workout to allow re-logging
            con.execute("DELETE FROM strength_sets WHERE workout_id = %s", (workout_id,))
            for s in sets:
                con.execute(
                    """
                    INSERT INTO strength_sets
                        (workout_id, exercise_name, set_number, weight_kg, reps, duration_sec, rpe, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        workout_id,
                        s["exercise_name"].lower().replace(" ", "_"),
                        s["set_number"],
                        s.get("weight_kg"),
                        s.get("reps"),
                        s.get("duration_sec"),
                        s.get("rpe"),
                        s.get("notes"),
                    ),
                )
            con.commit()

        # Build confirmation summary grouped by exercise
        by_exercise: dict[str, list] = {}
        for s in sets:
            by_exercise.setdefault(s["exercise_name"], []).append(s)

        lines = [f"Logged {len(sets)} sets for workout {workout_id}:"]
        for ex, ex_sets in by_exercise.items():
            set_strs = []
            for s in ex_sets:
                if s.get("weight_kg") and s.get("reps"):
                    est_1rm = epley_1rm(s["weight_kg"], s["reps"])
                    set_strs.append(f"{s['weight_kg']}kg×{s['reps']} (est. 1RM {est_1rm:.1f}kg)")
                elif s.get("duration_sec"):
                    set_strs.append(f"{s['duration_sec']}s hold")
                else:
                    set_strs.append(f"{s.get('reps', '?')} reps (bodyweight)")
            lines.append(f"  {ex}: " + " | ".join(set_strs))
        return "\n".join(lines)

    except Exception as exc:
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"


@tool
def get_recent_strength_sets(exercise_name: str, sessions: int = 4) -> str:
    """
    Retrieve the last N sessions of a specific exercise to inform progressive overload.
    Call this before prescribing a strength session to know the athlete's current level.

    Args:
        exercise_name: e.g. 'squat', 'bench_press', 'deadlift', 'pull_up', 'overhead_press'
        sessions: number of past sessions to return (default 4, max 10)
    """
    sessions = min(max(sessions, 1), 10)
    name = exercise_name.lower().replace(" ", "_")

    try:
        with db_ro() as con:
            # Get the distinct workout dates for this exercise
            workout_ids = con.execute(
                """
                SELECT DISTINCT ss.workout_id, w.start_date
                FROM strength_sets ss
                JOIN workouts w ON w.id = ss.workout_id
                WHERE ss.exercise_name = %s
                ORDER BY w.start_date DESC
                LIMIT %s
                """,
                (name, sessions),
            ).fetchall()

            if not workout_ids:
                return (
                    f"No logged sets found for '{exercise_name}'. "
                    f"Use log_strength_sets after a session to start tracking."
                )

            lines = [f"--- {exercise_name.replace('_', ' ').title()} — Last {len(workout_ids)} Sessions ---\n"]
            all_1rms = []

            for wrow in workout_ids:
                sets = con.execute(
                    """
                    SELECT set_number, weight_kg, reps, duration_sec, rpe
                    FROM strength_sets
                    WHERE workout_id = %s AND exercise_name = %s
                    ORDER BY set_number
                    """,
                    (wrow["workout_id"], name),
                ).fetchall()

                session_1rms = []
                set_strs = []
                for s in sets:
                    if s["weight_kg"] and s["reps"]:
                        est = epley_1rm(s["weight_kg"], s["reps"])
                        session_1rms.append(est)
                        rpe_str = f" @RPE{s['rpe']}" if s["rpe"] else ""
                        set_strs.append(f"{s['weight_kg']}kg×{s['reps']}{rpe_str}")
                    elif s["duration_sec"]:
                        set_strs.append(f"{s['duration_sec']}s")
                    else:
                        set_strs.append(f"{s['reps']} reps BW")

                best_1rm = max(session_1rms) if session_1rms else None
                if best_1rm:
                    all_1rms.append((wrow["start_date"][:10], best_1rm))
                lines.append(
                    f"{wrow['start_date'][:10]}: {' | '.join(set_strs)}"
                    + (f"  → est. 1RM {best_1rm:.1f} kg" if best_1rm else "")
                )

            # Progressive overload recommendation
            if len(all_1rms) >= 2:
                delta = all_1rms[0][1] - all_1rms[-1][1]
                lines.append("")
                if delta > 0:
                    lines.append(f"Strength trend: +{delta:.1f} kg est. 1RM over {len(all_1rms)} sessions (PROGRESSING)")
                elif delta < 0:
                    lines.append(f"Strength trend: {delta:.1f} kg est. 1RM (DECLINING — check fatigue/technique)")
                else:
                    lines.append("Strength trend: PLATEAUED — consider technique focus or deload")

                last_sets = con.execute(
                    """
                    SELECT weight_kg, reps FROM strength_sets
                    WHERE workout_id = %s AND exercise_name = %s
                    ORDER BY set_number DESC LIMIT 1
                    """,
                    (workout_ids[0]["workout_id"], name),
                ).fetchone()
                if last_sets and last_sets["weight_kg"] and last_sets["reps"]:
                    if last_sets["reps"] >= 5:
                        next_weight = last_sets["weight_kg"] + (2.5 if last_sets["weight_kg"] < 60 else 5.0)
                        lines.append(f"Next session suggestion: {next_weight}kg (standard +5% overload)")

        return "\n".join(lines)

    except Exception as exc:
        logger.exception("Tool error: %s", exc)
        return f"Database error: {exc}"
