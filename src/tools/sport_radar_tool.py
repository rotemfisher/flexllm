"""
sport_radar_tool.py — Proactive situation-detection tools for the coaching graph.

Two tools:
  check_upcoming_race_or_test()  — scan planned_workouts for imminent races/tests
  check_training_anomaly()       — statistical + declared anomaly detection on last session
"""

import logging
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from src.tools._utils import db_ro

logger = logging.getLogger(__name__)

# ── Race / test keyword detection ─────────────────────────────────────────────

_RACE_KEYWORDS = frozenset({
    "race", "competition", "marathon", "half marathon", "triathlon",
    "10k", "5k", "ironman", "duathlon", "parkrun", "championship",
    "qualifier", "gran fondo", "event", "half-marathon", "ultra",
})
_TEST_KEYWORDS = frozenset({
    "time trial", "assessment", "time_trial", "cooper test",
    "vdot test", "3rm", "1rm", "time-trial", "physical exam",
    "fitness test", "benchmark",
})


def _classify_row(row) -> tuple[str, str]:
    """Return (event_class, label): event_class is 'race' | 'test' | 'other'."""
    desc  = (row["description"]  or "").lower()
    notes = (row["notes"]        or "").lower()
    phase = (row["phase"]        or "").lower()
    wtype = (row["workout_type"] or "").lower()

    if phase == "race":
        return "race", "Race Day"
    if row["is_assessment"] and any(k in desc or k in notes for k in _RACE_KEYWORDS):
        return "race", "Race/Competition"
    if row["is_assessment"]:
        return "test", f"Fitness Test ({wtype or 'assessment'})"
    if any(k in desc or k in notes for k in _RACE_KEYWORDS):
        return "race", "Race/Competition"
    if any(k in desc or k in notes for k in _TEST_KEYWORDS):
        return "test", f"Fitness Test ({wtype or 'test'})"
    return "other", wtype or "session"


@tool
def check_upcoming_race_or_test() -> str:
    """
    Scan the planned workout calendar for any race, test, or competition
    in the next 2 days (today or tomorrow). Call this at the very start of
    EVERY session — before get_daily_readiness and before any other tool.

    Returns:
      "⚠ TRIGGER: PRE_RACE — ..."  race is today or tomorrow → ACTION REQUIRED
      "⚠ TRIGGER: PRE_TEST — ..."  fitness test is today or tomorrow → ACTION REQUIRED
      "UPCOMING — ..."              race/test in 3–7 days (awareness only)
      "NONE"                        no race or test imminent

    When a TRIGGER is returned you MUST immediately call
    trainer_transfer(target='psychologist') with the provided reason — do not skip this.
    """
    now      = datetime.now(timezone.utc)
    today    = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    in_7d    = (now + timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        with db_ro() as con:
            rows = con.execute(
                """
                SELECT day_date, workout_type, description, phase, is_assessment, notes
                FROM planned_workouts
                WHERE day_date BETWEEN ? AND ?
                  AND status = 'planned'
                ORDER BY day_date ASC
                """,
                (today, in_7d),
            ).fetchall()
    except Exception as exc:
        logger.exception("check_upcoming_race_or_test DB error: %s", exc)
        return f"NONE (DB unavailable: {exc})"

    if not rows:
        return "NONE — no planned sessions found in the next 7 days."

    imminent_race = None
    imminent_test = None
    upcoming: list[dict] = []

    for row in rows:
        event_class, label = _classify_row(row)
        if event_class == "other":
            continue
        is_imminent = row["day_date"] in (today, tomorrow)
        entry = {
            "date":        row["day_date"],
            "label":       label,
            "description": (row["description"] or "")[:120],
            "is_today":    row["day_date"] == today,
            "is_tomorrow": row["day_date"] == tomorrow,
            "class":       event_class,
        }
        if is_imminent:
            if event_class == "race" and imminent_race is None:
                imminent_race = entry
            elif event_class == "test" and imminent_test is None:
                imminent_test = entry
        else:
            upcoming.append(entry)

    def _when(e: dict) -> str:
        if e["is_today"]:    return "TODAY"
        if e["is_tomorrow"]: return "TOMORROW"
        return e["date"]

    if imminent_race:
        when = _when(imminent_race)
        reason = (
            f"PRE_RACE: Athlete has a race {when} "
            f"({imminent_race['description']}). "
            "Pre-race psychological preparation is required — confidence, "
            "race-plan focus, and arousal management."
        )
        return (
            f"⚠ TRIGGER: PRE_RACE\n"
            f"Event   : {imminent_race['label']} — {when} ({imminent_race['date']})\n"
            f"Details : {imminent_race['description']}\n"
            f"ACTION REQUIRED: Call trainer_transfer(target='psychologist', reason='{reason}')"
        )

    if imminent_test:
        when = _when(imminent_test)
        reason = (
            f"PRE_TEST: Athlete has a fitness test {when} "
            f"({imminent_test['description']}). "
            "Pre-test psychological preparation is required — confidence, "
            "focus, and arousal regulation."
        )
        return (
            f"⚠ TRIGGER: PRE_TEST\n"
            f"Event   : {imminent_test['label']} — {when} ({imminent_test['date']})\n"
            f"Details : {imminent_test['description']}\n"
            f"ACTION REQUIRED: Call trainer_transfer(target='psychologist', reason='{reason}')"
        )

    if upcoming:
        lines = ["UPCOMING — race/test detected within 7 days (no immediate handoff):"]
        for e in upcoming:
            lines.append(f"  • {e['date']} ({e['label']}): {e['description']}")
        return "\n".join(lines)

    return "NONE — no race or test detected in the next 7 days."


# ── Anomaly detection ─────────────────────────────────────────────────────────

_FATIGUE_WORDS = frozenset({
    "tired", "heavy", "exhausted", "dead legs", "couldn't", "could not",
    "failed", "struggle", "rough", "awful", "terrible", "sluggish", "drained",
    "no energy", "hit the wall", "bonked", "fell off", "couldn't finish",
    "harder than usual", "harder than expected", "couldn't keep up",
})
_FAST_WORDS = frozenset({
    "pb", "pr", "personal best", "personal record", "best time", "flew",
    "amazing run", "crushed it", "fastest", "new record", "smashed it",
})
_SLOW_WORDS = frozenset({
    "so slow", "really slow", "way slower", "couldn't keep pace",
    "fell off pace", "heavy legs", "dragged", "dragging",
})
_OVER_LIFTING_WORDS = frozenset({
    "overdid", "over did", "pushed too hard", "too heavy", "way more weight",
    "maxed out", "went too heavy", "twice the weight",
})
_LIFTING_FAIL_WORDS = frozenset({
    "couldn't lift", "failed the set", "dropped the weight",
    "couldn't finish", "missed reps",
})


def _client_anomaly_tags(report: str) -> list[tuple[str, str]]:
    """Return list of (anomaly_tag, description) from free-text client report."""
    if not report:
        return []
    lower = report.lower()
    tags: list[tuple[str, str]] = []
    if any(w in lower for w in _FATIGUE_WORDS):
        tags.append(("fatigue_declared", f'Athlete reported: "{report.strip()}"'))
    if any(w in lower for w in _FAST_WORDS):
        tags.append(("fast_run_declared", f'Athlete reported: "{report.strip()}"'))
    if any(w in lower for w in _SLOW_WORDS):
        tags.append(("slow_run_declared", f'Athlete reported: "{report.strip()}"'))
    if any(w in lower for w in _OVER_LIFTING_WORDS):
        tags.append(("over_lifting_declared", f'Athlete reported: "{report.strip()}"'))
    if any(w in lower for w in _LIFTING_FAIL_WORDS):
        tags.append(("fatigue_lifting_declared", f'Athlete reported: "{report.strip()}"'))
    return tags


def _zscore(value: float | None, mean: float | None, std: float | None) -> float | None:
    if None in (value, mean, std) or std < 0.01:
        return None
    return (value - mean) / std


def _fmt_pace(mins: float | None) -> str:
    if mins is None:
        return "N/A"
    m = int(mins)
    s = round((mins - m) * 60)
    return f"{m}:{s:02d}/km"


@tool
def check_training_anomaly(client_report: str = "") -> str:
    """
    Analyse the most recent training session for statistical or declared anomalies
    against the athlete's 28-day personal baseline.

    Call this whenever the athlete reports back after completing a session.
    Pass the athlete's own description of how it felt in client_report.

    client_report: optional free-text from the athlete about the session.
      Examples:
        "legs felt really heavy, had to slow down after km 5"
        "ran a new 5k PB by 40 seconds — felt incredible"
        "couldn't finish the last two sets of squats, way heavier than usual"
        "session felt normal, no issues"

    Returns:
      "⚠ TRIGGER: ANOMALY_TRAINING — <type>"  → ACTION REQUIRED: handoff to psychologist
      "✓ NORMAL — session within expected range"
      "✗ INSUFFICIENT_DATA — less than 3 sessions in the last 28 days"

    When a TRIGGER is returned you MUST immediately call
    trainer_transfer(target='psychologist') with the provided reason — do not skip this.
    """
    now      = datetime.now(timezone.utc)
    cutoff   = (now - timedelta(days=28)).strftime("%Y-%m-%d")

    anomalies: list[tuple[str, str]] = []   # (tag, detail_string)

    # ── Running anomaly ────────────────────────────────────────────────────────
    try:
        with db_ro() as con:
            last_run = con.execute(
                """
                SELECT start_date, distance_km,
                       ROUND(duration_min / NULLIF(distance_km, 0), 4) AS pace,
                       avg_heart_rate_bpm AS hr, rpe
                FROM workouts
                WHERE activity_type = 'running'
                  AND distance_km > 0.5
                ORDER BY start_date DESC LIMIT 1
                """
            ).fetchone()

            if last_run:
                baseline = con.execute(
                    """
                    SELECT
                        AVG(pace)              AS avg_pace,
                        SQRT(MAX(0.0,
                            AVG(pace * pace) - AVG(pace) * AVG(pace)
                        ))                     AS std_pace,
                        AVG(hr)                AS avg_hr,
                        COUNT(*)               AS n
                    FROM (
                        SELECT ROUND(duration_min / NULLIF(distance_km, 0), 4) AS pace,
                               avg_heart_rate_bpm AS hr
                        FROM workouts
                        WHERE activity_type = 'running'
                          AND distance_km > 0.5
                          AND start_date >= ?
                          AND start_date < ?
                    )
                    """,
                    (cutoff, last_run["start_date"]),
                ).fetchone()
            else:
                baseline = None
    except Exception as exc:
        logger.exception("Running anomaly query failed: %s", exc)
        last_run = baseline = None

    if last_run and baseline and (baseline["n"] or 0) >= 3:
        pace     = last_run["pace"]
        avg_pace = baseline["avg_pace"]
        std_pace = baseline["std_pace"]
        z        = _zscore(pace, avg_pace, std_pace)
        pct      = ((pace - avg_pace) / avg_pace * 100) if avg_pace else None

        if z is not None and (z > 2.0 or (pct and pct > 15)):
            anomalies.append((
                "slow_run",
                f"Pace {_fmt_pace(pace)} vs 28-day avg {_fmt_pace(avg_pace)} "
                f"(+{pct:.1f}%, z-score {z:.1f})"
            ))
        elif z is not None and (z < -2.0 or (pct and pct < -8)):
            anomalies.append((
                "fast_run",
                f"Pace {_fmt_pace(pace)} vs 28-day avg {_fmt_pace(avg_pace)} "
                f"({pct:.1f}%, z-score {z:.1f})"
            ))

        if last_run["hr"] and baseline["avg_hr"]:
            hr_pct = (last_run["hr"] - baseline["avg_hr"]) / baseline["avg_hr"] * 100
            if hr_pct > 12:
                anomalies.append((
                    "elevated_hr",
                    f"HR {round(last_run['hr'])} bpm vs baseline "
                    f"{round(baseline['avg_hr'])} bpm (+{hr_pct:.1f}%)"
                ))

    elif last_run and baseline and (baseline["n"] or 0) < 3:
        # Not enough data for stats, but will still check client signals
        pass

    # ── Strength anomaly ───────────────────────────────────────────────────────
    try:
        with db_ro() as con:
            last_str = con.execute(
                """
                SELECT w.id, w.start_date, w.rpe,
                       AVG(ss.weight_kg) AS avg_w,
                       COUNT(ss.id)      AS n_sets
                FROM workouts w
                JOIN strength_sets ss ON ss.workout_id = w.id
                WHERE w.activity_type = 'strength'
                  AND ss.weight_kg IS NOT NULL
                GROUP BY w.id
                ORDER BY w.start_date DESC LIMIT 1
                """
            ).fetchone()

            if last_str and last_str["id"]:
                str_base = con.execute(
                    """
                    SELECT
                        AVG(w.rpe)    AS avg_rpe,
                        SQRT(MAX(0.0,
                            AVG(w.rpe * w.rpe) - AVG(w.rpe) * AVG(w.rpe)
                        ))            AS std_rpe,
                        AVG(agg.avg_w)  AS avg_weight,
                        AVG(agg.n_sets) AS avg_sets,
                        COUNT(DISTINCT w.id) AS n
                    FROM workouts w
                    JOIN (
                        SELECT workout_id,
                               AVG(weight_kg) AS avg_w,
                               COUNT(*)       AS n_sets
                        FROM strength_sets WHERE weight_kg IS NOT NULL
                        GROUP BY workout_id
                    ) agg ON agg.workout_id = w.id
                    WHERE w.activity_type = 'strength'
                      AND w.start_date >= ?
                      AND w.start_date < ?
                      AND w.rpe IS NOT NULL
                    """,
                    (cutoff, last_str["start_date"]),
                ).fetchone()
            else:
                str_base = None
    except Exception as exc:
        logger.exception("Strength anomaly query failed: %s", exc)
        last_str = str_base = None

    if last_str and last_str["id"] and str_base and (str_base["n"] or 0) >= 3:
        rpe, avg_rpe, std_rpe = last_str["rpe"], str_base["avg_rpe"], str_base["std_rpe"]
        if rpe and avg_rpe:
            z_rpe = _zscore(rpe, avg_rpe, std_rpe)
            if (rpe - avg_rpe) >= 2 or (z_rpe is not None and z_rpe > 2.0):
                anomalies.append((
                    "fatigue_lifting",
                    f"Session RPE {rpe}/10 vs 28-day avg "
                    f"{avg_rpe:.1f} (+{rpe - avg_rpe:.1f}, z-score {z_rpe:.1f})"
                ))

        aw, base_w = last_str["avg_w"], str_base["avg_weight"]
        if aw and base_w and base_w > 0:
            wpct = (aw - base_w) / base_w * 100
            if wpct > 15:
                anomalies.append((
                    "over_lifting",
                    f"Avg weight {aw:.1f} kg vs baseline "
                    f"{base_w:.1f} kg (+{wpct:.1f}%)"
                ))

        ns, base_ns = last_str["n_sets"], str_base["avg_sets"]
        if ns and base_ns and base_ns > 0 and (ns / base_ns) < 0.75:
            anomalies.append((
                "fatigue_lifting",
                f"Completed {ns} sets vs baseline avg "
                f"{base_ns:.0f} ({ns / base_ns * 100:.0f}% of normal volume)"
            ))

    # ── Client-declared signals ────────────────────────────────────────────────
    declared = _client_anomaly_tags(client_report)
    existing_tags = {a[0] for a in anomalies}
    for tag, detail in declared:
        # Merge: strengthen statistical finding, or add as standalone declaration
        if tag not in existing_tags:
            anomalies.append((tag, detail))
            existing_tags.add(tag)

    # ── Insufficient data guard ────────────────────────────────────────────────
    if not anomalies:
        has_stat_data = (
            (last_run  and baseline  and (baseline["n"]  or 0) >= 3) or
            (last_str and last_str["id"] and str_base and (str_base["n"] or 0) >= 3)
        )
        if not has_stat_data and not client_report:
            return (
                "✗ INSUFFICIENT_DATA — less than 3 sessions in the last 28 days. "
                "Pass client_report to detect declared anomalies even without baseline."
            )
        return "✓ NORMAL — last session within expected range."

    # ── Build TRIGGER output ───────────────────────────────────────────────────
    _priority = [
        "slow_run", "fast_run", "fatigue_lifting", "over_lifting",
        "elevated_hr", "fatigue_declared", "slow_run_declared",
        "fast_run_declared", "over_lifting_declared", "fatigue_lifting_declared",
    ]
    tags_only = [a[0] for a in anomalies]
    primary   = next((p for p in _priority if p in tags_only), tags_only[0])

    _labels = {
        "slow_run":                  "SLOW RUN — significantly below baseline pace",
        "fast_run":                  "FAST RUN — significantly above baseline pace",
        "fatigue_lifting":           "FATIGUE IN LIFTING — elevated RPE or volume drop",
        "over_lifting":              "OVER-LIFTING — load significantly above baseline",
        "elevated_hr":               "ELEVATED HR — unusual cardiac strain during run",
        "fatigue_declared":          "ATHLETE-DECLARED FATIGUE",
        "slow_run_declared":         "ATHLETE-DECLARED SLOW RUN",
        "fast_run_declared":         "ATHLETE-DECLARED FAST RUN",
        "over_lifting_declared":     "ATHLETE-DECLARED OVER-LIFTING",
        "fatigue_lifting_declared":  "ATHLETE-DECLARED LIFTING FAILURE",
    }
    primary_label = _labels.get(primary, primary)
    detail_str    = " | ".join(d for _, d in anomalies)

    reason = (
        f"ANOMALY_TRAINING — {primary_label}. "
        f"Details: {detail_str}. "
        "Psychological check-in needed: reframe the session, restore confidence, "
        "prevent catastrophising."
    )

    return (
        f"⚠ TRIGGER: ANOMALY_TRAINING — {primary_label}\n"
        f"Anomalies : {', '.join(tags_only)}\n"
        + "".join(f"  • {d}\n" for _, d in anomalies)
        + f"ACTION REQUIRED: Call trainer_transfer(target='psychologist', "
        f"reason='{reason}')"
    )
