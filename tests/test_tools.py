"""
tests/test_tools.py — Unit tests for all src/tools/*.py functions.

Each test that needs database access uses the `tools_db` fixture which:
  - Creates a fresh SQLite file with the production schema
  - Seeds minimal reference data (vdot_paces, athlete_profile, workouts, daily_health)
  - Patches config.DB_PATH so all tools read/write the temp DB
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

from src.config import config
from src.tools._utils import epley_1rm
from src.tools.assessment_tool import (
    get_fitness_assessments,
    get_onboarding_status,
    log_fitness_assessment,
)
from src.tools.injury_tool import get_active_injuries, get_injury_recovery_trend
from src.tools.injury_write_tool import log_injury, log_injury_checkin, resolve_injury
from src.tools.log_workout_feedback_tool import log_workout_rpe_and_notes
from src.tools.plan_tool import (
    get_current_workout_plan,
    replace_day_in_plan,
    save_workout_plan,
    update_planned_workout_status,
)
from src.tools.profile_tool import update_athlete_profile
from src.tools.readiness_tool import get_daily_readiness
from src.tools.sql_tool import query_running_database
from src.tools.strength_tool import get_recent_strength_sets, log_strength_sets
from src.tools.vdot_tool import get_vdot_paces
from src.tools.nutrition_tool import get_nutrition_profile
from src.tools.workout_history_tool import get_recent_workouts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _week_start() -> str:
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=(now.weekday() + 1) % 7)).strftime("%Y-%m-%d")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Fixture ───────────────────────────────────────────────────────────────────

_VDOT_ROWS = [
    (50, 372, 353, 343, 330, 313, 299),
    (55, 342, 326, 316, 305, 290, 277),
    (60, 320, 304, 295, 285, 271, 259),
]


@pytest.fixture()
def tools_db(tmp_path, monkeypatch):
    """
    Fresh SQLite DB with the production schema and minimal seed data.
    Patches config.DB_PATH for the duration of each test.
    """
    db = tmp_path / "tools_test.db"
    schema_sql = (ROOT / "sql" / "schema.sql").read_text()
    con = sqlite3.connect(str(db))
    con.executescript(schema_sql)

    con.executemany("INSERT INTO vdot_paces VALUES (?,?,?,?,?,?,?)", _VDOT_ROWS)

    con.execute(
        "INSERT INTO athlete_profile"
        " (date_of_birth, fitness_level, onboarding_complete, current_goal)"
        " VALUES ('1990-01-01', 'intermediate', 0, 'marathon_prep')"
    )

    con.execute(
        """INSERT INTO daily_health
               (date, atl, ctl, tsb, resting_heart_rate_bpm, hrv_sdnn_ms,
                body_mass_kg, sleep_total_min, sleep_deep_min, sleep_rem_min)
           VALUES ('2026-01-10', 45.0, 55.0, 10.0, 52.0, 65.0, 72.5, 420, 90, 120)"""
    )

    con.execute(
        """INSERT INTO workouts
               (id, activity_type, start_date, end_date,
                duration_min, distance_km, avg_heart_rate_bpm, training_stress_score)
           VALUES (1, 'running', '2026-01-10 07:00:00', '2026-01-10 08:00:00',
                   60.0, 10.0, 155.0, 65.0)"""
    )
    con.execute(
        """INSERT INTO workouts
               (id, activity_type, start_date, end_date, duration_min)
           VALUES (2, 'strength', '2026-01-08 10:00:00', '2026-01-08 11:00:00', 60.0)"""
    )

    con.commit()
    con.close()

    monkeypatch.setattr(config, "DB_PATH", str(db))
    yield db


@pytest.fixture()
def empty_db(tmp_path, monkeypatch):
    """Schema-only DB with no seed data, for testing not-found paths."""
    db = tmp_path / "empty.db"
    schema_sql = (ROOT / "sql" / "schema.sql").read_text()
    con = sqlite3.connect(str(db))
    con.executescript(schema_sql)
    con.commit()
    con.close()
    monkeypatch.setattr(config, "DB_PATH", str(db))
    yield db


# ═══════════════════════════════════════════════════════════════════════════════
# _utils — epley_1rm  (pure function, no DB needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpley1RM:
    def test_single_rep_returns_weight_unchanged(self):
        assert epley_1rm(100.0, 1) == 100.0

    def test_five_reps(self):
        assert epley_1rm(80.0, 5) == pytest.approx(80.0 * (1 + 5 / 30))

    def test_ten_reps(self):
        assert epley_1rm(60.0, 10) == pytest.approx(60.0 * (1 + 10 / 30))

    def test_higher_weight_increases_1rm(self):
        assert epley_1rm(100.0, 5) > epley_1rm(80.0, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# sql_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryRunningDatabase:
    @pytest.mark.parametrize("stmt", [
        "INSERT INTO workouts VALUES (1)",
        "UPDATE workouts SET rpe=5",
        "DELETE FROM workouts",
        "DROP TABLE workouts",
        "ALTER TABLE workouts ADD COLUMN x TEXT",
        "CREATE TABLE x (id INTEGER)",
        "REPLACE INTO workouts VALUES (1)",
        "TRUNCATE TABLE workouts",
    ])
    def test_blocks_write_operations(self, tools_db, stmt):
        result = query_running_database.invoke({"query": stmt})
        assert result == "Error: only SELECT queries are permitted."

    def test_adds_limit_when_missing(self, tools_db):
        result = query_running_database.invoke({"query": "SELECT * FROM vdot_paces"})
        data = json.loads(result)
        assert len(data) == len(_VDOT_ROWS)

    def test_respects_explicit_limit(self, tools_db):
        result = query_running_database.invoke({"query": "SELECT * FROM vdot_paces LIMIT 1"})
        data = json.loads(result)
        assert len(data) == 1

    def test_no_results_message(self, tools_db):
        result = query_running_database.invoke({"query": "SELECT * FROM shoes LIMIT 1"})
        assert result == "No results found."

    def test_bad_sql_returns_error(self, tools_db):
        result = query_running_database.invoke({"query": "SELECT * FROM nonexistent_table LIMIT 1"})
        assert result.startswith("Query error:")

    def test_returns_json_rows(self, tools_db):
        result = query_running_database.invoke({"query": "SELECT vdot FROM vdot_paces LIMIT 3"})
        data = json.loads(result)
        vdots = [row["vdot"] for row in data]
        assert set(vdots) == {50, 55, 60}


# ═══════════════════════════════════════════════════════════════════════════════
# vdot_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetVdotPaces:
    def test_out_of_range_low(self, tools_db):
        result = get_vdot_paces.invoke({"vdot": 29})
        assert "VDOT must be between 30 and 85" in result

    def test_out_of_range_high(self, tools_db):
        result = get_vdot_paces.invoke({"vdot": 86})
        assert "VDOT must be between 30 and 85" in result

    def test_exact_match_no_note(self, tools_db):
        result = get_vdot_paces.invoke({"vdot": 50})
        assert "VDOT 50" in result
        assert "nearest available" not in result

    def test_exact_match_returns_all_pace_zones(self, tools_db):
        result = get_vdot_paces.invoke({"vdot": 50})
        for zone in ("Easy:", "Marathon:", "Threshold:", "Interval:", "Repetition:"):
            assert zone in result

    def test_nearest_fallback_adds_note(self, tools_db):
        # VDOT 52 not in table — nearest is 50 (|50-52|=2 < |55-52|=3)
        result = get_vdot_paces.invoke({"vdot": 52})
        assert "nearest available" in result

    def test_pace_format_is_mm_ss(self, tools_db):
        result = get_vdot_paces.invoke({"vdot": 50})
        # All pace values should be formatted as M:SS/km
        assert "/km" in result


# ═══════════════════════════════════════════════════════════════════════════════
# readiness_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetDailyReadiness:
    def test_no_data_returns_not_found(self, tools_db):
        result = get_daily_readiness.invoke({"date": "2020-01-01"})
        assert "No health or readiness data found" in result

    def test_returns_all_sections(self, tools_db):
        result = get_daily_readiness.invoke({"date": "2026-01-10"})
        assert "Readiness Report" in result
        assert "CTL" in result
        assert "ATL" in result
        assert "TSB" in result
        assert "Resting HR" in result
        assert "Sleep" in result

    def test_fresh_form_when_tsb_positive(self, tools_db):
        # Seeded TSB = 10.0 > 0
        result = get_daily_readiness.invoke({"date": "2026-01-10"})
        assert "Fresh" in result

    def test_fatigued_form_when_tsb_below_minus_10(self, tools_db):
        con = sqlite3.connect(str(tools_db))
        con.execute(
            "INSERT INTO daily_health (date, atl, ctl, tsb) VALUES ('2026-02-01', 60.0, 40.0, -20.0)"
        )
        con.commit()
        con.close()
        result = get_daily_readiness.invoke({"date": "2026-02-01"})
        assert "Fatigued" in result

    def test_optimal_form_when_tsb_between_minus10_and_0(self, tools_db):
        con = sqlite3.connect(str(tools_db))
        con.execute(
            "INSERT INTO daily_health (date, atl, ctl, tsb) VALUES ('2026-03-01', 50.0, 45.0, -5.0)"
        )
        con.commit()
        con.close()
        result = get_daily_readiness.invoke({"date": "2026-03-01"})
        assert "Optimal/Neutral" in result

    def test_returns_most_recent_row_on_or_before_date(self, tools_db):
        # Ask for 2026-01-11 — should fall back to the 2026-01-10 row
        result = get_daily_readiness.invoke({"date": "2026-01-11"})
        assert "2026-01-10" in result


# ═══════════════════════════════════════════════════════════════════════════════
# injury_tool (read)
# ═══════════════════════════════════════════════════════════════════════════════

def _insert_injury(db_path, *, body_part="knee", side="left", severity="moderate",
                   status="active", pain_scale=6, pain_context="workout"):
    con = sqlite3.connect(str(db_path))
    cur = con.execute(
        """INSERT INTO injuries
               (onset_date, body_part, side, severity, status, pain_scale, pain_context)
           VALUES ('2026-01-01', ?, ?, ?, ?, ?, ?)""",
        (body_part, side, severity, status, pain_scale, pain_context),
    )
    con.commit()
    injury_id = cur.lastrowid
    con.close()
    return injury_id


class TestGetActiveInjuries:
    def test_no_injuries(self, tools_db):
        result = get_active_injuries.invoke({})
        assert "NO active injuries" in result

    def test_active_injury_triggers_warning(self, tools_db):
        _insert_injury(tools_db, body_part="achilles", side="right")
        result = get_active_injuries.invoke({})
        assert "WARNING" in result
        assert "achilles" in result
        assert "right" in result

    def test_resolved_injury_not_shown(self, tools_db):
        _insert_injury(tools_db, status="resolved")
        result = get_active_injuries.invoke({})
        assert "NO active injuries" in result


class TestGetInjuryRecoveryTrend:
    def test_injury_not_found(self, tools_db):
        result = get_injury_recovery_trend.invoke({"injury_id": 999})
        assert "not found" in result

    def test_no_checkins_prompts_tracking(self, tools_db):
        iid = _insert_injury(tools_db)
        result = get_injury_recovery_trend.invoke({"injury_id": iid})
        assert "No check-ins" in result

    def test_improving_trend_detected(self, tools_db):
        iid = _insert_injury(tools_db)
        pain_series = [8, 7, 5, 3, 2, 1, 1, 1]
        con = sqlite3.connect(str(tools_db))
        for i, pain in enumerate(pain_series):
            dt = (datetime.now(timezone.utc) - timedelta(days=len(pain_series) - 1 - i)).strftime("%Y-%m-%d")
            con.execute(
                "INSERT INTO injury_checks (injury_id, check_date, pain_scale, pain_context)"
                " VALUES (?, ?, ?, 'rest')",
                (iid, dt, pain),
            )
        con.commit()
        con.close()
        result = get_injury_recovery_trend.invoke({"injury_id": iid})
        assert "IMPROVING" in result

    def test_return_to_train_cleared_after_three_low_pain_days(self, tools_db):
        iid = _insert_injury(tools_db)
        con = sqlite3.connect(str(tools_db))
        for i in range(3):
            dt = (datetime.now(timezone.utc) - timedelta(days=2 - i)).strftime("%Y-%m-%d")
            con.execute(
                "INSERT INTO injury_checks (injury_id, check_date, pain_scale, pain_context)"
                " VALUES (?, ?, 1, 'rest')",
                (iid, dt),
            )
        con.commit()
        con.close()
        result = get_injury_recovery_trend.invoke({"injury_id": iid})
        assert "RETURN-TO-TRAIN CLEARED" in result


# ═══════════════════════════════════════════════════════════════════════════════
# injury_write_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogInjury:
    def test_invalid_side_returns_error(self, tools_db):
        result = log_injury.invoke({
            "body_part": "knee", "side": "invalid",
            "severity": "mild", "pain_scale": 3, "pain_context": "workout",
        })
        assert "Error" in result and "side" in result

    def test_invalid_severity_returns_error(self, tools_db):
        result = log_injury.invoke({
            "body_part": "knee", "side": "left",
            "severity": "extreme", "pain_scale": 3, "pain_context": "workout",
        })
        assert "Error" in result and "severity" in result

    def test_pain_scale_above_10_rejected(self, tools_db):
        result = log_injury.invoke({
            "body_part": "knee", "side": "left",
            "severity": "mild", "pain_scale": 11, "pain_context": "workout",
        })
        assert "Error" in result and "pain_scale" in result

    def test_invalid_pain_context_returns_error(self, tools_db):
        result = log_injury.invoke({
            "body_part": "knee", "side": "left",
            "severity": "mild", "pain_scale": 3, "pain_context": "jogging",
        })
        assert "Error" in result and "pain_context" in result

    def test_success_returns_injury_id(self, tools_db):
        result = log_injury.invoke({
            "body_part": "achilles", "side": "right",
            "severity": "mild", "pain_scale": 4, "pain_context": "workout",
        })
        assert "Injury logged" in result
        assert "achilles" in result

    def test_persists_to_db(self, tools_db):
        log_injury.invoke({
            "body_part": "shin", "side": "left",
            "severity": "moderate", "pain_scale": 5, "pain_context": "rest",
        })
        con = sqlite3.connect(str(tools_db))
        row = con.execute("SELECT body_part, status FROM injuries WHERE body_part='shin'").fetchone()
        con.close()
        assert row is not None
        assert row[1] == "active"


class TestLogInjuryCheckin:
    def test_pain_scale_out_of_range(self, tools_db):
        iid = _insert_injury(tools_db)
        result = log_injury_checkin.invoke({"injury_id": iid, "pain_scale": -1, "pain_context": "rest"})
        assert "Error" in result

    def test_invalid_pain_context(self, tools_db):
        iid = _insert_injury(tools_db)
        result = log_injury_checkin.invoke({"injury_id": iid, "pain_scale": 3, "pain_context": "sleep"})
        assert "Error" in result

    def test_injury_not_found(self, tools_db):
        result = log_injury_checkin.invoke({"injury_id": 9999, "pain_scale": 3, "pain_context": "rest"})
        assert "not found" in result

    def test_success_shows_improving_trend(self, tools_db):
        # Seeded pain_scale = 6; check-in with 3 → improving
        iid = _insert_injury(tools_db, pain_scale=6)
        result = log_injury_checkin.invoke({"injury_id": iid, "pain_scale": 3, "pain_context": "rest"})
        assert "Check-in recorded" in result
        assert "improving" in result

    def test_updates_injury_pain_scale(self, tools_db):
        iid = _insert_injury(tools_db, pain_scale=7)
        log_injury_checkin.invoke({"injury_id": iid, "pain_scale": 4, "pain_context": "workout"})
        con = sqlite3.connect(str(tools_db))
        row = con.execute("SELECT pain_scale FROM injuries WHERE id=?", (iid,)).fetchone()
        con.close()
        assert row[0] == 4


class TestResolveInjury:
    def test_injury_not_found(self, tools_db):
        result = resolve_injury.invoke({"injury_id": 999})
        assert "not found" in result

    def test_already_resolved_returns_message(self, tools_db):
        iid = _insert_injury(tools_db, status="resolved")
        result = resolve_injury.invoke({"injury_id": iid})
        assert "already marked as resolved" in result

    def test_success_marks_resolved(self, tools_db):
        iid = _insert_injury(tools_db)
        result = resolve_injury.invoke({"injury_id": iid})
        assert "Injury resolved" in result
        con = sqlite3.connect(str(tools_db))
        row = con.execute("SELECT status, resolved_date FROM injuries WHERE id=?", (iid,)).fetchone()
        con.close()
        assert row[0] == "resolved"
        assert row[1] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# plan_tool
# ═══════════════════════════════════════════════════════════════════════════════

_BASE_SESSION = {
    "day_date": "2026-06-02",
    "activity_type": "running",
    "workout_type": "easy",
    "description": "45 min easy run",
    "intensity": "easy",
    "target_duration_min": 45,
    "phase": "base",
}


class TestSaveWorkoutPlan:
    def test_invalid_json_returns_error(self, tools_db):
        result = save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": "not json"})
        assert "Error" in result and "JSON" in result

    def test_missing_required_field(self, tools_db):
        bad = [{"day_date": "2026-06-02", "activity_type": "running"}]
        result = save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps(bad)})
        assert "missing required fields" in result

    def test_success_saves_sessions(self, tools_db):
        result = save_workout_plan.invoke({
            "week_start": "2026-06-01",
            "sessions": json.dumps([_BASE_SESSION]),
        })
        assert "Saved 1 sessions" in result
        assert "2026-06-01" in result

    def test_replaces_existing_plan_atomically(self, tools_db):
        save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps([_BASE_SESSION])})
        new_session = dict(_BASE_SESSION, day_date="2026-06-03", description="Recovery run")
        save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps([new_session])})
        con = sqlite3.connect(str(tools_db))
        rows = con.execute(
            "SELECT day_date FROM planned_workouts WHERE week_start='2026-06-01'"
        ).fetchall()
        con.close()
        assert len(rows) == 1
        assert rows[0][0] == "2026-06-03"

    def test_assessment_count_in_summary(self, tools_db):
        session = dict(_BASE_SESSION, is_assessment=1)
        result = save_workout_plan.invoke({
            "week_start": "2026-06-01",
            "sessions": json.dumps([session]),
        })
        assert "1 assessment session" in result


class TestGetCurrentWorkoutPlan:
    def test_no_plan_returns_prompt(self, tools_db):
        result = get_current_workout_plan.invoke({})
        assert "No training plan found" in result

    def test_returns_current_week_plan(self, tools_db):
        ws = _week_start()
        today = _today()
        session = dict(_BASE_SESSION, day_date=today, week_start=ws)
        save_workout_plan.invoke({"week_start": ws, "sessions": json.dumps([session])})
        result = get_current_workout_plan.invoke({})
        assert "Training Plan" in result
        assert today in result

    def test_falls_back_to_future_plan(self, tools_db):
        future_ws = "2099-01-05"
        session = dict(_BASE_SESSION, day_date="2099-01-07")
        save_workout_plan.invoke({"week_start": future_ws, "sessions": json.dumps([session])})
        result = get_current_workout_plan.invoke({})
        assert "2099-01-07" in result


class TestReplaceDayInPlan:
    def test_invalid_json_returns_error(self, tools_db):
        result = replace_day_in_plan.invoke({
            "week_start": "2026-06-01", "day_date": "2026-06-02", "sessions": "bad",
        })
        assert "Error" in result and "JSON" in result

    def test_missing_fields_returns_error(self, tools_db):
        bad = [{"activity_type": "running"}]
        result = replace_day_in_plan.invoke({
            "week_start": "2026-06-01", "day_date": "2026-06-02",
            "sessions": json.dumps(bad),
        })
        assert "missing required fields" in result

    def test_success_replaces_day(self, tools_db):
        save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps([_BASE_SESSION])})
        replacement = {
            "activity_type": "rest",
            "workout_type": "rest",
            "description": "Rest day",
            "intensity": "rest",
        }
        result = replace_day_in_plan.invoke({
            "week_start": "2026-06-01",
            "day_date": "2026-06-02",
            "sessions": json.dumps([replacement]),
        })
        assert "Updated 1 session" in result
        con = sqlite3.connect(str(tools_db))
        row = con.execute(
            "SELECT workout_type FROM planned_workouts WHERE week_start='2026-06-01' AND day_date='2026-06-02'"
        ).fetchone()
        con.close()
        assert row[0] == "rest"


class TestUpdatePlannedWorkoutStatus:
    def test_invalid_status_returns_error(self, tools_db):
        result = update_planned_workout_status.invoke({"day_date": "2026-06-02", "status": "cancelled"})
        assert "Error" in result and "status" in result

    def test_no_sessions_for_date(self, tools_db):
        result = update_planned_workout_status.invoke({"day_date": "2020-01-01", "status": "skipped"})
        assert "No planned sessions found" in result

    def test_success_updates_status(self, tools_db):
        save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps([_BASE_SESSION])})
        result = update_planned_workout_status.invoke({"day_date": "2026-06-02", "status": "completed"})
        assert "completed" in result
        con = sqlite3.connect(str(tools_db))
        row = con.execute(
            "SELECT status FROM planned_workouts WHERE day_date='2026-06-02'"
        ).fetchone()
        con.close()
        assert row[0] == "completed"

    def test_reason_appended_to_notes(self, tools_db):
        save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps([_BASE_SESSION])})
        update_planned_workout_status.invoke({
            "day_date": "2026-06-02", "status": "skipped", "reason": "knee pain",
        })
        con = sqlite3.connect(str(tools_db))
        row = con.execute(
            "SELECT notes FROM planned_workouts WHERE day_date='2026-06-02'"
        ).fetchone()
        con.close()
        assert "knee pain" in row[0]


# ═══════════════════════════════════════════════════════════════════════════════
# workout_history_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetRecentWorkouts:
    def test_unknown_activity_type_returns_not_found(self, tools_db):
        result = get_recent_workouts.invoke({"activity_type": "cycling"})
        assert "No recent cycling workouts found" in result

    def test_returns_running_workouts(self, tools_db):
        result = get_recent_workouts.invoke({"limit": 5, "activity_type": "running"})
        assert "Running Workouts" in result
        assert "2026-01-10" in result

    def test_limit_larger_than_15_does_not_crash(self, tools_db):
        result = get_recent_workouts.invoke({"limit": 100, "activity_type": "running"})
        assert "Running Workouts" in result

    def test_strength_activity_returned(self, tools_db):
        result = get_recent_workouts.invoke({"limit": 5, "activity_type": "strength"})
        assert "Strength Workouts" in result
        assert "2026-01-08" in result


# ═══════════════════════════════════════════════════════════════════════════════
# log_workout_feedback_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogWorkoutRpeAndNotes:
    def test_rpe_zero_rejected(self, tools_db):
        result = log_workout_rpe_and_notes.invoke({"rpe": 0, "notes": "test"})
        assert "Error" in result and "RPE" in result

    def test_rpe_eleven_rejected(self, tools_db):
        result = log_workout_rpe_and_notes.invoke({"rpe": 11, "notes": "test"})
        assert "Error" in result and "RPE" in result

    def test_no_workout_on_date(self, tools_db):
        result = log_workout_rpe_and_notes.invoke({"rpe": 7, "notes": "felt good", "date": "2020-01-01"})
        assert "Error" in result and "No" in result

    def test_success_saves_rpe_and_notes(self, tools_db):
        result = log_workout_rpe_and_notes.invoke({
            "rpe": 7, "notes": "felt strong", "date": "2026-01-10",
        })
        assert "Successfully logged" in result
        con = sqlite3.connect(str(tools_db))
        row = con.execute("SELECT rpe, notes FROM workouts WHERE id=1").fetchone()
        con.close()
        assert row[0] == 7
        assert row[1] == "felt strong"


# ═══════════════════════════════════════════════════════════════════════════════
# strength_tool
# ═══════════════════════════════════════════════════════════════════════════════

_SQUAT_SETS = json.dumps([
    {"exercise_name": "squat", "set_number": 1, "weight_kg": 80.0, "reps": 5, "rpe": 7},
    {"exercise_name": "squat", "set_number": 2, "weight_kg": 80.0, "reps": 5, "rpe": 8},
])


class TestLogStrengthSets:
    def test_invalid_json_returns_error(self, tools_db):
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": "bad json"})
        assert "Error" in result and "JSON" in result

    def test_missing_exercise_name_returns_error(self, tools_db):
        bad = json.dumps([{"set_number": 1, "weight_kg": 80, "reps": 5}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": bad})
        assert "exercise_name" in result

    def test_workout_not_found(self, tools_db):
        result = log_strength_sets.invoke({"workout_id": 9999, "sets_json": _SQUAT_SETS})
        assert "not found" in result

    def test_success_includes_1rm_estimate(self, tools_db):
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": _SQUAT_SETS})
        assert "Logged 2 sets" in result
        assert "est. 1RM" in result

    def test_relog_replaces_previous_sets(self, tools_db):
        log_strength_sets.invoke({"workout_id": 2, "sets_json": _SQUAT_SETS})
        log_strength_sets.invoke({"workout_id": 2, "sets_json": _SQUAT_SETS})
        con = sqlite3.connect(str(tools_db))
        count = con.execute("SELECT COUNT(*) FROM strength_sets WHERE workout_id=2").fetchone()[0]
        con.close()
        assert count == 2  # not 4 — second call clears and re-inserts


class TestGetRecentStrengthSets:
    def test_no_sets_returns_helpful_message(self, tools_db):
        result = get_recent_strength_sets.invoke({"exercise_name": "bench_press"})
        assert "No logged sets found" in result

    def test_returns_sets_after_logging(self, tools_db):
        log_strength_sets.invoke({"workout_id": 2, "sets_json": _SQUAT_SETS})
        result = get_recent_strength_sets.invoke({"exercise_name": "squat"})
        assert "Squat" in result
        assert "80" in result

    def test_sessions_param_over_10_does_not_crash(self, tools_db):
        result = get_recent_strength_sets.invoke({"exercise_name": "deadlift", "sessions": 999})
        assert "No logged sets" in result or "Deadlift" in result


# ═══════════════════════════════════════════════════════════════════════════════
# assessment_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetOnboardingStatus:
    def test_no_profile(self, empty_db):
        result = get_onboarding_status.invoke({})
        assert "No athlete profile found" in result

    def test_onboarding_complete(self, tools_db):
        con = sqlite3.connect(str(tools_db))
        con.execute("UPDATE athlete_profile SET onboarding_complete=1")
        con.commit()
        con.close()
        result = get_onboarding_status.invoke({})
        assert "Onboarding complete" in result

    def test_beginner_shows_two_day_assessment_protocol(self, tools_db):
        con = sqlite3.connect(str(tools_db))
        con.execute("UPDATE athlete_profile SET fitness_level='beginner', onboarding_complete=0")
        con.commit()
        con.close()
        result = get_onboarding_status.invoke({})
        assert "beginner" in result.lower()
        assert "DAY 1" in result

    def test_intermediate_without_assessments_shows_missing(self, tools_db):
        result = get_onboarding_status.invoke({})
        assert "Missing baseline assessments" in result


class TestLogFitnessAssessment:
    def test_time_trial_derives_vdot(self, tools_db):
        # 600 sec for 2 km = 300 sec/km → closest to VDOT 55 (t_pace=305)
        result = log_fitness_assessment.invoke({
            "assessment_type": "time_trial",
            "metric_name": "time_sec",
            "metric_value": 600.0,
            "distance_km": 2.0,
        })
        assert "Assessment logged" in result
        assert "VDOT" in result

    def test_strength_1rm_derives_epley_estimate(self, tools_db):
        result = log_fitness_assessment.invoke({
            "assessment_type": "strength_1rm",
            "metric_name": "weight_kg",
            "metric_value": 100.0,
            "exercise_name": "squat",
            "reps": 5,
        })
        assert "Assessment logged" in result
        assert "1RM" in result
        assert "squat" in result

    def test_cooper_test_derives_vdot(self, tools_db):
        # 2800 m → VO2max ≈ 51.3 → closest VDOT 50
        result = log_fitness_assessment.invoke({
            "assessment_type": "cooper_test",
            "metric_name": "distance_m",
            "metric_value": 2800.0,
        })
        assert "Assessment logged" in result


class TestGetFitnessAssessments:
    def test_no_assessments_message(self, tools_db):
        result = get_fitness_assessments.invoke({})
        assert "No fitness assessments recorded yet" in result

    def test_returns_logged_assessments(self, tools_db):
        log_fitness_assessment.invoke({
            "assessment_type": "time_trial",
            "metric_name": "time_sec",
            "metric_value": 600.0,
            "distance_km": 2.0,
        })
        result = get_fitness_assessments.invoke({})
        assert "time_trial" in result

    def test_filter_by_type_excludes_others(self, tools_db):
        log_fitness_assessment.invoke({
            "assessment_type": "time_trial",
            "metric_name": "time_sec",
            "metric_value": 600.0,
            "distance_km": 2.0,
        })
        log_fitness_assessment.invoke({
            "assessment_type": "body_composition",
            "metric_name": "weight_kg",
            "metric_value": 72.5,
        })
        result = get_fitness_assessments.invoke({"assessment_type": "time_trial"})
        assert "time_trial" in result
        assert "body_composition" not in result

    def test_vdot_trend_shown_with_two_time_trials(self, tools_db):
        # Older: 720 sec / 2km = 360 sec/km → VDOT 50
        log_fitness_assessment.invoke({
            "assessment_type": "time_trial",
            "metric_name": "time_sec",
            "metric_value": 720.0,
            "distance_km": 2.0,
        })
        # Newer: 600 sec / 2km = 300 sec/km → VDOT 55
        log_fitness_assessment.invoke({
            "assessment_type": "time_trial",
            "metric_name": "time_sec",
            "metric_value": 600.0,
            "distance_km": 2.0,
        })
        result = get_fitness_assessments.invoke({"assessment_type": "time_trial"})
        assert "Running trend" in result


# ═══════════════════════════════════════════════════════════════════════════════
# profile_tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpdateAthleteProfile:
    def test_invalid_field_rejected(self, tools_db):
        result = update_athlete_profile.invoke({"field": "height_cm", "value": "180"})
        assert "not updatable" in result

    def test_success_updates_current_goal(self, tools_db):
        result = update_athlete_profile.invoke({"field": "current_goal", "value": "10k_prep"})
        assert "current_goal" in result
        assert "10k_prep" in result
        con = sqlite3.connect(str(tools_db))
        row = con.execute("SELECT current_goal FROM athlete_profile").fetchone()
        con.close()
        assert row[0] == "10k_prep"

    def test_success_marks_onboarding_complete(self, tools_db):
        update_athlete_profile.invoke({"field": "onboarding_complete", "value": "1"})
        con = sqlite3.connect(str(tools_db))
        row = con.execute("SELECT onboarding_complete FROM athlete_profile").fetchone()
        con.close()
        assert row[0] == 1

    def test_no_profile_returns_message(self, empty_db):
        result = update_athlete_profile.invoke({"field": "current_goal", "value": "marathon_prep"})
        assert "No athlete profile found" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Data Freshness
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataFreshness:
    """
    get_daily_readiness must prepend a sync warning when the most recent
    daily_health row is older than _STALE_DAYS, and must stay silent when
    the data is current.
    """

    def test_stale_data_prepends_sync_warning(self, tools_db):
        # The fixture seeds daily_health with date='2026-01-10', which is well
        # over 7 days in the past relative to today (2026-06-02+).
        result = get_daily_readiness.invoke({"date": "2026-01-10"})
        assert "STALE" in result
        assert "sync" in result.lower() or "Sync" in result

    def test_stale_warning_mentions_health_app(self, tools_db):
        result = get_daily_readiness.invoke({"date": "2026-01-10"})
        assert "Health" in result

    def test_fresh_data_has_no_stale_warning(self, tools_db):
        today = _today()
        con = sqlite3.connect(str(tools_db))
        con.execute(
            "INSERT OR REPLACE INTO daily_health (date, atl, ctl, tsb)"
            " VALUES (?, 40.0, 50.0, 10.0)",
            (today,),
        )
        con.commit()
        con.close()
        result = get_daily_readiness.invoke({"date": today})
        assert "STALE" not in result

    def test_readiness_content_still_present_even_when_stale(self, tools_db):
        """Stale warning must prepend, not replace, the readiness data."""
        result = get_daily_readiness.invoke({"date": "2026-01-10"})
        assert "Readiness Report" in result
        assert "CTL" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Agent-Write Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveWorkoutPlanValidation:
    """
    save_workout_plan must reject unknown activity_type / workout_type values
    before they reach the database.  These are the most common LLM hallucinations.
    """

    def test_unknown_activity_type_rejected(self, tools_db):
        bad = [dict(_BASE_SESSION, activity_type="yoga")]
        result = save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps(bad)})
        assert "Error" in result and "activity_type" in result

    def test_swimming_activity_type_rejected_for_planner(self, tools_db):
        # 'swimming' is valid in the workouts table but not in planned_workouts
        bad = [dict(_BASE_SESSION, activity_type="swimming")]
        result = save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps(bad)})
        assert "Error" in result and "activity_type" in result

    def test_unknown_workout_type_rejected(self, tools_db):
        bad = [dict(_BASE_SESSION, workout_type="fartlek")]
        result = save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps(bad)})
        assert "Error" in result and "workout_type" in result

    def test_valid_activity_and_type_accepted(self, tools_db):
        good = [dict(_BASE_SESSION)]
        result = save_workout_plan.invoke({"week_start": "2026-06-01", "sessions": json.dumps(good)})
        assert "Saved" in result

    def test_replace_day_rejects_unknown_activity_type(self, tools_db):
        bad = [{
            "activity_type": "yoga",
            "workout_type": "easy",
            "description": "yoga session",
            "intensity": "easy",
        }]
        result = replace_day_in_plan.invoke({
            "week_start": "2026-06-01",
            "day_date": "2026-06-02",
            "sessions": json.dumps(bad),
        })
        assert "Error" in result and "activity_type" in result


class TestLogStrengthSetsValidation:
    """
    log_strength_sets must reject physiologically impossible values before
    writing them.  These protect against LLM-hallucinated weights / rep counts.
    """

    def test_weight_above_500kg_rejected(self, tools_db):
        heavy = json.dumps([{"exercise_name": "squat", "set_number": 1,
                             "weight_kg": 1000.0, "reps": 3}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": heavy})
        assert "Error" in result and "weight_kg" in result

    def test_negative_weight_rejected(self, tools_db):
        neg = json.dumps([{"exercise_name": "deadlift", "set_number": 1,
                           "weight_kg": -20.0, "reps": 5}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": neg})
        assert "Error" in result and "weight_kg" in result

    def test_reps_above_100_rejected(self, tools_db):
        many = json.dumps([{"exercise_name": "squat", "set_number": 1,
                            "weight_kg": 80.0, "reps": 500}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": many})
        assert "Error" in result and "reps" in result

    def test_zero_reps_rejected(self, tools_db):
        zero = json.dumps([{"exercise_name": "squat", "set_number": 1,
                            "weight_kg": 80.0, "reps": 0}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": zero})
        assert "Error" in result and "reps" in result

    def test_set_number_above_50_rejected(self, tools_db):
        bad_set = json.dumps([{"exercise_name": "plank", "set_number": 99,
                               "duration_sec": 60.0}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": bad_set})
        assert "Error" in result and "set_number" in result

    def test_borderline_valid_weight_accepted(self, tools_db):
        edge = json.dumps([{"exercise_name": "squat", "set_number": 1,
                            "weight_kg": 500.0, "reps": 1}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": edge})
        assert "Logged" in result

    def test_bodyweight_set_no_weight_accepted(self, tools_db):
        bw = json.dumps([{"exercise_name": "pull_up", "set_number": 1, "reps": 10}])
        result = log_strength_sets.invoke({"workout_id": 2, "sets_json": bw})
        assert "Logged" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════

# Strings that must never appear in any tool output — they degrade LLM reasoning.
_FORBIDDEN_STRINGS = ("None", "NaN", "null", "Traceback", "TypeError",
                      "AttributeError", "KeyError", "IndexError")


class TestToolOutputFormatting:
    """
    Tools must never leak Python None values, NaN, or raw exception tracebacks
    into their string output.  An LLM reading 'Resting HR: None bpm' draws wrong
    conclusions; an exception traceback is even worse.
    """

    def _assert_clean(self, result: str, tool_name: str) -> None:
        for bad in _FORBIDDEN_STRINGS:
            assert bad not in result, (
                f"{tool_name} returned the forbidden string '{bad}'.\n"
                f"Full output: {result[:300]}"
            )

    # ── Empty / no-data paths ─────────────────────────────────────────────────

    def test_readiness_empty_db_is_clean(self, empty_db):
        self._assert_clean(get_daily_readiness.invoke({"date": "2025-01-01"}), "get_daily_readiness")

    def test_recent_workouts_empty_db_is_clean(self, empty_db):
        self._assert_clean(
            get_recent_workouts.invoke({"limit": 5, "activity_type": "running"}),
            "get_recent_workouts",
        )

    def test_nutrition_profile_empty_db_is_clean(self, empty_db):
        self._assert_clean(get_nutrition_profile.invoke({}), "get_nutrition_profile")

    def test_current_workout_plan_empty_db_is_clean(self, empty_db):
        self._assert_clean(get_current_workout_plan.invoke({}), "get_current_workout_plan")

    def test_strength_sets_empty_db_is_clean(self, empty_db):
        self._assert_clean(
            get_recent_strength_sets.invoke({"exercise_name": "squat"}),
            "get_recent_strength_sets",
        )

    # ── Sparse rows (only mandatory columns populated) ────────────────────────

    def test_readiness_sparse_row_is_clean(self, tools_db):
        """A daily_health row with only ATL/CTL/TSB set must render cleanly."""
        con = sqlite3.connect(str(tools_db))
        con.execute(
            "INSERT OR REPLACE INTO daily_health (date, atl, ctl, tsb)"
            " VALUES ('2026-03-15', 40.0, 50.0, 10.0)"
        )
        con.commit()
        con.close()
        result = get_daily_readiness.invoke({"date": "2026-03-15"})
        self._assert_clean(result, "get_daily_readiness (sparse row)")
        # N/A is the correct placeholder — confirm the tool didn't blank them out
        assert "N/A" in result

    def test_recent_workouts_null_fields_is_clean(self, tools_db):
        """A workout with null distance/HR/RPE must not produce 'None' in output."""
        con = sqlite3.connect(str(tools_db))
        con.execute(
            """INSERT INTO workouts
                   (id, activity_type, start_date, end_date, duration_min)
               VALUES (99, 'running', '2026-01-20 08:00:00', '2026-01-20 09:00:00', 55.0)"""
        )
        con.commit()
        con.close()
        result = get_recent_workouts.invoke({"limit": 10, "activity_type": "running"})
        self._assert_clean(result, "get_recent_workouts (sparse workout)")
