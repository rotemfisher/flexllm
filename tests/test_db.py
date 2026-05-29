"""
tests/test_db.py  —  Read-only data-quality tests for data/personal/running.db

All tests use the `prod_db` session fixture from conftest.py.
Nothing in this file writes to the database.

Run fast subset only:
    pytest tests/test_db.py -v
"""

from pathlib import Path

import pytest

# ─── Expected values ──────────────────────────────────────────────────────────

KNOWN_ACTIVITY_TYPES = {
    "running", "strength", "swimming", "walking",
    "cycling", "hiking", "yoga", "mindfulness",
    "elliptical", "stair_climbing", "hiit",
    "cross_training", "cooldown", "other",
}

DEDUP_INDICES = {
    "ux_workouts_start_type",
    "ux_sleep_window",
    "ux_health_rec",
    "ux_gps_track",
    "ux_laps",
}

# Stable date with known sleep aggregate (from production data, 2025-05-25)
SLEEP_CHECK_DATE = "2025-05-25"
SLEEP_CHECK_EXPECTED_MIN = 197.5

# VDOT 50 ground-truth values from Daniels' Running Formula 4th ed., Table 3.1
VDOT_50 = {
    "e_pace_slow_sec": 372,
    "e_pace_fast_sec": 353,
    "m_pace_sec":      343,
    "t_pace_sec":      330,
    "i_pace_sec":      313,
    "r_pace_sec":      299,
}

# ─── Schema ───────────────────────────────────────────────────────────────────

def test_dedup_indices_present(prod_db):
    """All five dedup unique indices must be in sqlite_master."""
    present = {
        row[0]
        for row in prod_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    missing = DEDUP_INDICES - present
    assert not missing, f"Missing dedup indices: {missing}"


def test_view_exists_and_returns_rows(prod_db):
    """v_running_overview must exist and return at least one row."""
    views = {
        row[0]
        for row in prod_db.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        )
    }
    assert "v_running_overview" in views, "v_running_overview view is missing"
    count = prod_db.execute("SELECT COUNT(*) FROM v_running_overview").fetchone()[0]
    assert count > 0, "v_running_overview returned 0 rows"


# ─── Coverage / counts ────────────────────────────────────────────────────────

def test_table_counts_within_expected_ranges(prod_db):
    """
    Each populated table must be within a generous range.
    Tables that are intentionally empty must be exactly 0.
    Ranges are wide enough to survive a future Apple Health sync.
    """
    ranges = {
        "workouts":       (120, 300),
        "running_form":   (60,  200),
        "gps_tracks":     (100_000, 400_000),
        "sleep_records":  (1_000,   6_000),
        "health_records": (400_000, 1_200_000),
        "daily_health":   (800,     3_000),
        "workout_laps":   (1_000,   6_000),
        "activity_rings": (800,     3_000),
        "vdot_paces":     (28,      28),
    }
    empty_tables = ["kilometer_splits", "shoes", "run_summaries"]

    for table, (lo, hi) in ranges.items():
        n = prod_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert lo <= n <= hi, f"{table}: expected {lo}–{hi} rows, got {n}"

    for table in empty_tables:
        n = prod_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert n == 0, f"{table}: expected 0 rows (not yet populated), got {n}"


# ─── Workout quality ──────────────────────────────────────────────────────────

def test_workout_dates_are_utc_format(prod_db):
    """
    Every start_date must match 'YYYY-MM-DD HH:MM:SS'.
    The ETL normalises all timestamps to UTC; a timezone offset or ISO-8601 T
    variant would corrupt date() / substr() aggregations.
    """
    bad = prod_db.execute(
        "SELECT COUNT(*) FROM workouts "
        "WHERE start_date NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] "
        "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]'"
    ).fetchone()[0]
    assert bad == 0, f"{bad} workouts have malformed start_date"


def test_running_workout_durations_positive(prod_db):
    """All running workouts must have duration_min > 0 (TRIMP input)."""
    bad = prod_db.execute(
        "SELECT COUNT(*) FROM workouts "
        "WHERE activity_type = 'running' "
        "AND (duration_min IS NULL OR duration_min <= 0)"
    ).fetchone()[0]
    assert bad == 0, f"{bad} running workouts have NULL/zero duration_min"


def test_running_workout_distances_reasonable(prod_db):
    """
    Non-null distances for land-based activities must be between 0.1 km and 150 km.
    Swimming is excluded: Apple Health reports HKQuantityTypeIdentifierDistanceSwimming
    in meters; the ETL converts new ingestions to km, but existing pool-swim rows in the
    production DB (750–2000 m stored raw) are expected to be > 150.
    """
    bad = prod_db.execute(
        "SELECT COUNT(*) FROM workouts "
        "WHERE distance_km IS NOT NULL "
        "AND activity_type IN ('running', 'walking', 'hiking', 'cycling') "
        "AND (distance_km < 0.1 OR distance_km > 150)"
    ).fetchone()[0]
    assert bad == 0, f"{bad} land-activity workouts have distance_km outside [0.1, 150]"


def test_activity_types_are_known_values(prod_db):
    """All activity_type values must be from the ETL's ACTIVITY_TYPES mapping."""
    found = {
        row[0]
        for row in prod_db.execute("SELECT DISTINCT activity_type FROM workouts")
    }
    unknown = found - KNOWN_ACTIVITY_TYPES
    assert not unknown, (
        f"Unknown activity_type values (update ACTIVITY_TYPES in ingest_health.py): {unknown}"
    )


# ─── Referential integrity ────────────────────────────────────────────────────

def test_running_form_linked_to_running_workouts(prod_db):
    """
    Every running_form row must reference a workout with activity_type='running'.
    The FK only checks existence, not type — this catches ingester routing bugs.
    """
    bad = prod_db.execute(
        "SELECT COUNT(*) FROM running_form rf "
        "LEFT JOIN workouts w ON w.id = rf.workout_id "
        "WHERE w.activity_type != 'running' OR w.id IS NULL"
    ).fetchone()[0]
    assert bad == 0, f"{bad} running_form rows linked to non-running or missing workouts"


def test_gps_tracks_linked_to_gpx_workouts(prod_db):
    """
    Every gps_tracks row must reference a workout that has a gpx_file_path.
    Orphaned GPS points or tracks for workouts without GPX files indicate a
    mis-matched GPX parse.
    """
    orphaned = prod_db.execute(
        "SELECT COUNT(*) FROM gps_tracks g "
        "LEFT JOIN workouts w ON w.id = g.workout_id "
        "WHERE w.id IS NULL"
    ).fetchone()[0]
    assert orphaned == 0, f"{orphaned} gps_tracks reference non-existent workout_id"

    no_gpx = prod_db.execute(
        "SELECT COUNT(*) FROM gps_tracks g "
        "JOIN workouts w ON w.id = g.workout_id "
        "WHERE w.gpx_file_path IS NULL"
    ).fetchone()[0]
    assert no_gpx == 0, f"{no_gpx} gps_tracks linked to workouts without a gpx_file_path"


# ─── Computed fields ──────────────────────────────────────────────────────────

def test_tss_set_on_all_hr_workouts(prod_db):
    """
    Every workout with avg_heart_rate_bpm must also have training_stress_score.
    If _compute_tss() ran to completion, no row satisfies (hr NOT NULL AND tss IS NULL).
    """
    missed = prod_db.execute(
        "SELECT COUNT(*) FROM workouts "
        "WHERE avg_heart_rate_bpm IS NOT NULL "
        "AND training_stress_score IS NULL"
    ).fetchone()[0]
    assert missed == 0, (
        f"{missed} workouts have HR data but no TSS — "
        "did you run the ingester's _compute_tss() phase?"
    )

    # Sanity floor: at least 100 workouts should have TSS set
    tss_count = prod_db.execute(
        "SELECT COUNT(*) FROM workouts WHERE training_stress_score IS NOT NULL"
    ).fetchone()[0]
    assert tss_count >= 100, f"Only {tss_count} workouts have TSS (expected ≥100)"


def test_atl_ctl_tsb_fully_populated(prod_db):
    """No daily_health row may have a NULL ATL, CTL, or TSB."""
    null_load = prod_db.execute(
        "SELECT COUNT(*) FROM daily_health "
        "WHERE atl IS NULL OR ctl IS NULL OR tsb IS NULL"
    ).fetchone()[0]
    assert null_load == 0, (
        f"{null_load} daily_health rows have NULL ATL/CTL/TSB — "
        "did you run _compute_load()?"
    )

    max_ctl = prod_db.execute("SELECT MAX(ctl) FROM daily_health").fetchone()[0]
    assert max_ctl is not None and max_ctl > 10, (
        f"MAX(ctl) = {max_ctl} — training load was never meaningfully computed"
    )


# ─── VDOT reference data ─────────────────────────────────────────────────────

def test_vdot_exact_count_and_spot_check(prod_db):
    """
    The vdot_paces table must have exactly 28 rows (VDOT 30–85).
    VDOT=50 values are verified against Daniels' Running Formula 4th ed., Table 3.1.
    """
    count = prod_db.execute("SELECT COUNT(*) FROM vdot_paces").fetchone()[0]
    assert count == 28, f"vdot_paces has {count} rows (expected 28)"

    row = prod_db.execute(
        "SELECT e_pace_slow_sec, e_pace_fast_sec, m_pace_sec, "
        "t_pace_sec, i_pace_sec, r_pace_sec "
        "FROM vdot_paces WHERE vdot = 50"
    ).fetchone()
    assert row is not None, "VDOT=50 row is missing"
    actual = dict(zip(VDOT_50.keys(), row))
    assert actual == VDOT_50, (
        f"VDOT=50 mismatch.\n  expected: {VDOT_50}\n  got:      {actual}"
    )


# ─── Sleep aggregate consistency ─────────────────────────────────────────────

def test_sleep_aggregate_matches_raw(prod_db):
    """
    daily_health.sleep_total_min must equal the sum of non-in_bed sleep_records
    for a stable historical date (2025-05-25).
    """
    row = prod_db.execute(
        "SELECT sleep_total_min FROM daily_health WHERE date = ?",
        (SLEEP_CHECK_DATE,),
    ).fetchone()
    assert row is not None, f"No daily_health row for {SLEEP_CHECK_DATE}"
    assert row[0] is not None, f"sleep_total_min is NULL for {SLEEP_CHECK_DATE}"

    raw_sum = prod_db.execute(
        "SELECT SUM(duration_min) FROM sleep_records "
        "WHERE date = ? AND stage != 'in_bed'",
        (SLEEP_CHECK_DATE,),
    ).fetchone()[0]
    assert raw_sum is not None, f"No sleep_records for {SLEEP_CHECK_DATE}"

    assert abs(row[0] - raw_sum) < 0.01, (
        f"sleep_total_min mismatch for {SLEEP_CHECK_DATE}: "
        f"daily_health={row[0]} vs raw_sum={raw_sum}"
    )
