"""
tests/test_ingest.py  —  ETL smoke tests for etl/ingest_health.py

Each test uses the `pg_temp_dsn` fixture (a fresh throw-away PostgreSQL schema)
and never touches the production database.

The three full-parse tests are marked @pytest.mark.slow because each requires
streaming the entire 2.8M-line Apple Health XML (~60–120 s).
Skip them during quick iteration with:
    pytest tests/test_ingest.py -m "not slow" -v
"""

import sys
from pathlib import Path

import psycopg2
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from etl.ingest_health import EXPORT_DIR, XML_FILE, HealthIngester


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _row_counts(dsn: str) -> dict[str, int]:
    con = psycopg2.connect(dsn)
    tables = [
        "workouts", "running_form", "workout_laps",
        "sleep_records", "health_records",
        "activity_rings", "daily_health",
    ]
    cur = con.cursor()
    counts = {}
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        counts[t] = cur.fetchone()[0]
    con.close()
    return counts


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_dedup_indices_exist(pg_temp_dsn):
    """All five required dedup unique indices must be present after DB init."""
    HealthIngester(database_url=pg_temp_dsn, xml=XML_FILE, export_dir=EXPORT_DIR)

    con = psycopg2.connect(pg_temp_dsn)
    cur = con.cursor()
    cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()")
    present = {row[0] for row in cur.fetchall()}
    con.close()

    required = {
        "ux_workouts_start_type",
        "ux_sleep_window",
        "ux_health_rec",
        "ux_gps_track",
        "ux_laps",
    }
    missing = required - present
    assert not missing, f"Missing dedup indices: {missing}"


@pytest.mark.slow
def test_xml_stream_populates_tables(pg_temp_dsn, apple_health_xml):
    """A single XML pass must produce non-empty rows in every key table."""
    HealthIngester(database_url=pg_temp_dsn, xml=apple_health_xml, export_dir=EXPORT_DIR)._stream_xml()
    counts = _row_counts(pg_temp_dsn)

    assert counts["workouts"]       > 0, "Expected workouts"
    assert counts["running_form"]   > 0, "Expected running_form rows for running workouts"
    assert counts["workout_laps"]   > 0, "Expected workout_laps"
    assert counts["sleep_records"]  > 0, "Expected sleep records"
    assert counts["health_records"] > 0, "Expected health_records (HR, speed …)"
    assert counts["activity_rings"] > 0, "Expected activity_rings from ActivitySummary"
    assert counts["daily_health"]   > 0, "Expected daily_health rows seeded from activity rings"


@pytest.mark.slow
def test_no_duplicates_on_rerun(pg_temp_dsn, apple_health_xml):
    """Running the ingester twice must leave every table count unchanged."""
    HealthIngester(database_url=pg_temp_dsn, xml=apple_health_xml, export_dir=EXPORT_DIR)._stream_xml()
    after_run1 = _row_counts(pg_temp_dsn)

    HealthIngester(database_url=pg_temp_dsn, xml=apple_health_xml, export_dir=EXPORT_DIR)._stream_xml()
    after_run2 = _row_counts(pg_temp_dsn)

    for table, n1 in after_run1.items():
        n2 = after_run2[table]
        assert n2 == n1, (
            f"{table}: run1={n1} run2={n2} delta={n2 - n1:+d} — DUPLICATE DETECTED"
        )


@pytest.mark.slow
def test_gpx_and_tss(pg_temp_dsn, apple_health_xml):
    """GPX tracks must be linked to workouts; TSS must be set on workouts with HR."""
    ingester = HealthIngester(database_url=pg_temp_dsn, xml=apple_health_xml, export_dir=EXPORT_DIR)
    ingester._stream_xml()
    ingester._stream_gpx()
    ingester._compute_tss()

    con = psycopg2.connect(pg_temp_dsn)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM gps_tracks")
    gps_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM workouts WHERE training_stress_score IS NOT NULL")
    tss_count = cur.fetchone()[0]
    con.close()

    assert gps_count > 0, "Expected GPS track points"
    assert tss_count > 0, "Expected workouts with TSS computed"
