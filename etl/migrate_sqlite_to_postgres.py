#!/usr/bin/env python3
"""
etl/migrate_sqlite_to_postgres.py — One-time migration: SQLite running.db → PostgreSQL

Copies all existing workout history from running.db into the PostgreSQL
database. Safe to re-run: uses ON CONFLICT DO NOTHING so nothing is duplicated
or lost. Resets BIGSERIAL sequences after bulk insert so new ETL runs don't
produce ID conflicts.

Usage:
    python etl/migrate_sqlite_to_postgres.py
    python etl/migrate_sqlite_to_postgres.py --sqlite /path/to/running.db
    DATABASE_URL=postgresql://user:pass@host/db python etl/migrate_sqlite_to_postgres.py
"""

import argparse
import logging
import os
import sqlite3
from pathlib import Path

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).parent.parent

_DEFAULT_SQLITE   = ROOT / "data" / "personal" / "running.db"
_DEFAULT_PG_URL   = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/flexllm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Tables in FK-dependency order.
# (name, has_bigserial_id_column)
TABLES = [
    ("athlete_profile",    True),
    ("shoes",              True),
    ("daily_health",       False),   # TEXT PK: date
    ("activity_rings",     False),   # TEXT PK: date
    ("sleep_records",      True),
    ("workouts",           True),
    ("running_form",       False),   # PK: workout_id (FK, not serial)
    ("workout_laps",       True),
    ("kilometer_splits",   True),
    ("health_records",     True),
    ("gps_tracks",         True),
    ("injuries",           True),
    ("injury_checks",      True),
    ("planned_workouts",   True),
    ("fitness_assessments", True),
    ("vdot_paces",         False),   # INTEGER PK: vdot
    ("conversation_summaries", True),
]


def _sqlite_columns(sqlite_con: sqlite3.Connection, table: str) -> list[str]:
    rows = sqlite_con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _table_exists_sqlite(sqlite_con: sqlite3.Connection, table: str) -> bool:
    row = sqlite_con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return bool(row and row[0])


def migrate_table(
    sqlite_con: sqlite3.Connection,
    pg_con,
    table: str,
) -> int:
    if not _table_exists_sqlite(sqlite_con, table):
        logger.info("  %-30s — not in SQLite, skipping", table)
        return 0

    cols = _sqlite_columns(sqlite_con, table)
    rows = sqlite_con.execute(
        f"SELECT {', '.join(cols)} FROM {table}"
    ).fetchall()

    if not rows:
        logger.info("  %-30s — empty, skipping", table)
        return 0

    placeholders = ", ".join(["%s"] * len(cols))
    cols_sql     = ", ".join(cols)
    sql = (
        f"INSERT INTO {table} ({cols_sql}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT DO NOTHING"
    )

    cur = pg_con.cursor()
    psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    pg_con.commit()

    logger.info("  %-30s — %d rows", table, len(rows))
    return len(rows)


def reset_sequences(pg_con) -> None:
    """Advance BIGSERIAL sequences to MAX(id) so new inserts don't collide."""
    cur = pg_con.cursor()
    for table, has_serial in TABLES:
        if not has_serial:
            continue
        try:
            cur.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"COALESCE(MAX(id), 1)) FROM {table}"
            )
            pg_con.commit()
        except Exception as e:
            pg_con.rollback()
            logger.warning("Could not reset sequence for %s: %s", table, e)
    logger.info("Sequences reset.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-time migration: SQLite running.db → PostgreSQL"
    )
    p.add_argument("--sqlite",       type=Path, default=_DEFAULT_SQLITE, metavar="PATH",
                   help="Path to running.db SQLite database")
    p.add_argument("--database-url", type=str,  default=_DEFAULT_PG_URL,  metavar="URL",
                   help="PostgreSQL connection URL")
    args = p.parse_args()

    if not args.sqlite.exists():
        logger.error("SQLite file not found: %s", args.sqlite)
        raise SystemExit(1)

    logger.info("Source : %s", args.sqlite)
    logger.info("Target : %s", args.database_url)

    sqlite_con = sqlite3.connect(args.sqlite)
    sqlite_con.row_factory = None  # return plain tuples

    pg_con = psycopg2.connect(args.database_url)

    total = 0
    for table, _ in TABLES:
        total += migrate_table(sqlite_con, pg_con, table)

    reset_sequences(pg_con)

    sqlite_con.close()
    pg_con.close()

    logger.info("Migration complete. %d rows copied.", total)


if __name__ == "__main__":
    main()
