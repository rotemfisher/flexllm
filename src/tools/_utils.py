from contextlib import contextmanager
import logging
from typing import Generator
import sqlite3

from src.config import config

logger = logging.getLogger(__name__)


def _apply_migrations() -> None:
    """Apply incremental schema changes to an existing database.

    Safe to call on every startup: each migration is idempotent.
    Skips silently when the database file has not been created yet (first run).
    """
    if not config.DB_PATH.exists():
        return
    try:
        con = sqlite3.connect(config.DB_PATH)
        con.execute("PRAGMA journal_mode=WAL")

        # v2 — soft deletes for planned_workouts
        try:
            con.execute("ALTER TABLE planned_workouts ADD COLUMN deleted_at TEXT")
            con.commit()
            logger.info("DB migration: added deleted_at column to planned_workouts")
        except sqlite3.OperationalError:
            pass  # column already exists

        # v2 — replace full unique index with a partial (active-rows-only) unique index
        con.execute("DROP INDEX IF EXISTS idx_planned_workouts_day_order")
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_planned_workouts_active_day_order "
            "ON planned_workouts(week_start, day_date, session_order) "
            "WHERE deleted_at IS NULL"
        )
        con.commit()
        con.close()
    except Exception as exc:
        logger.warning("DB migration check failed: %s", exc)


_apply_migrations()


def epley_1rm(weight_kg: float, reps: int) -> float:
    """Epley formula: estimates 1RM from a sub-maximal set."""
    return weight_kg if reps == 1 else weight_kg * (1 + reps / 30)


@contextmanager
def db_ro() -> Generator[sqlite3.Connection, None, None]:
    """Read-only SQLite connection as a context manager."""
    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


@contextmanager
def db_rw() -> Generator[sqlite3.Connection, None, None]:
    """Read-write SQLite connection; rolls back on exception before closing."""
    con = sqlite3.connect(config.DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
