from contextlib import contextmanager
from typing import Generator
import sqlite3

from src.config import config


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
