from contextlib import contextmanager
import logging
from typing import Generator

import psycopg
from psycopg.rows import dict_row

from src.config import config

logger = logging.getLogger(__name__)


def epley_1rm(weight_kg: float, reps: int) -> float:
    """Epley formula: estimates 1RM from a sub-maximal set."""
    return weight_kg if reps == 1 else weight_kg * (1 + reps / 30)


@contextmanager
def db_ro() -> Generator[psycopg.Connection, None, None]:
    """Read-only PostgreSQL connection. Autocommit — no transaction overhead."""
    with psycopg.connect(config.DATABASE_URL, row_factory=dict_row, autocommit=True) as con:
        yield con


@contextmanager
def db_rw() -> Generator[psycopg.Connection, None, None]:
    """Read-write PostgreSQL connection.

    Commits on clean exit, rolls back on exception.  Callers may call
    con.commit() mid-block for explicit checkpoints within a single transaction.
    """
    with psycopg.connect(config.DATABASE_URL, row_factory=dict_row) as con:
        yield con
