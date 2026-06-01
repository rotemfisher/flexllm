"""
tests/conftest.py  —  Shared pytest fixtures for the FlexLLM test suite.

Fixtures
────────
  prod_db      session-scoped, read-only sqlite3 connection to running.db
  qdrant_col   session-scoped (client, collection_name) tuple for the coaching_books
               Qdrant collection; dense + sparse models loaded once per session
  temp_db      function-scoped Path to a fresh temporary SQLite file;
               auto-cleaned by pytest's tmp_path machinery
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.tracing import setup_tracing
setup_tracing()  # activate LangSmith tracing if LANGSMITH_API_KEY is set


# ── prod_db ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def prod_db():
    """
    Read-only connection to data/personal/running.db.

    Uses SQLite URI mode (?mode=ro) so the engine refuses any accidental write
    at the OS level — tests cannot corrupt production data even if they try.
    row_factory = sqlite3.Row enables column access by name in assertions.
    """
    db_path = ROOT / "data" / "personal" / "running.db"
    if not db_path.exists():
        pytest.skip(
            "data/personal/running.db not found — run `python etl/ingest_health.py` first"
        )
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    yield con
    con.close()


# ── qdrant_col ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant_col():
    """
    Session-scoped (client, dense_model, sparse_model, collection_name) tuple.

    The BAAI/bge-large-en-v1.5 dense model and Qdrant/bm25 sparse model are
    loaded once per session; subsequent semantic-query tests reuse them.
    Skips loudly if the Qdrant database has not been created yet.
    """
    qdrant_path = ROOT / "data" / "qdrant_db"
    if not qdrant_path.exists():
        pytest.skip(
            "data/qdrant_db not found — run `python etl/embed_books.py` first"
        )
    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        from fastembed import SparseTextEmbedding
    except ImportError as e:
        pytest.skip(f"required package not installed: {e}")

    try:
        client = QdrantClient(path=str(qdrant_path))
    except RuntimeError as e:
        pytest.skip(f"Qdrant DB locked by another process — close other clients and retry: {e}")

    try:
        collections  = {c.name for c in client.get_collections().collections}
        if "coaching_books" not in collections:
            client.close()
            pytest.skip("coaching_books collection not found — run `python etl/embed_books.py` first")

        dense_model  = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        yield client, dense_model, sparse_model, "coaching_books"
    finally:
        client.close()


# ── temp_db ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def temp_db(tmp_path):
    """
    Path to a fresh temporary SQLite file for ETL smoke tests.

    tmp_path is pytest's built-in per-test isolated directory — unique,
    auto-cleaned after the session, and safe for parallel test runs.
    WAL sibling files (-shm, -wal) are also removed on teardown.
    """
    db = tmp_path / "test_running.db"
    yield db
    for suffix in ("-shm", "-wal"):
        sib = db.parent / (db.name + suffix)
        sib.unlink(missing_ok=True)
