"""
tests/conftest.py  —  Shared pytest fixtures for the FlexLLM test suite.

Fixtures
────────
  prod_db         session-scoped read-only psycopg connection to the production PostgreSQL DB
  pg_temp_dsn     function-scoped PostgreSQL URL pointing at a fresh throw-away schema;
                  schema is populated with sql/schema.sql and dropped on teardown
  qdrant_col      session-scoped (client, collection_name) tuple for coaching_books
  apple_health_xml session-scoped path to the Apple Health XML export
"""

import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlencode, urlunparse

import psycopg
from psycopg.rows import dict_row
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import config
from src.tracing import setup_tracing
setup_tracing(project="flexllm-test")


def _make_schema_url(base_url: str, schema: str) -> str:
    """Return a PostgreSQL URL with search_path set to *schema*."""
    p = urlparse(base_url)
    query = urlencode({"options": f"-csearch_path={schema}"})
    return urlunparse(p._replace(query=query))


_BASE_DSN = config.DATABASE_URL


# ── prod_db ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def prod_db():
    """Read-only psycopg connection to the production PostgreSQL DB."""
    try:
        con = psycopg.connect(_BASE_DSN, row_factory=dict_row, autocommit=True)
    except Exception as e:
        pytest.skip(f"PostgreSQL unavailable ({e}) — run docker compose up first")
    yield con
    con.close()


# ── pg_temp_dsn ───────────────────────────────────────────────────────────────

@pytest.fixture()
def pg_temp_dsn():
    """
    PostgreSQL URL pointing at a fresh schema, populated with sql/schema.sql.
    The schema is dropped on teardown so each test starts with a clean slate.
    """
    schema = f"test_{uuid.uuid4().hex[:8]}"

    with psycopg.connect(_BASE_DSN, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")

    dsn = _make_schema_url(_BASE_DSN, schema)
    schema_sql = (ROOT / "sql" / "schema.sql").read_text()

    with psycopg.connect(dsn) as con:
        for stmt in schema_sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                con.execute(stmt)

    yield dsn

    with psycopg.connect(_BASE_DSN, autocommit=True) as admin:
        admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


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
        import src.tools.rag_tool as _rt
        if _rt._client is not None:
            try:
                _rt._client.close()
            except Exception:
                pass
            _rt._client = None
    except Exception:
        pass

    try:
        client = QdrantClient(path=str(qdrant_path))
    except RuntimeError as e:
        pytest.skip(
            f"Qdrant DB locked by an external process (is the app running?). "
            f"Stop it and retry.\nDetail: {e}"
        )

    try:
        collections = {c.name for c in client.get_collections().collections}
        if "coaching_books" not in collections:
            client.close()
            pytest.skip("coaching_books collection not found — run `python etl/embed_books.py` first")

        dense_model  = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu",
                                           local_files_only=True)
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        yield client, dense_model, sparse_model, "coaching_books"
    finally:
        client.close()


# ── apple_health_xml ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def apple_health_xml():
    """Path to the Apple Health XML export; skips if absent."""
    from etl.ingest_health import XML_FILE
    if not XML_FILE.exists():
        pytest.skip(
            f"Apple Health XML not found ({XML_FILE.name}) "
            "— export from the Health app first"
        )
    return XML_FILE
