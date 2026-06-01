"""
tests/test_qdrant.py  —  Quality tests for the coaching_books Qdrant collection.

All tests use the `qdrant_col` session fixture from conftest.py which yields
(client, dense_model, sparse_model, collection_name).

Run:
    pytest tests/test_qdrant.py -v
"""

import uuid
import pytest
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    Prefetch, Fusion, FusionQuery, SparseVector,
)

# ─── Constants ────────────────────────────────────────────────────────────────

COLLECTION = "coaching_books"

EMBEDDED_BOOKS = {
    "daniels", "lore", "science_run",
    "strength", "periodization", "nsca",
    "sport_nutrition", "clinical_sports_nutrition",
}
PENDING_BOOKS = {"physiology_sport", "clinical_sports_medicine"}

EXPECTED_CATEGORIES   = {"running", "fitness", "nutrition", "physiology"}
EMBEDDING_DIM         = 1024
REQUIRED_PAYLOAD_KEYS = {"book", "book_title", "category", "chunk_type", "section",
                         "chunk_index", "text"}

MIN_CHUNKS_PER_BOOK = {
    "daniels":                   400,
    "lore":                     1400,
    "science_run":               650,
    "strength":                  600,
    "periodization":             700,
    "nsca":                      400,
    "sport_nutrition":           400,
    "clinical_sports_nutrition": 700,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _scroll_all(client, with_payload=True):
    """Return all points in the collection via paginated scroll."""
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            COLLECTION, limit=1000, offset=offset,
            with_payload=with_payload, with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _books_present(client) -> set[str]:
    points = _scroll_all(client, with_payload=True)
    return {p.payload["book"] for p in points if p.payload}


def _skip_if_book_missing(client, book_key):
    if book_key not in _books_present(client):
        pytest.skip(f"'{book_key}' not yet embedded — run `python etl/embed_books.py`")


def _hybrid_query(client, dense_model, sparse_model, query_text, n_results=3,
                  book_filter=None):
    dense_vec  = dense_model.encode([query_text], normalize_embeddings=True)[0].tolist()
    sparse_raw = list(sparse_model.embed([query_text]))[0]
    sparse_q   = SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )
    qfilter = None
    if book_filter:
        qfilter = Filter(must=[FieldCondition(key="book", match=MatchValue(value=book_filter))])

    return client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=n_results * 4),
            Prefetch(query=sparse_q,  using="sparse", limit=n_results * 4),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=n_results,
        query_filter=qfilter,
        with_payload=True,
    ).points


# ─── Collection-level sanity ──────────────────────────────────────────────────

def test_collection_not_empty(qdrant_col):
    client, *_ = qdrant_col
    n = client.count(COLLECTION).count
    assert n >= 6000, (
        f"coaching_books has only {n} chunks — expected ≥6 000. "
        "Re-run `python etl/embed_books.py` if books are missing."
    )


def test_no_duplicate_ids(qdrant_col):
    client, *_ = qdrant_col
    points = _scroll_all(client, with_payload=False)
    ids = [p.id for p in points]
    assert len(ids) == len(set(ids)), (
        f"Found {len(ids) - len(set(ids))} duplicate point IDs"
    )


# ─── Book coverage ────────────────────────────────────────────────────────────

def test_all_embedded_books_present(qdrant_col):
    client, *_ = qdrant_col
    found   = _books_present(client)
    missing = EMBEDDED_BOOKS - found
    assert not missing, (
        f"Books missing from Qdrant: {missing}. Re-run `python etl/embed_books.py`."
    )


def test_pending_books_status(qdrant_col):
    client, *_ = qdrant_col
    found    = _books_present(client)
    embedded = PENDING_BOOKS & found
    missing  = PENDING_BOOKS - found
    if missing:
        pytest.skip(
            f"Pending books not yet embedded: {missing}. "
            f"Already embedded: {embedded or 'none'}."
        )


def test_all_categories_represented(qdrant_col):
    client, *_ = qdrant_col
    points   = _scroll_all(client, with_payload=True)
    found    = {p.payload["category"] for p in points if p.payload}
    missing  = EXPECTED_CATEGORIES - found
    assert not missing, (
        f"Missing categories: {missing}. "
        "Run `python etl/embed_books.py` to embed physiology books."
    )


def test_each_embedded_book_has_text_chunks(qdrant_col):
    client, *_ = qdrant_col
    points     = _scroll_all(client, with_payload=True)
    text_books = {p.payload["book"] for p in points
                  if p.payload and p.payload.get("chunk_type") == "text"}
    missing    = EMBEDDED_BOOKS - text_books
    assert not missing, f"Books with no text chunks: {missing}"


def test_fitness_books_have_table_chunks(qdrant_col):
    client, *_ = qdrant_col
    fitness_books = {"nsca", "periodization", "strength"}
    points        = _scroll_all(client, with_payload=True)
    table_books   = {p.payload["book"] for p in points
                     if p.payload and p.payload.get("chunk_type") == "table"}
    missing       = fitness_books - table_books
    assert not missing, f"Fitness books missing table chunks: {missing}"


# ─── Per-book minimum chunk count ────────────────────────────────────────────

@pytest.mark.parametrize("book,min_n", MIN_CHUNKS_PER_BOOK.items())
def test_book_meets_minimum_chunk_count(qdrant_col, book, min_n):
    client, *_ = qdrant_col
    _skip_if_book_missing(client, book)
    n = client.count(
        COLLECTION,
        count_filter=Filter(must=[FieldCondition(key="book", match=MatchValue(value=book))]),
    ).count
    assert n >= min_n, (
        f"'{book}' has {n} chunks (expected ≥{min_n}). "
        + ("Run: python etl/embed_books.py --reembed science_run"
           if book == "science_run" else
           "Re-run `python etl/embed_books.py`.")
    )

# ─── Embedding dimension ─────────────────────────────────────────────────────

def test_embedding_dimension_is_1024(qdrant_col):
    client, *_ = qdrant_col
    points, _  = client.scroll(COLLECTION, limit=1, with_vectors=["dense"])
    assert points, "scroll() returned no points"
    dim = len(points[0].vector["dense"])
    assert dim == EMBEDDING_DIM, (
        f"Dense embedding dimension is {dim}, expected {EMBEDDING_DIM}."
    )


def test_sparse_vector_present(qdrant_col):
    client, *_ = qdrant_col
    points, _  = client.scroll(COLLECTION, limit=1, with_vectors=["sparse"])
    assert points, "scroll() returned no points"
    sv = points[0].vector.get("sparse")
    assert sv is not None, "No sparse vector stored for first point"
    assert len(sv.indices) > 0, "Sparse vector has no non-zero terms"


# ─── Stable ID format ────────────────────────────────────────────────────────

def test_stable_id_format(qdrant_col):
    """
    The first two chunks of every embedded book must be retrievable by their
    deterministic UUID (derived from '{book}_c00000' / '{book}_c00001').
    """
    client, *_ = qdrant_col
    for book in EMBEDDED_BOOKS:
        _skip_if_book_missing(client, book)
        expected_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_OID, f"{book}_c00000")),
            str(uuid.uuid5(uuid.NAMESPACE_OID, f"{book}_c00001")),
        ]
        found = client.retrieve(COLLECTION, ids=expected_ids, with_payload=False)
        assert len(found) == 2, (
            f"Could not retrieve both anchor chunks for '{book}'. "
            "stable_qdrant_id() or UUID namespace may have changed."
        )


# ─── Payload completeness ────────────────────────────────────────────────────

def test_payload_fields_complete(qdrant_col):
    """A sample of 50 chunks must carry all required payload keys including 'text'."""
    client, *_ = qdrant_col
    points, _  = client.scroll(COLLECTION, limit=50, with_payload=True)
    for i, point in enumerate(points):
        payload = point.payload or {}
        missing = REQUIRED_PAYLOAD_KEYS - set(payload.keys())
        assert not missing, (
            f"Point {i} (id={point.id}) missing payload keys: {missing}"
        )


# ─── Hybrid search quality ────────────────────────────────────────────────────

def test_hybrid_vdot_query(qdrant_col):
    """'VDOT easy pace training zones' must return daniels as the top result."""
    client, dense, sparse, _ = qdrant_col
    points = _hybrid_query(client, dense, sparse, "VDOT easy pace training zones")
    assert points, "Hybrid query returned no results"
    assert points[0].payload["book"] == "daniels", (
        f"Expected 'daniels' for VDOT query, got '{points[0].payload['book']}'"
    )


def test_hybrid_periodization_query(qdrant_col):
    """'periodization mesocycle macrocycle training plan' must surface the periodization book in top 3."""
    client, dense, sparse, _ = qdrant_col
    points = _hybrid_query(client, dense, sparse,
                           "periodization mesocycle macrocycle training plan", n_results=3)
    books = [p.payload.get("book") for p in points if p.payload]
    assert "periodization" in books, (
        f"Expected 'periodization' in top 3, got {books}"
    )


def test_hybrid_vo2max_query(qdrant_col):
    """'VO2max running economy oxygen consumption' must return a running book."""
    client, dense, sparse, _ = qdrant_col
    points = _hybrid_query(client, dense, sparse,
                           "VO2max running economy oxygen consumption")
    assert points and points[0].payload["book"] in {"lore", "science_run", "daniels"}, (
        f"Expected a running book for VO2max query, got '{points[0].payload.get('book')}'"
    )


def test_hybrid_nutrition_carbohydrate_query(qdrant_col):
    """
    'carbohydrate loading glycogen marathon fueling' must surface at least one
    nutrition chunk in the top 5.
    """
    client, dense, sparse, _ = qdrant_col
    points = _hybrid_query(client, dense, sparse,
                           "carbohydrate loading glycogen marathon fueling", n_results=5)
    cats = [p.payload.get("category") for p in points if p.payload]
    assert "nutrition" in cats, (
        f"No nutrition result in top 5 for carbohydrate query. "
        f"Got: {[p.payload.get('book') for p in points]}"
    )


def test_hybrid_protein_recovery_query(qdrant_col):
    """'protein intake muscle recovery endurance athletes' must return nutrition, fitness, or physiology."""
    client, dense, sparse, _ = qdrant_col
    points = _hybrid_query(client, dense, sparse,
                           "protein intake muscle recovery endurance athletes")
    assert points and points[0].payload.get("category") in {"nutrition", "fitness", "physiology"}, (
        f"Expected nutrition, fitness, or physiology, got '{points[0].payload.get('category')}'"
    )


def test_hybrid_strength_training_query(qdrant_col):
    """'one rep max progressive overload hypertrophy' must return a fitness or physiology book."""
    client, dense, sparse, _ = qdrant_col
    points = _hybrid_query(client, dense, sparse,
                           "one rep max progressive overload hypertrophy resistance training")
    assert points and points[0].payload.get("category") in {"fitness", "physiology"}, (
        f"Expected fitness or physiology for strength query, got '{points[0].payload.get('category')}'"
    )


def test_hybrid_physiology_query(qdrant_col):
    """'cardiac output stroke volume heart rate exercise' must return a physiology book."""
    client, *_ = qdrant_col
    found = _books_present(client)
    if not ({"physiology_sport", "clinical_sports_medicine"} & found):
        pytest.skip("Physiology books not yet embedded")
    client, dense, sparse, _ = qdrant_col
    points = _hybrid_query(client, dense, sparse,
                           "cardiac output stroke volume heart rate during exercise")
    assert points and points[0].payload.get("category") == "physiology", (
        f"Expected physiology for cardiac query, got '{points[0].payload.get('book')}'"
    )

# ─── Science of Running Specific Tests ────────────────────────────────────────

@pytest.mark.parametrize("query_text, expected_keyword", [
    ("central governor theory brain regulation fatigue", "governor"),
    ("muscle fiber recruitment specific endurance", "fiber"),
    ("vVO2max interval training protocols", "interval"),
    ("aerobic capacity neuromuscular fatigue running", "fatigue")
])
def test_science_run_internal_retrieval(qdrant_col, query_text, expected_keyword):
    """
    Test the embeddings of 'Science of Running' in isolation.
    By applying a book_filter, we prevent massive textbooks from out-scoring it.
    This proves the chunks themselves are semantically meaningful.
    """
    client, dense, sparse, _ = qdrant_col
    
    points = _hybrid_query(
        client, dense, sparse, query_text, 
        n_results=3, 
        book_filter="science_run"
    )
    
    assert len(points) > 0, f"No chunks returned for {query_text} inside science_run!"
    
    retrieved_texts = [p.payload.get("text", "").lower() for p in points if p.payload]
    keyword_found = any(expected_keyword in text for text in retrieved_texts)
    
    assert keyword_found, (
        f"The embeddings for 'science_run' might be diluted. \n"
        f"Queried: '{query_text}'.\n"
        f"Expected to find a chunk containing '{expected_keyword}', but got:\n"
        f"{[t[:100] + '...' for t in retrieved_texts]}"
    )