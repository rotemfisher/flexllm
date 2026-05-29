from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    Prefetch, Fusion, SparseVector,
)
from langchain_core.tools import tool

_QDRANT_PATH  = Path(__file__).parent.parent.parent / "data" / "qdrant_db"
_COLLECTION   = "coaching_books"
_EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
_SPARSE_MODEL = "Qdrant/bm25"
_RERANK_MODEL = "BAAI/bge-reranker-large"

# Loaded once per process
_client       = None
_dense_model  = None
_sparse_model = None
_rerank_model = None


def _get_models():
    global _client, _dense_model, _sparse_model, _rerank_model
    if _client is None:
        _client       = QdrantClient(path=str(_QDRANT_PATH))
        _dense_model  = SentenceTransformer(_EMBED_MODEL, device="cpu")
        _sparse_model = SparseTextEmbedding(model_name=_SPARSE_MODEL)
        _rerank_model = CrossEncoder(_RERANK_MODEL, device="cpu")
    return _client, _dense_model, _sparse_model, _rerank_model


@tool
def search_coaching_books(query: str, book_filter: str | None = None, n_results: int = 5) -> str:
    """
    Hybrid semantic + keyword search across coaching books and sports science literature.
    Use for physiology concepts, training principles, nutrition, and evidence-based advice.

    book_filter values (exact string) — omit to search all books:
    Running:
    - "daniels"                   → Daniels' Running Formula (VDOT, training phases)
    - "lore"                      → Lore of Running (endurance physiology)
    - "science_run"               → Science of Running (biomechanics, race execution)
    Fitness:
    - "strength"                  → Science and Practice of Strength Training
    - "periodization"             → Periodization Theory and Methodology
    - "nsca"                      → Essentials of Strength Training and Conditioning
    Nutrition:
    - "sport_nutrition"           → Sport Nutrition (Jeukendrup & Gleeson)
    - "clinical_sports_nutrition" → Clinical Sports Nutrition (Burke & Deakin)
    Physiology:
    - "physiology_sport"          → Physiology of Sport and Exercise
    - "clinical_sports_medicine"  → Clinical Sports Medicine (Khan)

    For ISSN position-stand articles (caffeine, creatine, protein, beta-alanine,
    omega-3, ketogenic diets, etc.) do NOT use book_filter — query broadly and
    the relevant article will surface by semantic similarity.

    Examples:
    - query="threshold training benefits"                        → all books
    - query="VDOT interval paces", book_filter="daniels"         → Daniels only
    - query="carbohydrate loading race day", book_filter="sport_nutrition"
    """
    client, dense_model, sparse_model, rerank_model = _get_models()

    dense_vec  = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
    sparse_raw = list(sparse_model.embed([query]))[0]
    sparse_q   = SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )

    query_filter = None
    if book_filter:
        query_filter = Filter(
            must=[FieldCondition(key="book", match=MatchValue(value=book_filter))]
        )

    # Fetch a wider candidate pool so the reranker has more to work with
    rerank_pool = max(n_results * 6, 30)
    results = client.query_points(
        collection_name=_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=rerank_pool),
            Prefetch(query=sparse_q,  using="sparse", limit=rerank_pool),
        ],
        query=Fusion.RRF,
        limit=rerank_pool,
        query_filter=query_filter,
        with_payload=True,
    )

    points = results.points
    if points:
        texts  = [p.payload.get("text", "") for p in points]
        scores = rerank_model.predict([(query, t) for t in texts])
        points = [p for _, p in sorted(zip(scores, points), key=lambda x: x[0], reverse=True)]
        points = points[:n_results]
    if not points:
        return "No relevant passages found."

    parts = []
    for point in points:
        p          = point.payload or {}
        book_title = p.get("book_title", p.get("book", "Unknown"))
        section    = p.get("section", "")
        header     = f"[{book_title}" + (f" | {section}]" if section else "]")
        parts.append(f"{header}\n{p.get('text', '')}")

    return "\n\n---\n\n".join(parts)
