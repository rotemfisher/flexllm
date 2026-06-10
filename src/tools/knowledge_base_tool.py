"""
General-purpose Qdrant knowledge base search tool.

This is the primary source of truth for all domain knowledge in the system.
The Qdrant collection holds coaching books, sports science literature, and
position-stand articles covering running, strength, nutrition, physiology,
and psychology. All agents should prefer this tool for evidence-based advice.
"""

import logging

from langchain_core.tools import tool
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from src.config import config
from src.tools.rag_tool import _get_models

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {"running", "strength", "nutrition", "physiology", "psychology"}


@tool
def search_knowledge_base(query: str, category: str | None = None, n_results: int = 5) -> str:
    """
    Search the Qdrant knowledge base — the single source of truth for all domain knowledge.

    Use this tool whenever you need evidence-based information on any coaching,
    training, nutrition, physiotherapy, recovery, or psychology topic. It performs
    hybrid semantic + keyword search across the full coaching library and
    sports science literature, then reranks results for relevance.

    category — optional filter to scope the search (omit to search everything):
      "running"     → running training: periodization, VDOT, VO2max, tempo, intervals,
                       race execution, biomechanics (Daniels, Lore of Running, Science of Running)
      "strength"    → strength & conditioning: periodization, hypertrophy, powerlifting,
                       NSCA guidelines (Strength Training, Periodization, NSCA Essentials)
      "nutrition"   → sports nutrition: macros, fueling, hydration, supplements, race-day
                       nutrition, ISSN position stands (Sport Nutrition, Clinical Sports Nutrition,
                       and ISSN caffeine / creatine / protein / beta-alanine / omega-3 articles)
      "physiology"  → exercise physiology and sports medicine: energy systems, injury anatomy,
                       return-to-sport protocols (Physiology of Sport & Exercise,
                       Clinical Sports Medicine)
      "psychology"  → sport psychology: mental toughness, anxiety, confidence, imagery,
                       motivation, mindset (Champion's Mind, Applied Sport Psych,
                       Foundations of Sport Psychology)

    Examples:
      search_knowledge_base("threshold training adaptations")
      search_knowledge_base("caffeine timing and dosage", category="nutrition")
      search_knowledge_base("ACL rehabilitation protocol", category="physiology")
      search_knowledge_base("pre-race anxiety management", category="psychology")
      search_knowledge_base("progressive overload principles", category="strength")
    """
    try:
        client, dense_model, sparse_model, rerank_model = _get_models()

        dense_vec  = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
        sparse_raw = list(sparse_model.embed([query]))[0]
        sparse_q   = SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist(),
        )

        query_filter = None
        if category:
            category = category.strip().lower()
            if category not in _VALID_CATEGORIES:
                return (
                    f"Unknown category '{category}'. "
                    f"Valid values: {', '.join(sorted(_VALID_CATEGORIES))}. "
                    "Omit the category argument to search all knowledge."
                )
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        rerank_pool = max(n_results * 6, 30)
        results = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=rerank_pool),
                Prefetch(query=sparse_q,  using="sparse", limit=rerank_pool),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=rerank_pool,
            query_filter=query_filter,
            with_payload=True,
        )

        points = results.points
        if not points:
            return (
                "No relevant knowledge found for this query"
                + (f" in category '{category}'" if category else "")
                + ". Try broadening the query or omitting the category filter."
            )

        texts  = [p.payload.get("text", "") for p in points]
        scores = rerank_model.predict([(query, t) for t in texts])
        ranked = [p for _, p in sorted(zip(scores, points), key=lambda x: x[0], reverse=True)]
        ranked = ranked[:n_results]

        parts = []
        for point in ranked:
            p          = point.payload or {}
            book_title = p.get("book_title", p.get("book", "Unknown source"))
            section    = p.get("section", "")
            cat_label  = p.get("category", "")
            header     = f"[{book_title}" + (f" | {section}" if section else "") + (f" | {cat_label}" if cat_label else "") + "]"
            parts.append(f"{header}\n{p.get('text', '')}")

        return "\n\n---\n\n".join(parts)

    except Exception as exc:
        logger.exception("search_knowledge_base error: %s", exc)
        if "doesn't exist" in str(exc) or "Not found" in str(exc):
            return (
                "Knowledge base is not yet available (collection not initialised). "
                "Proceed without citing the knowledge base for now."
            )
        return f"Knowledge base search failed: {exc}"
