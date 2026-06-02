#!/usr/bin/env python3
"""
evals/rag_eval.py — Ragas evaluation of the knowledge-base RAG pipeline.

Evaluates whether the Qdrant hybrid search returns relevant, faithful context
for typical coaching questions using three Ragas metrics:

  • context_precision  — are the top-K chunks actually relevant to the query?
  • faithfulness       — does the generated answer stay within the retrieved chunks?
  • answer_relevancy   — is the answer on-topic with the user's question?

Prerequisites:
  1. Qdrant DB must be built:  python etl/embed_books.py
  2. Ollama must be running with the model set in .env (or MODEL_ID env var)
  3. ragas >= 0.2.0 installed

Usage:
    python evals/rag_eval.py
    python evals/rag_eval.py --model qwen2.5:32b --ollama-url http://localhost:11434
    python evals/rag_eval.py --metrics context_precision faithfulness
    python evals/rag_eval.py --no-generate   # skip LLM generation, eval context quality only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Make src/ importable from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Test dataset ──────────────────────────────────────────────────────────────
# Each entry: (user_query, ground_truth_answer, domain_hint)
# Ground truth is used by context_recall (optional) and as a reference for
# faithfulness calibration.  It should be a factually correct, concise answer.

RAG_TEST_CASES: list[dict] = [
    {
        "query": "What is the recommended carbohydrate intake per hour during a marathon?",
        "reference": (
            "Research supports consuming 30–60g of carbohydrate per hour during "
            "endurance events lasting more than 90 minutes. Multiple transportable "
            "carbohydrates (glucose + fructose) allow up to 90g/hr in trained athletes."
        ),
        "domain": "dietitian",
    },
    {
        "query": "How does training load periodisation work using ATL and CTL?",
        "reference": (
            "Chronic Training Load (CTL) represents long-term fitness (~42-day EMA of TSS). "
            "Acute Training Load (ATL) is short-term fatigue (~7-day EMA). "
            "Training Stress Balance (TSB = CTL - ATL) measures readiness. "
            "A positive TSB near race day (tapering) indicates peak form."
        ),
        "domain": "recovery_coach",
    },
    {
        "query": "What is VDOT and how is it used to set training paces?",
        "reference": (
            "VDOT is Jack Daniels' estimate of effective VO2max derived from race performance. "
            "It is used to assign five training intensities: Easy (E), Marathon (M), "
            "Threshold (T), Interval (I), and Repetition (R), each with specific pace ranges."
        ),
        "domain": "trainer",
    },
    {
        "query": "What are the signs and symptoms of plantar fasciitis and how is it treated?",
        "reference": (
            "Plantar fasciitis presents as heel pain (especially in the morning or after rest), "
            "tenderness at the medial calcaneal tubercle, and pain that eases with activity "
            "but worsens after. First-line treatment: stretching, load management, supportive "
            "footwear, and gradual return to running."
        ),
        "domain": "physiotherapist",
    },
    {
        "query": "How can visualisation techniques improve running performance?",
        "reference": (
            "Mental imagery (visualisation) activates the same neural pathways as physical "
            "practice. Runners use it for race execution rehearsal, managing anxiety, and "
            "reinforcing positive self-talk. Best practice: vivid, multi-sensory imagery "
            "in a relaxed state, 5–10 min daily."
        ),
        "domain": "psychologist",
    },
    {
        "query": "What are the physiological adaptations to easy aerobic running?",
        "reference": (
            "Easy aerobic running stimulates: increased mitochondrial density, greater "
            "capillary density in muscle fibres, improved fat oxidation, increased "
            "stroke volume, reduced resting heart rate, and enhanced slow-twitch "
            "muscle fibre recruitment efficiency."
        ),
        "domain": "trainer",
    },
    {
        "query": "How much protein does a runner need per kg of body weight?",
        "reference": (
            "Endurance athletes typically require 1.4–1.7g protein per kg body weight daily, "
            "rising to 1.6–2.2g/kg during high-volume or high-intensity training blocks. "
            "Protein should be distributed across 3–4 meals, with 20–40g per serving."
        ),
        "domain": "dietitian",
    },
    {
        "query": "What is overtraining syndrome and how is it diagnosed?",
        "reference": (
            "Overtraining syndrome (OTS) is a maladaptive response to excessive training "
            "without adequate recovery. Markers include persistent performance decline, "
            "elevated resting HR, suppressed HRV, mood disturbances, and immune "
            "suppression. Diagnosis requires ruling out other causes; treatment is rest."
        ),
        "domain": "recovery_coach",
    },
]


# ── Retrieval helper ──────────────────────────────────────────────────────────

def _retrieve_contexts(query: str, n: int = 5) -> list[str]:
    """Call the existing RAG tool and split the result into individual chunks."""
    from src.tools.rag_tool import search_coaching_books
    raw: str = search_coaching_books.invoke({"query": query, "n_results": n})
    if raw.startswith("No relevant") or raw.startswith("Knowledge base"):
        return []
    return [chunk.strip() for chunk in raw.split("---") if chunk.strip()]


# ── Generation helper ─────────────────────────────────────────────────────────

def _generate_answer(query: str, contexts: list[str], llm) -> str:
    """Generate a grounded answer from retrieved contexts using the LLM."""
    context_block = "\n\n".join(contexts[:3])
    prompt = (
        f"Using only the following excerpts from coaching literature, answer the question "
        f"concisely and factually (2–4 sentences).\n\n"
        f"Excerpts:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ── Ragas evaluation ──────────────────────────────────────────────────────────

def _build_ragas_dataset(
    cases: list[dict],
    generate: bool,
    llm,
) -> "EvaluationDataset":
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for case in cases:
        print(f"  Retrieving: {case['query'][:60]} …")
        contexts = _retrieve_contexts(case["query"])
        if not contexts:
            print(f"  WARNING: no contexts returned for: {case['query'][:60]}")
            contexts = ["No relevant passages found."]

        response = ""
        if generate and llm:
            print(f"  Generating answer …")
            response = _generate_answer(case["query"], contexts, llm)

        samples.append(
            SingleTurnSample(
                user_input=case["query"],
                retrieved_contexts=contexts,
                response=response,
                reference=case["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def _run_evaluation(
    dataset: "EvaluationDataset",
    metric_names: list[str],
    llm_wrapper,
    embeddings_wrapper,
) -> None:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    metric_map = {
        "context_precision": context_precision,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    metrics = [metric_map[m] for m in metric_names if m in metric_map]

    print(f"\nRunning Ragas evaluation ({', '.join(metric_names)}) …")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
        raise_exceptions=False,
    )
    print("\n" + "═" * 60)
    print("  Ragas Results")
    print("═" * 60)
    for metric_name, score in result.items():
        print(f"  {metric_name:<25} {score:.4f}")
    print("═" * 60)

    df = result.to_pandas()
    out_path = Path(__file__).parent / "rag_eval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Ragas RAG quality evaluation for FlexLLM")
    p.add_argument("--model", default=None,
                   help="Ollama model ID (default: MODEL_ID from .env)")
    p.add_argument("--ollama-url", default=None,
                   help="Ollama base URL (default: OLLAMA_BASE_URL from .env)")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["context_precision", "faithfulness", "answer_relevancy"],
        choices=["context_precision", "faithfulness", "answer_relevancy"],
        help="Ragas metrics to evaluate",
    )
    p.add_argument("--no-generate", action="store_true",
                   help="Skip LLM answer generation; only evaluate context quality "
                        "(disables faithfulness and answer_relevancy)")
    p.add_argument("--cases", type=int, default=len(RAG_TEST_CASES),
                   help=f"Number of test cases to run (default: all {len(RAG_TEST_CASES)})")
    args = p.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    from src.config import config
    model_id = args.model or config.MODEL_ID
    ollama_url = args.ollama_url or config.OLLAMA_BASE_URL

    # ── Verify Ragas is installed ─────────────────────────────────────────────
    try:
        import ragas  # noqa: F401
    except ImportError:
        print("ERROR: ragas is not installed. Run: pip install ragas>=0.2.0")
        sys.exit(1)

    # ── Build LLM + embeddings wrappers for Ragas ─────────────────────────────
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    chat_llm = ChatOllama(model=model_id, base_url=ollama_url)
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
    llm_wrapper = LangchainLLMWrapper(chat_llm)
    embeddings_wrapper = LangchainEmbeddingsWrapper(ollama_embeddings)

    generate = not args.no_generate
    metrics = args.metrics
    if args.no_generate:
        metrics = [m for m in metrics if m == "context_precision"]
        if not metrics:
            metrics = ["context_precision"]
        print("--no-generate: limiting to context_precision only.")

    cases = RAG_TEST_CASES[: args.cases]
    print(f"Building dataset from {len(cases)} test cases …\n")

    llm_for_generation = chat_llm if generate else None
    dataset = _build_ragas_dataset(cases, generate, llm_for_generation)

    _run_evaluation(dataset, metrics, llm_wrapper, embeddings_wrapper)


if __name__ == "__main__":
    main()
