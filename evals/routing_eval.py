#!/usr/bin/env python3
"""
evals/routing_eval.py — Real-model routing accuracy evaluation.

Unlike tests/test_eval.py (which mocks embeddings to verify graph wiring),
this script runs the live BAAI/bge-small-en-v1.5 model against a curated
dataset of 40 labeled queries and reports:

  • Overall accuracy
  • Per-agent precision / recall / F1
  • Confusion matrix
  • Lowest-confidence predictions (easy to improve anchors from these)

No LLM or database required — only fastembed and numpy.

Usage:
    python evals/routing_eval.py
    python evals/routing_eval.py --verbose    # show every prediction
    python evals/routing_eval.py --threshold 0.85  # flag uncertain predictions
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make sure src/ is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.router import _DOMAIN_ANCHORS, _ROUTER_MODEL, _cosine

# ── Dataset ───────────────────────────────────────────────────────────────────
# Each tuple is (query, expected_agent).
# Queries are intentionally varied: some are clear-cut, others are ambiguous
# boundary cases that stress-test the anchor coverage.

ROUTING_CASES: list[tuple[str, str]] = [
    # ── physiotherapist ────────────────────────────────────────────────────────
    ("My right knee has been aching for three days after my long run", "physiotherapist"),
    ("I have a sharp stabbing pain in my foot when I step out of bed", "physiotherapist"),
    ("My achilles tendon is swollen and very tender to touch", "physiotherapist"),
    ("I twisted my ankle badly during a trail run and it is now bruised", "physiotherapist"),
    ("There is constant tightness in my calf that does not go away after warming up", "physiotherapist"),
    ("I think I have shin splints — the pain is along the inner edge of my tibia", "physiotherapist"),
    ("My IT band is giving me serious grief on the outside of my knee", "physiotherapist"),
    ("I pulled my hamstring during a sprint and now I cannot walk properly", "physiotherapist"),
    ("My lower back seizes up every time I increase my weekly mileage", "physiotherapist"),
    # ── recovery_coach ────────────────────────────────────────────────────────
    ("My HRV this morning was 12ms below my 7-day average — should I train?", "recovery_coach"),
    ("I have been sleeping under 6 hours for five nights and feel completely drained", "recovery_coach"),
    ("My resting heart rate has been elevated by 8 bpm since Monday", "recovery_coach"),
    ("I feel completely overtrained and have no energy for any workout", "recovery_coach"),
    ("My TSB is very negative — am I accumulating too much fatigue?", "recovery_coach"),
    ("I cannot recover between sessions — my legs feel heavy even on easy runs", "recovery_coach"),
    ("How should I adjust this week based on my poor sleep and low readiness score?", "recovery_coach"),
    ("My form score is terrible today — do I still do the interval session?", "recovery_coach"),
    # ── trainer ───────────────────────────────────────────────────────────────
    ("Build me a 16-week marathon training plan starting from base phase", "trainer"),
    ("What interval workout should I do this week to improve my 5K time?", "trainer"),
    ("I completed a 25-minute 5K — what is my VDOT and what pace should I train at?", "trainer"),
    ("How do I periodise strength training alongside marathon prep?", "trainer"),
    ("What should my easy run pace be based on my current fitness?", "trainer"),
    ("I want to add tempo runs to my schedule — how often and at what effort?", "trainer"),
    ("Design a base-building phase for the next 8 weeks", "trainer"),
    ("I logged a 32km long run today — can you add it to my training log?", "trainer"),
    ("What is the right progression for increasing weekly mileage safely?", "trainer"),
    # ── dietitian ─────────────────────────────────────────────────────────────
    ("What should I eat the night before a half marathon for optimal glycogen stores?", "dietitian"),
    ("How much protein do I need daily to support muscle recovery and adaptation?", "dietitian"),
    ("I keep bonking at 30km in my long runs — what is my fueling strategy missing?", "dietitian"),
    ("Can you calculate my calorie needs for the heavy training week ahead?", "dietitian"),
    ("What carbohydrate loading protocol should I follow three days before my race?", "dietitian"),
    ("I am trying to lose 3kg while maintaining my running performance — help me plan meals", "dietitian"),
    ("What should I eat and drink during a 3-hour long run?", "dietitian"),
    ("Are there any supplements that genuinely help endurance performance?", "dietitian"),
    # ── psychologist ──────────────────────────────────────────────────────────
    ("I keep giving up during hard intervals — I cannot push through the pain", "psychologist"),
    ("My confidence is shattered after a terrible race last weekend", "psychologist"),
    ("I get severe pre-race anxiety the night before competitions and cannot sleep", "psychologist"),
    ("How do I use visualisation techniques to improve my race performance?", "psychologist"),
    ("I have been feeling completely unmotivated to train for three weeks", "psychologist"),
    ("I freeze at the start line and run way too conservatively out of fear", "psychologist"),
    ("I have very negative self-talk during hard efforts — how do I fix that?", "psychologist"),
    ("My goal-setting feels vague and I am not sure how to structure it properly", "psychologist"),
]

AGENTS = sorted(set(expected for _, expected in ROUTING_CASES))


# ── Router (reuses the same logic as router.py but runs standalone) ───────────

def _load_router():
    """Load fastembed model and pre-compute mean anchor vectors."""
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=_ROUTER_MODEL)
    domain_vecs: dict[str, np.ndarray] = {}
    for domain, sentences in _DOMAIN_ANCHORS.items():
        embs = np.array(list(model.embed(sentences)))
        domain_vecs[domain] = embs.mean(axis=0)
    return model, domain_vecs


def _predict(query: str, model, domain_vecs: dict[str, np.ndarray]) -> tuple[str, dict[str, float]]:
    """Return (predicted_agent, {agent: cosine_score}) for a single query."""
    vec = np.array(next(iter(model.embed([query]))))
    scores = {domain: _cosine(vec, dv) for domain, dv in domain_vecs.items()}
    predicted = max(scores, key=scores.get)
    return predicted, scores


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class PerClassMetrics:
    agent: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _compute_metrics(
    predictions: list[tuple[str, str, dict[str, float]]]
) -> tuple[float, dict[str, PerClassMetrics]]:
    """Return (accuracy, {agent: PerClassMetrics})."""
    correct = sum(1 for pred, exp, _ in predictions if pred == exp)
    accuracy = correct / len(predictions)

    per_class: dict[str, PerClassMetrics] = {a: PerClassMetrics(a) for a in AGENTS}
    confusion: dict[tuple[str, str], int] = defaultdict(int)  # (expected, predicted)

    for pred, exp, _ in predictions:
        confusion[(exp, pred)] += 1
        if pred == exp:
            per_class[exp].tp += 1
        else:
            per_class[exp].fn += 1
            per_class[pred].fp += 1

    return accuracy, per_class, confusion


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_confusion_matrix(confusion: dict, agents: list[str]) -> None:
    col_w = max(len(a) for a in agents) + 2
    header = " " * col_w + "".join(a[:col_w].ljust(col_w) for a in agents)
    print(header)
    print("-" * len(header))
    for exp in agents:
        row = exp.ljust(col_w)
        for pred in agents:
            cell = str(confusion.get((exp, pred), 0)).ljust(col_w)
            row += cell
        print(row)


def _print_low_confidence(
    cases: list[tuple[str, str]],
    predictions: list[tuple[str, str, dict[str, float]]],
    threshold: float,
) -> None:
    flagged = [
        (q, exp, pred, scores)
        for (q, exp), (pred, exp2, scores) in zip(cases, predictions)
        if max(scores.values()) < threshold
    ]
    if not flagged:
        print(f"All predictions scored ≥ {threshold:.2f}.")
        return
    for q, exp, pred, scores in flagged:
        top = max(scores.values())
        print(f"  score={top:.3f}  expected={exp}  predicted={pred}")
        print(f"    \"{q}\"")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate semantic router accuracy")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print every prediction")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="Flag predictions whose top score is below this value")
    args = p.parse_args()

    print(f"Loading {_ROUTER_MODEL} …")
    model, domain_vecs = _load_router()

    print(f"Running {len(ROUTING_CASES)} routing cases …\n")
    predictions: list[tuple[str, str, dict[str, float]]] = []
    for query, expected in ROUTING_CASES:
        pred, scores = _predict(query, model, domain_vecs)
        predictions.append((pred, expected, scores))
        if args.verbose:
            mark = "✓" if pred == expected else "✗"
            top_score = max(scores.values())
            print(f"  {mark} [{top_score:.3f}] expected={expected:<20} got={pred:<20}")
            print(f"    \"{query}\"")

    accuracy, per_class, confusion = _compute_metrics(predictions)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Overall accuracy: {accuracy:.1%}  ({sum(1 for p,e,_ in predictions if p==e)}/{len(predictions)})")
    print("═" * 60)

    print("\nPer-agent metrics:")
    print(f"  {'Agent':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 52)
    for agent in AGENTS:
        m = per_class[agent]
        print(f"  {agent:<22} {m.precision:>10.1%} {m.recall:>10.1%} {m.f1:>10.1%}")

    macro_f1 = sum(per_class[a].f1 for a in AGENTS) / len(AGENTS)
    print(f"\n  Macro F1: {macro_f1:.1%}")

    print("\nConfusion matrix (rows=expected, cols=predicted):")
    _print_confusion_matrix(confusion, AGENTS)

    print(f"\nLow-confidence predictions (top score < {args.threshold:.2f}):")
    _print_low_confidence(ROUTING_CASES, predictions, args.threshold)

    # Non-zero exit if accuracy below 90 % so CI can catch regressions.
    if accuracy < 0.90:
        print(f"\nFAIL: accuracy {accuracy:.1%} < 90% threshold")
        sys.exit(1)
    else:
        print(f"\nPASS: accuracy {accuracy:.1%} ≥ 90%")


if __name__ == "__main__":
    main()
