"""
Semantic entry router for the multi-agent coaching graph.

Uses BAAI/bge-small-en-v1.5 (67MB ONNX, via fastembed) to embed the user
message and compute cosine similarity against per-domain anchor sentences.
The closest domain wins. Model and anchor embeddings are computed once and
cached for the lifetime of the process.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.models.agent_state import CoachState

# ── Domain anchors ────────────────────────────────────────────────────────────
# Representative sentences that describe each domain's scope.
# More diverse anchors → better coverage of real user phrasing.

_DOMAIN_ANCHORS: dict[str, list[str]] = {
    "physiotherapist": [
        "I have pain in my knee",
        "I injured myself during a run",
        "My shoulder is sore and inflamed",
        "I twisted my ankle and it hurts",
        "There is a sharp ache in my tendon",
        "I am dealing with an overuse injury",
        "My calf feels tight and very painful",
        "I think I have a stress fracture",
        "There is swelling around my joint",
        "I strained my hamstring",
        "My shin is really bothering me",
        "I have plantar fasciitis",
        "My achilles is giving me trouble",
        "I feel a sharp stabbing in my foot",
        "My back hurts when I run",
    ],
    "recovery_coach": [
        "I am completely exhausted and cannot recover",
        "My HRV has been very low this week",
        "I have not been sleeping well lately",
        "I feel overtrained and burned out",
        "My body feels heavy and fatigued",
        "My training load is too high right now",
        "I need to check my readiness before training",
        "I have no energy and feel totally run down",
        "My resting heart rate is elevated",
        "I feel like I need a rest day",
        "My sleep quality has been terrible",
        "I feel like I am not recovering between sessions",
        "My form score is terrible today",
        "I feel drained and sluggish",
    ],
    "dietitian": [
        "What should I eat before a long run",
        "I need help planning my nutrition",
        "How many calories should I eat",
        "What foods help with recovery after training",
        "I want to lose weight while maintaining performance",
        "What macros do I need for muscle building",
        "How do I fuel during a race",
        "I need a meal plan for the week",
        "What is the best pre-workout nutrition",
        "How much protein should I be eating",
        "I want to improve my diet for running",
        "What should I eat on rest days versus training days",
        "I am always hungry after my long runs",
        "Can you help me with carbohydrate loading",
        "What supplements should I take",
        "I need help fuelling for my event",
        "How do I fuel properly for a race or competition",
        "I am struggling to manage or maintain my body weight",
        "I need a nutrition strategy for my upcoming race",
        "What should I eat to help my body recover",
        "I want to change my eating habits to support training",
    ],
    "trainer": [
        "Build me a training plan for a race",
        "What pace should I be running at",
        "I want to improve my running performance",
        "Create a weekly workout schedule for me",
        "What strength exercises should I include in training",
        "How should I progress my workouts over time",
        "I completed a workout today and want to log it",
        "Plan my training for the next month",
        "What is my VDOT score",
        "How do I structure my easy and hard runs",
        "I want to do a strength training program",
        "Can you build me a base training phase",
        "What should my tempo run pace be",
        "I am preparing for a marathon",
        "How do I periodise my training",
    ],
}

_ROUTER_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _load() -> tuple:
    """Load model and pre-compute mean anchor embeddings. Called once, cached."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=_ROUTER_MODEL)
    domain_vecs: dict[str, np.ndarray] = {}
    for domain, sentences in _DOMAIN_ANCHORS.items():
        embs = np.array(list(model.embed(sentences)))  # (N, D)
        domain_vecs[domain] = embs.mean(axis=0)        # (D,)
    return model, domain_vecs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


def route_entry(state: CoachState) -> str:
    """
    Return the name of the agent node that should handle this turn.

    Mid-session (active_agent already set): stay with the current agent so
    handoffs are honoured across turns.
    New session: embed the user message and pick the closest domain.
    """
    if state.get("active_agent"):
        return state["active_agent"]

    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    if last_human is None:
        return "trainer"

    model, domain_vecs = _load()
    msg_vec = np.array(next(iter(model.embed([last_human.content]))))

    scores = {domain: _cosine(msg_vec, vec) for domain, vec in domain_vecs.items()}
    return max(scores, key=scores.get)
