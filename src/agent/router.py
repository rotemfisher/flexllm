"""
Multi-intent supervisor router for the multi-agent coaching graph.

Primary path: an LLM call that returns JSON identifying which specialist(s)
should handle the current message.  Supports multi-intent queries by returning
a primary agent plus an ordered list of secondary agents that will be activated
automatically after the primary finishes.

Fallback path: BAAI/bge-small-en-v1.5 cosine-similarity (original single-agent
router) — used when the LLM call fails or returns unparseable output.

make_supervisor_node(llm) returns a LangGraph node function suitable for
graph.add_node("supervisor", make_supervisor_node(llm)).
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

import numpy as np

from src.models.agent_state import CoachState

logger = logging.getLogger(__name__)

# ── Valid agent names ─────────────────────────────────────────────────────────

_VALID_AGENTS = frozenset(
    {"trainer", "physiotherapist", "recovery_coach", "dietitian", "psychologist"}
)

# ── LLM supervisor ────────────────────────────────────────────────────────────

_SUPERVISOR_PROMPT = """\
You are a routing supervisor for a sports coaching AI with five specialists.

Specialists:
- "trainer": training plans, workouts, running performance, pace, VDOT, periodization, strength programs
- "physiotherapist": pain, injury, soreness, swelling, strain, sprain, fracture, rehabilitation
- "recovery_coach": fatigue, HRV, sleep, readiness, overtraining, rest days
- "dietitian": nutrition, food, calories, macros, meal plan, weight management, fueling, supplements
- "psychologist": motivation, anxiety, mental blocks, confidence, performance mindset, race nerves

Analyse the user message. Return ONLY valid JSON — no markdown, no explanation:
{"primary": "<specialist>", "secondary": [<up to 2 additional specialists if multiple distinct topics>]}

Rules:
1. "secondary" is [] when the message covers a single topic.
2. List at most 2 secondary agents.
3. Only use the exact specialist names above.
4. Any message asking to build, create, or generate a "training plan", "workout plan", "running programme", "workout schedule", or "exercise programme" → primary MUST always be "trainer", even if the message mentions personal data, BMR, or profile.
5. References to "my personal data", "my profile", "my stats", or "my goal" alone do NOT indicate dietitian — route based on the main request topic.

Examples:
"I twisted my ankle" → {"primary": "physiotherapist", "secondary": []}
"I twisted my ankle and I need to update my nutrition plan for tomorrow" → {"primary": "physiotherapist", "secondary": ["dietitian"]}
"Build me a training plan for my 10k" → {"primary": "trainer", "secondary": []}
"I have no motivation and I'm not sleeping well" → {"primary": "psychologist", "secondary": ["recovery_coach"]}
"I'm injured and feeling burned out and need diet help" → {"primary": "physiotherapist", "secondary": ["recovery_coach", "dietitian"]}
"According to my personal data build me a professional training plan to reach my goal" → {"primary": "trainer", "secondary": ["dietitian"]}
"Create a workout schedule based on my profile" → {"primary": "trainer", "secondary": []}\
"""


def _route_by_llm(message: str, llm) -> dict | None:
    """Call the LLM supervisor and return {"primary": ..., "secondary": [...]}.

    Returns None on any error so the caller can fall back to embedding routing.
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = llm.invoke(
            [SystemMessage(content=_SUPERVISOR_PROMPT), HumanMessage(content=message)]
        )
        raw = response.content.strip() if isinstance(response.content, str) else ""

        # Try direct parse first, then extract the first JSON object.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if not m:
                logger.debug("Supervisor LLM: no JSON object found in %r", raw[:120])
                return None
            data = json.loads(m.group(0))

        primary = data.get("primary", "")
        if primary not in _VALID_AGENTS:
            logger.debug("Supervisor LLM: unknown primary %r", primary)
            return None

        secondary = [
            a for a in (data.get("secondary") or [])
            if a in _VALID_AGENTS and a != primary
        ][:2]

        logger.debug("Supervisor LLM: primary=%s secondary=%s", primary, secondary)
        return {"primary": primary, "secondary": secondary}

    except Exception:
        logger.debug("Supervisor LLM call failed", exc_info=True)
        return None


# ── Embedding fallback ────────────────────────────────────────────────────────

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
    "psychologist": [
        "I have no motivation to train",
        "I am scared of failing in my race",
        "I feel anxious before competitions",
        "I cannot stay focused during my runs",
        "I lost my confidence after a bad race",
        "I keep giving up when it gets hard",
        "I feel mentally burned out from sport",
        "I have negative self-talk during workouts",
        "I need help with mental toughness",
        "I am struggling with performance anxiety",
        "I freeze under pressure during races",
        "How do I use visualization or mental imagery",
        "I am dealing with a performance slump",
        "I feel like I am not good enough",
        "How do I set better goals for my training",
        "I struggle to stay consistent with my training",
        "I cannot handle the pressure of competition",
    ],
}

_ROUTER_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _load() -> tuple:
    """Load embedding model and pre-compute mean anchor vectors. Called once, cached."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=_ROUTER_MODEL)
    domain_vecs: dict[str, np.ndarray] = {}
    for domain, sentences in _DOMAIN_ANCHORS.items():
        embs = np.array(list(model.embed(sentences)))
        domain_vecs[domain] = embs.mean(axis=0)
    return model, domain_vecs


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


def _route_by_embedding(message: str) -> str:
    """Return the closest domain agent for *message* using cosine similarity."""
    model, domain_vecs = _load()
    msg_vec = np.array(next(iter(model.embed([message]))))
    scores = {domain: _cosine(msg_vec, vec) for domain, vec in domain_vecs.items()}
    return max(scores, key=scores.get)


# ── Supervisor node factory ───────────────────────────────────────────────────

def make_supervisor_node(llm):
    """Return a LangGraph node function that routes the first message of a session.

    Mid-session (active_agent already set): no-op — the current agent continues.
    New session: calls the LLM supervisor for multi-intent detection; falls back
    to embedding similarity when the LLM fails or returns invalid JSON.
    """

    def supervisor(state: CoachState) -> dict:
        if state.get("active_agent"):
            return {}

        last_human = next(
            (m for m in reversed(state["messages"]) if m.type == "human"), None
        )
        if last_human is None:
            return {"active_agent": "trainer", "pending_agents": []}

        message = last_human.content

        result = _route_by_llm(message, llm)
        if result is not None:
            return {
                "active_agent": result["primary"],
                "pending_agents": result["secondary"],
            }

        # LLM failed — fall back to embedding (single agent, no secondary)
        logger.warning("Supervisor LLM unavailable — falling back to embedding router")
        primary = _route_by_embedding(message)
        return {"active_agent": primary, "pending_agents": []}

    return supervisor
