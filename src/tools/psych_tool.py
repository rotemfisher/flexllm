"""
Psychologist-specific tools:
  1. get_situational_psych_tips  — structured evidence-based tips for eight trigger situations.
  2. search_psychology_books     — hybrid RAG search scoped to the psychology book category.
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


# ── Situational guidance library ──────────────────────────────────────────────
# Each entry is a dict with four sections the LLM uses to structure its response.

_SITUATION_GUIDANCE: dict[str, dict] = {
    "onboarding": {
        "label": "Onboarding — First psychological contact",
        "psychological_profile": (
            "The athlete is forming their first impression of the coaching relationship. "
            "Key psychological drivers are unclear. Intrinsic vs extrinsic motivation, "
            "baseline self-efficacy, and relationship to failure and pressure must be assessed."
        ),
        "key_questions": [
            "What are your main goals, and why do they matter to you personally?",
            "Have you ever worked with a sports psychologist or mental coach before?",
            "Describe how you respond when training or racing doesn't go as planned.",
            "On a scale of 1–10, how would you rate your confidence in achieving your goal?",
            "What would success look like for you, beyond the performance number?",
        ],
        "recommended_techniques": [
            "Goal clarification: distinguish outcome, performance, and process goals.",
            "Baseline confidence inventory: identify high-confidence and low-confidence areas.",
            "Motivation mapping: explore autonomy (choice), competence (growth), and relatedness (belonging).",
            "Values clarification: help the athlete articulate why sport matters beyond results.",
        ],
        "watch_for": [
            "Perfectionism and all-or-nothing thinking about performance.",
            "Exclusively extrinsic motivation (prizes, approval) with no intrinsic anchors.",
            "History of sport burnout or dropout.",
            "High social anxiety around team or competitive contexts.",
        ],
        "approach_note": (
            "Prioritise building trust and safety. Ask open questions, reflect back, "
            "and avoid prescribing techniques in session one — assess first."
        ),
    },

    "pre_test": {
        "label": "Pre-test — Fitness test or time trial imminent",
        "psychological_profile": (
            "Fitness tests create acute performance anxiety. The athlete's self-worth "
            "may be tied to the result. Arousal level must be calibrated — too high "
            "triggers panic, too low leads to underperformance."
        ),
        "key_questions": [
            "How are you feeling about the test right now (physically and mentally)?",
            "What result would you consider a success, and what would disappoint you?",
            "Have you done this test before? How did it go?",
            "What's your biggest fear or concern going into today?",
        ],
        "recommended_techniques": [
            "Pre-performance routine: 3–5 min ritual (breathing → cue word → action) to anchor optimal arousal.",
            "Arousal regulation: box breathing (4-4-4-4) or 4-7-8 breathing to reduce over-activation.",
            "Process goal focus: redirect attention from outcome ('hit X pace') to process ('stay relaxed first km').",
            "Mental rehearsal: 2–3 min vivid visualisation of executing the test well.",
            "Cue word: choose one word (e.g., 'smooth', 'strong') to return focus if it drifts during the test.",
        ],
        "watch_for": [
            "Catastrophising ('if I fail this test everything is ruined').",
            "Over-arousal: fast speech, muscle tension, digestive distress.",
            "Identity fusion with the result — athlete IS their test score.",
            "Avoidance signals: finding reasons to delay or withdraw.",
        ],
        "approach_note": (
            "Keep this conversation brief and grounding. Do not overload the athlete "
            "with techniques — pick one regulation strategy and one process goal. "
            "End with confidence-affirming language about their preparation."
        ),
    },

    "pre_race": {
        "label": "Pre-race — Competition is imminent",
        "psychological_profile": (
            "Race anxiety is normal and functional up to a point. The athlete must "
            "shift from broad to narrow attentional focus, lock in their race plan, "
            "and regulate arousal into their individual optimal performance zone."
        ),
        "key_questions": [
            "Walk me through your race plan — what are the key execution points?",
            "Which part of the race worries you the most?",
            "How does it feel when you compete at your best?",
            "Are there any distractions (other competitors, conditions, expectations) on your mind?",
        ],
        "recommended_techniques": [
            "Race-plan review: confirm 3–4 concrete execution checkpoints (not time splits).",
            "Visualisation: close eyes, 5 min rehearsal of the whole race including adversity response ('if X happens, I do Y').",
            "Arousal regulation: identify personal optimal zone ('I perform best when I feel _____').",
            "Distraction control: 'park it' technique — write the distraction down, then close the notebook.",
            "Pre-race mantra: one sentence that captures the athlete's competitive identity.",
        ],
        "watch_for": [
            "Excessive comparison with competitors (upward social comparison).",
            "Paralysis by analysis — too many tactical decisions unresolved.",
            "Avoidance coping: over-eating, excessive socialising, phone scrolling before warm-up.",
            "Negative self-talk cascade in the 30 min before gun.",
        ],
        "approach_note": (
            "Focus on controllables. Do not introduce new strategies the day of the race. "
            "Anchor the athlete to what they've trained and what they know about themselves."
        ),
    },

    "post_race": {
        "label": "Post-race / post-test — Debrief after competition or assessment",
        "psychological_profile": (
            "After performance the athlete is in an emotionally raw state — either "
            "euphoric (risk: overconfidence) or disappointed (risk: rumination, shame). "
            "Accurate attribution and growth-minded reflection are the targets."
        ),
        "key_questions": [
            "How do you feel right now — emotionally and physically?",
            "Walk me through the race from your perspective. What happened?",
            "What are you most proud of from today, even if the result wasn't what you wanted?",
            "If you could change one thing about your mental approach, what would it be?",
            "What does this result tell you about where you are right now?",
        ],
        "recommended_techniques": [
            "Attribution retraining: guide the athlete to stable, controllable internal attributions ('I ran out of glycogen because my fuelling plan wasn't precise' not 'I'm just slow').",
            "3:1 reflection: for every criticism, identify three things that went well.",
            "Growth framing: 'What did this race teach you that training couldn't?'",
            "Emotional acceptance: normalise post-race emotional flatness or disappointment.",
            "Goal revision: use performance data to reset the next process goal cycle.",
        ],
        "watch_for": [
            "Catastrophising a poor result ('I'll never be good enough').",
            "Externalising failure consistently (blaming weather, course, competitors).",
            "Externalising success (dismissing a good result as luck).",
            "Post-race identity crisis — esp. after a peak race that is now over.",
        ],
        "approach_note": (
            "Wait 30–60 min after the race before the full debrief — "
            "acute emotions distort perception. Begin with empathy, not analysis."
        ),
    },

    "anomaly_training": {
        "label": "Anomaly training — Unexpected session result",
        "psychological_profile": (
            "Anomalous sessions (much slower/faster than expected, unusual fatigue, "
            "strength regression, or a surprising personal best) destabilise the athlete's "
            "internal model of fitness and progress. The primary risk is over-generalisation: "
            "treating one data point as a verdict on their entire trajectory."
        ),
        "key_questions": [
            "Tell me about the session — what happened and how did it feel?",
            "What's your interpretation of what caused this? (sleep, stress, nutrition, day in the cycle?)",
            "How does this compare to your baseline over the past 2–4 weeks?",
            "What story are you telling yourself about what this session means?",
            "How much is this affecting your confidence for future sessions?",
        ],
        "recommended_techniques": [
            "Data normalisation: explain that training response varies ±10–15% naturally; one session is noise, not signal.",
            "Attribution specificity: help identify concrete, specific causes (e.g., 'I slept 5h, dehydrated, high-stress week') rather than global ones ('I'm overtrained').",
            "Reframe awareness: 'Noticing something is off IS a skill — you're learning to read your body.'",
            "Process goal reset: redirect focus to the next session's execution, not the anomaly.",
            "Confidence anchor: recall 2–3 recent strong sessions to restore context.",
        ],
        "watch_for": [
            "Rapid identity threat: 'I've lost my fitness / I'm not a real athlete.'",
            "Obsessive re-analysis: athlete replaying the session repeatedly.",
            "Over-compensation plan: athlete wants to train harder to 'make up for it.'",
            "Denial of genuine warning signs (masking injury or illness as a 'bad day').",
        ],
        "approach_note": (
            "Validate the frustration first. Then introduce data perspective — "
            "share readiness metrics (TSB, HRV, sleep) to provide an objective anchor "
            "before addressing the psychological interpretation."
        ),
    },

    "new_injury": {
        "label": "New injury — Athlete has just been injured",
        "psychological_profile": (
            "Injury triggers acute grief-like responses: denial, anger, bargaining. "
            "Threats to athletic identity, loss of routine, and fear of permanent limitation "
            "are central. Early psychological intervention significantly improves "
            "adherence to rehab and long-term confidence."
        ),
        "key_questions": [
            "How are you feeling right now — beyond the physical pain?",
            "What worries you most about this injury?",
            "Has your sport or training identity been shaken by this?",
            "Do you have a history of previous injuries? How did you cope?",
            "What does your daily routine look like without training, and how does that feel?",
        ],
        "recommended_techniques": [
            "Emotional validation: normalise frustration, grief, and anger — these are adaptive.",
            "Rehab goal-setting: shift goal focus to daily rehab milestones instead of racing goals.",
            "Mental rehearsal for rehab: visualise completing physio exercises with perfect form to maintain neural pathways.",
            "Identity broadening: explore roles beyond 'athlete' to reduce identity fragility.",
            "Social connection: encourage team/peer connection to maintain sense of belonging.",
            "Healing visualisation: guide athlete through brief daily imagery of tissue healing and recovery.",
        ],
        "watch_for": [
            "Catastrophising: 'My season is over / I'll never run again.'",
            "Premature return pressure: athlete pushes to return before physiotherapist clears them.",
            "Clinical depression or anxiety symptoms — refer to licensed mental health professional if persistent.",
            "Complete disengagement from sport community (social withdrawal).",
        ],
        "approach_note": (
            "Session 1 post-injury: 80% emotional support, 20% practical planning. "
            "Do not overwhelm with rehab psychology techniques in the first session. "
            "Coordinate with the physiotherapist on recovery timeline to give the athlete realistic anchors."
        ),
    },

    "return_to_training": {
        "label": "Return to training — Athlete cleared after injury",
        "psychological_profile": (
            "Physical clearance does not equal psychological readiness. Re-injury fear, "
            "loss of fitness confidence, and altered body trust are common. "
            "The athlete may overcautiously avoid load OR recklessly push to 'prove' recovery."
        ),
        "key_questions": [
            "How does it feel to be back — what emotions come up?",
            "On a scale of 1–10, how confident are you that your body will handle training?",
            "Is there any fear or worry about re-injuring the same area?",
            "Do you feel pressure (internal or external) to come back faster than feels right?",
            "What would a successful first week back look like for you?",
        ],
        "recommended_techniques": [
            "Graded exposure: plan sessions that start well below capacity to rebuild trust in the body.",
            "Re-injury anxiety management: distinguish realistic precaution from catastrophic fear; use breathing if anxiety spikes during session.",
            "Confidence ladder: identify the specific first milestone (e.g., '10 min easy jog with no pain') and celebrate it explicitly.",
            "Body trust rituals: 2 min body-scan check-in before and after each session to track sensation non-judgementally.",
            "Return narrative: help the athlete construct a positive comeback story ('I came back stronger because...').",
        ],
        "watch_for": [
            "Fear-avoidance pattern: athlete reports physical symptoms that stop training but physio finds nothing.",
            "Over-compensation: athlete training harder than prescribed to 'prove' fitness.",
            "Renewed performance anxiety: athlete now anxious about fitness level relative to peers.",
            "Kinesiophobia (fear of movement): requires graduated exposure therapy.",
        ],
        "approach_note": (
            "Check in after every session for the first two weeks. The psychological "
            "return curve often lags the physical by 2–4 weeks. "
            "Coordinate milestone celebrations with the trainer."
        ),
    },
}

_VALID_SITUATIONS = list(_SITUATION_GUIDANCE.keys())


# ── Tool 1: Situational psychological tips ────────────────────────────────────

@tool
def get_situational_psych_tips(situation: str, context: str = "") -> str:
    """
    Return structured, evidence-based psychological guidance for a specific athletic situation.

    Call this tool whenever one of the following triggering events occurs — it gives you
    a ready-made psychological framework to adapt into your response.

    situation — choose exactly one of:
      "onboarding"         : first contact with the athlete (goal setting, motivation mapping)
      "pre_test"           : athlete is about to do a fitness test or time trial
      "pre_race"           : athlete is preparing for a race or competition
      "post_race"          : debrief after a race or test (good or bad result)
      "anomaly_training"   : unexpected session — too slow, too fast, unusual fatigue, or
                             over-lifting / strength regression
      "new_injury"         : athlete has just been injured (acute psychological response)
      "return_to_training" : athlete is cleared by physio and returning after injury

    context — optional extra detail to include in the tip summary.
              Examples: "athlete ran 45 s/km slower than threshold", "second injury this season",
              "athlete is 3 days out from their first marathon".

    Returns a structured guidance block with:
      - Psychological profile of the situation
      - Key questions to ask the athlete
      - Evidence-based techniques to offer
      - Warning signals to watch for
      - Approach note
    """
    situation = situation.strip().lower()
    if situation not in _SITUATION_GUIDANCE:
        valid = ", ".join(f'"{s}"' for s in _VALID_SITUATIONS)
        return (
            f"Unknown situation '{situation}'. "
            f"Valid values are: {valid}"
        )

    g = _SITUATION_GUIDANCE[situation]

    lines = [
        f"=== PSYCHOLOGICAL GUIDANCE: {g['label'].upper()} ===",
        "",
    ]

    if context:
        lines += [f"CONTEXT PROVIDED: {context}", ""]

    lines += [
        "PSYCHOLOGICAL PROFILE:",
        g["psychological_profile"],
        "",
        "KEY QUESTIONS TO ASK THE ATHLETE:",
    ]
    for i, q in enumerate(g["key_questions"], 1):
        lines.append(f"  {i}. {q}")

    lines += [
        "",
        "EVIDENCE-BASED TECHNIQUES:",
    ]
    for t in g["recommended_techniques"]:
        lines.append(f"  • {t}")

    lines += [
        "",
        "WATCH FOR (RED FLAGS):",
    ]
    for w in g["watch_for"]:
        lines.append(f"  ⚠ {w}")

    lines += [
        "",
        "APPROACH NOTE:",
        g["approach_note"],
    ]

    return "\n".join(lines)


# ── Tool 2: Psychology book Q&A ───────────────────────────────────────────────

@tool
def search_psychology_books(query: str, n_results: int = 5) -> str:
    """
    Hybrid semantic + keyword search scoped exclusively to sport psychology books.

    Use this for any free-form psychological question where you need evidence from
    the sport psychology literature. This is the primary tool for the Q&A use case.

    The search covers three books:
      - "champions_mind"         → The Champion's Mind (mindset, mental toughness, peak performance)
      - "applied_sport_psych"    → Applied Sport Psychology (anxiety, confidence, imagery, team dynamics)
      - "foundations_sport_psych"→ Foundations of Sport and Exercise Psychology (theory, motivation, arousal)

    Examples:
      query="how to build mental toughness in endurance athletes"
      query="cognitive restructuring techniques for performance anxiety"
      query="self-determination theory intrinsic motivation sport"
      query="imagery and mental rehearsal protocols"
      query="coping strategies for athletic slumps"
    """
    try:
        client, dense_model, sparse_model, rerank_model = _get_models()

        dense_vec  = dense_model.encode([query], normalize_embeddings=True)[0].tolist()
        sparse_raw = list(sparse_model.embed([query]))[0]
        sparse_q   = SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist(),
        )

        # Filter to psychology category only
        psych_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="psychology"))]
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
            query_filter=psych_filter,
            with_payload=True,
        )

        points = results.points
        if not points:
            return (
                "No psychology book passages found for this query. "
                "Psychology books may not yet be embedded — run `python etl/embed_books.py`."
            )

        texts  = [p.payload.get("text", "") for p in points]
        scores = rerank_model.predict([(query, t) for t in texts])
        ranked = [p for _, p in sorted(zip(scores, points), key=lambda x: x[0], reverse=True)]
        ranked = ranked[:n_results]

        parts = []
        for point in ranked:
            p          = point.payload or {}
            book_title = p.get("book_title", p.get("book", "Unknown"))
            section    = p.get("section", "")
            header     = f"[{book_title}" + (f" | {section}]" if section else "]")
            parts.append(f"{header}\n{p.get('text', '')}")

        return "\n\n---\n\n".join(parts)

    except Exception as exc:
        logger.exception("search_psychology_books error: %s", exc)
        return f"Psychology book search failed: {exc}"
