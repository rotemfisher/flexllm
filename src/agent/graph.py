from contextlib import asynccontextmanager
import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

from src.config import config
from src.models.agent_state import CoachState
from src.agent.trainer import TRAINER_TOOLS
from src.agent.physiotherapist import PHYSIO_TOOLS
from src.agent.recovery_coach import RECOVERY_TOOLS
from src.agent.dietitian import DIETITIAN_TOOLS
from src.agent.psychologist import PSYCHOLOGIST_TOOLS
from src.agent.prompts import (
    build_trainer_prompt,
    build_physio_prompt,
    build_recovery_prompt,
    build_dietitian_prompt,
    build_psychologist_prompt,
)
from src.agent.router import route_entry

# Names of the five handoff tools — used by the conflict resolver.
_HANDOFF_TOOL_NAMES = frozenset({
    "trainer_transfer",
    "physio_transfer",
    "recovery_transfer",
    "dietitian_transfer",
    "psychologist_transfer",
})

# Write operations that MUST execute before a handoff — they are not read-loops
# and should never be stripped when paired with a handoff.
_WRITE_WITH_HANDOFF = frozenset({
    "save_workout_plan",
    "replace_day_in_plan",
    "update_planned_workout_status",
    "log_injury",
    "log_injury_checkin",
    "resolve_injury",
    "log_strength_sets",
    "log_fitness_assessment",
    "update_athlete_profile",
})

# Keep at most this many messages in the context window sent to the LLM.
# Enough to cover several back-and-forth exchanges while staying well within
# Qwen 30B's NUM_CTX=16384 token budget.
_MAX_HISTORY_MESSAGES = 30


# ── Trainer context pre-fetch ─────────────────────────────────────────────────

def _gather_trainer_context(state: CoachState) -> dict:
    """Pre-fetch all trainer session-startup data at the Python level.

    Runs before the trainer LLM call so the model receives fresh session data
    as injected context instead of having to emit 6 parallel tool calls.
    The VDOT is resolved from the latest fitness_assessments row; defaults to
    35 (Daniels' beginner baseline) when no assessment exists yet.
    """
    from src.tools.sport_radar_tool import check_upcoming_race_or_test
    from src.tools.assessment_tool import get_onboarding_status
    from src.tools.readiness_tool import get_daily_readiness
    from src.tools.plan_tool import get_current_workout_plan
    from src.tools.workout_history_tool import get_recent_workouts
    from src.tools.vdot_tool import get_vdot_paces
    from src.tools._utils import db_ro

    vdot = 35
    try:
        with db_ro() as con:
            row = con.execute(
                "SELECT estimated_vdot FROM fitness_assessments "
                "WHERE estimated_vdot IS NOT NULL "
                "ORDER BY assessment_date DESC LIMIT 1"
            ).fetchone()
            if row and row["estimated_vdot"]:
                vdot = int(row["estimated_vdot"])
    except Exception:
        pass

    startup_calls = [
        ("check_upcoming_race_or_test", check_upcoming_race_or_test, {}),
        ("get_onboarding_status",       get_onboarding_status,       {}),
        ("get_daily_readiness",         get_daily_readiness,         {}),
        ("get_current_workout_plan",    get_current_workout_plan,    {}),
        ("get_recent_workouts",         get_recent_workouts,         {"limit": 10}),
        ("get_vdot_paces",              get_vdot_paces,              {"vdot": vdot}),
    ]

    sections: list[str] = []
    for name, fn, kwargs in startup_calls:
        try:
            result = fn.invoke(kwargs)
        except Exception as exc:
            result = f"(unavailable: {exc})"
        sections.append(f"[{name}]\n{result}")

    block = "\n\n".join(sections)
    prefetched = (
        f"\n\n=== SESSION STARTUP DATA (pre-fetched — do NOT call these tools again) ===\n"
        f"{block}\n"
        f"=== END SESSION STARTUP DATA ==="
    )
    logger.debug("gather_trainer_context: pre-fetched %d startup results (VDOT=%d)", len(startup_calls), vdot)
    return {"prefetched_context": prefetched}


def _gather_dietitian_context(state: CoachState) -> dict:
    from src.tools.nutrition_tool import get_nutrition_profile
    from src.tools.readiness_tool import get_daily_readiness

    calls = [
        ("get_nutrition_profile", get_nutrition_profile, {}),
        ("get_daily_readiness",   get_daily_readiness,   {}),
    ]
    sections = []
    for name, fn, kwargs in calls:
        try:
            result = fn.invoke(kwargs)
        except Exception as exc:
            result = f"(unavailable: {exc})"
        sections.append(f"[{name}]\n{result}")

    prefetched = (
        "\n\n=== SESSION STARTUP DATA (pre-fetched — do NOT call these tools again) ===\n"
        + "\n\n".join(sections)
        + "\n=== END SESSION STARTUP DATA ==="
    )
    logger.debug("gather_dietitian_context: pre-fetched %d results", len(calls))
    return {"prefetched_context": prefetched}


def _gather_physio_context(state: CoachState) -> dict:
    from src.tools.injury_tool import get_active_injuries
    from src.tools.workout_history_tool import get_recent_workouts

    calls = [
        ("get_active_injuries", get_active_injuries, {}),
        ("get_recent_workouts", get_recent_workouts, {"limit": 10}),
    ]
    sections = []
    for name, fn, kwargs in calls:
        try:
            result = fn.invoke(kwargs)
        except Exception as exc:
            result = f"(unavailable: {exc})"
        sections.append(f"[{name}]\n{result}")

    prefetched = (
        "\n\n=== SESSION STARTUP DATA (pre-fetched — do NOT call these tools again) ===\n"
        + "\n\n".join(sections)
        + "\n=== END SESSION STARTUP DATA ==="
    )
    logger.debug("gather_physio_context: pre-fetched %d results", len(calls))
    return {"prefetched_context": prefetched}


def _gather_recovery_context(state: CoachState) -> dict:
    from src.tools.readiness_tool import get_daily_readiness
    from src.tools.plan_tool import get_current_workout_plan

    calls = [
        ("get_daily_readiness",      get_daily_readiness,      {}),
        ("get_current_workout_plan", get_current_workout_plan, {}),
    ]
    sections = []
    for name, fn, kwargs in calls:
        try:
            result = fn.invoke(kwargs)
        except Exception as exc:
            result = f"(unavailable: {exc})"
        sections.append(f"[{name}]\n{result}")

    prefetched = (
        "\n\n=== SESSION STARTUP DATA (pre-fetched — do NOT call these tools again) ===\n"
        + "\n\n".join(sections)
        + "\n=== END SESSION STARTUP DATA ==="
    )
    logger.debug("gather_recovery_context: pre-fetched %d results", len(calls))
    return {"prefetched_context": prefetched}


def _gather_psychologist_context(state: CoachState) -> dict:
    from src.tools.readiness_tool import get_daily_readiness
    from src.tools.workout_history_tool import get_recent_workouts

    calls = [
        ("get_daily_readiness", get_daily_readiness, {}),
        ("get_recent_workouts", get_recent_workouts, {"limit": 10}),
    ]
    sections = []
    for name, fn, kwargs in calls:
        try:
            result = fn.invoke(kwargs)
        except Exception as exc:
            result = f"(unavailable: {exc})"
        sections.append(f"[{name}]\n{result}")

    prefetched = (
        "\n\n=== SESSION STARTUP DATA (pre-fetched — do NOT call these tools again) ===\n"
        + "\n\n".join(sections)
        + "\n=== END SESSION STARTUP DATA ==="
    )
    logger.debug("gather_psychologist_context: pre-fetched %d results", len(calls))
    return {"prefetched_context": prefetched}


# ── RAG auto-injection ────────────────────────────────────────────────────────

def _keyword_rag_queries(human_message: str, athlete_context: str) -> list[str]:
    """Keyword-based Qdrant query builder — used as a fallback when LLM generation fails."""
    ctx = athlete_context.lower()
    queries: list[str] = []

    if human_message.strip():
        queries.append(human_message)

    level = (
        "beginner" if "beginner" in ctx else
        "intermediate" if "intermediate" in ctx else
        "advanced"
    )

    if any(k in ctx for k in ["10k", "5k", "run", "marathon", "half marathon", "pace", "jog"]):
        dist = (
            "marathon" if "marathon" in ctx else
            "10k" if "10k" in ctx else
            "5k" if "5k" in ctx else
            "running"
        )
        queries.append(f"{level} {dist} training periodization base phase aerobic development VDOT")

    if any(k in ctx for k in ["muscle", "strength", "shredded", "hypertrophy", "muscular", "lean"]):
        queries.append(f"{level} hypertrophy strength training concurrent endurance periodization")

    if any(k in ctx for k in ["fat", "shredded", "lean", "weight loss", "body comp"]):
        queries.append("concurrent training fat loss body composition performance nutrition")

    return queries[:4]


def _build_rag_queries(human_message: str, athlete_context: str, query_llm=None) -> list[str]:
    """Generate targeted Qdrant queries for the user message.

    When a query_llm is provided the model rewrites the user message into up to
    3 precise exercise-science search queries — far more accurate than keyword
    matching.  Falls back to keyword extraction on any LLM error so RAG always
    runs even if the query model is unavailable.
    """
    if not human_message.strip():
        return []

    if query_llm is not None:
        try:
            from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

            system = (
                "You are a search-query optimizer for a sports science knowledge base "
                "(books: Daniels' Running Formula, NSCA Essentials, Sports Nutrition, etc.).\n"
                "Given a user question and athlete profile, output EXACTLY 3 search queries, "
                "one per line — no numbering, no bullets, no explanation.\n"
                "Rules:\n"
                "- Each query must be 6–15 words.\n"
                "- Use technical exercise-science terminology.\n"
                "- Target physiology, training methodology, or nutrition research.\n"
                "- Queries must be diverse: capture the core concept plus two related angles."
            )
            user_prompt = (
                f"Athlete profile (summary): {athlete_context[:300]}\n\n"
                f"User question: {human_message}\n\n"
                "Output 3 search queries:"
            )
            response = query_llm.invoke([_SM(content=system), _HM(content=user_prompt)])
            raw_lines = response.content.strip().split("\n")
            queries = [
                line.lstrip("0123456789.-) \t").strip()
                for line in raw_lines
                if line.strip() and len(line.strip()) > 5
            ][:3]
            if queries:
                logger.debug("LLM RAG queries: %s", queries)
                return queries
        except Exception:
            logger.debug("LLM query generation failed — falling back to keywords", exc_info=True)

    return _keyword_rag_queries(human_message, athlete_context)


def _rag_context_for(queries: list[str], n_per_query: int = 5) -> str:
    """Fetch targeted passages from Qdrant for multiple queries.

    Runs each query independently, deduplicates results by content hash, and
    returns a formatted knowledge block ready to inject into the system prompt.
    Returns an empty string on any error so it never blocks the graph.
    """
    if not queries:
        return ""
    try:
        from src.tools.knowledge_base_tool import search_knowledge_base
        seen: set[int] = set()
        sections: list[str] = []
        for q in queries:
            result: str = search_knowledge_base.invoke({"query": q, "n_results": n_per_query})
            if not result or "No relevant" in result or "failed" in result.lower():
                continue
            key = hash(result[:300])
            if key in seen:
                continue
            seen.add(key)
            sections.append(f"[search: {q}]\n{result}")
        if not sections:
            # Searched but found nothing — inject a hard red-flag so the model
            # cannot silently fall back to parametric knowledge.
            logger.debug("RAG search returned no results for queries: %s", queries)
            return (
                "\n\n=== KNOWLEDGE BASE — NO RESULTS FOUND ===\n"
                "⛔ The knowledge base returned NO relevant passages for this question.\n"
                "You MUST NOT provide any professional recommendation, training prescription, "
                "or nutrition advice. Respond with exactly:\n"
                "'I don't currently have scientifically backed information on this topic in my "
                "knowledge library. Please refocus your question or ask me to search for "
                "additional sources.'\n"
                "=== END KNOWLEDGE BASE ==="
            )
        body = "\n\n---\n\n".join(sections)
        return (
            f"\n\n=== KNOWLEDGE BASE — research evidence (treat as primary source) ===\n"
            f"{body}\n"
            f"=== END KNOWLEDGE BASE ===\n"
            f"⛔ Every recommendation MUST be grounded in the passages above. "
            f"Cite the book title for each prescription. Never write advice that contradicts these sources."
        )
    except Exception:
        logger.debug("RAG auto-inject skipped", exc_info=True)
    return ""


# ── Output validator helpers ──────────────────────────────────────────────────

_PROFESSIONAL_KEYWORDS = (
    "pace", "km/h", "min/km", "calories", "kcal", "protein", "carbohydrate",
    "sets", "reps", "intensity", "threshold", "interval", "zone ", "vdot",
    "tss", "hrv", "periodiz", "training load", "volume", "recovery protocol",
    "bmr", "tdee", "macro", "g/kg", "training plan", "rehabilitation",
    "aerobic", "anaerobic", "lactate", "vo2", "heart rate zone",
)

# Matches parenthetical book/author citations: capital letter, letters/apostrophes/spaces,
# at least 10 chars — catches "(Daniels' Running Formula)" but not "(RPE 8)" or "(e.g.)".
_CITATION_RE = re.compile(r"\([A-Z][A-Za-z'\s]{9,60}\)")


def _has_professional_content(text: str) -> bool:
    """Return True when the response is long enough and contains professional claims."""
    if len(text) < 180:
        return False
    text_lower = text.lower()
    matches = sum(1 for kw in _PROFESSIONAL_KEYWORDS if kw in text_lower)
    return matches >= 2


def _has_valid_citations(text: str) -> bool:
    """Return True when the response contains at least one (Book Title) citation."""
    return bool(_CITATION_RE.search(text))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _trim_messages(messages: list[BaseMessage], max_messages: int = _MAX_HISTORY_MESSAGES) -> list[BaseMessage]:
    """Return the most recent up-to-max_messages messages.

    Never starts the resulting slice with a ToolMessage: an orphaned tool
    result (without its AIMessage parent that contains tool_calls) confuses
    the Ollama chat format and can cause an API error.
    """
    # working with tools is always in pairs of AIMessage + ToolMessage, so max_messages should be even to avoid cutting off in the middle of a pair.
    if len(messages) <= max_messages:
        return messages
    trimmed = list(messages[-max_messages:])
    # Walk forward until we're past any orphaned ToolMessages at the head.
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed = trimmed[1:]
    return trimmed


def _resolve_tool_conflicts(response) -> object:
    """If the LLM returns a handoff tool call mixed with domain tool calls,
    keep write operations (they must persist before routing) but drop read-only
    domain calls.  Read-only calls alongside a handoff cause an extra loop that
    LangGraph handles unpredictably.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls or len(tool_calls) <= 1:
        return response
    handoffs = [tc for tc in tool_calls if tc["name"] in _HANDOFF_TOOL_NAMES]
    if not handoffs:
        return response  # Multiple domain calls — fine, no conflict.
    # Keep: write-once operations (must execute before handoff) + first handoff.
    # Drop: read-only domain tools that would create an unwanted extra loop.
    writes  = [tc for tc in tool_calls if tc["name"] in _WRITE_WITH_HANDOFF]
    return response.model_copy(update={"tool_calls": writes + handoffs[:1]})


# ── Agent node factory ────────────────────────────────────────────────────────

def _make_agent_node(llm_with_tools, prompt_builder,
                     startup_tools: frozenset = frozenset(),
                     handoff_startup_tools: frozenset = frozenset(),
                     query_llm=None):
    def call_model(state: CoachState) -> dict:
        from langchain_core.messages import AIMessage as _AIMessage

        def _safe_invoke(msgs, label: str):
            """Invoke the LLM; return a fallback AIMessage on any error."""
            try:
                return llm_with_tools.invoke(msgs)
            except Exception as exc:
                logger.error("LLM invoke failed (%s): %s", label, exc, exc_info=True)
                return _AIMessage(
                    content="I ran into a technical issue. Please try again in a moment."
                )

        athlete_ctx = state.get("athlete_context", "")
        system_prompt = prompt_builder(athlete_ctx)
        prefetched = state.get("prefetched_context", "")
        if prefetched:
            system_prompt += prefetched
        history = _trim_messages(list(state["messages"]))
        reason = state.get("handoff_reason")
        if reason:
            system_prompt += f"\n\nCRITICAL CONTEXT: You have just received control of the session. The previous agent's handoff note: '{reason}'"

        last_human = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)
        last_human_content = last_human.content if last_human else ""

        is_first_call = not any(isinstance(m, ToolMessage) for m in history)

        last_hist = history[-1] if history else None
        is_after_tool = isinstance(last_hist, ToolMessage)
        last_tool_content = getattr(last_hist, "content", "") or ""

        # Fresh handoff activation: the LAST message the agent received is a
        # handoff ToolMessage ("Transferred to …").  Intercept BEFORE the first
        # LLM call so the model cannot write hallucinated data.
        # Note: the shared history always contains the previous agent's tool
        # results, so we only look at the LAST message, not all ToolMessages.
        is_fresh_handoff = (
            is_after_tool
            and last_tool_content.startswith("Transferred to")
        )

        # ── Qdrant knowledge injection ────────────────────────────────────────
        # Injection 1: first call of each user turn — targeted queries derived
        # from the athlete profile so the model starts with relevant book passages.
        if is_first_call and not is_fresh_handoff:
            if last_human:
                rag_queries = _build_rag_queries(last_human_content, athlete_ctx, query_llm)
                system_prompt += _rag_context_for(rag_queries)

        # Injection 2: call immediately after startup tools complete — this is
        # when the plan is actually written, so evidence must be in context here.
        # Signal: the last AIMessage called get_vdot_paces (startup just finished).
        if is_after_tool and not is_fresh_handoff and startup_tools:
            _last_ai = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
            if _last_ai and any(
                tc.get("name") == "get_vdot_paces"
                for tc in (getattr(_last_ai, "tool_calls", None) or [])
            ):
                rag_queries = _build_rag_queries(last_human_content, athlete_ctx, query_llm)
                system_prompt += _rag_context_for(rag_queries)
        # ─────────────────────────────────────────────────────────────────────

        # On the very first call of a session (no tool results yet, not a handoff),
        # inject the mandatory startup tool list directly into the system prompt so
        # the model cannot claim it didn't know what to call.
        if is_first_call and not is_fresh_handoff and startup_tools:
            tools_list = "\n".join(f"  - {t}()" for t in startup_tools)
            system_prompt += (
                f"\n\n⛔ MANDATORY FIRST RESPONSE: Your entire output MUST be ONLY "
                f"these {len(startup_tools)} simultaneous tool calls — zero text:\n{tools_list}"
            )

        if is_fresh_handoff:
            messages = [SystemMessage(content=system_prompt)] + history + [
                HumanMessage(content=(
                    "You have just been activated via handoff. "
                    "Your session startup data has been pre-loaded in the SESSION STARTUP DATA block above — "
                    "do NOT call startup tools again. Read the data and respond to the athlete now."
                ))
            ]
        else:
            messages = [SystemMessage(content=system_prompt)] + history

        response = _safe_invoke(messages, "initial")
        response = _resolve_tool_conflicts(response)

        has_tool_calls = bool(getattr(response, "tool_calls", None))
        called_tool_names = {tc["name"] for tc in (getattr(response, "tool_calls", None) or [])}
        content = response.content if isinstance(response.content, str) else ""
        is_empty = not content.strip()

        # Detect "announcement without action": model described what it will do but
        # did not actually call any tool.  Fire when after a ToolMessage (catches
        # both the first activation-via-handoff and mid-chain stalls) OR when this
        # is the very first call and the response has no tool calls at all.
        _ANNOUNCEMENT_PHRASES = (
            # "let me …" family
            "let me fetch", "let me check", "let me get", "let me look",
            "let me retrieve", "let me pull", "let me gather",
            "let me start", "let me begin", "let me now",
            # "i will / i'll …" family
            "i will now", "i'll now", "i will fetch", "i'll fetch",
            "i will check", "i'll check", "i will get", "i'll get",
            "i will start", "i'll start", "i will begin", "i'll begin",
            # "i need to …" family
            "i need to fetch", "i need to check", "i need to get",
            # "let's …" family  ← the missing ones that caught the dietitian
            "let's start by", "let's begin", "let's first",
            "let's get started", "let's proceed",
            # other announcement patterns
            "first, let", "first, i'll", "first, i will",
            "allow me to", "we'll start", "we'll begin",
            "to get started", "to begin,",
        )
        is_announcement = (
            is_after_tool
            and not has_tool_calls
            and not is_empty
            and any(p in content.lower() for p in _ANNOUNCEMENT_PHRASES)
        )
        # Detect a failed tool call: model responded after a ToolNode error either
        # with no tool calls, or by trying to escape via handoff instead of retrying.
        is_after_tool_error = is_after_tool and "Error invoking tool" in last_tool_content
        has_handoff_only = (
            has_tool_calls
            and all(tc["name"] in _HANDOFF_TOOL_NAMES for tc in (getattr(response, "tool_calls", None) or []))
        )
        is_tool_error = is_after_tool_error and (not has_tool_calls or has_handoff_only)

        if is_first_call and not is_fresh_handoff and startup_tools:
            missing = startup_tools - called_tool_names
            if missing:
                logger.warning("Missing startup tools: %s — requesting via nudge", missing)
                missing_str = ", ".join(f"{t}()" for t in missing)
                startup_nudge = messages + [
                    response,
                    HumanMessage(content=(
                        f"You missed required startup tool calls: {missing_str}. "
                        f"Call ONLY these missing tools right now — no text."
                    )),
                ]
                nudge_resp = _safe_invoke(startup_nudge, "missing-startup-tools nudge")
                orig_calls = list(getattr(response, "tool_calls", None) or [])
                new_calls = [
                    tc for tc in (getattr(nudge_resp, "tool_calls", None) or [])
                    if tc["name"] in missing
                ]
                if new_calls:
                    response = response.model_copy(update={"tool_calls": orig_calls + new_calls})
        elif is_fresh_handoff and handoff_startup_tools:
            missing_handoff = handoff_startup_tools - called_tool_names
            if missing_handoff:
                logger.warning("Fresh handoff missing required tools: %s — enforcing via nudge", missing_handoff)
                missing_str = ", ".join(f"{t}()" for t in missing_handoff)
                handoff_nudge = messages + [
                    response,
                    HumanMessage(content=(
                        f"You must call these tools BEFORE writing any text: {missing_str}. "
                        f"Call ONLY these tools right now — no text."
                    )),
                ]
                nudge_resp = _safe_invoke(handoff_nudge, "fresh-handoff startup nudge")
                nudge_calls = getattr(nudge_resp, "tool_calls", None) or []
                # Strip handoff escapes — agent must call startup tools, not route away.
                valid_new = [
                    tc for tc in nudge_calls
                    if tc["name"] in missing_handoff and tc["name"] not in _HANDOFF_TOOL_NAMES
                ]
                orig_calls = list(getattr(response, "tool_calls", None) or [])
                # Clear fake text — the real response will be written after tool results arrive.
                response = response.model_copy(update={
                    "tool_calls": orig_calls + valid_new,
                    "content": "",
                })
        elif is_tool_error:
            logger.warning("Tool error detected — nudging model to retry with correct args")
            nudge = messages + [
                response,
                HumanMessage(content=(
                    "The previous tool call failed. Re-read the error message above, "
                    "fix the arguments, and call the tool again now. "
                    "Output ONLY the corrected tool call — no text."
                )),
            ]
            response = _safe_invoke(nudge, "tool-error retry nudge")
            response = _resolve_tool_conflicts(response)
        elif is_empty and is_after_tool:
            logger.warning("Empty response after tool results — nudging model to write reply")
            nudge = messages + [
                response,
                HumanMessage(content=(
                    "You have received all the tool results above. "
                    "Now write your complete, substantive response to the athlete. "
                    "Do not call any more tools — just write the answer."
                )),
            ]
            response = _safe_invoke(nudge, "empty-after-tool nudge")
            response = _resolve_tool_conflicts(response)
        elif is_announcement:
            logger.warning("Announcement without tool call — nudging model to act")
            nudge = messages + [
                response,
                HumanMessage(content=(
                    "Call the required tool now. "
                    "Output ONLY the tool call — no text before or after it."
                )),
            ]
            response = _safe_invoke(nudge, "announcement nudge")
            response = _resolve_tool_conflicts(response)
        elif is_empty:
            logger.warning("Empty response (no prior tool) — nudging model to reply")
            nudge = messages + [
                response,
                HumanMessage(content=(
                    "Your response was empty. Please write your response to the athlete now."
                )),
            ]
            response = _safe_invoke(nudge, "empty nudge")
            response = _resolve_tool_conflicts(response)

        # Citation validator — runs last, after all other nudges are resolved.
        # Only fires on final text responses (no tool calls) that contain professional
        # content without any book-title citation in the (Title) format.
        final_content = response.content if isinstance(response.content, str) else ""
        final_has_tools = bool(getattr(response, "tool_calls", None))
        if (
            not final_has_tools
            and _has_professional_content(final_content)
            and not _has_valid_citations(final_content)
        ):
            logger.warning("Professional content without citations — triggering validator nudge")
            citation_nudge = messages + [
                response,
                HumanMessage(content=(
                    "You made professional recommendations without citing the knowledge base. "
                    "Rewrite your entire response and add citations in the format "
                    "(Book Title) after every professional claim (e.g. '(Daniels' Running Formula)'). "
                    "If you cannot cite a passage from the injected KNOWLEDGE BASE for a claim, "
                    "replace that claim with the strict abstention phrase instead of inventing a citation."
                )),
            ]
            response = _safe_invoke(citation_nudge, "citation validator nudge")
            response = _resolve_tool_conflicts(response)

        return {"messages": [response], "handoff_reason": None}
    return call_model


def _should_continue(tools_node_name: str):
    def router(state: CoachState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return tools_node_name
        return END
    return router


# ── Graph builder ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def build_multi_agent_graph():
    llm = ChatOllama(
        model=config.MODEL_ID,
        temperature=0,
        base_url=config.OLLAMA_BASE_URL,
        num_ctx=16384,
        request_timeout=180,
    )

    # Query LLM: separate model for RAG query generation when QUERY_MODEL_ID is set.
    # Reuses the main LLM when unset so no extra model is loaded into VRAM.
    if config.QUERY_MODEL_ID and config.QUERY_MODEL_ID != config.MODEL_ID:
        query_llm = ChatOllama(
            model=config.QUERY_MODEL_ID,
            temperature=0,
            base_url=config.OLLAMA_BASE_URL,
            num_ctx=1024,
            request_timeout=15,
        )
        logger.info("RAG query model: %s", config.QUERY_MODEL_ID)
    else:
        query_llm = llm

    trainer_llm      = llm.bind_tools(TRAINER_TOOLS)
    physio_llm       = llm.bind_tools(PHYSIO_TOOLS)
    recovery_llm     = llm.bind_tools(RECOVERY_TOOLS)
    dietitian_llm    = llm.bind_tools(DIETITIAN_TOOLS)
    psychologist_llm = llm.bind_tools(PSYCHOLOGIST_TOOLS)

    graph = StateGraph(CoachState)

    # Context pre-fetch nodes — each runs before its agent to avoid read-only tool calls
    graph.add_node("gather_trainer_context",     _gather_trainer_context)
    graph.add_node("gather_dietitian_context",   _gather_dietitian_context)
    graph.add_node("gather_physio_context",      _gather_physio_context)
    graph.add_node("gather_recovery_context",    _gather_recovery_context)
    graph.add_node("gather_psychologist_context",_gather_psychologist_context)

    # Agent nodes
    graph.add_node("trainer",         _make_agent_node(trainer_llm,      build_trainer_prompt,      query_llm=query_llm))
    graph.add_node("physiotherapist", _make_agent_node(physio_llm,       build_physio_prompt,       query_llm=query_llm))
    graph.add_node("recovery_coach",  _make_agent_node(recovery_llm,     build_recovery_prompt,     query_llm=query_llm))
    graph.add_node("dietitian",       _make_agent_node(dietitian_llm,    build_dietitian_prompt,    query_llm=query_llm))
    graph.add_node("psychologist",    _make_agent_node(psychologist_llm, build_psychologist_prompt, query_llm=query_llm))

    # Tool nodes — separate per agent so each only exposes its own tools
    graph.add_node("trainer_tools",      ToolNode(TRAINER_TOOLS))
    graph.add_node("physio_tools",       ToolNode(PHYSIO_TOOLS))
    graph.add_node("recovery_tools",     ToolNode(RECOVERY_TOOLS))
    graph.add_node("dietitian_tools",    ToolNode(DIETITIAN_TOOLS))
    graph.add_node("psychologist_tools", ToolNode(PSYCHOLOGIST_TOOLS))

    # Entry: semantic router at START — all agents go through their gather nodes first.
    graph.add_conditional_edges(START, route_entry, {
        "trainer":         "gather_trainer_context",
        "physiotherapist": "gather_physio_context",
        "recovery_coach":  "gather_recovery_context",
        "dietitian":       "gather_dietitian_context",
        "psychologist":    "gather_psychologist_context",
    })
    graph.add_edge("gather_trainer_context",      "trainer")
    graph.add_edge("gather_dietitian_context",    "dietitian")
    graph.add_edge("gather_physio_context",       "physiotherapist")
    graph.add_edge("gather_recovery_context",     "recovery_coach")
    graph.add_edge("gather_psychologist_context", "psychologist")

    # Each agent routes to its tool node or END
    graph.add_conditional_edges(
        "trainer", _should_continue("trainer_tools"),
        {"trainer_tools": "trainer_tools", END: END},
    )
    graph.add_conditional_edges(
        "physiotherapist", _should_continue("physio_tools"),
        {"physio_tools": "physio_tools", END: END},
    )
    graph.add_conditional_edges(
        "recovery_coach", _should_continue("recovery_tools"),
        {"recovery_tools": "recovery_tools", END: END},
    )
    graph.add_conditional_edges(
        "dietitian", _should_continue("dietitian_tools"),
        {"dietitian_tools": "dietitian_tools", END: END},
    )
    graph.add_conditional_edges(
        "psychologist", _should_continue("psychologist_tools"),
        {"psychologist_tools": "psychologist_tools", END: END},
    )

    # Normal tool calls loop back to their owning agent.
    # When a handoff tool returns Command(goto=...), LangGraph overrides these edges.
    graph.add_edge("trainer_tools",      "trainer")
    graph.add_edge("physio_tools",       "physiotherapist")
    graph.add_edge("recovery_tools",     "recovery_coach")
    graph.add_edge("dietitian_tools",    "dietitian")
    graph.add_edge("psychologist_tools", "psychologist")

    checkpoint_db = config.DB_PATH.parent / "checkpoints.db"
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as checkpointer:
        yield graph.compile(checkpointer=checkpointer)
