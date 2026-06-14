from contextlib import asynccontextmanager
import asyncio
import logging
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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
from src.agent.router import make_supervisor_node

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
            sections.append(result)
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

# Matches "[Book Title | Section]" or "[Book Title]" headers injected by _rag_context_for().
_BOOK_TITLE_IN_CONTEXT_RE = re.compile(r"\[([^\|\]\n]+?)(?:\s*\|[^\]]+)?\]")

_CITATION_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "in", "for", "to", "by", "on", "at", "with",
})


def _extract_injected_titles(rag_ctx: str) -> set[str]:
    """Return the set of book titles (lower-cased) present in a RAG context block."""
    titles: set[str] = set()
    for m in _BOOK_TITLE_IN_CONTEXT_RE.finditer(rag_ctx):
        title = m.group(1).strip()
        if not title.startswith("search:"):
            titles.add(title.lower())
    return titles


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


def _citations_are_grounded(text: str, injected_titles: set[str]) -> bool:
    """Return True when at least one citation in *text* matches a book that was
    actually injected from Qdrant this turn.

    Falls back to pure regex when no RAG context was injected (so the validator
    still fires on professional responses that should have triggered a search).
    """
    if not injected_titles:
        return _has_valid_citations(text)
    for m in _CITATION_RE.finditer(text):
        cited = m.group(0)[1:-1].strip().lower()
        cited_words = {w for w in cited.split() if w not in _CITATION_STOP_WORDS}
        for title in injected_titles:
            title_words = {w for w in title.split() if w not in _CITATION_STOP_WORDS}
            if len(cited_words & title_words) >= 2:
                return True
    return False


# ── Utilities ─────────────────────────────────────────────────────────────────

def _is_premature_analysis(text: str) -> bool:
    """Return True when a response looks like a leaked internal analysis.

    Catches the pattern where the model writes multi-line bullet-point topics
    (e.g. 'improving lactate threshold for beginner runners') instead of calling
    tools or writing a complete coaching response.  Only fires on short responses
    that would normally be too brief to constitute real coaching advice.
    """
    stripped = text.strip()
    return (
        stripped.count("\n") >= 1          # multi-line → likely a list
        and len(stripped) < 350            # too short for a real coaching response
        and any(kw in stripped.lower() for kw in _PROFESSIONAL_KEYWORDS)
    )


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


# ── Conversation summarization ────────────────────────────────────────────────

_SUMMARIZE_EVERY_N_TURNS = 8   # prune + summarise after every N human messages
_KEEP_RECENT_TOOL_MSGS   = 10  # keep this many recent ToolMessages intact


def _maybe_summarize(
    llm,
    full_messages: list[BaseMessage],
) -> tuple[list[BaseMessage], str]:
    """Identify stale ToolMessages (and their paired AIMessages) for pruning.

    Returns (messages_to_remove, summary_text).  The caller decides whether this
    turn qualifies for summarisation and only calls this function when it does.
    Returns ([], "") on LLM error so the main flow is never blocked.
    """
    all_tool_msgs = [m for m in full_messages if isinstance(m, ToolMessage)]
    stale_tools   = all_tool_msgs[:-_KEEP_RECENT_TOOL_MSGS]
    if not stale_tools:
        return [], ""

    # Collect AIMessages whose *entire* tool-call set maps to stale ToolMessages.
    stale_tc_ids = {getattr(m, "tool_call_id", None) for m in stale_tools}
    stale_ais = [
        m for m in full_messages
        if isinstance(m, AIMessage)
        and (getattr(m, "tool_calls", None) or [])
        and {tc.get("id") for tc in (getattr(m, "tool_calls", None) or [])}.issubset(stale_tc_ids)
    ]

    context = "\n\n".join(
        f"[{getattr(m, 'name', 'tool')}]\n{str(m.content)[:500]}"
        for m in stale_tools
    )
    try:
        resp = llm.invoke([
            SystemMessage(content=(
                "You are a concise data summariser for a sports-coaching AI. "
                "Below are tool results from older conversation turns. "
                "Write 4–8 bullet points covering the key athlete metrics, recent training, "
                "injuries, and coaching decisions made so far. "
                "Output ONLY the bullet list — no headers, no intro, no tool calls."
            )),
            HumanMessage(content=f"Summarise these tool results:\n\n{context}"),
        ])
        summary = resp.content.strip() if isinstance(resp.content, str) else ""
    except Exception:
        logger.warning("Conversation summarisation LLM call failed — skipping prune")
        return [], ""

    to_remove = stale_tools + stale_ais
    logger.info(
        "Summarised %d tool msgs and %d AI msgs at turn boundary",
        len(stale_tools), len(stale_ais),
    )
    return to_remove, summary


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

        # ── Conversation summarisation ────────────────────────────────────────
        # Count human messages in the full (untruncated) state to detect turn boundaries.
        full_messages  = list(state["messages"])
        n_human        = sum(1 for m in full_messages if isinstance(m, HumanMessage))
        last_full_msg  = full_messages[-1] if full_messages else None
        _should_summarise = (
            isinstance(last_full_msg, HumanMessage)
            and n_human > 0
            and n_human % _SUMMARIZE_EVERY_N_TURNS == 0
        )
        msgs_to_remove: list[BaseMessage] = []
        new_summary    = ""
        if _should_summarise:
            msgs_to_remove, new_summary = _maybe_summarize(llm_with_tools, full_messages)

        removed_ids = {m.id for m in msgs_to_remove if m.id}

        history = _trim_messages(list(state["messages"]))
        # Strip pruned messages from the window sent to the LLM this turn
        if removed_ids:
            history = [m for m in history if m.id not in removed_ids]

        # Inject existing rolling summary when non-empty
        existing_summary = state.get("conversation_summary", "")
        if existing_summary:
            system_prompt += (
                "\n\n=== PRIOR CONVERSATION SUMMARY (compressed older turns) ===\n"
                + existing_summary
                + "\n=== END SUMMARY ===\n"
                "(The above replaces older tool results that have been compressed out "
                "of the message history to save context.)"
            )
        # ─────────────────────────────────────────────────────────────────────

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
        # Track which book titles were actually injected so the citation validator
        # can reject hallucinated book names that were never in the Qdrant results.
        injected_book_titles: set[str] = set()

        # Injection 1: first call of each user turn — targeted queries derived
        # from the athlete profile so the model starts with relevant book passages.
        if is_first_call and not is_fresh_handoff:
            if last_human:
                rag_queries = _build_rag_queries(last_human_content, athlete_ctx, query_llm)
                rag_ctx = _rag_context_for(rag_queries, n_per_query=3)
                system_prompt += rag_ctx
                injected_book_titles |= _extract_injected_titles(rag_ctx)

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
                rag_ctx = _rag_context_for(rag_queries)
                system_prompt += rag_ctx
                injected_book_titles |= _extract_injected_titles(rag_ctx)
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
        elif (
            is_first_call
            and not is_fresh_handoff
            and not has_tool_calls
            and not is_empty
            and _is_premature_analysis(content)
        ):
            logger.warning(
                "Premature analysis detected (%d chars) — nudging for complete action",
                len(content.strip()),
            )
            nudge = messages + [
                response,
                HumanMessage(content=(
                    "You wrote a short topic outline instead of taking action. "
                    "Do NOT list topics or themes. Based on the SESSION STARTUP DATA "
                    "and KNOWLEDGE BASE already in your context, proceed NOW: "
                    "call the required tools (e.g. save_workout_plan) and write the "
                    "complete coaching response. No topic lists, no preambles, no outlines."
                )),
            ]
            response = _safe_invoke(nudge, "premature-analysis nudge")
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
            and not _citations_are_grounded(final_content, injected_book_titles)
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

            # Hard override: if the model still ignored the nudge, replace the
            # response entirely rather than letting uncited professional advice reach
            # the athlete.
            post_nudge = response.content if isinstance(response.content, str) else ""
            if (
                not getattr(response, "tool_calls", None)
                and _has_professional_content(post_nudge)
                and not _citations_are_grounded(post_nudge, injected_book_titles)
            ):
                logger.warning("Citation nudge ignored — overriding with abstention message")
                response = response.model_copy(update={
                    "content": (
                        "I want to make sure my recommendations are grounded in the coaching "
                        "literature before I share them. Could you give me a bit more detail — "
                        "for example your specific goal, current fitness level, or the exact "
                        "topic you'd like help with? That helps me pull the right evidence "
                        "from the knowledge base and give you a fully cited answer."
                    )
                })

        # Build the output message list: RemoveMessages first (state cleanup),
        # then the actual LLM response.
        out_messages = (
            [RemoveMessage(id=m.id) for m in msgs_to_remove if m.id] + [response]
        )
        out: dict = {"messages": out_messages, "handoff_reason": None}
        if new_summary:
            out["conversation_summary"] = new_summary
        return out
    return call_model


def _should_continue(tools_node_name: str):
    def router(state: CoachState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return tools_node_name
        if state.get("pending_agents"):
            return "route_pending"
        return END
    return router


def _route_pending(state: CoachState) -> dict:
    """Pop the first queued agent and promote it to active."""
    pending = list(state.get("pending_agents") or [])
    if not pending:
        return {}
    next_agent = pending[0]
    return {
        "active_agent": next_agent,
        "pending_agents": pending[1:],
        "handoff_reason": None,
    }


def _route_from_pending(state: CoachState) -> str:
    return state["active_agent"]


# ── Tool error formatter ──────────────────────────────────────────────────────

_PLAN_TOOL_ALLOWED: dict[str, str] = {
    "activity_type": "'running', 'strength', 'rest', 'cross_training'",
    "workout_type":  "'easy', 'tempo', 'interval', 'long_run', 'recovery', 'strength', 'rest', 'assessment'",
    "intensity":     "'easy', 'moderate', 'hard', 'rest'",
    "phase":         "'onboarding', 'base', 'build', 'peak', 'race', 'recovery', 'return_to_run' (or null)",
}


def _format_tool_error(error: Exception) -> str:
    """Custom ToolNode error handler.

    Rewrites Pydantic ValidationErrors into a short, model-readable message that
    names the bad field, the received value, and the allowed literals.  All other
    exceptions fall through to LangGraph's default handler so existing behaviour
    is unchanged.
    """
    from langgraph.prebuilt.tool_node import ToolInvocationError, _default_handle_tool_errors
    from pydantic import ValidationError as PydanticValidationError

    if isinstance(error, ToolInvocationError) and isinstance(error.source, PydanticValidationError):
        errors = error.filtered_errors or error.source.errors()
        lines: list[str] = []
        for e in errors[:5]:
            loc  = " → ".join(str(p) for p in e.get("loc", ()))
            msg  = e.get("msg", "invalid value")
            inp  = e.get("input", "?")
            lines.append(f"  • {loc}: {msg} (you sent: {inp!r})")
        bad_fields = {str(e.get("loc", ("",))[-1]) for e in errors if e.get("loc")}
        hint_lines = [
            f"    {k}: {v}"
            for k, v in _PLAN_TOOL_ALLOWED.items()
            if k in bad_fields
        ]
        out = f"Tool '{error.tool_name}' validation error — fix and retry:\n" + "\n".join(lines)
        if hint_lines:
            out += "\nAllowed values:\n" + "\n".join(hint_lines)
        return out

    return _default_handle_tool_errors(error)


# ── Graph builder ─────────────────────────────────────────────────────────────

async def _warmup_ollama(base_url: str, model_id: str) -> None:
    """Preload a model into Ollama memory so the first real request skips cold-start.

    Sends keep_alive=-1 (stay loaded indefinitely) with no prompt — Ollama loads
    the weights and returns immediately without generating any tokens.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            await client.post(
                f"{base_url}/api/generate",
                json={"model": model_id, "keep_alive": -1},
            )
        logger.info("Ollama warmup complete: %s", model_id)
    except Exception as exc:
        logger.warning("Ollama warmup failed for %s: %s", model_id, exc)


async def _warmup_rag_models() -> None:
    """Pre-load sentence-transformer RAG models in a background thread.

    _get_models() initialises the dense embedder, sparse model, reranker, and
    Qdrant client.  Running it once at startup means the first search_knowledge_base
    call doesn't pay the ~60 s model-loading penalty.
    """
    try:
        from src.tools.rag_tool import _get_models
        await asyncio.to_thread(_get_models)
        logger.info("RAG model warmup complete (bge-large + reranker loaded)")
    except Exception as exc:
        logger.warning("RAG model warmup failed: %s", exc)


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

    # Supervisor LLM: small context, JSON mode, used only for intent routing at
    # the start of each new session (not called for mid-session messages).
    supervisor_llm = ChatOllama(
        model=config.QUERY_MODEL_ID or config.MODEL_ID,
        temperature=0,
        format="json",
        base_url=config.OLLAMA_BASE_URL,
        num_ctx=1024,
        request_timeout=20,
    )
    logger.info("Supervisor model: %s", config.QUERY_MODEL_ID or config.MODEL_ID)

    trainer_llm      = llm.bind_tools(TRAINER_TOOLS)
    physio_llm       = llm.bind_tools(PHYSIO_TOOLS)
    recovery_llm     = llm.bind_tools(RECOVERY_TOOLS)
    dietitian_llm    = llm.bind_tools(DIETITIAN_TOOLS)
    psychologist_llm = llm.bind_tools(PSYCHOLOGIST_TOOLS)

    graph = StateGraph(CoachState)

    # ── Supervisor ────────────────────────────────────────────────────────────
    # Detects multi-intent on the first message of a session; no-op mid-session.
    graph.add_node("supervisor", make_supervisor_node(supervisor_llm))

    # ── Context pre-fetch nodes ───────────────────────────────────────────────
    graph.add_node("gather_trainer_context",     _gather_trainer_context)
    graph.add_node("gather_dietitian_context",   _gather_dietitian_context)
    graph.add_node("gather_physio_context",      _gather_physio_context)
    graph.add_node("gather_recovery_context",    _gather_recovery_context)
    graph.add_node("gather_psychologist_context",_gather_psychologist_context)

    # ── Agent nodes ───────────────────────────────────────────────────────────
    graph.add_node("trainer",         _make_agent_node(trainer_llm,      build_trainer_prompt,      query_llm=query_llm))
    graph.add_node("physiotherapist", _make_agent_node(physio_llm,       build_physio_prompt,       query_llm=query_llm))
    graph.add_node("recovery_coach",  _make_agent_node(recovery_llm,     build_recovery_prompt,     query_llm=query_llm))
    graph.add_node("dietitian",       _make_agent_node(dietitian_llm,    build_dietitian_prompt,    query_llm=query_llm))
    graph.add_node("psychologist",    _make_agent_node(psychologist_llm, build_psychologist_prompt, query_llm=query_llm))

    # ── Tool nodes ────────────────────────────────────────────────────────────
    graph.add_node("trainer_tools",      ToolNode(TRAINER_TOOLS,      handle_tool_errors=_format_tool_error))
    graph.add_node("physio_tools",       ToolNode(PHYSIO_TOOLS,        handle_tool_errors=_format_tool_error))
    graph.add_node("recovery_tools",     ToolNode(RECOVERY_TOOLS,      handle_tool_errors=_format_tool_error))
    graph.add_node("dietitian_tools",    ToolNode(DIETITIAN_TOOLS,     handle_tool_errors=_format_tool_error))
    graph.add_node("psychologist_tools", ToolNode(PSYCHOLOGIST_TOOLS,  handle_tool_errors=_format_tool_error))

    # ── Pending-agent relay ───────────────────────────────────────────────────
    # After a primary agent finishes, if the supervisor queued secondary agents,
    # route_pending promotes the next one and routes to its gather node.
    graph.add_node("route_pending", _route_pending)

    # ── Entry edges ───────────────────────────────────────────────────────────
    # START → supervisor → gather node (based on active_agent set by supervisor)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["active_agent"], {
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

    # Each agent routes to its tool node, the pending relay, or END.
    _agent_edges = {"trainer_tools": "trainer_tools", "route_pending": "route_pending", END: END}
    graph.add_conditional_edges("trainer",         _should_continue("trainer_tools"),      _agent_edges)
    _agent_edges = {"physio_tools": "physio_tools", "route_pending": "route_pending", END: END}
    graph.add_conditional_edges("physiotherapist", _should_continue("physio_tools"),       _agent_edges)
    _agent_edges = {"recovery_tools": "recovery_tools", "route_pending": "route_pending", END: END}
    graph.add_conditional_edges("recovery_coach",  _should_continue("recovery_tools"),     _agent_edges)
    _agent_edges = {"dietitian_tools": "dietitian_tools", "route_pending": "route_pending", END: END}
    graph.add_conditional_edges("dietitian",       _should_continue("dietitian_tools"),    _agent_edges)
    _agent_edges = {"psychologist_tools": "psychologist_tools", "route_pending": "route_pending", END: END}
    graph.add_conditional_edges("psychologist",    _should_continue("psychologist_tools"), _agent_edges)

    # Normal tool calls loop back to their owning agent.
    # When a handoff tool returns Command(goto=...), LangGraph overrides these edges.
    graph.add_edge("trainer_tools",      "trainer")
    graph.add_edge("physio_tools",       "physiotherapist")
    graph.add_edge("recovery_tools",     "recovery_coach")
    graph.add_edge("dietitian_tools",    "dietitian")
    graph.add_edge("psychologist_tools", "psychologist")

    # route_pending promotes the next queued agent and routes to its gather node.
    graph.add_conditional_edges("route_pending", _route_from_pending, {
        "trainer":         "gather_trainer_context",
        "physiotherapist": "gather_physio_context",
        "recovery_coach":  "gather_recovery_context",
        "dietitian":       "gather_dietitian_context",
        "psychologist":    "gather_psychologist_context",
    })

    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as checkpointer:
        await checkpointer.setup()
        compiled = graph.compile(checkpointer=checkpointer)

        # Preload the lite Ollama model so the supervisor + RAG query steps
        # on the first message don't pay the cold-start penalty.
        lite_model = config.QUERY_MODEL_ID or config.MODEL_ID
        asyncio.ensure_future(_warmup_ollama(config.OLLAMA_BASE_URL, lite_model))

        yield compiled


async def inject_bot_message_into_thread(text: str) -> None:
    """Store a proactive bot message in the LangGraph conversation thread.

    Called after every scheduled job send so the interactive agent has full
    context when the user replies to a proactive message.
    """
    plain = re.sub(r"<[^>]+>", "", text).strip()
    if not plain:
        return

    thread_cfg = {"configurable": {"thread_id": str(config.TELEGRAM_ALLOWED_USER_ID)}}

    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as checkpointer:
        # Minimal graph — only the state schema matters for aupdate_state.
        g: StateGraph = StateGraph(CoachState)
        g.add_node("_noop", lambda s: {})
        g.add_edge(START, "_noop")
        g.add_edge("_noop", END)
        compiled = g.compile(checkpointer=checkpointer)
        await compiled.aupdate_state(thread_cfg, {"messages": [AIMessage(content=plain)]})

    logger.debug("Stored proactive message in thread %s", config.TELEGRAM_ALLOWED_USER_ID)
