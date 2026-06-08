from contextlib import asynccontextmanager
import logging

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

# Tools the trainer MUST call on the very first response of every session.
_TRAINER_STARTUP_TOOLS = frozenset({
    "check_upcoming_race_or_test",
    "get_onboarding_status",
    "get_daily_readiness",
    "get_current_workout_plan",
    "get_recent_workouts",
    "get_vdot_paces",   # required to have real pace zones before building any plan
})

# Tools the dietitian MUST call immediately on fresh-handoff activation (before any text).
_DIETITIAN_HANDOFF_STARTUP_TOOLS = frozenset({
    "get_nutrition_profile",
    "get_daily_readiness",
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


# ── RAG auto-injection ────────────────────────────────────────────────────────

def _build_rag_queries(human_message: str, athlete_context: str) -> list[str]:
    """Derive 2–4 targeted Qdrant queries from the user message and athlete profile.

    One generic query (the raw message) plus goal-specific queries derived from
    the athlete context give the model directly relevant book passages instead of
    generic results from a vague user sentence.
    """
    ctx = athlete_context.lower()
    queries: list[str] = []

    if human_message.strip():
        queries.append(human_message)

    level = (
        "beginner" if "beginner" in ctx else
        "intermediate" if "intermediate" in ctx else
        "advanced"
    )

    # Running goal
    if any(k in ctx for k in ["10k", "5k", "run", "marathon", "half marathon", "pace", "jog"]):
        dist = (
            "marathon" if "marathon" in ctx else
            "10k" if "10k" in ctx else
            "5k" if "5k" in ctx else
            "running"
        )
        queries.append(f"{level} {dist} training periodization base phase aerobic development VDOT")

    # Strength / body composition goal
    if any(k in ctx for k in ["muscle", "strength", "shredded", "hypertrophy", "muscular", "lean"]):
        queries.append(f"{level} hypertrophy strength training concurrent endurance periodization")

    # Fat loss goal
    if any(k in ctx for k in ["fat", "shredded", "lean", "weight loss", "body comp"]):
        queries.append("concurrent training fat loss body composition performance nutrition")

    return queries[:4]


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
            return ""
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
                     handoff_startup_tools: frozenset = frozenset()):
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
                rag_queries = _build_rag_queries(last_human_content, athlete_ctx)
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
                rag_queries = _build_rag_queries(last_human_content, athlete_ctx)
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
                    "You have just been activated. "
                    "IMMEDIATELY call your required startup tools. "
                    "Output ONLY the tool calls — no text whatsoever."
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

    trainer_llm      = llm.bind_tools(TRAINER_TOOLS)
    physio_llm       = llm.bind_tools(PHYSIO_TOOLS)
    recovery_llm     = llm.bind_tools(RECOVERY_TOOLS)
    dietitian_llm    = llm.bind_tools(DIETITIAN_TOOLS)
    psychologist_llm = llm.bind_tools(PSYCHOLOGIST_TOOLS)

    graph = StateGraph(CoachState)

    # Agent nodes
    graph.add_node("trainer",         _make_agent_node(trainer_llm,      build_trainer_prompt, startup_tools=_TRAINER_STARTUP_TOOLS))
    graph.add_node("physiotherapist", _make_agent_node(physio_llm,       build_physio_prompt))
    graph.add_node("recovery_coach",  _make_agent_node(recovery_llm,     build_recovery_prompt))
    graph.add_node("dietitian",       _make_agent_node(dietitian_llm,    build_dietitian_prompt, handoff_startup_tools=_DIETITIAN_HANDOFF_STARTUP_TOOLS))
    graph.add_node("psychologist",    _make_agent_node(psychologist_llm, build_psychologist_prompt))

    # Tool nodes — separate per agent so each only exposes its own tools
    graph.add_node("trainer_tools",      ToolNode(TRAINER_TOOLS))
    graph.add_node("physio_tools",       ToolNode(PHYSIO_TOOLS))
    graph.add_node("recovery_tools",     ToolNode(RECOVERY_TOOLS))
    graph.add_node("dietitian_tools",    ToolNode(DIETITIAN_TOOLS))
    graph.add_node("psychologist_tools", ToolNode(PSYCHOLOGIST_TOOLS))

    # Entry: semantic router at START (embedding cosine similarity)
    graph.add_conditional_edges(START, route_entry, {
        "trainer":         "trainer",
        "physiotherapist": "physiotherapist",
        "recovery_coach":  "recovery_coach",
        "dietitian":       "dietitian",
        "psychologist":    "psychologist",
    })

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
