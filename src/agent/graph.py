from contextlib import asynccontextmanager

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

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

# Keep at most this many messages in the context window sent to the LLM.
# Enough to cover several back-and-forth exchanges while staying well within
# Qwen 30B's NUM_CTX=16384 token budget.
_MAX_HISTORY_MESSAGES = 30


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
    discard the domain calls and keep only the handoff.

    LLMs occasionally ignore the 'handoff must be the only tool call'
    instruction. Keeping both would send a ToolMessage back to the ToolNode
    alongside a Command, which LangGraph handles unpredictably.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls or len(tool_calls) <= 1:
        return response
    handoffs = [tc for tc in tool_calls if tc["name"] in _HANDOFF_TOOL_NAMES]
    if not handoffs:
        return response  # Multiple domain calls — fine, no conflict.
    # Conflict detected: keep only the first handoff, drop domain calls.
    return response.model_copy(update={"tool_calls": handoffs[:1]})


# ── Agent node factory ────────────────────────────────────────────────────────

def _make_agent_node(llm_with_tools, prompt_builder):
    def call_model(state: CoachState) -> dict:
        system_prompt = prompt_builder(state.get("athlete_context", ""))
        history = _trim_messages(list(state["messages"]))
        reason = state.get("handoff_reason")
        if reason:
            system_prompt += f"\n\nCRITICAL CONTEXT: You have just received control of the session. The previous agent's handoff note: '{reason}'"
        messages = [SystemMessage(content=system_prompt)] + history
        response = llm_with_tools.invoke(messages)
        response = _resolve_tool_conflicts(response)

        has_tool_calls = bool(getattr(response, "tool_calls", None))
        content = response.content if isinstance(response.content, str) else ""
        is_empty = not content.strip()

        # Detect "announcement without action": model wrote "let me fetch X" after
        # receiving tool results but did not actually call the tool.
        last_hist = history[-1] if history else None
        is_mid_sequence = isinstance(last_hist, ToolMessage)
        _ANNOUNCEMENT_PHRASES = (
            "let me fetch", "let me check", "let me get", "let me look",
            "i will now", "i'll now", "i need to fetch", "i need to check",
            "let me retrieve", "let me pull", "let me gather",
        )
        is_announcement = (
            is_mid_sequence
            and not has_tool_calls
            and not is_empty
            and any(p in content.lower() for p in _ANNOUNCEMENT_PHRASES)
        )

        if is_empty or is_announcement:
            reason_tag = "empty response" if is_empty else "announcement without tool call"
            logger.warning("Model produced %s — retrying with nudge", reason_tag)
            nudge = messages + [
                response,
                HumanMessage(content=(
                    "Stop announcing what you will do — call the required tool right now. "
                    "Do not write any text until all needed tool calls are complete."
                )),
            ]
            response = llm_with_tools.invoke(nudge)
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
    llm = ChatOllama(model=config.MODEL_ID, temperature=0, base_url=config.OLLAMA_BASE_URL, num_ctx=32768)

    trainer_llm      = llm.bind_tools(TRAINER_TOOLS)
    physio_llm       = llm.bind_tools(PHYSIO_TOOLS)
    recovery_llm     = llm.bind_tools(RECOVERY_TOOLS)
    dietitian_llm    = llm.bind_tools(DIETITIAN_TOOLS)
    psychologist_llm = llm.bind_tools(PSYCHOLOGIST_TOOLS)

    graph = StateGraph(CoachState)

    # Agent nodes
    graph.add_node("trainer",         _make_agent_node(trainer_llm,      build_trainer_prompt))
    graph.add_node("physiotherapist", _make_agent_node(physio_llm,       build_physio_prompt))
    graph.add_node("recovery_coach",  _make_agent_node(recovery_llm,     build_recovery_prompt))
    graph.add_node("dietitian",       _make_agent_node(dietitian_llm,    build_dietitian_prompt))
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

    async with AsyncSqliteSaver.from_conn_string(str(config.DB_PATH)) as checkpointer:
        yield graph.compile(checkpointer=checkpointer)
