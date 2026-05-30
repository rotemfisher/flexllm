from contextlib import contextmanager

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import config
from src.models.agent_state import CoachState
from src.agent.trainer import TRAINER_TOOLS
from src.agent.physiotherapist import PHYSIO_TOOLS
from src.agent.recovery_coach import RECOVERY_TOOLS
from src.agent.dietitian import DIETITIAN_TOOLS
from src.agent.prompts import (
    build_trainer_prompt,
    build_physio_prompt,
    build_recovery_prompt,
    build_dietitian_prompt,
)
from src.agent.router import route_entry

# ── Agent node factory ────────────────────────────────────────────────────────

def _make_agent_node(llm_with_tools, prompt_builder):
    def call_model(state: CoachState) -> dict:
        system_prompt = prompt_builder(state.get("athlete_context", ""))
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    return call_model


def _should_continue(tools_node_name: str):
    def router(state: CoachState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return tools_node_name
        return END
    return router


# ── Graph builder ─────────────────────────────────────────────────────────────

@contextmanager
def build_multi_agent_graph():
    llm = ChatOllama(model=config.MODEL_ID, temperature=0)

    trainer_llm   = llm.bind_tools(TRAINER_TOOLS)
    physio_llm    = llm.bind_tools(PHYSIO_TOOLS)
    recovery_llm  = llm.bind_tools(RECOVERY_TOOLS)
    dietitian_llm = llm.bind_tools(DIETITIAN_TOOLS)

    graph = StateGraph(CoachState)

    # Agent nodes
    graph.add_node("trainer",         _make_agent_node(trainer_llm,   build_trainer_prompt))
    graph.add_node("physiotherapist", _make_agent_node(physio_llm,    build_physio_prompt))
    graph.add_node("recovery_coach",  _make_agent_node(recovery_llm,  build_recovery_prompt))
    graph.add_node("dietitian",       _make_agent_node(dietitian_llm, build_dietitian_prompt))

    # Tool nodes — separate per agent so each only exposes its own tools
    graph.add_node("trainer_tools",   ToolNode(TRAINER_TOOLS))
    graph.add_node("physio_tools",    ToolNode(PHYSIO_TOOLS))
    graph.add_node("recovery_tools",  ToolNode(RECOVERY_TOOLS))
    graph.add_node("dietitian_tools", ToolNode(DIETITIAN_TOOLS))

    # Entry: semantic router at START (embedding cosine similarity)
    graph.add_conditional_edges(START, route_entry, {
        "trainer":         "trainer",
        "physiotherapist": "physiotherapist",
        "recovery_coach":  "recovery_coach",
        "dietitian":       "dietitian",
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

    # Normal tool calls loop back to their owning agent.
    # When a handoff tool returns Command(goto=...), LangGraph overrides these edges.
    graph.add_edge("trainer_tools",   "trainer")
    graph.add_edge("physio_tools",    "physiotherapist")
    graph.add_edge("recovery_tools",  "recovery_coach")
    graph.add_edge("dietitian_tools", "dietitian")

    with SqliteSaver.from_conn_string(config.DB_PATH) as checkpointer:
        yield graph.compile(checkpointer=checkpointer)
