from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CoachState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    athlete_context: str
    active_agent: str  # "trainer" | "physiotherapist" | "recovery_coach" | "dietitian"
    handoff_reason: str | None
    prefetched_context: str  # populated by gather_trainer_context; empty for non-trainer agents
