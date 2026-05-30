from langchain_core.tools import tool
from langgraph.types import Command


# ── FROM TRAINER ──────────────────────────────────────────────────────────────

@tool
def transfer_to_physiotherapist(reason: str) -> Command:
    """Transfer to the Physiotherapist. Use when the athlete reports pain, injury,
    or movement contraindications that require specialist assessment."""
    return Command(goto="physiotherapist", update={"active_agent": "physiotherapist"})


@tool
def transfer_to_recovery_coach(reason: str) -> Command:
    """Transfer to the Recovery Coach. Use when TSB < -20, HRV is critically low,
    sleep is under 5h, or the athlete is asking about fatigue or load management."""
    return Command(goto="recovery_coach", update={"active_agent": "recovery_coach"})


@tool
def transfer_to_dietitian(reason: str) -> Command:
    """Transfer to the Dietitian. Use when the athlete asks about nutrition,
    meal planning, macros, weight management, or fuelling strategy."""
    return Command(goto="dietitian", update={"active_agent": "dietitian"})


# ── FROM PHYSIOTHERAPIST ──────────────────────────────────────────────────────

@tool
def physio_transfer_to_trainer(reason: str) -> Command:
    """Return control to the Trainer. Use after the injury has been addressed
    and it is safe to resume or modify training — include the return-to-train
    protocol in the reason so the Trainer can rebuild the plan accordingly."""
    return Command(goto="trainer", update={"active_agent": "trainer"})


@tool
def physio_transfer_to_recovery_coach(reason: str) -> Command:
    """Transfer to the Recovery Coach. Use when fatigue or accumulated load
    appears to be the root cause of the injury or is slowing recovery."""
    return Command(goto="recovery_coach", update={"active_agent": "recovery_coach"})


@tool
def physio_transfer_to_dietitian(reason: str) -> Command:
    """Transfer to the Dietitian. Use when the injury management requires
    dietary support (collagen synthesis, anti-inflammatory nutrition)."""
    return Command(goto="dietitian", update={"active_agent": "dietitian"})


# ── FROM RECOVERY COACH ───────────────────────────────────────────────────────

@tool
def recovery_transfer_to_trainer(reason: str) -> Command:
    """Return to the Trainer. Use after the readiness assessment is complete
    and the appropriate session intensity or modification has been determined."""
    return Command(goto="trainer", update={"active_agent": "trainer"})


@tool
def recovery_transfer_to_physiotherapist(reason: str) -> Command:
    """Transfer to the Physiotherapist. Use when fatigue symptoms may indicate
    an underlying injury rather than normal training stress."""
    return Command(goto="physiotherapist", update={"active_agent": "physiotherapist"})


@tool
def recovery_transfer_to_dietitian(reason: str) -> Command:
    """Transfer to the Dietitian. Use when poor recovery appears driven by
    under-fuelling, caloric deficit, or inadequate sleep nutrition."""
    return Command(goto="dietitian", update={"active_agent": "dietitian"})


# ── FROM DIETITIAN ────────────────────────────────────────────────────────────

@tool
def dietitian_transfer_to_trainer(reason: str) -> Command:
    """Return to the Trainer. Use after the nutrition plan is set and the
    athlete has questions about training, paces, or workout planning."""
    return Command(goto="trainer", update={"active_agent": "trainer"})


@tool
def dietitian_transfer_to_physiotherapist(reason: str) -> Command:
    """Transfer to the Physiotherapist. Use when a dietary topic intersects
    with injury (e.g. bone stress, tendon health, anti-inflammatory protocol)."""
    return Command(goto="physiotherapist", update={"active_agent": "physiotherapist"})


@tool
def dietitian_transfer_to_recovery_coach(reason: str) -> Command:
    """Transfer to the Recovery Coach. Use when nutrition questions relate to
    sleep quality, HRV, or overall recovery capacity."""
    return Command(goto="recovery_coach", update={"active_agent": "recovery_coach"})
