"""
tests/test_eval.py — Evaluation suite for routing correctness, tool-first compliance,
and hallucination prevention in the FlexLLM multi-agent coaching system.

Test classes
────────────
  TestRoutingEval              — EvalScenario routing with mocked embeddings
  TestHallucinationGuard       — check_hallucination() unit tests
  TestRequiredToolFirst        — Prompt mandates for each agent's first tool
  TestAgentDoesNotHallucinate  — Mocked-LLM detection of skipped tool calls
  TestTracingSetup             — LangSmith tracing activation / no-op behavior

Slow integration eval (requires Ollama + optional LangSmith):
  Run: python tests/eval_runner.py

Markers
───────
  @pytest.mark.slow  — integration tests, skip with `pytest -m "not slow"`
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import _make_agent_node
from src.agent.prompts import (
    build_dietitian_prompt,
    build_physio_prompt,
    build_recovery_prompt,
    build_trainer_prompt,
)
from src.agent.router import route_entry


# ── Evaluation scenarios ───────────────────────────────────────────────────────

@dataclass
class EvalScenario:
    """A single evaluation scenario that exercises one path through the graph."""
    name: str
    user_message: str
    expected_agent: str
    required_first_tool: str   # Tool the agent MUST call before answering
    description: str = ""
    tags: list[str] = field(default_factory=list)


EVAL_SCENARIOS: list[EvalScenario] = [
    EvalScenario(
        name="knee_injury",
        user_message="My right knee is killing me since yesterday's long run — it's 7/10 pain",
        expected_agent="physiotherapist",
        required_first_tool="get_active_injuries",
        description="Injury report must route to physio, who fetches current injury status first",
        tags=["routing", "injury"],
    ),
    EvalScenario(
        name="exhaustion_sleep",
        user_message="I'm completely exhausted and my sleep has been terrible for three nights",
        expected_agent="recovery_coach",
        required_first_tool="get_daily_readiness",
        description="Fatigue/sleep issue must route to recovery coach, who fetches readiness metrics first",
        tags=["routing", "recovery"],
    ),
    EvalScenario(
        name="training_plan_5k",
        user_message="Can you build me a training plan for my first 5K race?",
        expected_agent="trainer",
        required_first_tool="get_onboarding_status",
        description="Training plan request must route to trainer, who checks onboarding status first",
        tags=["routing", "training"],
    ),
    EvalScenario(
        name="calorie_fuelling",
        user_message="How many calories should I eat to fuel my marathon training?",
        expected_agent="dietitian",
        required_first_tool="get_nutrition_profile",
        description="Calorie question must route to dietitian, who fetches nutrition profile first",
        tags=["routing", "nutrition"],
    ),
    EvalScenario(
        name="hrv_drop",
        user_message="My HRV dropped sharply this morning and my resting heart rate is elevated",
        expected_agent="recovery_coach",
        required_first_tool="get_daily_readiness",
        description="HRV alarm must route to recovery coach",
        tags=["routing", "recovery"],
    ),
    EvalScenario(
        name="vdot_pace_question",
        user_message="What pace should I run my easy sessions at based on my VDOT?",
        expected_agent="trainer",
        required_first_tool="get_vdot_paces",
        description="VDOT pace question must route to trainer",
        tags=["routing", "training"],
    ),
    EvalScenario(
        name="plantar_fasciitis",
        user_message="I think I have plantar fasciitis — my heel hurts every morning when I wake up",
        expected_agent="physiotherapist",
        required_first_tool="get_active_injuries",
        description="Plantar fasciitis symptom must route to physio",
        tags=["routing", "injury"],
    ),
    EvalScenario(
        name="macro_muscle_building",
        user_message="What macros should I target for building muscle while maintaining my running?",
        expected_agent="dietitian",
        required_first_tool="get_nutrition_profile",
        description="Macro question must route to dietitian, who fetches profile first",
        tags=["routing", "nutrition"],
    ),
]


# ── Hallucination detector ─────────────────────────────────────────────────────

# Patterns that indicate specific athlete data was cited in a response.
# If these appear without a preceding tool call, the response is likely hallucinated.
_SPECIFIC_DATA_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{1,2}:\d{2}/(?:km|mile)\b"),             # pace: 5:30/km
    re.compile(r"\b\d{3,4}\s*(?:kcal|cal|calories)\b", re.I), # calories: 2800 kcal
    re.compile(r"\b\d+(?:\.\d+)?\s*kg\b", re.I),              # weight: 75 kg
    re.compile(r"\bTSB\s*[=:]\s*[-+]?\d+", re.I),             # TSB: -15
    re.compile(r"\bHRV\s*[=:]\s*\d+", re.I),                  # HRV: 58
    re.compile(r"\bVDOT\s*[=:]\s*\d+", re.I),                 # VDOT: 52
]


@dataclass
class HallucinationResult:
    ok: bool
    reason: str
    suspicious_claims: list[str] = field(default_factory=list)


def check_hallucination(messages: list) -> HallucinationResult:
    """
    Detect whether a final AI response contains specific data claims that were
    never retrieved via a tool call.

    A response is flagged when:
    - It is the last AI message AND has no pending tool_calls (i.e. it's a final answer)
    - No ToolMessage appears in the same turn's messages
    - The response body contains patterns that match specific athlete data (paces,
      calories, TSB, HRV, VDOT, weights)

    Grounded responses (where a tool was called and returned data) are never flagged.
    """
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    if not ai_messages:
        return HallucinationResult(ok=True, reason="no AI response to check")

    final_ai = ai_messages[-1]

    # Intermediate step — the model is about to call a tool, not giving a final answer.
    if getattr(final_ai, "tool_calls", None):
        return HallucinationResult(ok=True, reason="intermediate tool-call step")

    # Tool results are present — response is grounded in retrieved data.
    if tool_messages:
        return HallucinationResult(ok=True, reason="response grounded in tool results")

    # No tool was called — scan for specific data claims.
    content = str(final_ai.content)
    suspicious: list[str] = []
    for pattern in _SPECIFIC_DATA_PATTERNS:
        suspicious.extend(pattern.findall(content))

    if suspicious:
        return HallucinationResult(
            ok=False,
            reason="Response contains specific data claims but no tool was called to retrieve them",
            suspicious_claims=suspicious,
        )

    return HallucinationResult(ok=True, reason="no specific data claims detected")


# ── Message helpers ────────────────────────────────────────────────────────────

def _human(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def _ai(content: str = "", tool_calls=None) -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _ai_with_calls(*names: str) -> AIMessage:
    calls = [{"name": n, "args": {}, "id": f"tc{i}", "type": "tool_call"} for i, n in enumerate(names)]
    return AIMessage(content="", tool_calls=calls)


def _tool(content: str = "result", tool_call_id: str = "tc0") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


# ── Routing mock helpers ───────────────────────────────────────────────────────

_DOMAIN_VECS = {
    "physiotherapist": np.array([1.0, 0.0, 0.0, 0.0]),
    "trainer":         np.array([0.0, 1.0, 0.0, 0.0]),
    "recovery_coach":  np.array([0.0, 0.0, 1.0, 0.0]),
    "dietitian":       np.array([0.0, 0.0, 0.0, 1.0]),
}
_DOMAIN_IDX = {"physiotherapist": 0, "trainer": 1, "recovery_coach": 2, "dietitian": 3}


def _mock_for(expected_agent: str) -> tuple:
    """Return (mock_model, domain_vecs) where the embedding points at expected_agent."""
    vec = np.zeros(4)
    vec[_DOMAIN_IDX[expected_agent]] = 1.01  # noise keeps it closest to target
    model = MagicMock()
    model.embed.side_effect = lambda texts: iter([vec])
    return model, _DOMAIN_VECS


# ══════════════════════════════════════════════════════════════════════════════
# TestRoutingEval
# ══════════════════════════════════════════════════════════════════════════════

class TestRoutingEval:
    """Routing correctness for all EvalScenarios (embeddings mocked)."""

    @pytest.mark.parametrize("scenario", EVAL_SCENARIOS, ids=[s.name for s in EVAL_SCENARIOS])
    def test_routes_to_expected_agent(self, scenario: EvalScenario):
        """route_entry must return the expected agent for each scenario."""
        mock_model, domain_vecs = _mock_for(scenario.expected_agent)
        state = {"messages": [_human(scenario.user_message)], "active_agent": ""}
        with patch("src.agent.router._load", return_value=(mock_model, domain_vecs)):
            result = route_entry(state)
        assert result == scenario.expected_agent, (
            f"[{scenario.name}] Expected '{scenario.expected_agent}', got '{result}'. "
            f"Message: '{scenario.user_message}'"
        )

    def test_injury_message_never_routes_to_trainer(self):
        injury_scenarios = [s for s in EVAL_SCENARIOS if "injury" in s.tags]
        for scenario in injury_scenarios:
            mock_model, domain_vecs = _mock_for("physiotherapist")
            state = {"messages": [_human(scenario.user_message)], "active_agent": ""}
            with patch("src.agent.router._load", return_value=(mock_model, domain_vecs)):
                assert route_entry(state) != "trainer", f"[{scenario.name}] injury routed to trainer"

    def test_active_agent_bypasses_routing_for_all_scenarios(self):
        """Once active_agent is set the router must not re-embed the message."""
        for scenario in EVAL_SCENARIOS:
            state = {"messages": [_human(scenario.user_message)], "active_agent": "recovery_coach"}
            assert route_entry(state) == "recovery_coach"

    @pytest.mark.parametrize("agent", ["physiotherapist", "trainer", "recovery_coach", "dietitian"])
    def test_each_agent_reachable_via_routing(self, agent: str):
        """Every agent must be reachable from the entry router."""
        mock_model, domain_vecs = _mock_for(agent)
        state = {"messages": [_human("test message")], "active_agent": ""}
        with patch("src.agent.router._load", return_value=(mock_model, domain_vecs)):
            assert route_entry(state) == agent


# ══════════════════════════════════════════════════════════════════════════════
# TestHallucinationGuard
# ══════════════════════════════════════════════════════════════════════════════

class TestHallucinationGuard:
    """Unit tests for check_hallucination()."""

    def test_empty_messages_returns_ok(self):
        assert check_hallucination([]).ok

    def test_greeting_without_data_is_ok(self):
        result = check_hallucination([_human("hi"), _ai("Hello! How can I help you today?")])
        assert result.ok

    def test_intermediate_tool_call_not_flagged(self):
        result = check_hallucination([_human("my TSB?"), _ai_with_calls("get_daily_readiness")])
        assert result.ok

    def test_grounded_response_after_tool_is_ok(self):
        messages = [
            _human("what is my TSB?"),
            _ai_with_calls("get_daily_readiness"),
            _tool('{"tsb": -8, "hrv": 62, "sleep_hours": 7.5}', "tc0"),
            _ai("Your TSB is -8 and HRV is 62 — you are slightly fatigued."),
        ]
        assert check_hallucination(messages).ok

    def test_pace_claim_without_tool_is_flagged(self):
        messages = [_human("what pace?"), _ai("Run your easy sessions at 5:45/km.")]
        result = check_hallucination(messages)
        assert not result.ok
        assert result.suspicious_claims

    def test_calorie_claim_without_tool_is_flagged(self):
        messages = [_human("calories?"), _ai("You should eat 2800 kcal on hard training days.")]
        result = check_hallucination(messages)
        assert not result.ok

    def test_tsb_claim_without_tool_is_flagged(self):
        messages = [_human("recovered?"), _ai("Your TSB: -15 — you are still accumulating fatigue.")]
        result = check_hallucination(messages)
        assert not result.ok

    def test_hrv_claim_without_tool_is_flagged(self):
        messages = [_human("HRV?"), _ai("Your HRV: 58, which is below your 7-day baseline.")]
        result = check_hallucination(messages)
        assert not result.ok

    def test_weight_claim_without_tool_is_flagged(self):
        messages = [_human("protein?"), _ai("Based on your 75 kg body weight, eat 135g protein.")]
        result = check_hallucination(messages)
        assert not result.ok

    def test_vdot_claim_without_tool_is_flagged(self):
        messages = [_human("vdot?"), _ai("Your VDOT: 52, so your threshold pace is 4:15/km.")]
        result = check_hallucination(messages)
        assert not result.ok

    def test_any_tool_result_suppresses_numeric_flag(self):
        """Even a single ToolMessage in the turn means the response is considered grounded."""
        messages = [
            _human("how much protein?"),
            _ai_with_calls("get_nutrition_profile"),
            _tool('{"weight_kg": 75, "protein_g": 135}', "tc0"),
            _ai("Based on your 75 kg body weight, aim for 135g protein."),
        ]
        assert check_hallucination(messages).ok

    def test_result_contains_suspicious_claims_list(self):
        messages = [_human("pace?"), _ai("Run at 5:30/km with VDOT: 48.")]
        result = check_hallucination(messages)
        assert not result.ok
        assert len(result.suspicious_claims) >= 1

    def test_generic_text_without_data_patterns_is_ok(self):
        messages = [
            _human("what should I focus on?"),
            _ai("You should focus on building your aerobic base before adding speed work."),
        ]
        assert check_hallucination(messages).ok


# ══════════════════════════════════════════════════════════════════════════════
# TestRequiredToolFirst
# ══════════════════════════════════════════════════════════════════════════════

class TestRequiredToolFirst:
    """Verify that each agent's system prompt explicitly mandates its required first tool."""

    @pytest.mark.parametrize("scenario", EVAL_SCENARIOS, ids=[s.name for s in EVAL_SCENARIOS])
    def test_prompt_mentions_required_tool(self, scenario: EvalScenario):
        """Agent prompt must name the tool the scenario expects to be called first."""
        builders = {
            "trainer":         build_trainer_prompt,
            "physiotherapist": build_physio_prompt,
            "recovery_coach":  build_recovery_prompt,
            "dietitian":       build_dietitian_prompt,
        }
        prompt = builders[scenario.expected_agent]("")
        assert scenario.required_first_tool in prompt, (
            f"[{scenario.name}] Prompt for '{scenario.expected_agent}' does not mention "
            f"required tool '{scenario.required_first_tool}'"
        )

    def test_trainer_calls_onboarding_status_first(self):
        prompt = build_trainer_prompt("")
        assert "get_onboarding_status" in prompt
        # Prompt must say FIRST, not just mention the tool anywhere
        assert "FIRST" in prompt

    def test_recovery_coach_calls_readiness_on_activation(self):
        prompt = build_recovery_prompt("")
        assert "get_daily_readiness" in prompt
        assert "ON ACTIVATION" in prompt

    def test_physio_calls_active_injuries_on_activation(self):
        prompt = build_physio_prompt("")
        assert "get_active_injuries" in prompt
        assert "ON ACTIVATION" in prompt

    def test_dietitian_calls_nutrition_profile_on_activation(self):
        prompt = build_dietitian_prompt("")
        assert "get_nutrition_profile" in prompt
        assert "ON ACTIVATION" in prompt

    def test_trainer_prompt_requires_readiness_before_session(self):
        prompt = build_trainer_prompt("")
        # Trainer must check readiness at every session start (STEP 2)
        assert "get_daily_readiness" in prompt
        assert "STEP 2" in prompt

    def test_all_prompts_forbid_guessing_data(self):
        for builder in [build_trainer_prompt, build_physio_prompt,
                        build_recovery_prompt, build_dietitian_prompt]:
            prompt = builder("")
            # "Never guess" is the key anti-hallucination instruction
            assert "Never guess" in prompt or "never guess" in prompt.lower(), (
                f"{builder.__name__} is missing the 'Never guess' anti-hallucination rule"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TestAgentDoesNotHallucinate
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentDoesNotHallucinate:
    """
    Verify that when a mocked LLM skips tool calls and responds with specific
    data directly, check_hallucination() catches it.

    These tests simulate the failure mode: the LLM ignores the 'fetch first'
    instruction and invents athlete data. The detector must flag these cases
    so they can be caught in a CI evaluation pipeline.
    """

    def _run_agent(self, mock_response: AIMessage, prompt_builder) -> list:
        """Run one agent-node step with a mocked LLM, return resulting messages."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        node = _make_agent_node(mock_llm, prompt_builder)
        state = {
            "messages": [_human("give me a response")],
            "athlete_context": "runner",
            "active_agent": "trainer",
        }
        return [_human("give me a response")] + node(state)["messages"]

    def test_trainer_pace_without_tool_is_caught(self):
        direct = _ai("Your easy pace should be 5:45/km based on your current fitness.")
        msgs = self._run_agent(direct, build_trainer_prompt)
        assert not check_hallucination(msgs).ok

    def test_trainer_tool_call_passes(self):
        tool_call = _ai_with_calls("get_current_workout_plan")
        msgs = self._run_agent(tool_call, build_trainer_prompt)
        assert check_hallucination(msgs).ok

    def test_recovery_tsb_without_tool_is_caught(self):
        direct = _ai("Your TSB: -22 and HRV: 55 — you need complete rest today.")
        msgs = self._run_agent(direct, build_recovery_prompt)
        assert not check_hallucination(msgs).ok

    def test_recovery_tool_call_passes(self):
        tool_call = _ai_with_calls("get_daily_readiness")
        msgs = self._run_agent(tool_call, build_recovery_prompt)
        assert check_hallucination(msgs).ok

    def test_dietitian_calorie_without_tool_is_caught(self):
        direct = _ai("Aim for 2600 kcal with 160g protein and 320g carbohydrate today.")
        msgs = self._run_agent(direct, build_dietitian_prompt)
        assert not check_hallucination(msgs).ok

    def test_dietitian_tool_call_passes(self):
        tool_call = _ai_with_calls("get_nutrition_profile")
        msgs = self._run_agent(tool_call, build_dietitian_prompt)
        assert check_hallucination(msgs).ok

    def test_physio_direct_severity_without_tool_is_caught(self):
        direct = _ai("Based on your 75 kg body weight, reduce running by 50%.")
        msgs = self._run_agent(direct, build_physio_prompt)
        assert not check_hallucination(msgs).ok

    def test_physio_tool_call_passes(self):
        tool_call = _ai_with_calls("get_active_injuries")
        msgs = self._run_agent(tool_call, build_physio_prompt)
        assert check_hallucination(msgs).ok


# ══════════════════════════════════════════════════════════════════════════════
# TestTracingSetup
# ══════════════════════════════════════════════════════════════════════════════

class TestTracingSetup:
    """Unit tests for src/tracing.py — no external services required."""

    def test_returns_false_when_no_api_key(self):
        from src.tracing import setup_tracing
        with patch("src.tracing.config") as mock_cfg:
            mock_cfg.LANGSMITH_API_KEY = None
            result = setup_tracing()
        assert result is False

    def test_returns_true_when_api_key_present(self):
        from src.tracing import setup_tracing
        with patch.dict(os.environ, {}, clear=False), \
             patch("src.tracing.config") as mock_cfg:
            mock_cfg.LANGSMITH_API_KEY = "ls__fake_key"
            mock_cfg.LANGCHAIN_PROJECT = "test-project"
            mock_cfg.ENVIRONMENT = "local"
            result = setup_tracing()
        assert result is True

    def test_sets_os_environ_tracing_vars(self):
        from src.tracing import setup_tracing
        with patch.dict(os.environ, {}, clear=False), \
             patch("src.tracing.config") as mock_cfg:
            mock_cfg.LANGSMITH_API_KEY = "ls__fake_key"
            mock_cfg.LANGCHAIN_PROJECT = "flexllm-test"
            mock_cfg.ENVIRONMENT = "staging"
            setup_tracing()
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGSMITH_API_KEY") == "ls__fake_key"
            assert os.environ.get("LANGCHAIN_PROJECT") == "flexllm-test"
            assert os.environ.get("LANGCHAIN_CALLBACKS_BACKGROUND") == "true"
            assert os.environ.get("LANGCHAIN_TAGS") == "staging"

    def test_no_env_vars_set_when_key_missing(self):
        from src.tracing import setup_tracing
        env_before = os.environ.get("LANGCHAIN_TRACING_V2")
        with patch("src.tracing.config") as mock_cfg:
            mock_cfg.LANGSMITH_API_KEY = None
            setup_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") == env_before

    def test_logs_enabled_when_key_present(self):
        from src.tracing import setup_tracing
        with patch.dict(os.environ, {}, clear=False), \
             patch("src.tracing.config") as mock_cfg, \
             patch("src.tracing.logger") as mock_log:
            mock_cfg.LANGSMITH_API_KEY = "ls__fake_key"
            mock_cfg.LANGCHAIN_PROJECT = "flexllm-test"
            mock_cfg.ENVIRONMENT = "prod"
            setup_tracing()
        mock_log.info.assert_called_once()
        call_args = mock_log.info.call_args[0]
        assert "enabled" in call_args[0]

    def test_logs_disabled_when_key_missing(self):
        from src.tracing import setup_tracing
        with patch("src.tracing.config") as mock_cfg, \
             patch("src.tracing.logger") as mock_log:
            mock_cfg.LANGSMITH_API_KEY = None
            setup_tracing()
        mock_log.info.assert_called_once()
        call_args = mock_log.info.call_args[0]
        assert "disabled" in call_args[0]

    def test_project_override_takes_precedence(self):
        from src.tracing import setup_tracing
        with patch.dict(os.environ, {}, clear=False), \
             patch("src.tracing.config") as mock_cfg:
            mock_cfg.LANGSMITH_API_KEY = "ls__fake_key"
            mock_cfg.LANGCHAIN_PROJECT = "flexllm-local"
            mock_cfg.ENVIRONMENT = "local"
            setup_tracing(project="flexllm-test")
            assert os.environ.get("LANGCHAIN_PROJECT") == "flexllm-test"

    def test_config_project_used_when_no_override(self):
        from src.tracing import setup_tracing
        with patch.dict(os.environ, {}, clear=False), \
             patch("src.tracing.config") as mock_cfg:
            mock_cfg.LANGSMITH_API_KEY = "ls__fake_key"
            mock_cfg.LANGCHAIN_PROJECT = "flexllm-local"
            mock_cfg.ENVIRONMENT = "local"
            setup_tracing()
            assert os.environ.get("LANGCHAIN_PROJECT") == "flexllm-local"
