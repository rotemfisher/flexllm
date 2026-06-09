"""
tests/test_agent.py — Unit tests for src/agent/ modules.

Covers:
  - router.py:   _cosine, route_entry
  - handoffs.py: all four transfer tools
  - graph.py:    _trim_messages, _resolve_tool_conflicts, _should_continue, _make_agent_node
  - memory.py:   _week_start, _messages_to_text, SummaryStore, generate_*, save_session_summary
  - prompts.py:  build_*_prompt
"""
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from src.agent.graph import (
    _make_agent_node,
    _resolve_tool_conflicts,
    _should_continue,
    _trim_messages,
)
from src.agent.handoffs import (
    dietitian_transfer,
    physio_transfer,
    psychologist_transfer,
    recovery_transfer,
    trainer_transfer,
)
from src.agent.memory import (
    SummaryStore,
    _messages_to_text,
    _week_start,
    generate_daily_summary,
    generate_weekly_summary,
    save_session_summary,
)
from src.agent.prompts import (
    build_dietitian_prompt,
    build_physio_prompt,
    build_psychologist_prompt,
    build_recovery_prompt,
    build_trainer_prompt,
)
from src.agent.router import _cosine, route_entry


# ── Message helpers ────────────────────────────────────────────────────────────

def _human(content=""):
    return HumanMessage(content=content)


def _ai(content="", tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls or [])


def _tool(content="result", tool_call_id="tc0"):
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def _ai_with_calls(*names):
    calls = [
        {"name": n, "args": {}, "id": f"tc{i}", "type": "tool_call"}
        for i, n in enumerate(names)
    ]
    return AIMessage(content="", tool_calls=calls)


# ══════════════════════════════════════════════════════════════════════════════
# graph._trim_messages
# ══════════════════════════════════════════════════════════════════════════════

class TestTrimMessages:
    def test_fewer_than_max_returns_unchanged(self):
        msgs = [_human("q"), _ai("a")]
        assert _trim_messages(msgs, max_messages=10) == msgs

    def test_exactly_max_returns_unchanged(self):
        msgs = [_human(f"q{i}") for i in range(5)]
        result = _trim_messages(msgs, max_messages=5)
        assert result == msgs

    def test_more_than_max_trims_to_max(self):
        msgs = [_human(f"q{i}") for i in range(10)]
        result = _trim_messages(msgs, max_messages=5)
        assert len(result) == 5
        assert result == msgs[-5:]

    def test_strips_orphaned_tool_message_at_head_after_trim(self):
        # After trimming to 2: [Tool, AI] → ToolMessage stripped → [AI]
        msgs = [_ai("setup"), _tool("tool result"), _ai("final answer")]
        result = _trim_messages(msgs, max_messages=2)
        assert len(result) == 1
        assert result[0].content == "final answer"

    def test_multiple_leading_tool_messages_stripped(self):
        # After trimming to 3: [Tool, Tool, AI] → all Tools stripped → [AI]
        msgs = [_ai("start"), _tool("r1", "tc1"), _tool("r2", "tc2"), _ai("final")]
        result = _trim_messages(msgs, max_messages=3)
        assert len(result) == 1
        assert result[0].content == "final"

    def test_empty_list_returns_empty(self):
        assert _trim_messages([], max_messages=10) == []

    def test_does_not_strip_tool_message_when_no_trim_needed(self):
        # [ToolMessage] fits in max — returned as-is (no stripping unless trimming)
        msgs = [_tool("result")]
        result = _trim_messages(msgs, max_messages=5)
        assert len(result) == 1


# ══════════════════════════════════════════════════════════════════════════════
# graph._resolve_tool_conflicts
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveToolConflicts:
    def test_no_tool_calls_returns_same_object(self):
        msg = _ai("plain text")
        assert _resolve_tool_conflicts(msg) is msg

    def test_single_domain_tool_returns_unchanged(self):
        msg = _ai_with_calls("get_recent_workouts")
        result = _resolve_tool_conflicts(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_recent_workouts"

    def test_multiple_domain_tools_no_conflict(self):
        msg = _ai_with_calls("get_recent_workouts", "get_vdot_paces")
        result = _resolve_tool_conflicts(msg)
        assert len(result.tool_calls) == 2

    def test_single_handoff_returns_unchanged(self):
        msg = _ai_with_calls("trainer_transfer")
        result = _resolve_tool_conflicts(msg)
        assert result.tool_calls[0]["name"] == "trainer_transfer"

    def test_handoff_followed_by_domain_keeps_only_handoff(self):
        msg = _ai_with_calls("trainer_transfer", "get_recent_workouts")
        result = _resolve_tool_conflicts(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "trainer_transfer"

    def test_domain_followed_by_handoff_keeps_only_handoff(self):
        msg = _ai_with_calls("get_recent_workouts", "physio_transfer")
        result = _resolve_tool_conflicts(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "physio_transfer"

    def test_multiple_handoffs_keeps_first(self):
        msg = _ai_with_calls("trainer_transfer", "physio_transfer")
        result = _resolve_tool_conflicts(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "trainer_transfer"


# ══════════════════════════════════════════════════════════════════════════════
# graph._should_continue
# ══════════════════════════════════════════════════════════════════════════════

class TestShouldContinue:
    def test_no_tool_calls_returns_end(self):
        router = _should_continue("trainer_tools")
        state = {"messages": [_ai("plain answer")]}
        assert router(state) == END

    def test_empty_tool_calls_returns_end(self):
        router = _should_continue("trainer_tools")
        state = {"messages": [_ai("answer", tool_calls=[])]}
        assert router(state) == END

    def test_with_tool_calls_returns_node_name(self):
        router = _should_continue("trainer_tools")
        state = {"messages": [_ai_with_calls("get_recent_workouts")]}
        assert router(state) == "trainer_tools"

    def test_node_name_propagated(self):
        router = _should_continue("physio_tools")
        state = {"messages": [_ai_with_calls("log_injury")]}
        assert router(state) == "physio_tools"


# ══════════════════════════════════════════════════════════════════════════════
# graph._make_agent_node
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeAgentNode:
    def _state(self, handoff_reason=None, athlete_context="runner"):
        s = {
            "messages": [_human("How do I train?")],
            "athlete_context": athlete_context,
            "active_agent": "trainer",
        }
        if handoff_reason is not None:
            s["handoff_reason"] = handoff_reason
        return s

    def test_returns_messages_and_clears_handoff_reason(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai("Here is your plan.")
        node = _make_agent_node(mock_llm, lambda ctx: "System prompt")
        result = node(self._state())
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["handoff_reason"] is None

    def test_handoff_reason_injected_into_system_message(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai("Noted.")
        node = _make_agent_node(mock_llm, lambda ctx: "Base prompt")
        node(self._state(handoff_reason="athlete reported knee pain"))
        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "knee pain" in system_content
        assert "CRITICAL CONTEXT" in system_content

    def test_no_handoff_reason_omits_critical_context(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai("OK.")
        node = _make_agent_node(mock_llm, lambda ctx: "Base prompt")
        node(self._state())
        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        assert "CRITICAL CONTEXT" not in system_content

    def test_athlete_context_forwarded_to_prompt_builder(self):
        captured = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _ai("OK.")
        node = _make_agent_node(mock_llm, lambda ctx: captured.append(ctx) or "prompt")
        node(self._state(athlete_context="elite marathon runner"))
        assert captured == ["elite marathon runner"]

    def test_llm_response_conflict_resolved_before_returning(self):
        # LLM returns handoff + domain in same response → resolved to handoff only
        conflicted = _ai_with_calls("trainer_transfer", "get_recent_workouts")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = conflicted
        node = _make_agent_node(mock_llm, lambda ctx: "prompt")
        result = node(self._state())
        assert len(result["messages"][0].tool_calls) == 1
        assert result["messages"][0].tool_calls[0]["name"] == "trainer_transfer"


# ══════════════════════════════════════════════════════════════════════════════
# handoffs
# ══════════════════════════════════════════════════════════════════════════════

class TestHandoffs:
    def test_trainer_transfer_returns_command(self):
        result = trainer_transfer.invoke({"target": "physiotherapist", "reason": "knee pain"})
        assert isinstance(result, Command)

    def test_trainer_transfer_target_and_update(self):
        result = trainer_transfer.invoke({"target": "recovery_coach", "reason": "fatigue"})
        assert result.goto == "recovery_coach"
        assert result.update["active_agent"] == "recovery_coach"
        assert result.update["handoff_reason"] == "fatigue"

    def test_physio_transfer_returns_command_with_correct_goto(self):
        result = physio_transfer.invoke({"target": "trainer", "reason": "cleared for training"})
        assert isinstance(result, Command)
        assert result.goto == "trainer"
        assert result.update["handoff_reason"] == "cleared for training"

    def test_recovery_transfer_returns_command(self):
        result = recovery_transfer.invoke({"target": "dietitian", "reason": "underfuelling"})
        assert isinstance(result, Command)
        assert result.goto == "dietitian"

    def test_dietitian_transfer_returns_command(self):
        result = dietitian_transfer.invoke({"target": "trainer", "reason": "nutrition plan set"})
        assert isinstance(result, Command)
        assert result.goto == "trainer"

    def test_psychologist_transfer_returns_command(self):
        result = psychologist_transfer.invoke({"target": "trainer", "reason": "mental skills addressed"})
        assert isinstance(result, Command)
        assert result.goto == "trainer"

    def test_psychologist_transfer_to_recovery(self):
        result = psychologist_transfer.invoke({"target": "recovery_coach", "reason": "burnout"})
        assert result.goto == "recovery_coach"
        assert result.update["active_agent"] == "recovery_coach"
        assert result.update["handoff_reason"] == "burnout"

    def test_trainer_can_transfer_to_psychologist(self):
        result = trainer_transfer.invoke({"target": "psychologist", "reason": "motivation loss"})
        assert isinstance(result, Command)
        assert result.goto == "psychologist"
        assert result.update["active_agent"] == "psychologist"

    def test_physio_can_transfer_to_psychologist(self):
        result = physio_transfer.invoke({"target": "psychologist", "reason": "fear of re-injury"})
        assert result.goto == "psychologist"

    def test_recovery_can_transfer_to_psychologist(self):
        result = recovery_transfer.invoke({"target": "psychologist", "reason": "burnout stress"})
        assert result.goto == "psychologist"

    def test_dietitian_can_transfer_to_psychologist(self):
        result = dietitian_transfer.invoke({"target": "psychologist", "reason": "disordered eating concern"})
        assert result.goto == "psychologist"

    @pytest.mark.parametrize("fn,target", [
        (trainer_transfer,      "physiotherapist"),
        (physio_transfer,       "trainer"),
        (recovery_transfer,     "trainer"),
        (dietitian_transfer,    "trainer"),
        (psychologist_transfer, "trainer"),
    ])
    def test_all_transfers_store_reason_in_update(self, fn, target):
        result = fn.invoke({"target": target, "reason": "test reason"})
        assert result.update["handoff_reason"] == "test reason"

    @pytest.mark.parametrize("fn,target", [
        (trainer_transfer,      "physiotherapist"),
        (physio_transfer,       "trainer"),
        (recovery_transfer,     "trainer"),
        (dietitian_transfer,    "trainer"),
        (psychologist_transfer, "trainer"),
    ])
    def test_all_transfers_update_active_agent(self, fn, target):
        result = fn.invoke({"target": target, "reason": "reason"})
        assert result.update["active_agent"] == target


# ══════════════════════════════════════════════════════════════════════════════
# router._cosine
# ══════════════════════════════════════════════════════════════════════════════

class TestCosine:
    def test_identical_vectors_returns_1(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_0(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_returns_negative_1(self):
        v = np.array([1.0, 2.0])
        assert _cosine(v, -v) == pytest.approx(-1.0)

    def test_known_similarity(self):
        a = np.array([3.0, 4.0])
        b = np.array([4.0, 3.0])
        # dot(a,b) = 24, |a|=|b|=5, cos = 24/25
        assert _cosine(a, b) == pytest.approx(24 / 25)

    def test_zero_vector_does_not_crash(self):
        result = _cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert abs(result) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# router.route_entry
# ══════════════════════════════════════════════════════════════════════════════

# Orthogonal unit vectors — each clearly closest to its own domain.
_MOCK_DOMAIN_VECS = {
    "physiotherapist": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    "trainer":         np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    "recovery_coach":  np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    "dietitian":       np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
    "psychologist":    np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
}


def _mock_load(domain: str):
    """Return (mock_model, domain_vecs) where message embedding points to *domain*."""
    target_vec = _MOCK_DOMAIN_VECS[domain].copy()
    target_vec += 0.01  # slight noise, still closest to target domain
    mock_model = MagicMock()
    mock_model.embed.side_effect = lambda texts: iter([target_vec])
    return mock_model, _MOCK_DOMAIN_VECS


class TestRouteEntry:
    def test_active_agent_bypasses_embedding(self):
        state = {"messages": [_human("anything")], "active_agent": "physiotherapist"}
        assert route_entry(state) == "physiotherapist"

    def test_no_human_message_defaults_to_trainer(self):
        state = {"messages": [_ai("bot message")], "active_agent": ""}
        assert route_entry(state) == "trainer"

    def test_empty_active_agent_uses_embedding(self):
        mock_model, domain_vecs = _mock_load("dietitian")
        state = {"messages": [_human("meal plan")], "active_agent": ""}
        with patch("src.agent.router._load", return_value=(mock_model, domain_vecs)):
            result = route_entry(state)
        assert result == "dietitian"

    @pytest.mark.parametrize("domain", ["physiotherapist", "trainer", "recovery_coach", "dietitian", "psychologist"])
    def test_semantic_routing_picks_closest_domain(self, domain):
        mock_model, domain_vecs = _mock_load(domain)
        state = {"messages": [_human("some message")], "active_agent": ""}
        with patch("src.agent.router._load", return_value=(mock_model, domain_vecs)):
            result = route_entry(state)
        assert result == domain

    def test_last_human_message_is_used(self):
        # Multiple human messages — embedding should use the last one
        mock_model, domain_vecs = _mock_load("recovery_coach")
        calls = []
        mock_model.embed.side_effect = lambda texts: (
            calls.append(texts[0]) or iter([_MOCK_DOMAIN_VECS["recovery_coach"]])
        )
        state = {
            "messages": [_human("old question"), _ai("answer"), _human("sleep quality")],
            "active_agent": "",
        }
        with patch("src.agent.router._load", return_value=(mock_model, domain_vecs)):
            route_entry(state)
        assert calls[-1] == "sleep quality"


# ══════════════════════════════════════════════════════════════════════════════
# memory._week_start
# ══════════════════════════════════════════════════════════════════════════════

class TestWeekStart:
    def test_sunday_returns_same_day(self):
        sunday = date(2026, 5, 31)  # Known Sunday
        assert _week_start(sunday) == sunday

    def test_monday_returns_previous_sunday(self):
        assert _week_start(date(2026, 6, 1)) == date(2026, 5, 31)

    def test_saturday_returns_preceding_sunday(self):
        assert _week_start(date(2026, 6, 6)) == date(2026, 5, 31)

    def test_midweek_returns_correct_sunday(self):
        assert _week_start(date(2026, 6, 3)) == date(2026, 5, 31)

    def test_next_week_sunday_returns_itself(self):
        next_sunday = date(2026, 6, 7)
        assert _week_start(next_sunday) == next_sunday


# ══════════════════════════════════════════════════════════════════════════════
# memory._messages_to_text
# ══════════════════════════════════════════════════════════════════════════════

class TestMessagesToText:
    def test_empty_list_returns_empty_string(self):
        assert _messages_to_text([]) == ""

    def test_formats_role_and_content(self):
        text = _messages_to_text([_human("hello"), _ai("world")])
        assert "HUMAN: hello" in text
        assert "AI: world" in text

    def test_skips_messages_with_empty_content(self):
        assert _messages_to_text([_ai("")]) == ""

    def test_preserves_message_order(self):
        text = _messages_to_text([_human("first"), _ai("second"), _human("third")])
        assert text.index("first") < text.index("second") < text.index("third")

    def test_multiple_messages_joined_by_newlines(self):
        text = _messages_to_text([_human("a"), _ai("b")])
        assert "\n" in text


# ══════════════════════════════════════════════════════════════════════════════
# memory.SummaryStore
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def store(pg_temp_dsn, monkeypatch):
    from src.config import config
    monkeypatch.setattr(config, "DATABASE_URL", pg_temp_dsn)
    return SummaryStore()


class TestSummaryStore:
    def test_schema_created_on_init(self, pg_temp_dsn, monkeypatch):
        import psycopg
        from psycopg.rows import dict_row
        from src.config import config
        monkeypatch.setattr(config, "DATABASE_URL", pg_temp_dsn)
        SummaryStore()
        with psycopg.connect(pg_temp_dsn, row_factory=dict_row, autocommit=True) as con:
            row = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'conversation_summaries' AND table_schema = current_schema()"
            ).fetchone()
        assert row["count"] > 0

    def test_save_and_retrieve_daily(self, store):
        store.save(date.today(), "trainer", "daily", "• Ran 10k")
        rows = store.get_recent_daily(days=1)
        assert len(rows) == 1
        assert rows[0]["domain"] == "trainer"
        assert "10k" in rows[0]["content"]

    def test_get_recent_daily_excludes_old_rows(self, store):
        old = date.today() - timedelta(days=10)
        store.save(old, "trainer", "daily", "old note")
        rows = store.get_recent_daily(days=7)
        assert all(r["content"] != "old note" for r in rows)

    def test_save_replaces_existing_entry_same_date_and_domain(self, store):
        today = date.today()
        store.save(today, "trainer", "daily", "original")
        store.save(today, "trainer", "daily", "updated")
        rows = store.get_recent_daily(days=1)
        contents = [r["content"] for r in rows]
        assert "updated" in contents
        assert "original" not in contents

    def test_multiple_domains_returned_for_same_date(self, store):
        today = date.today()
        store.save(today, "trainer", "daily", "training note")
        store.save(today, "dietitian", "daily", "nutrition note")
        rows = store.get_recent_daily(days=1)
        domains = {r["domain"] for r in rows}
        assert domains == {"trainer", "dietitian"}

    def test_get_current_week_summary_none_when_empty(self, store):
        assert store.get_current_week_summary() is None

    def test_get_current_week_summary_with_data(self, store):
        ws = _week_start(date.today())
        store.save(ws, "all", "weekly", "Weekly overview")
        assert store.get_current_week_summary() == "Weekly overview"

    def test_format_for_context_empty_db(self, store):
        assert store.format_for_context() == ""

    def test_format_for_context_includes_weekly_header(self, store):
        ws = _week_start(date.today())
        store.save(ws, "all", "weekly", "Weekly content")
        result = store.format_for_context()
        assert "THIS WEEK'S COACHING SUMMARY" in result
        assert "Weekly content" in result

    def test_format_for_context_includes_daily_notes(self, store):
        store.save(date.today(), "trainer", "daily", "• 10k run")
        result = store.format_for_context()
        assert "RECENT SESSION NOTES" in result
        assert "10k run" in result

    def test_format_for_context_includes_both_weekly_and_daily(self, store):
        ws = _week_start(date.today())
        store.save(ws, "all", "weekly", "Weekly summary")
        store.save(date.today(), "trainer", "daily", "Daily note")
        result = store.format_for_context()
        assert "THIS WEEK'S COACHING SUMMARY" in result
        assert "RECENT SESSION NOTES" in result


# ══════════════════════════════════════════════════════════════════════════════
# memory.generate_daily_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateDailySummary:
    def test_empty_messages_returns_empty_string(self):
        assert generate_daily_summary([], "trainer", "test_model") == ""

    def test_messages_with_no_content_returns_empty(self):
        assert generate_daily_summary([_ai("")], "trainer", "test_model") == ""

    def test_non_empty_transcript_calls_llm_and_returns_content(self):
        msgs = [_human("How do I train?"), _ai("Here is a plan.")]
        with patch("src.agent.memory.ChatOllama") as MockOllama:
            MockOllama.return_value.invoke.return_value = MagicMock(content="• Key point\n• Action item")
            result = generate_daily_summary(msgs, "trainer", "test_model")
        assert "Key point" in result
        assert "Action item" in result

    def test_llm_response_stripped_of_whitespace(self):
        msgs = [_human("q"), _ai("a")]
        with patch("src.agent.memory.ChatOllama") as MockOllama:
            MockOllama.return_value.invoke.return_value = MagicMock(content="  • trimmed  ")
            result = generate_daily_summary(msgs, "trainer", "test_model")
        assert result == "• trimmed"


# ══════════════════════════════════════════════════════════════════════════════
# memory.generate_weekly_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateWeeklySummary:
    def test_empty_list_returns_empty_string(self):
        assert generate_weekly_summary([], "test_model") == ""

    def test_non_empty_list_calls_llm(self):
        daily = [{"date": "2026-05-28", "domain": "trainer", "content": "• 10k run"}]
        with patch("src.agent.memory.ChatOllama") as MockOllama:
            MockOllama.return_value.invoke.return_value = MagicMock(content="Weekly overview")
            result = generate_weekly_summary(daily, "test_model")
        assert result == "Weekly overview"


# ══════════════════════════════════════════════════════════════════════════════
# memory.save_session_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveSessionSummary:
    def test_fewer_than_4_messages_skips(self, store):
        msgs = [_human("hi"), _ai("hello"), _human("ok")]
        save_session_summary(msgs, "trainer", store, "test_model")
        assert store.get_recent_daily(days=1) == []

    def test_exactly_3_messages_skips(self, store):
        msgs = [_human("q"), _ai("a"), _human("q2")]
        save_session_summary(msgs, "trainer", store, "test_model")
        assert store.get_recent_daily(days=1) == []

    def test_4_messages_triggers_save(self, store):
        msgs = [_human("q1"), _ai("a1"), _human("q2"), _ai("a2")]
        with patch("src.agent.memory.ChatOllama") as MockOllama:
            MockOllama.return_value.invoke.return_value = MagicMock(content="• Summary")
            save_session_summary(msgs, "trainer", store, "test_model")
        rows = store.get_recent_daily(days=1)
        assert len(rows) == 1
        assert "Summary" in rows[0]["content"]

    def test_empty_generated_summary_not_persisted(self, store):
        msgs = [_human("q1"), _ai("a1"), _human("q2"), _ai("a2")]
        with patch("src.agent.memory.ChatOllama") as MockOllama:
            MockOllama.return_value.invoke.return_value = MagicMock(content="")
            save_session_summary(msgs, "trainer", store, "test_model")
        assert store.get_recent_daily(days=1) == []

    def test_custom_session_date_stored_correctly(self, store):
        msgs = [_human("q1"), _ai("a1"), _human("q2"), _ai("a2")]
        custom_date = date(2026, 1, 15)
        with patch("src.agent.memory.ChatOllama") as MockOllama:
            MockOllama.return_value.invoke.return_value = MagicMock(content="• Noted")
            save_session_summary(msgs, "trainer", store, "test_model", session_date=custom_date)
        rows = store.get_recent_daily(days=3000)
        assert any(r["date"] == "2026-01-15" for r in rows)


# ══════════════════════════════════════════════════════════════════════════════
# prompts
# ══════════════════════════════════════════════════════════════════════════════

class TestPrompts:
    @pytest.mark.parametrize("builder,label", [
        (build_trainer_prompt,      "TRAINER"),
        (build_physio_prompt,       "PHYSIOTHERAPIST"),
        (build_recovery_prompt,     "RECOVERY COACH"),
        (build_dietitian_prompt,    "DIETITIAN"),
        (build_psychologist_prompt, "PSYCHOLOGIST"),
    ])
    def test_prompt_contains_agent_label(self, builder, label):
        assert label in builder("")

    @pytest.mark.parametrize("builder", [
        build_trainer_prompt,
        build_physio_prompt,
        build_recovery_prompt,
        build_dietitian_prompt,
        build_psychologist_prompt,
    ])
    def test_prompt_includes_athlete_context(self, builder):
        ctx = "Athlete: 32yo marathon runner"
        assert ctx in builder(ctx)

    @pytest.mark.parametrize("builder", [
        build_trainer_prompt,
        build_physio_prompt,
        build_recovery_prompt,
        build_dietitian_prompt,
        build_psychologist_prompt,
    ])
    def test_prompt_includes_behaviour_section(self, builder):
        assert "BEHAVIOUR" in builder("")

    @pytest.mark.parametrize("builder", [
        build_trainer_prompt,
        build_physio_prompt,
        build_recovery_prompt,
        build_dietitian_prompt,
        build_psychologist_prompt,
    ])
    def test_prompt_includes_calendar_convention(self, builder):
        assert "CALENDAR CONVENTION" in builder("")

    @pytest.mark.parametrize("builder", [
        build_trainer_prompt,
        build_physio_prompt,
        build_recovery_prompt,
        build_dietitian_prompt,
        build_psychologist_prompt,
    ])
    def test_prompt_includes_handoff_triggers(self, builder):
        assert "HANDOFF TRIGGERS" in builder("")

    def test_trainer_prompt_contains_tool_rules(self):
        assert "TOOL RULES" in build_trainer_prompt("")

    def test_physio_prompt_contains_return_to_train_protocol(self):
        assert "RETURN-TO-TRAIN PROTOCOL" in build_physio_prompt("")

    def test_recovery_prompt_contains_readiness_thresholds(self):
        assert "READINESS THRESHOLDS" in build_recovery_prompt("")

    def test_dietitian_prompt_contains_calculation_protocol(self):
        assert "CALCULATION PROTOCOL" in build_dietitian_prompt("")

    def test_psychologist_prompt_contains_assessment_framework(self):
        assert "ASSESSMENT FRAMEWORK" in build_psychologist_prompt("")

    def test_psychologist_prompt_contains_intervention_menu(self):
        assert "INTERVENTION MENU" in build_psychologist_prompt("")
