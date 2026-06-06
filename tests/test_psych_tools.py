"""
tests/test_psych_tools.py — Unit and integration tests for the psychologist tools.

Covers:
  - get_situational_psych_tips: all 7 situations, unknown situation, context injection
  - search_psychology_books: skipped if qdrant not built, category filter, reranking
"""

import pytest

from src.tools.psych_tool import get_situational_psych_tips, search_psychology_books

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_SITUATIONS = [
    "onboarding",
    "pre_test",
    "pre_race",
    "post_race",
    "anomaly_training",
    "new_injury",
    "return_to_training",
]

# Expected section headers that every situation response must contain
REQUIRED_SECTIONS = [
    "PSYCHOLOGICAL PROFILE:",
    "KEY QUESTIONS TO ASK THE ATHLETE:",
    "EVIDENCE-BASED TECHNIQUES:",
    "WATCH FOR (RED FLAGS):",
    "APPROACH NOTE:",
]


# ══════════════════════════════════════════════════════════════════════════════
# get_situational_psych_tips
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSituationalPsychTips:

    @pytest.mark.parametrize("situation", VALID_SITUATIONS)
    def test_valid_situation_returns_structured_output(self, situation):
        result = get_situational_psych_tips.invoke({"situation": situation})
        assert isinstance(result, str)
        assert len(result) > 100

    @pytest.mark.parametrize("situation", VALID_SITUATIONS)
    def test_all_sections_present(self, situation):
        result = get_situational_psych_tips.invoke({"situation": situation})
        for section in REQUIRED_SECTIONS:
            assert section in result, (
                f"Section '{section}' missing from '{situation}' output."
            )

    @pytest.mark.parametrize("situation", VALID_SITUATIONS)
    def test_situation_label_in_header(self, situation):
        result = get_situational_psych_tips.invoke({"situation": situation})
        assert "PSYCHOLOGICAL GUIDANCE:" in result

    def test_unknown_situation_returns_error_message(self):
        result = get_situational_psych_tips.invoke({"situation": "weightlifting_pr"})
        assert "Unknown situation" in result
        assert "weightlifting_pr" in result

    def test_unknown_situation_lists_valid_options(self):
        result = get_situational_psych_tips.invoke({"situation": "xyz"})
        for s in VALID_SITUATIONS:
            assert s in result, f"Valid situation '{s}' not listed in error message."

    def test_context_injected_into_output(self):
        ctx = "athlete's very first marathon, 3 days out"
        result = get_situational_psych_tips.invoke({
            "situation": "pre_race",
            "context": ctx,
        })
        assert ctx in result

    def test_context_section_absent_when_empty(self):
        result = get_situational_psych_tips.invoke({
            "situation": "onboarding",
            "context": "",
        })
        assert "CONTEXT PROVIDED:" not in result

    def test_context_section_present_when_given(self):
        result = get_situational_psych_tips.invoke({
            "situation": "new_injury",
            "context": "stress fracture, second injury this season",
        })
        assert "CONTEXT PROVIDED:" in result

    def test_situation_is_case_insensitive(self):
        lower = get_situational_psych_tips.invoke({"situation": "pre_race"})
        upper = get_situational_psych_tips.invoke({"situation": "PRE_RACE"})
        assert lower == upper

    # ── Situation-specific content checks ─────────────────────────────────────

    def test_onboarding_mentions_goal_setting(self):
        result = get_situational_psych_tips.invoke({"situation": "onboarding"})
        assert "goal" in result.lower()

    def test_pre_test_mentions_arousal_regulation(self):
        result = get_situational_psych_tips.invoke({"situation": "pre_test"})
        assert "arousal" in result.lower()

    def test_pre_race_mentions_visualisation(self):
        result = get_situational_psych_tips.invoke({"situation": "pre_race"})
        assert "visuali" in result.lower()

    def test_post_race_mentions_attribution(self):
        result = get_situational_psych_tips.invoke({"situation": "post_race"})
        assert "attribution" in result.lower()

    def test_anomaly_training_mentions_overgeneralisation(self):
        result = get_situational_psych_tips.invoke({"situation": "anomaly_training"})
        result_lower = result.lower()
        assert "generalisation" in result_lower or "generalization" in result_lower or "noise" in result_lower

    def test_new_injury_mentions_identity(self):
        result = get_situational_psych_tips.invoke({"situation": "new_injury"})
        assert "identity" in result.lower()

    def test_return_to_training_mentions_fear(self):
        result = get_situational_psych_tips.invoke({"situation": "return_to_training"})
        assert "fear" in result.lower()

    def test_new_injury_has_clinical_referral_note(self):
        result = get_situational_psych_tips.invoke({"situation": "new_injury"})
        # Should warn about clinical-level symptoms
        assert "depress" in result.lower() or "clinical" in result.lower() or "licensed" in result.lower()

    @pytest.mark.parametrize("situation,expected_word", [
        ("pre_race",           "controllable"),
        ("post_race",          "growth"),
        ("return_to_training", "confidence"),
    ])
    def test_situation_key_concept_present(self, situation, expected_word):
        result = get_situational_psych_tips.invoke({"situation": situation})
        assert expected_word in result.lower(), (
            f"Expected '{expected_word}' in '{situation}' output."
        )


# ══════════════════════════════════════════════════════════════════════════════
# search_psychology_books
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchPsychologyBooks:

    def test_returns_string(self):
        result = search_psychology_books.invoke({"query": "mental toughness confidence"})
        assert isinstance(result, str)
        assert len(result) > 0

    @staticmethod
    def _make_mocks(points=None, raise_exc=None):
        """Build (mock_client, mock_dense, mock_sparse, mock_rerank) for patching _get_models."""
        import numpy as np
        from unittest.mock import MagicMock

        # sparse embed result must have numpy arrays so .tolist() works
        sparse_raw = MagicMock()
        sparse_raw.indices = np.array([1, 2])
        sparse_raw.values  = np.array([0.5, 0.3])

        mock_client = MagicMock()
        if raise_exc:
            mock_client.query_points.side_effect = raise_exc
        else:
            mock_client.query_points.return_value = MagicMock(points=points or [])

        mock_dense  = MagicMock()
        mock_dense.encode.return_value = np.array([[0.1] * 1024])
        mock_sparse = MagicMock()
        mock_sparse.embed.return_value = iter([sparse_raw])
        mock_rerank = MagicMock()
        mock_rerank.predict.return_value = [0.9] * len(points or [])

        return mock_client, mock_dense, mock_sparse, mock_rerank

    def test_no_books_returns_graceful_message(self):
        """When no psychology chunks exist the tool returns a helpful message, not a crash."""
        from unittest.mock import patch

        mocks = self._make_mocks(points=[])
        with patch("src.tools.psych_tool._get_models", return_value=mocks):
            result = search_psychology_books.invoke({"query": "mental toughness"})

        assert "No psychology" in result or "not yet embedded" in result

    def test_search_failure_returns_error_string_not_exception(self):
        """A Qdrant error must be caught and returned as a string, never raised."""
        from unittest.mock import patch

        mocks = self._make_mocks(raise_exc=RuntimeError("simulated DB failure"))
        with patch("src.tools.psych_tool._get_models", return_value=mocks):
            result = search_psychology_books.invoke({"query": "confidence"})

        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()

    # ── Live DB tests (skipped when psychology books not yet embedded) ─────────

    @staticmethod
    def _psych_count() -> int:
        """
        Count psychology chunks using the already-open global client.
        Returns -1 when either the client or the embedding models are not ready
        (no Qdrant DB, or models not downloaded/cached).
        """
        try:
            import src.tools.rag_tool as rag_module
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            client, dense_model, _, _ = rag_module._get_models()
            if client is None or dense_model is None:
                return -1
            return client.count(
                "coaching_books",
                count_filter=Filter(
                    must=[FieldCondition(key="category", match=MatchValue(value="psychology"))]
                ),
            ).count
        except Exception:
            return -1

    @pytest.mark.parametrize("query", [
        "mental toughness resilience athlete",
        "anxiety arousal pre-competition performance",
        "goal setting motivation intrinsic extrinsic",
        "imagery visualization mental rehearsal",
        "self-efficacy confidence sport psychology",
    ])
    def test_psychology_queries_return_non_empty_result(self, query):
        if self._psych_count() <= 0:
            pytest.skip("No psychology chunks in DB — run `python etl/embed_books.py`")
        result = search_psychology_books.invoke({"query": query})
        assert len(result) > 100, f"Too short result for '{query}': {result[:80]}"

    def test_n_results_parameter_limits_output(self):
        if self._psych_count() <= 0:
            pytest.skip("No psychology chunks in DB")
        result_3 = search_psychology_books.invoke({
            "query": "confidence sport athlete", "n_results": 3
        })
        result_1 = search_psychology_books.invoke({
            "query": "confidence sport athlete", "n_results": 1
        })
        assert len(result_1) <= len(result_3)
