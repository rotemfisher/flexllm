"""
tests/test_sport_radar.py — Unit tests for check_upcoming_race_or_test and check_training_anomaly.

All DB interaction is mocked so tests run without real training data.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.tools.sport_radar_tool import (
    _classify_row,
    _client_anomaly_tags,
    _fmt_pace,
    _zscore,
    check_training_anomaly,
    check_upcoming_race_or_test,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _row(**kwargs):
    """Build a dict-like MagicMock from keyword arguments."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, k: kwargs.get(k)
    mock.keys = lambda: list(kwargs)
    for k, v in kwargs.items():
        setattr(mock, k, v)
    return mock


def _make_planned_row(
    day_date, description="", phase=None, workout_type="easy",
    is_assessment=0, notes="",
):
    return _row(
        day_date=day_date,
        description=description,
        phase=phase,
        workout_type=workout_type,
        is_assessment=is_assessment,
        notes=notes,
    )


TODAY     = date.today().strftime("%Y-%m-%d")
TOMORROW  = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
IN_3_DAYS = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════════
# Pure-logic helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestZScore:
    def test_basic_computation(self):
        assert _zscore(12.0, 10.0, 2.0) == pytest.approx(1.0)

    def test_zero_std_returns_none(self):
        assert _zscore(10.0, 10.0, 0.0) is None

    def test_none_inputs_return_none(self):
        assert _zscore(None, 10.0, 2.0) is None
        assert _zscore(10.0, None, 2.0) is None
        assert _zscore(10.0, 10.0, None) is None

    def test_negative_z_for_below_mean(self):
        assert _zscore(8.0, 10.0, 2.0) == pytest.approx(-1.0)


class TestFmtPace:
    def test_exact_minute(self):
        assert _fmt_pace(6.0) == "6:00/km"

    def test_with_seconds(self):
        assert _fmt_pace(6.5) == "6:30/km"

    def test_none_returns_na(self):
        assert _fmt_pace(None) == "N/A"

    def test_rounding(self):
        result = _fmt_pace(6.25)
        assert result == "6:15/km"


class TestClassifyRow:
    def test_phase_race(self):
        row = _make_planned_row(TODAY, phase="race", description="City Marathon")
        cls, label = _classify_row(row)
        assert cls == "race"

    def test_is_assessment_without_race_keywords(self):
        row = _make_planned_row(TODAY, is_assessment=1, description="3km time trial")
        cls, label = _classify_row(row)
        assert cls == "test"

    def test_race_keyword_in_description(self):
        row = _make_planned_row(TODAY, description="Half marathon race — goal pace")
        cls, label = _classify_row(row)
        assert cls == "race"

    def test_test_keyword_in_description(self):
        row = _make_planned_row(TODAY, description="VDOT time trial 2km")
        cls, label = _classify_row(row)
        assert cls == "test"

    def test_ordinary_session_returns_other(self):
        row = _make_planned_row(TODAY, description="45 min easy run", workout_type="easy")
        cls, label = _classify_row(row)
        assert cls == "other"

    def test_race_keyword_in_notes(self):
        row = _make_planned_row(TODAY, description="hard effort", notes="10k competition today")
        cls, label = _classify_row(row)
        assert cls == "race"

    @pytest.mark.parametrize("keyword", ["marathon", "triathlon", "5k", "10k", "ironman"])
    def test_race_keywords_detected(self, keyword):
        row = _make_planned_row(TODAY, description=f"do a {keyword} today")
        cls, _ = _classify_row(row)
        assert cls == "race"

    @pytest.mark.parametrize("keyword", ["time trial", "cooper test", "1rm", "3rm"])
    def test_test_keywords_detected(self, keyword):
        row = _make_planned_row(TODAY, description=f"perform a {keyword} test")
        cls, _ = _classify_row(row)
        assert cls == "test"


class TestClientAnomalyTags:
    def test_fatigue_words_detected(self):
        tags = _client_anomaly_tags("legs felt really tired and heavy")
        assert any(t == "fatigue_declared" for t, _ in tags)

    def test_fast_words_detected(self):
        tags = _client_anomaly_tags("hit a new personal best today!")
        assert any(t == "fast_run_declared" for t, _ in tags)

    def test_slow_words_detected(self):
        tags = _client_anomaly_tags("so slow, fell off pace after 3km")
        assert any(t == "slow_run_declared" for t, _ in tags)

    def test_over_lifting_detected(self):
        tags = _client_anomaly_tags("overdid it with the squats today")
        assert any(t == "over_lifting_declared" for t, _ in tags)

    def test_lifting_fail_detected(self):
        tags = _client_anomaly_tags("couldn't finish the last set of squats")
        assert any(t == "fatigue_lifting_declared" for t, _ in tags)

    def test_empty_report_returns_empty(self):
        assert _client_anomaly_tags("") == []

    def test_normal_report_returns_empty(self):
        assert _client_anomaly_tags("session felt normal, no issues at all") == []

    def test_report_text_included_in_detail(self):
        report = "legs were really tired"
        tags = _client_anomaly_tags(report)
        assert any(report in detail for _, detail in tags)


# ══════════════════════════════════════════════════════════════════════════════
# check_upcoming_race_or_test
# ══════════════════════════════════════════════════════════════════════════════

def _patch_race_radar(rows):
    """Context manager: patch db_ro so the tool sees the given rows."""
    mock_con = MagicMock()
    mock_con.__enter__ = lambda s: mock_con
    mock_con.__exit__  = MagicMock(return_value=False)
    mock_con.execute.return_value.fetchall.return_value = rows
    return patch("src.tools.sport_radar_tool.db_ro", return_value=mock_con)


class TestCheckUpcomingRaceOrTest:
    def test_no_sessions_returns_none(self):
        with _patch_race_radar([]):
            result = check_upcoming_race_or_test.invoke({})
        assert "NONE" in result

    def test_race_tomorrow_returns_pre_race_trigger(self):
        row = _make_planned_row(TOMORROW, description="City Marathon race", phase="race")
        with _patch_race_radar([row]):
            result = check_upcoming_race_or_test.invoke({})
        assert "TRIGGER: PRE_RACE" in result
        assert "ACTION REQUIRED" in result
        assert "trainer_transfer" in result
        assert "psychologist" in result

    def test_race_today_returns_pre_race_trigger(self):
        row = _make_planned_row(TODAY, description="10k race today", phase="race")
        with _patch_race_radar([row]):
            result = check_upcoming_race_or_test.invoke({})
        assert "TRIGGER: PRE_RACE" in result
        assert "TODAY" in result

    def test_test_tomorrow_returns_pre_test_trigger(self):
        row = _make_planned_row(TOMORROW, description="2km VDOT time trial", is_assessment=1)
        with _patch_race_radar([row]):
            result = check_upcoming_race_or_test.invoke({})
        assert "TRIGGER: PRE_TEST" in result
        assert "ACTION REQUIRED" in result

    def test_race_in_3_days_returns_upcoming(self):
        row = _make_planned_row(IN_3_DAYS, description="half marathon", phase=None)
        with _patch_race_radar([row]):
            result = check_upcoming_race_or_test.invoke({})
        assert "UPCOMING" in result
        assert "ACTION REQUIRED" not in result

    def test_ordinary_session_returns_none(self):
        row = _make_planned_row(TOMORROW, description="45 min easy run", workout_type="easy")
        with _patch_race_radar([row]):
            result = check_upcoming_race_or_test.invoke({})
        assert "TRIGGER" not in result

    def test_race_takes_priority_over_test_when_both_imminent(self):
        race = _make_planned_row(TOMORROW, description="marathon competition", phase="race")
        test = _make_planned_row(TODAY, description="1rm test", is_assessment=1)
        with _patch_race_radar([race, test]):
            result = check_upcoming_race_or_test.invoke({})
        assert "PRE_RACE" in result

    def test_trigger_reason_contains_event_description(self):
        row = _make_planned_row(TOMORROW, description="City Half Marathon race day", phase="race")
        with _patch_race_radar([row]):
            result = check_upcoming_race_or_test.invoke({})
        assert "Half Marathon" in result or "City" in result

    def test_db_error_returns_none_gracefully(self):
        mock_con = MagicMock()
        mock_con.__enter__ = lambda s: mock_con
        mock_con.__exit__  = MagicMock(return_value=False)
        mock_con.execute.side_effect = RuntimeError("DB locked")
        with patch("src.tools.sport_radar_tool.db_ro", return_value=mock_con):
            result = check_upcoming_race_or_test.invoke({})
        assert "NONE" in result
        assert isinstance(result, str)

    def test_returns_string(self):
        with _patch_race_radar([]):
            result = check_upcoming_race_or_test.invoke({})
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# check_training_anomaly
# ══════════════════════════════════════════════════════════════════════════════

def _mock_anomaly_db(last_run=None, run_baseline=None,
                     last_str=None, str_base=None):
    """
    Patch db_ro with TWO separate context managers — one per 'with db_ro()' block
    in check_training_anomaly (running block first, strength block second).

    Using a shared call counter would go out of sync when last_run=None causes
    the running block to make only 1 execute call instead of 2.
    """
    def _make_ctx(*rows):
        idx = {"n": 0}
        con = MagicMock()
        con.__enter__ = lambda s: con
        con.__exit__  = MagicMock(return_value=False)

        def exec_fn(*a, **k):
            m = MagicMock()
            m.fetchone.return_value = rows[idx["n"]] if idx["n"] < len(rows) else None
            idx["n"] += 1
            return m

        con.execute.side_effect = exec_fn
        return con

    ctx_run = _make_ctx(last_run, run_baseline)
    ctx_str = _make_ctx(last_str, str_base)
    outer   = {"n": 0}

    def db_ro_factory():
        ctx = ctx_run if outer["n"] == 0 else ctx_str
        outer["n"] += 1
        return ctx

    return patch("src.tools.sport_radar_tool.db_ro", side_effect=db_ro_factory)


class TestCheckTrainingAnomaly:

    def _run_row(self, pace=6.0, hr=150.0, rpe=7, start_date=TODAY):
        return _row(start_date=start_date, distance_km=10.0,
                    pace=pace, hr=hr, rpe=rpe)

    def _run_base(self, avg_pace=6.0, std_pace=0.3, avg_hr=148.0, n=10):
        return _row(avg_pace=avg_pace, std_pace=std_pace, avg_hr=avg_hr, n=n)

    def _str_row(self, id=1, rpe=7, avg_w=80.0, n_sets=15, start_date=TODAY):
        return _row(id=id, start_date=start_date, rpe=rpe,
                    avg_w=avg_w, n_sets=n_sets)

    def _str_base(self, avg_rpe=6.5, std_rpe=0.8, avg_weight=78.0,
                  avg_sets=15.0, n=8):
        return _row(avg_rpe=avg_rpe, std_rpe=std_rpe, avg_weight=avg_weight,
                    avg_sets=avg_sets, n=n)

    # ── Normal sessions ────────────────────────────────────────────────────────

    def test_normal_run_returns_normal(self):
        with _mock_anomaly_db(
            last_run=self._run_row(pace=6.05),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=0.3, n=10),
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({})
        assert "NORMAL" in result
        assert "TRIGGER" not in result

    # ── Slow run ──────────────────────────────────────────────────────────────

    def test_slow_run_detected_by_zscore(self):
        # pace = 7.0, baseline avg=6.0, std=0.3 → z = (7-6)/0.3 = 3.33
        with _mock_anomaly_db(
            last_run=self._run_row(pace=7.0),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=0.3, n=10),
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({})
        assert "TRIGGER" in result
        assert "slow_run" in result.lower() or "SLOW" in result

    def test_slow_run_detected_by_pct(self):
        # pace = 7.1, baseline = 6.0 → 18% slower (>15% threshold)
        with _mock_anomaly_db(
            last_run=self._run_row(pace=7.1),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=5.0, n=10),  # high std → z small
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({})
        assert "TRIGGER" in result

    def test_trigger_contains_action_required(self):
        with _mock_anomaly_db(
            last_run=self._run_row(pace=7.5),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=0.3, n=10),
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({})
        assert "ACTION REQUIRED" in result
        assert "trainer_transfer" in result
        assert "psychologist" in result

    # ── Fast run ──────────────────────────────────────────────────────────────

    def test_fast_run_detected(self):
        # pace = 4.8, baseline = 6.0, std = 0.3 → z = -4.0
        with _mock_anomaly_db(
            last_run=self._run_row(pace=4.8),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=0.3, n=10),
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({})
        assert "TRIGGER" in result
        assert "fast_run" in result.lower() or "FAST" in result

    # ── Elevated HR ───────────────────────────────────────────────────────────

    def test_elevated_hr_detected(self):
        # HR 170 vs baseline 148 → +14.9%
        with _mock_anomaly_db(
            last_run=self._run_row(pace=6.0, hr=170.0),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=0.3, avg_hr=148.0, n=10),
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({})
        assert "TRIGGER" in result
        assert "elevated_hr" in result.lower() or "HR" in result

    # ── Strength anomalies ────────────────────────────────────────────────────

    def test_fatigue_lifting_detected_by_rpe(self):
        # RPE 9, baseline 6.5, diff = 2.5 (> threshold of 2)
        with _mock_anomaly_db(
            last_run=None,
            run_baseline=None,
            last_str=self._str_row(rpe=9, avg_w=80.0, n_sets=15),
            str_base=self._str_base(avg_rpe=6.5, std_rpe=0.5, n=5),
        ):
            result = check_training_anomaly.invoke({})
        assert "TRIGGER" in result
        assert "fatigue_lifting" in result.lower() or "FATIGUE" in result

    def test_over_lifting_detected(self):
        # avg_w = 100 vs baseline 80 → +25%
        with _mock_anomaly_db(
            last_run=None,
            run_baseline=None,
            last_str=self._str_row(rpe=7, avg_w=100.0, n_sets=15),
            str_base=self._str_base(avg_rpe=7.0, std_rpe=0.5,
                                     avg_weight=80.0, n=5),
        ):
            result = check_training_anomaly.invoke({})
        assert "TRIGGER" in result
        assert "over_lifting" in result.lower() or "OVER" in result

    def test_volume_drop_triggers_fatigue(self):
        # only 8 sets vs baseline 18 → 44% of normal
        with _mock_anomaly_db(
            last_run=None,
            run_baseline=None,
            last_str=self._str_row(rpe=7, avg_w=80.0, n_sets=8),
            str_base=self._str_base(avg_rpe=6.5, std_rpe=0.5,
                                     avg_weight=80.0, avg_sets=18.0, n=5),
        ):
            result = check_training_anomaly.invoke({})
        assert "TRIGGER" in result

    # ── Client declarations ───────────────────────────────────────────────────

    def test_client_declared_fatigue_triggers_anomaly(self):
        with _mock_anomaly_db(last_run=None, run_baseline=None,
                              last_str=_row(id=None), str_base=None):
            result = check_training_anomaly.invoke({
                "client_report": "legs felt completely dead and exhausted"
            })
        assert "TRIGGER" in result
        assert "fatigue_declared" in result.lower() or "FATIGUE" in result

    def test_client_declared_pb_triggers_fast_anomaly(self):
        with _mock_anomaly_db(last_run=None, run_baseline=None,
                              last_str=_row(id=None), str_base=None):
            result = check_training_anomaly.invoke({
                "client_report": "hit a new personal best today by 45 seconds!"
            })
        assert "TRIGGER" in result

    def test_normal_client_report_no_trigger(self):
        with _mock_anomaly_db(
            last_run=self._run_row(pace=6.05),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=0.3, n=10),
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({
                "client_report": "session felt normal, no issues at all"
            })
        assert "TRIGGER" not in result

    def test_insufficient_data_message(self):
        with _mock_anomaly_db(
            last_run=self._run_row(pace=6.0),
            run_baseline=self._run_base(avg_pace=6.0, std_pace=0.3, n=2),
            last_str=_row(id=None),
            str_base=None,
        ):
            result = check_training_anomaly.invoke({})
        # 2 sessions < 3 minimum: no trigger, either NORMAL or INSUFFICIENT
        assert "TRIGGER" not in result

    def test_returns_string(self):
        with _mock_anomaly_db(last_run=None, run_baseline=None,
                              last_str=_row(id=None), str_base=None):
            result = check_training_anomaly.invoke({})
        assert isinstance(result, str)

    def test_db_error_returns_string_not_exception(self):
        mock_con = MagicMock()
        mock_con.__enter__ = lambda s: mock_con
        mock_con.__exit__  = MagicMock(return_value=False)
        mock_con.execute.side_effect = RuntimeError("DB locked")
        with patch("src.tools.sport_radar_tool.db_ro", return_value=mock_con):
            result = check_training_anomaly.invoke({})
        assert isinstance(result, str)

    # ── Prompt compliance ──────────────────────────────────────────────────────

    def test_physio_prompt_contains_new_injury_mandatory_protocol(self):
        from src.agent.prompts import build_physio_prompt
        prompt = build_physio_prompt("")
        assert "MANDATORY" in prompt or "NON-NEGOTIABLE" in prompt
        assert "psychologist" in prompt.lower()
        assert "log_injury" in prompt

    def test_trainer_prompt_contains_step0_detection(self):
        from src.agent.prompts import build_trainer_prompt
        prompt = build_trainer_prompt("")
        assert "check_upcoming_race_or_test" in prompt
        assert "STEP 0" in prompt

    def test_trainer_prompt_contains_anomaly_detection(self):
        from src.agent.prompts import build_trainer_prompt
        prompt = build_trainer_prompt("")
        assert "check_training_anomaly" in prompt
        assert "ANOMALY DETECTION" in prompt
