"""Tests for expectations.py and verdict.py."""

import json

import pytest

from trade_drift import expectations as e
from trade_drift import verdict as v


# ------------------------------------------------------- expectations

def test_new_expectations_valid():
    exp = e.new_expectations("s1", expected_sharpe=1.5, expected_win_rate=0.55)
    assert exp["schema_version"] == 1
    assert exp["strategy_id"] == "s1"
    assert "expected_payoff" in exp["missing"]


def test_new_expectations_rejects_bad():
    with pytest.raises(ValueError):
        e.new_expectations("s1", bogus_field=1.0)
    with pytest.raises(ValueError):
        e.new_expectations("s1", expected_win_rate=1.5)
    with pytest.raises(ValueError):
        e.new_expectations("s1", expected_max_drawdown=-0.1)


def _fake_report():
    return {
        "schema_version": 1,
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "strategy": {"name": "swing-trend"},
        "backtest": {
            "n_trades": 200,
            "metrics": {"sharpe_ratio": 1.8, "win_rate": 0.6,
                        "profit_factor": 1.8, "max_drawdown": 0.05},
        },
    }


def test_from_validation_report_maps_fields():
    exp = e.from_validation_report(_fake_report(), n_days=400.0)
    assert exp["expected_sharpe"] == pytest.approx(1.8)
    assert exp["expected_win_rate"] == pytest.approx(0.6)
    # payoff = profit_factor * (1-p)/p = 1.8 * 0.4/0.6 = 1.2
    assert exp["expected_payoff"] == pytest.approx(1.2)
    assert exp["expected_max_drawdown"] == pytest.approx(0.05)
    assert exp["expected_trades_per_day"] == pytest.approx(0.5)
    assert exp["n_backtest_trades"] == 200


def test_from_validation_report_without_n_days_skips_tpd():
    exp = e.from_validation_report(_fake_report())
    assert exp["expected_trades_per_day"] is None
    assert "expected_trades_per_day" in exp["missing"]


def test_from_validation_report_rejects_bad_schema():
    with pytest.raises(ValueError):
        e.from_validation_report({"schema_version": 99})


def test_expectations_json_round_trip():
    exp = e.new_expectations("s1", expected_sharpe=1.2)
    assert e.from_json(e.to_json(exp))["expected_sharpe"] == pytest.approx(1.2)
    with pytest.raises(ValueError):
        e.from_json(json.dumps({"schema_version": 99}))


# ------------------------------------------------------------ verdict

def _t(metric, statistic, skipped=False, detail="d", skip_reason=""):
    return {"metric": metric, "statistic": statistic, "skipped": skipped,
            "detail": detail, "skip_reason": skip_reason,
            "observed": None, "expected": None, "se": None, "p_value": None}


def test_grade_z_bands_balanced():
    assert v.grade_z(1.9) == "OK"
    assert v.grade_z(2.0) == "WATCH"
    assert v.grade_z(2.99) == "WATCH"
    assert v.grade_z(3.0) == "VIOLATION"
    assert v.grade_z(-2.5) == "WATCH"  # two-sided


def test_grade_drawdown_bands():
    assert v.grade_drawdown(1.24) == "OK"
    assert v.grade_drawdown(1.25) == "WATCH"
    assert v.grade_drawdown(1.5) == "VIOLATION"


def test_grade_cusum():
    assert v.grade_cusum(0) == "OK"
    assert v.grade_cusum(2) == "WATCH"  # never a lone VIOLATION


def test_grade_test_skipped():
    assert v.grade_test(_t("sharpe", 0.0, skipped=True)) == "SKIPPED"
    assert v.grade_test(_t("max_drawdown", 2.0)) == "VIOLATION"
    assert v.grade_test(_t("cusum", 1)) == "WATCH"
    assert v.grade_test(_t("sharpe", -2.2)) == "WATCH"


def test_composite_rules():
    assert v.composite({"a": "OK"})[0] == "OK"
    assert v.composite({"a": "WATCH"})[0] == "WATCH"
    assert v.composite({"a": "WATCH", "b": "WATCH"})[0] == "VIOLATION"
    assert v.composite({"a": "OK", "b": "VIOLATION"})[0] == "VIOLATION"
    assert v.composite({"a": "SKIPPED"})[0] == "OK"
    assert v.composite({})[0] == "OK"


def test_watch_state_escalates_on_third(tmp_path):
    path = str(tmp_path / "state.json")
    state = v.WatchState(path, streak_threshold=3)
    assert state.record("s", {"sharpe": "WATCH"}) == []
    assert state.record("s", {"sharpe": "WATCH"}) == []
    esc = state.record("s", {"sharpe": "WATCH"})
    assert esc == [{"metric": "sharpe", "streak": 3}]
    # persists across instances
    state2 = v.WatchState(path, streak_threshold=3)
    assert state2.streaks("s")["sharpe"] == 3


def test_watch_state_ok_resets_and_skip_untouched(tmp_path):
    state = v.WatchState(str(tmp_path / "s.json"), streak_threshold=3)
    state.record("s", {"sharpe": "WATCH"})
    state.record("s", {"sharpe": "OK"})
    assert state.streaks("s")["sharpe"] == 0
    state.record("s", {"sharpe": "WATCH"})
    state.record("s", {"sharpe": "SKIPPED"})
    assert state.streaks("s")["sharpe"] == 1  # skip neither confirms nor breaks


def test_verdict_report_escalation(tmp_path):
    state = v.WatchState(str(tmp_path / "s.json"), streak_threshold=2)
    tests = [_t("sharpe", -2.2, detail="mild underperformance")]
    r1 = v.verdict_report("s", tests, state=state)
    assert r1["verdict"] == "WATCH" and r1["escalated"] == []
    r2 = v.verdict_report("s", tests, state=state)
    assert r2["verdict"] == "VIOLATION"
    assert r2["escalated"] == [{"metric": "sharpe", "streak": 2}]
    assert any("consecutive" in r for r in r2["reasons"])


def test_verdict_report_rejects_bad_profile():
    with pytest.raises(ValueError):
        v.verdict_report("s", [], profile="nope")


def test_profiles_move_bands():
    assert v.grade_z(2.2, "strict") == "WATCH"
    assert v.grade_z(2.2, "balanced") == "WATCH"
    assert v.grade_z(2.2, "lenient") == "OK"
    assert v.grade_z(2.6, "strict") == "VIOLATION"
    assert v.grade_z(2.6, "lenient") == "WATCH"


def test_verdict_report_carries_reasons_and_schema():
    rep = v.verdict_report("s9", [_t("sharpe", 0.5, detail="fine")])
    assert rep["schema_version"] == 1
    assert rep["verdict"] == "OK"
    assert any("fine" in r for r in rep["reasons"])
