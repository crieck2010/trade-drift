"""Tests for the demo theater and the CLI surface."""

import json
import random

import pytest

from trade_drift.demo import run_demo
from trade_drift import expectations as e
from trade_drift.cli import main as cli_main


def _scrub(obj):
    # verdict reports carry wall-clock timestamps; everything else must be
    # deterministic across runs.
    if isinstance(obj, dict):
        return {k: ("<t>" if k in ("as_of", "at") else _scrub(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    return obj


def test_demo_deterministic():
    r1 = json.dumps(_scrub(run_demo()), sort_keys=True, default=str)
    r2 = json.dumps(_scrub(run_demo()), sort_keys=True, default=str)
    assert r1 == r2


def test_demo_outcomes():
    result = run_demo()
    scenes = {s["title"]: s for s in result["scenes"]}
    h = scenes["steady-9: 60-day check (healthy)"]["report"]
    assert h["verdict"] == "OK"
    assert set(h["grades"].values()) == {"OK"}

    d60 = scenes["decay-4: 20-day window ending day 60"]["report"]
    assert d60["verdict"] == "VIOLATION"
    d40 = scenes["decay-4: 20-day window ending day 40"]["report"]
    assert d40["verdict"] == "WATCH"  # CUSUM early warning before confirmation

    esc = scenes["watch-state: WATCH x3 → escalated"]["escalation"]
    assert [x["verdict"] for x in esc] == ["WATCH", "WATCH", "VIOLATION"]
    assert esc[2]["escalated"] == [{"metric": "sharpe", "streak": 3}]

    s = scenes["break-2: 60-day check (shock on day 51)"]["report"]
    assert s["verdict"] == "VIOLATION"
    assert s["grades"]["max_drawdown"] == "VIOLATION"
    assert "DRIFT ALERT" in result["digest"]


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj, default=str))
    return str(p)


def _healthy_ledger(seed=5, n=61):
    rng = random.Random(seed)
    eq = [100000.0]
    for _ in range(n - 1):
        eq.append(eq[-1] * (1 + rng.gauss(0.0009, 0.01)))
    trades = [{"pnl": rng.gauss(160, 25) if rng.random() < 0.55
               else -abs(rng.gauss(100, 18))} for _ in range(54)]
    return {"equity": eq, "trades": trades}


def _expectations_file(tmp_path):
    exp = e.new_expectations(
        "t1", expected_sharpe=1.5, sharpe_se=0.35, expected_win_rate=0.55,
        expected_payoff=1.6, expected_max_drawdown=0.06,
        expected_trades_per_day=0.9)
    return _write(tmp_path, "exp.json", exp)


def test_cli_check_healthy_exits_zero(tmp_path):
    ledger = _write(tmp_path, "ledger.json", _healthy_ledger())
    exp = _expectations_file(tmp_path)
    code = cli_main(["check", "--ledger", ledger, "--expect", exp,
                     "--strategy-id", "t1"])
    assert code == 0


def test_cli_check_violation_exits_two(tmp_path):
    led = _healthy_ledger()
    led["equity"] = led["equity"][:51] + [led["equity"][50] * 0.82] + \
        [led["equity"][50] * 0.82] * 9  # -18% shock, no recovery
    ledger = _write(tmp_path, "ledger.json", led)
    exp = _expectations_file(tmp_path)
    code = cli_main(["check", "--ledger", ledger, "--expect", exp,
                     "--strategy-id", "t1"])
    assert code == 2


def test_cli_watch_and_presets_and_demo(tmp_path):
    ledger = _write(tmp_path, "ledger.json", _healthy_ledger())
    exp = _expectations_file(tmp_path)
    # A 20-day window on noisy data can legitimately trip bands — the
    # command must run cleanly and return a valid verdict exit code.
    code = cli_main(["watch", "--ledger", ledger, "--expect", exp,
                     "--window", "20", "--step", "20"])
    assert code in (0, 1, 2)
    assert cli_main(["presets"]) == 0
    assert cli_main(["demo", "--json"]) == 0


def test_cli_expect_from_report(tmp_path):
    report = {"schema_version": 1, "strategy": {"name": "x"},
              "backtest": {"n_trades": 120,
                           "metrics": {"sharpe_ratio": 1.3, "win_rate": 0.53,
                                       "profit_factor": 1.4,
                                       "max_drawdown": 0.05}}}
    rp = _write(tmp_path, "report.json", report)
    out = str(tmp_path / "exp.json")
    assert cli_main(["expect", "--report", rp, "--out", out,
                     "--n-days", "240"]) == 0
    exp = json.loads((tmp_path / "exp.json").read_text())
    assert exp["expected_trades_per_day"] == pytest.approx(0.5)


def test_cli_alerts_from_report_file(tmp_path):
    from trade_drift import drift as dmod
    from trade_drift import stats as smod
    from trade_drift import verdict as vmod
    led = _healthy_ledger()
    led["equity"] = led["equity"][:51] + [led["equity"][50] * 0.80] + \
        [led["equity"][50] * 0.80] * 9
    paper = smod.paper_stats(led["equity"], led["trades"])
    exp = e.new_expectations("t1", expected_sharpe=1.5, sharpe_se=0.35,
                             expected_win_rate=0.55, expected_payoff=1.6,
                             expected_max_drawdown=0.06,
                             expected_trades_per_day=0.9)
    rep = vmod.verdict_report("t1", dmod.run_all(paper, exp))
    rp = _write(tmp_path, "verdict.json", rep)
    assert cli_main(["alerts", "--report", rp]) == 0


def test_cli_license_and_update_check_do_not_crash():
    assert cli_main(["license"]) == 0
    assert cli_main(["update-check"]) == 0  # never raises, even offline


def test_cli_check_json_and_hac_lags(tmp_path):
    ledger = _write(tmp_path, "ledger.json", _healthy_ledger())
    exp = _expectations_file(tmp_path)
    assert cli_main(["check", "--ledger", ledger, "--expect", exp,
                     "--json", "--hac-lags", "3"]) == 0
