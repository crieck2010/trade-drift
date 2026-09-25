"""Tests for alerts.py and adapters.py."""

import json
import os
import sqlite3

import pytest

from trade_drift import alerts as a
from trade_drift import adapters as ad


# --------------------------------------------------------------- alerts

def _report():
    return {
        "schema_version": 1, "strategy_id": "s1", "as_of": "2026-01-01",
        "profile": "balanced",
        "grades": {"sharpe": "WATCH", "max_drawdown": "VIOLATION",
                   "win_rate": "OK"},
        "tests": [
            {"metric": "sharpe", "detail": "z=-2.2"},
            {"metric": "max_drawdown", "detail": "ratio 1.6x"},
            {"metric": "win_rate", "detail": "fine"},
        ],
    }


def test_make_alert_rejects_non_alert_levels():
    with pytest.raises(ValueError):
        a.make_alert("s", "OK", "sharpe", "nope")


def test_alerts_from_report_picks_non_ok():
    alerts = a.alerts_from_report(_report())
    assert {x["metric"] for x in alerts} == {"sharpe", "max_drawdown"}
    assert {x["level"] for x in alerts} == {"WATCH", "VIOLATION"}


def test_digest_empty_and_ordering():
    assert "all clear" in a.digest([])
    alerts = a.alerts_from_report(_report())
    text = a.digest(alerts)
    assert text.index("[VIOLATION]") < text.index("[WATCH]")
    assert "s1" in text


def test_notify_needs_channel():
    alerts = a.alerts_from_report(_report())
    r = a.notify(alerts)
    assert r["sent"] is False and "never sends" in r["reason"]
    r2 = a.notify(alerts, channel="nope")
    assert r2["sent"] is False
    a.register_notifier("test-ch", lambda al: {"delivered": len(al)})
    r3 = a.notify(alerts, channel="test-ch")
    assert r3["sent"] is True and r3["receipt"] == {"delivered": 2}
    assert a.notify([])["sent"] is False


# ------------------------------------------------------------- adapters

def test_sibling_available():
    assert ad.sibling_available("trade_drift") is True
    assert ad.sibling_available("no_such_package_xyz_123") is False


def test_expectations_from_bad_report():
    assert ad.expectations_from_trade_swing("/no/such/file.json")["ok"] is False
    assert ad.expectations_from_trade_swing({"nope": 1})["ok"] is False


def test_expectations_from_report_dict():
    report = {"schema_version": 1, "strategy": {"name": "x"},
              "backtest": {"n_trades": 100,
                           "metrics": {"sharpe_ratio": 1.4, "win_rate": 0.52,
                                       "profit_factor": 1.5,
                                       "max_drawdown": 0.04}}}
    rec = ad.expectations_from_trade_swing(report, n_days=200.0)
    assert rec["ok"] is True
    assert rec["expectations"]["expected_trades_per_day"] == pytest.approx(0.5)


def test_ledger_from_trade_incubate_jsonl(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps({"equity": 100000.0,
                             "trades": [{"pnl": 50.0}]}) + "\n")
        fh.write(json.dumps({"equity": 100050.0, "trades": []}) + "\n")
    rec = ad.ledger_from_trade_incubate(path)
    assert rec["ok"] is True
    assert rec["equity"] == [100000.0, 100050.0]
    assert rec["trades"] == [{"pnl": 50.0}]
    assert ad.ledger_from_trade_incubate("/no/such.jsonl")["ok"] is False


def _make_paper_db(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE equity_snapshots (id INTEGER PRIMARY KEY, at TEXT,
            equity REAL NOT NULL, cash REAL NOT NULL, note TEXT DEFAULT '');
        CREATE TABLE fills (id INTEGER PRIMARY KEY, client_order_id TEXT,
            symbol TEXT, side TEXT, quantity REAL, price REAL,
            commission REAL DEFAULT 0, filled_at TEXT);
    """)
    db.executemany("INSERT INTO equity_snapshots (at, equity, cash) VALUES (?,?,?)",
                   [("2026-01-01", 100000.0, 100000.0),
                    ("2026-01-02", 100500.0, 100500.0)])
    db.executemany("INSERT INTO fills (client_order_id, symbol, side, quantity,"
                   " price, filled_at) VALUES (?,?,?,?,?,?)",
                   [("o1", "AAPL", "buy", 10, 100.0, "2026-01-01"),
                    ("o2", "AAPL", "sell", 10, 105.0, "2026-01-02")])
    db.commit()
    db.close()


def test_ledger_from_trade_paper(tmp_path):
    path = str(tmp_path / "ledger.db")
    _make_paper_db(path)
    rec = ad.ledger_from_trade_paper(path)
    assert rec["ok"] is True
    assert rec["equity"] == [100000.0, 100500.0]
    # FIFO: bought 10 @100, sold 10 @105 → +50
    assert rec["trades"] == [{"pnl": pytest.approx(50.0), "symbol": "AAPL"}]
    assert ad.ledger_from_trade_paper("/no/such.db")["ok"] is False


def test_kill_condition_input():
    rep = _report()
    ki = ad.kill_condition_input("s1", rep)
    assert ki["condition"] == "drift_violation"
    assert ki["metrics"] == ["max_drawdown"]
    assert ki["strategy_id"] == "s1"
    ok_rep = dict(rep, grades={"sharpe": "OK"})
    assert ad.kill_condition_input("s1", ok_rep)["condition"] == "drift_watch"
