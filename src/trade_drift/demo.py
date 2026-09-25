"""Seeded deterministic drift theater: three strategies, one engine.

- H ``steady-9`` : healthy the whole 60 days  → all OK.
- D ``decay-4``  : edge decays from day 31    → rolling watch goes
  OK → VIOLATION with the degraded metrics named (plus a consecutive-
  WATCH escalation vignette showing the streak mechanism).
- S ``break-2``  : sudden shock on day 51     → fast VIOLATION.

Deterministic: every number below comes from random.Random(7).  The
tests assert two runs agree exactly.
"""

from __future__ import annotations

import random

from . import drift as _drift
from . import stats as _stats
from . import verdict as _verdict
from . import alerts as _alerts
from . import expectations as _exp

SEED = 7
SIGMA = 0.01          # daily vol of the synthetic strategies
MU_HEALTHY = 0.000945  # ≈ Sharpe 1.5 at SIGMA=0.01


def _gen_returns(rng: random.Random, n: int, mu: float,
                 sigma: float = SIGMA) -> list[float]:
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _equity(returns: list[float], start: float = 100_000.0) -> list[float]:
    eq, out = start, [start]
    for r in returns:
        eq *= (1.0 + r)
        out.append(eq)
    return out


def _gen_trades(rng: random.Random, n_days: int, trades_per_day: float,
                win_p: float, avg_win: float, avg_loss: float,
                win_sd: float = 25.0, loss_sd: float = 18.0) -> list[dict]:
    n = int(trades_per_day * n_days)
    trades = []
    for _ in range(n):
        if rng.random() < win_p:
            trades.append({"pnl": max(rng.gauss(avg_win, win_sd), 1.0)})
        else:
            trades.append({"pnl": -max(rng.gauss(avg_loss, loss_sd), 1.0)})
    return trades


def build_ledgers(seed: int = SEED) -> dict:
    """{strategy_id: {"equity": [...], "trades": [...]}} — 60 trading days."""
    rng = random.Random(seed)
    n = 60

    h_ret = _gen_returns(rng, n, MU_HEALTHY)
    d_ret = _gen_returns(rng, 30, MU_HEALTHY) + \
        _gen_returns(rng, 30, -0.005)          # edge inverts from day 31
    s_ret = _gen_returns(rng, 50, MU_HEALTHY) + [-0.12] + \
        _gen_returns(rng, 9, 0.0, 0.008)       # shock on day 51

    ledgers = {
        "steady-9": {
            "equity": _equity(h_ret),
            "trades": _gen_trades(rng, n, 0.9, 0.55, 160.0, 100.0),
        },
        "decay-4": {
            "equity": _equity(d_ret),
            "trades": (_gen_trades(rng, 30, 0.9, 0.55, 160.0, 100.0)
                       + _gen_trades(rng, 30, 0.9, 0.45, 130.0, 100.0)),
        },
        "break-2": {
            "equity": _equity(s_ret),
            "trades": _gen_trades(rng, n, 0.9, 0.55, 160.0, 100.0),
        },
    }
    return ledgers


def build_expectations() -> dict:
    """What the (fictional) backtest promised."""
    return _exp.new_expectations(
        "demo", source_report_id="demo-validation-report",
        expected_sharpe=1.5, sharpe_se=0.35,
        expected_win_rate=0.55, expected_payoff=1.6,
        expected_max_drawdown=0.06, expected_trades_per_day=0.9,
        n_backtest_trades=480, horizon_days=504)


def check_window(equity: list[float], trades: list[dict], exp: dict,
                 window: int | None = None) -> dict:
    """Check the last ``window`` days (or the whole ledger).

    Demo schema carries no trade timestamps, so windowed checks take the
    proportional slice of the trade list (trades are uniform over the
    demo horizon).  Real ledgers filter by date — see adapters.
    """
    n_days = len(equity) - 1
    eq = equity[-(window + 1):] if window else equity
    if window and n_days > window:
        lo = int(len(trades) * (n_days - window) / n_days)
        hi = int(len(trades) * n_days / n_days)
        tr = trades[lo:hi] or trades[:1]
    else:
        tr = trades
    paper = _stats.paper_stats(eq, tr)
    mu_daily = 1.5 * SIGMA / (252.0 ** 0.5)  # expected daily drift ≈ Sharpe 1.5
    tests = _drift.run_all(paper, exp, daily_equity=eq,
                           expected_daily_drift=mu_daily)
    return _verdict.verdict_report("demo", tests)


def run_demo(seed: int = SEED) -> dict:
    """Full theater.  Returns plain data; printing is the caller's job."""
    ledgers = build_ledgers(seed)
    exp = build_expectations()
    out: dict = {"seed": seed, "expectations": exp, "scenes": []}

    # Scene 1 — healthy strategy, full 60-day check: all OK.
    h = ledgers["steady-9"]
    rep_h = check_window(h["equity"], h["trades"], exp)
    rep_h["strategy_id"] = "steady-9"
    out["scenes"].append({"title": "steady-9: 60-day check (healthy)",
                          "report": rep_h})

    # Scene 2 — decaying strategy, rolling 20-day watch.
    d = ledgers["decay-4"]
    for end in (30, 40, 50, 60):
        # Re-slice the ledger to end at the right day for the demo; the
        # windowed check takes the proportional trade slice (uniform demo
        # trades; real ledgers filter by date — see adapters).
        eq = d["equity"][:end + 1]
        n_all = len(d["equity"]) - 1
        lo = int(len(d["trades"]) * (end - 20) / n_all)
        hi = int(len(d["trades"]) * end / n_all)
        tr = d["trades"][max(lo, 0):hi] or d["trades"][:1]
        win_eq = eq[-21:] if len(eq) > 21 else eq
        paper = _stats.paper_stats(win_eq, tr)
        mu_daily = 1.5 * SIGMA / (252.0 ** 0.5)
        tests = _drift.run_all(paper, exp, daily_equity=win_eq,
                               expected_daily_drift=mu_daily)
        rep = _verdict.verdict_report("decay-4", tests)
        out["scenes"].append(
            {"title": f"decay-4: 20-day window ending day {end}", "report": rep})

    # Scene 3 — consecutive-WATCH escalation, mechanism vignette.
    state = _verdict.WatchState(streak_threshold=3)
    esc_reports = []
    for i in range(3):
        rep = _verdict.verdict_report(
            "decay-4",
            [{"metric": "sharpe", "statistic": -2.2, "skipped": False,
              "detail": "synthetic persistent mild underperformance",
              "observed": 0.4, "expected": 1.5, "se": 0.5, "p_value": 0.03,
              "skip_reason": ""}],
            state=state)
        esc_reports.append({"check": i + 1, "verdict": rep["verdict"],
                            "grades": rep["grades"],
                            "escalated": rep["escalated"]})
    out["scenes"].append({"title": "watch-state: WATCH x3 → escalated",
                          "escalation": esc_reports})

    # Scene 4 — sudden break: fast VIOLATION.
    s = ledgers["break-2"]
    rep_s = check_window(s["equity"], s["trades"], exp)
    rep_s["strategy_id"] = "break-2"
    out["scenes"].append({"title": "break-2: 60-day check (shock on day 51)",
                          "report": rep_s})

    # Alert digest across every non-OK scene.
    all_alerts = []
    for scene in out["scenes"]:
        if "report" in scene:
            all_alerts.extend(_alerts.alerts_from_report(scene["report"]))
    out["digest"] = _alerts.digest(all_alerts)
    return out


def print_demo(result: dict) -> None:
    for scene in result["scenes"]:
        print(f"\n=== {scene['title']} ===")
        if "report" in scene:
            rep = scene["report"]
            print(f"verdict: {rep['verdict']}  (profile: {rep['profile']})")
            for metric, grade in rep["grades"].items():
                print(f"  {metric:15s} {grade}")
            for reason in rep["reasons"]:
                print(f"  - {reason}")
        else:
            for e in scene["escalation"]:
                print(f"  review {e['check']}: {e['verdict']} "
                      f"{e['grades']} escalated={e['escalated']}")
    print("\n=== alert digest ===")
    print(result["digest"])
