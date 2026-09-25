"""End-to-end drift check on a synthetic track record."""

from trade_drift import alerts, drift, expectations, stats, verdict

# What the (fictional) backtest promised.
exp = expectations.new_expectations(
    "demo-strategy", expected_sharpe=1.5, sharpe_se=0.35,
    expected_win_rate=0.55, expected_payoff=1.6,
    expected_max_drawdown=0.06, expected_trades_per_day=0.9)

# Realized history: 60 days of equity + closed trades.
import random
rng = random.Random(7)
equity = [100_000.0]
for _ in range(60):
    equity.append(equity[-1] * (1 + rng.gauss(0.0009, 0.01)))
trades = [{"pnl": rng.gauss(160, 25) if rng.random() < 0.55
           else -abs(rng.gauss(100, 18))} for _ in range(54)]

paper = stats.paper_stats(equity, trades)
tests = drift.run_all(paper, exp, daily_equity=equity,
                      expected_daily_drift=0.0009)
report = verdict.verdict_report("demo-strategy", tests)

print("verdict:", report["verdict"])
for reason in report["reasons"]:
    print(" -", reason)
print()
print(alerts.digest(alerts.alerts_from_report(report)))
