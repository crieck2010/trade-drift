> [!IMPORTANT]
> **Consolidated into [TradeSuite](https://github.com/crieck2010/trade-suite)**  
> This engine has been consolidated into the unified [TradeSuite monorepo](https://github.com/crieck2010/trade-suite). Active development, bug fixes, releases, and unified testing now live in `trade-suite`. This repository is preserved as an archive.

# trade-drift

Canonical model-drift detection for the trading suite: strategies decay,
and the most expensive failure is one that passed validation and then
quietly stops working. This engine compares realized paper/live behavior
against backtest expectations — Sharpe, win rate, payoff ratio, drawdown
depth, trade frequency — plus a CUSUM degradation scan, and grades
**OK / WATCH / VIOLATION** with named reasons and a phone-readable alert
digest. Stdlib-only, plain data in / plain data out.

Part of the [trade-suite](https://github.com/crieck2010/trade-suite)
algorithmic trading system.

## The problem it solves

`trade-incubate` v0.1.0 embedded basic drift checks in its 63-day
graduation program (Sharpe bands, win-rate, payoff, drawdown ratio, trade
frequency). That logic belongs in a canonical standalone engine:
general-purpose (any strategy, paper or live), rolling windows (not just
the 63-day program), stateful escalation, and alerts. **trade-drift is
that engine.** `trade-incubate` should delegate to it in a future v0.2.0 —
the statistics here are consistent with (and improve on) what it
embedded; see [Interop](#interop).

## Install

```bash
pip install trade-drift
# or from source
git clone https://github.com/crieck2010/trade-drift
cd trade-drift
pip install -e .
```

Requires Python 3.10+. No third-party dependencies — not even NumPy.

## Quick start

```python
from trade_drift import expectations, stats, drift, verdict, alerts

# 1. Encode what the backtest promised (or derive it from a report).
exp = expectations.new_expectations(
    "my-strategy", expected_sharpe=1.5, sharpe_se=0.35,
    expected_win_rate=0.55, expected_payoff=1.6,
    expected_max_drawdown=0.06, expected_trades_per_day=0.9)

# 2. Summarize realized history: equity curve + closed trades [{pnl}].
paper = stats.paper_stats(daily_equity, trades)

# 3. Run every applicable test.
tests = drift.run_all(paper, exp, daily_equity=daily_equity,
                      expected_daily_drift=0.0009)

# 4. Grade it.  Consecutive WATCH streaks escalate via the state file.
state = verdict.WatchState("drift_state.json")
report = verdict.verdict_report("my-strategy", tests, state=state)
print(report["verdict"], report["reasons"])

# 5. Phone-readable digest of anything non-OK.
print(alerts.digest(alerts.alerts_from_report(report)))
```

From a trade-swing ValidationReport:

```python
from trade_drift import adapters
rec = adapters.expectations_from_trade_swing("validation_report.json",
                                             n_days=504)
exp = rec["expectations"]  # missing fields → those checks skip, loudly
```

CLI:

```bash
trade-drift expect --report validation_report.json --out exp.json --n-days 504
trade-drift check --ledger ledger.json --expect exp.json --strategy-id my-strat
trade-drift check --ledger incubate.jsonl --expect exp.json --state drift_state.json
trade-drift watch --ledger ledger.json --expect exp.json --window 20 --step 5
trade-drift alerts --report verdict.json
trade-drift demo            # seeded drift theater: healthy / decaying / shock
trade-drift presets
```

Exit codes: 0 = OK, 1 = WATCH, 2 = VIOLATION, 3 = usage/data error.
VIOLATION refuses silently nowhere — it exits non-zero and names metrics.

See `examples/drift_example.py` for the full walkthrough.

## The maths

**What you learn.** Whether the strategy running today is still the
strategy you validated: is realized performance statistically consistent
with the backtest, or has the edge decayed, the regime shifted, or the
implementation drifted?

**Why it matters.** A validated strategy that quietly stops working is the
most expensive failure mode in the system — it keeps taking risk while the
edge is gone. The question is never *whether* strategies decay but whether
you notice in weeks or months. Formal comparison against recorded
expectations beats gut feel and beats staring at the equity curve.

**The maths.**

- *Sharpe z-test.* Realized Sharpe vs expected, two-sided:
  `z = (S_paper − S_exp) / √(SE_paper² + SE_exp²)`. The SE is Lo (2002):
  `SE = √((1 + S_p²/2)/N) · √252`, with an optional Newey–West-style
  Bartlett-kernel inflation for autocorrelated return series (trend
  followers: your Sharpe is less precisely estimated than Lo's iid
  formula says). Two-sided deliberately — a surprisingly *good* Sharpe
  is also divergence (expectations were wrong, or a regime tailwind is
  flattering you).
- *Win-rate binomial test.* Continuity-corrected normal approximation:
  `z = (|x − n·p₀| − 0.5) / √(n·p₀(1−p₀))`, two-sided.
- *Payoff delta method.* `R = avg_win/avg_loss`; on the log scale
  `Var(log R) ≈ (sd_w/(avg_win·√n_w))² + (sd_l/(avg_loss·√n_l))²`, so
  `SE(R) = R·√Var(log R)`. An approximation — honest about it — that
  needs ≥5 trades and at least one loss (a payoff with no losing trades
  is *undefined*, not infinite).
- *Drawdown exceedance.* One-sided by construction: `ratio = DD_paper /
  DD_exp`. Better-than-expected drawdown is never drift. Bands 1.25x /
  1.50x are conventions shared with trade-incubate, not theorems — there
  is no sampling distribution for "max drawdown of a different sample
  path", and the docs say so.
- *Trade frequency Poisson test.* Count vs `λ = tpd_exp · days`, two-sided
  by pmf-ordering. Both directions matter: a strategy that stops trading
  and one that starts churning have both diverged.
- *CUSUM degradation scan.* One-sided lower CUSUM on residuals vs the
  expected daily drift: `S_t = max(0, S_{t−1} + (μ_exp − r_t) − k)`,
  signal when `S_t > h`, with textbook `k = 0.5σ, h = 4σ` defaults
  calibrated on the series' own early history. A *heuristic* early-warning
  scan, not a formal change-point proof — it can only ever raise WATCH.
- *Verdict composition.* Any VIOLATION → VIOLATION; ≥2 WATCH →
  VIOLATION (correlated smoke is treated as fire); a metric WATCH on 3
  consecutive checks escalates to VIOLATION (quiet persistent
  underperformance is exactly what this engine exists to catch).
- *The multiple-testing problem.* Six metrics at 5% each is not a 5%
  test. The composite rule leans into this honestly: it treats clustered
  weak signals as evidence rather than pretending each test stands alone.

**Honest limitations.** Statistical power is low in the first weeks —
with 20 daily returns the Sharpe SE is ≈3.5, so only catastrophes move
it; the engine says "can't tell yet" (skip/OK with reasons) rather than
crying wolf, and `docs/DRIFT.md` gives minimum-sample guidance per
metric. Correlated metrics double-count the same bad luck (a drawdown
*is* a Sharpe event). Nonstationary markets make "expectations" a moving
target — a VIOLATION can mean the world changed, not that the backtest
lied. This detects *divergence*, never *causes*: it tells you the
strategy stopped behaving like its backtest, not why. The CUSUM scan is
uncalibrated on real return series — early warning only. FIFO-paired
paper fills are an approximation of trade pnl.

## Interop

| Direction | Sibling | Contract |
|---|---|---|
| Expectations in | `trade-swing` ValidationReport schema v1 | `adapters.expectations_from_trade_swing` (payoff recovered from profit_factor × (1−p)/p) |
| History in | `trade-incubate` daily JSONL | `adapters.ledger_from_trade_incubate` (equity + trades[{pnl}]) |
| History in | `trade-paper` SQLite audit ledger | `adapters.ledger_from_trade_paper` (equity_snapshots + FIFO-paired fills) |
| Pause signal out | `trade-lifecycle` gate_pause | `adapters.kill_condition_input` — **informs only**; the committee decides |
| Delegation (future) | `trade-incubate` v0.2.0 | should replace its embedded drift checks with this engine |

All adapters are lazy and fail-soft: missing siblings, unreadable files,
or unrecognized schemas return `{"ok": False, "reason": ...}` — never an
exception, never a silent zero. This engine pauses nothing and sends
nothing; detection and action live in separate failure domains by design.

## Docs

- `docs/ARCHITECTURE.md` — module map, data flow, scaling, contracts
- `docs/METHODOLOGY.md` — every test's formula, assumptions, failure modes
- `docs/DRIFT.md` — practitioner's guide: which metric degrades first,
  alert-fatigue management, minimum samples, what to do on VIOLATION
- `CHANGELOG.md` — release history

## License

MIT. See `LICENSE`.
