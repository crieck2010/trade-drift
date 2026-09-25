# ARCHITECTURE — trade-drift

## Module map

```
src/trade_drift/
  stats.py         shared primitives: returns, Sharpe, Lo SE (+HAC option),
                   normal/binomial/Poisson tests, delta-method payoff SE,
                   max drawdown, paper_stats()
  expectations.py  Expectation schema v1; new_expectations(); mapping from
                   trade-swing ValidationReport (payoff via profit_factor)
  drift.py         per-metric tests → {metric, observed, expected, statistic,
                   se, p_value, skipped, skip_reason, detail}; run_all()
  verdict.py       profiles (strict/balanced/lenient), grade_*, composite
                   rule, WatchState (consecutive-WATCH escalation),
                   verdict_report() schema v1
  alerts.py        make_alert(), alerts_from_report(), digest(),
                   notify() hook spec (never sends by itself)
  adapters.py      lazy fail-soft readers: trade-swing reports,
                   trade-incubate JSONL, trade-paper SQLite (FIFO fills),
                   kill_condition_input() for trade-lifecycle
  demo.py          seeded 3-strategy theater (healthy / decaying / shock)
  presets.py       profile descriptions
  cli.py           expect / check / watch / alerts / demo / presets /
                   license / update-check
  licensing.py     license-key + update-check hooks (suite convention)
```

## Data flow

```
ValidationReport (trade-swing)
        │ from_validation_report()
        ▼
Expectation {expected_sharpe, win_rate, payoff, max_dd, tpd, ...}
        │
daily equity + closed trades ──► paper_stats() ──► run_all() ──► tests[]
        │                                              │
        │                                         verdict_report()
        │                                              │
        │                                    {verdict, grades, reasons}
        │                                         │         │
        │                                    alerts/digest  kill_condition_input()
        │                                                    (informs only)
        ▼
  trade-lifecycle gate_pause (human/committee decides)
```

## Contracts

- **Expectation**: `schema_version: 1`, fields in `expectations.FIELDS`.
  Any field may be `None` → dependent checks skip with reasons. `missing[]`
  lists them explicitly.
- **Test result**: `{metric, observed, expected, statistic, se, p_value,
  skipped, skip_reason, detail}`. `statistic` is a z-score (sampling
  tests), an exceedance ratio (drawdown), or a signal count (CUSUM).
- **Verdict report**: `schema_version: 1`,
  `{strategy_id, as_of, profile, grades, escalated, verdict, reasons,
  tests}`. JSON-serializable throughout.
- **Alert**: `{schema_version, at, strategy_id, level, metric, message}`.

## Scaling

Everything is O(N) in the length of the history; the whole engine runs in
milliseconds on multi-year daily histories. The expensive-looking part —
the Poisson two-sided sum — walks at most ~λ + 12√λ terms. No state except
the tiny `WatchState` JSON file (per-strategy WATCH streaks). For
intraday/fleet use: shard by strategy, one state file per strategy, and
call `run_all()` per shard — there is no shared mutable state.

## Relation to trade-incubate

trade-incubate v0.1.0 embeds drift checks in its graduation program
(`drift.py`: Sharpe band, win-rate band, payoff humility, drawdown ratio,
trade-frequency floor). Those checks are *consistent* with this engine's
defaults (balanced profile: |z| 2.0/3.0, drawdown 1.25x/1.50x, <5-day and
<5-trade humility) and should be **replaced by delegation** in a future
trade-incubate v0.2.0: `trade_drift.run_all()` + `verdict_report()` cover
the same ground with proper p-values, rolling windows, escalation, and
alerts. This repo was deliberately built as the canonical home so the
program layer doesn't own statistics. Do not duplicate the formulas —
import them.

## Failure-domain discipline

- Detection never acts: no pause, no order, no notification is ever sent
  by this engine. `notify()` without a registered channel returns
  `sent: False` and says so.
- `kill_condition_input()` is advisory payload for trade-lifecycle's
  pause gate. A broken drift engine cannot halt a healthy strategy; a
  healthy drift engine cannot be bypassed by anything except the human
  operating the committee.
- Adapters fail soft (`ok: False` + reason). A missing sibling degrades
  the check to "skipped with reason", never to a silent pass.
