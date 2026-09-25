# Changelog — trade-drift

## v0.1.0 — 2026-09-24

Initial release: the canonical model-drift detection engine.

- `stats`: returns, Sharpe, Lo (2002) SE with optional HAC inflation,
  normal/binomial/Poisson tests, delta-method payoff SE, max drawdown,
  `paper_stats()` (same contract as trade-incubate's, plus win/loss SDs)
- `expectations`: schema v1; `new_expectations()` with range validation;
  `from_validation_report()` mapping trade-swing schema v1 (payoff
  recovered from profit_factor × (1−p)/p); missing fields → checks skip
- `drift`: six tests — Sharpe z, win-rate binomial, payoff delta-method,
  drawdown exceedance (one-sided), trade-frequency Poisson, CUSUM
  degradation scan (heuristic); every test returns
  {metric, observed, expected, statistic, se, p_value, skipped,
  skip_reason, detail}
- `verdict`: strict/balanced/lenient profiles; composite rule (any
  VIOLATION → VIOLATION; ≥2 WATCH → VIOLATION); consecutive-WATCH
  escalation via tiny JSON `WatchState`; `verdict_report()` schema v1
- `alerts`: alert records, phone-readable `digest()`, `notify()` hook
  spec that never sends by itself
- `adapters`: lazy fail-soft readers for trade-swing reports,
  trade-incubate JSONL, trade-paper SQLite (FIFO-paired fills), and
  `kill_condition_input()` advisory payload for trade-lifecycle's pause
  gate (informs only — separate failure domains)
- `demo`: seeded theater — healthy all-OK; decay OK → WATCH (CUSUM early
  warning) → VIOLATION with named metrics; WATCH×3 escalation vignette;
  sudden shock → fast drawdown VIOLATION; alert digest
- CLI: `expect`, `check`, `watch`, `alerts`, `demo`, `presets`,
  `license`, `update-check`; exits 0/1/2 for OK/WATCH/VIOLATION
- Docs: README (with "The maths"), ARCHITECTURE, METHODOLOGY, DRIFT
  practitioner's guide
- 65 tests, stdlib-only, MIT
