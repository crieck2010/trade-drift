# DRIFT — the practitioner's guide

How to actually use this engine without drowning in alerts or missing
the decay that matters.

## Which metric degrades first (in practice)

1. **Trade frequency** — the canary. A strategy that stops trading (or
   starts churning) tells you the market stopped offering its setup
   before the P&L does. Watch pace weekly.
2. **CUSUM** — the smoke alarm. A sustained downward shift in mean return
   shows up here weeks before any sampling test has the power to
   confirm it. A lone CUSUM WATCH means "look closer", not "kill it".
3. **Win rate / payoff** — need 30–100+ trades to move. When they move,
   the edge itself has changed (costs up, fills worse, alpha decayed).
4. **Drawdown** — the fire alarm. By the time the exceedance ratio
   trips, you're already living the pain. Fast and one-sided: it never
   false-alarms on good luck.
5. **Sharpe** — the obituary. Needs the most data; confirms what the
   other metrics already told you.

## Running it

- **Daily:** `check` on the full track record during incubation, with a
  `--state` file so consecutive-WATCH escalation works.
- **Weekly:** `watch --window 20 --step 5` — rolling windows catch recent
  decay that full-history averages dilute. Size the window to the decay
  you fear: 20 days for sudden breaks, 60 for slow bleed.
- **On alert:** read the digest on your phone, then open the full
  report. Every grade carries its detail string — the numbers, not just
  the color.

## Alert-fatigue management

- A lone WATCH on one metric in the first month is usually "not enough
  data", not "strategy broken". The engine says this in the reasons;
  believe it.
- The composite rule (≥2 WATCH → VIOLATION) exists because single-metric
  WATCHes are noisy and correlated WATCHes are not.
- If you get desensitized, switch to the `strict` profile deliberately
  for a week rather than ignoring `balanced` — fatigue is a configuration
  problem, not a character flaw.
- Register a real notification channel (`alerts.register_notifier`) only
  when the digest is worth interrupting a field day. Until then,
  `notify()` correctly reports `sent: False`.

## What to do on VIOLATION

1. **Don't trade through it.** A VIOLATION is the engine telling you the
   strategy is no longer the one you validated. In the lifecycle, feed
   `adapters.kill_condition_input()` to the pause gate — the committee
   (you) decides, but the default motion is pause.
2. **Diagnose before you tweak.** Divergence is not a diagnosis. Check:
   did costs/fills change? Did the regime shift (ask trade-regime)? Did
   the implementation change (bad deploy)? Is the backtest expectation
   stale?
3. **Never "fix" it by widening the bands.** Moving from balanced to
   lenient because balanced complained is the statistical equivalent of
   breaking the thermometer. Amend the *expectations* (with a recorded
   reason) or retire the strategy — both are first-class operations.
4. **Re-validation, not re-optimization.** If you change the strategy in
   response, it goes back through the lifecycle gates (trade-lifecycle)
   as a new candidate. The old expectations are void.

## What this engine will not do

- Tell you *why* performance diverged (regime? costs? bug? decay?).
- Distinguish "edge decayed" from "normal bad luck inside the bands" —
  that's what the bands are for, and they're probabilistic, not certain.
- Replace the incubation program: this is the *instrument*, the 63-day
  program is the *procedure*. Instruments don't run procedures.
