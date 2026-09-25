# METHODOLOGY — trade-drift

Every formula in the engine, its assumptions, and how it can mislead you.

## 1. Sharpe comparison (two-sided z-test)

`z = (S_paper − S_exp) / √(SE_paper² + SE_exp²)`.

- **SE (Lo 2002, iid):** for per-period Sharpe `s_p`,
  `SE_ann = √((1 + s_p²/2)/N) · √252`. Asymptotic, assumes iid returns.
- **HAC option:** with `hac_lags > 0`, multiply by
  `√(1 + 2·Σ_{l=1..L} (1−l/(L+1))·ρ_l)` (Bartlett kernel, Newey–West
  spirit). Widen the bands for autocorrelated strategies — trend
  followers' Sharpes are less precisely estimated than Lo's formula says.
- **Two-sided on purpose:** outperformance beyond the band is flagged.
  Divergence is divergence: either the backtest understated the edge or
  a regime tailwind is flattering the strategy. Investigate, don't
  celebrate blindly.
- **Failure modes:** with N=20 daily returns, SE ≈ 3.5 — only
  catastrophes register. Non-normal returns (fat tails) make the normal
  p-value optimistic; the z-grade bands (|z| ≥ 2/3) are the real decision
  rule, p-values are supporting detail.

## 2. Win-rate binomial test

Continuity-corrected normal approximation,
`z = (|x − n·p₀| − 0.5)/√(n·p₀(1−p₀))`, two-sided. Skips when n = 0 or
p₀ ∈ {0, 1}. For n < ~30 the normal approximation is rough — the
continuity correction helps, but treat marginal p-values with suspicion.

## 3. Payoff ratio (delta method)

`R = avg_win/avg_loss`. Via the log scale:
`Var(log R) ≈ (sd_w/(avg_win·√n_w))² + (sd_l/(avg_loss·√n_l))²`,
`SE(R) = R·√Var(log R)`. This is a first-order approximation that
assumes wins and losses are each roughly lognormal-ish and independent.
Requires ≥5 trades and ≥1 loss: fewer trades and the ratio is
luck-dominated; zero losses and the ratio is *undefined*, not infinite
(a common lie in marketing material). Returns `None` (→ skip) rather
than a fabricated SE when unsupported.

## 4. Drawdown exceedance (heuristic band)

`ratio = DD_paper / DD_backtest`, one-sided — better is never drift.
Bands 1.25x (WATCH) / 1.50x (VIOLATION) are **conventions**, shared with
trade-incubate v0.1.0, not theorems: there is no sampling distribution
for the maximum drawdown of a *different* sample path, and pretending
otherwise would be false precision. What the ratio does capture
honestly: the backtest's worst pain is the only empirical anchor for
"how bad does this strategy get", and exceeding it by 50% means you are
outside your own experience.

## 5. Trade frequency (Poisson two-sided test)

Count `k` vs `λ = tpd_exp · days`, two-sided by pmf-ordering (sum the
probability mass of all outcomes with pmf ≤ pmf(k)). Both tails matter:
silence and churning are both divergence. Needs ≥5 days of record.
Assumes trades arrive roughly independently — clustered signal
generation (e.g. all signals on Fed days) violates this mildly; the
bands are wide enough that it rarely matters.

## 6. CUSUM degradation scan (heuristic early warning)

One-sided lower tabular CUSUM on residuals `e_t = μ_exp − r_t`:
`S_t = max(0, S_{t−1} + e_t − k)`, signal when `S_t > h`.
Defaults `k = 0.5σ, h = 4σ` (Montgomery's textbook starting point),
σ estimated from the series' own first half so a late break doesn't
inflate the calibration. Detects *sustained* downward shifts in mean
return — the signature of quiet decay that sampling tests need months
to confirm.

**Honest status:** heuristic, uncalibrated false-alarm rate on real
(nonstationary, fat-tailed) return series, blind to shifts smaller than
~0.5σ sustained. It can only ever raise WATCH, never a lone VIOLATION.
Use it as the smoke alarm, not the fire marshal.

## 7. Verdict composition

- Per-metric: |z| bands per profile (balanced: 2.0 WATCH / 3.0 VIOLATION);
  drawdown ratio bands; CUSUM signals → WATCH.
- Composite: any VIOLATION → VIOLATION; ≥2 WATCH → VIOLATION; else worst.
- Consecutive-WATCH escalation: same metric WATCH on 3 straight checks →
  VIOLATION. Skipped checks freeze streaks (missing data neither confirms
  nor breaks them); OK/VIOLATION resets.
- **Multiple testing, handled honestly:** six metrics at 5% each is not
  a 5% test, and we don't pretend it is. The composite rule treats
  clustered weak signals as evidence (correlated smoke is fire) instead
  of applying a Bonferroni correction that would guarantee blindness to
  exactly the gradual decay this engine hunts.

## 8. Power and minimum samples (the uncomfortable table)

| Metric | Needs for a meaningful check |
|---|---|
| Sharpe | ≥60 daily returns before |z| means much; ≥252 to be serious |
| Win rate | ≥30 trades (≥100 to catch a 0.55→0.45 decay) |
| Payoff | ≥20 trades with both wins and losses |
| Drawdown | any history, but one number — no distribution |
| Frequency | ≥5 days; ≥20 days for pace |
| CUSUM | ≥20 returns; sensitive only to ≥0.5σ sustained shifts |

Early in a track record the engine reports "can't tell yet" (skip/OK
with reasons). That is a feature: a drift detector that cries wolf in
week two gets uninstalled in week three.
