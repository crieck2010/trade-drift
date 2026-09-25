"""Per-metric drift tests: realized vs expected, with honest statistics.

Each test is a pure function returning a plain-data dict:

    {metric, observed, expected, statistic, se, p_value, skipped,
     skip_reason, detail}

``statistic`` is a z-score for the sampling tests, an exceedance ratio for
drawdown, and a signal count for the CUSUM scan.  ``skipped`` is True
whenever the inputs can't support the test — absence of evidence is
reported, never filled in.

Consistency note: the Sharpe SE, win-rate band, payoff humility band,
drawdown ratios, and frequency floors here generalize what
trade-incubate v0.1.0 embedded in its graduation program (same Lo 2002
Sharpe SE, same 1.25/1.5 drawdown ratios, same <5-day/<5-trade humility).
trade-incubate should delegate to this module in a future v0.2.0.
"""

from __future__ import annotations

import math

from . import stats as _s


def _base(metric: str) -> dict:
    return {"metric": metric, "observed": None, "expected": None,
            "statistic": None, "se": None, "p_value": None,
            "skipped": False, "skip_reason": "", "detail": ""}


def test_sharpe(paper: dict, expected: dict) -> dict:
    """Realized Sharpe vs expected Sharpe: two-sided z-test.

    Combines the paper SE with the backtest SE when both are finite
    (independent samples); otherwise uses whichever is finite.
    """
    r = _base("sharpe")
    exp_sharpe = expected.get("expected_sharpe")
    if exp_sharpe is None:
        r["skipped"], r["skip_reason"] = True, "no expected Sharpe recorded"
        return r
    obs = float(paper.get("sharpe", 0.0))
    se_p = float(paper.get("sharpe_se", float("inf")))
    se_e = expected.get("sharpe_se")
    se_e = float(se_e) if se_e is not None else float("inf")
    if not math.isfinite(se_p) and not math.isfinite(se_e):
        r["skipped"], r["skip_reason"] = True, "Sharpe SE undefined (no returns)"
        return r
    se = math.sqrt((se_p ** 2 if math.isfinite(se_p) else 0.0)
                   + (se_e ** 2 if math.isfinite(se_e) else 0.0))
    if se <= 0:
        r["skipped"], r["skip_reason"] = True, "zero Sharpe SE"
        return r
    z = (obs - float(exp_sharpe)) / se
    r.update(observed=obs, expected=float(exp_sharpe), statistic=z, se=se,
             p_value=_s.two_sided_p(z),
             detail=(f"paper Sharpe {obs:.2f} vs expected {exp_sharpe:.2f} "
                     f"(z={z:+.2f}, two-sided p={_s.two_sided_p(z):.3f})"))
    return r


def test_win_rate(paper: dict, expected: dict) -> dict:
    """Realized win rate vs expected: continuity-corrected binomial z-test."""
    r = _base("win_rate")
    p0 = expected.get("expected_win_rate")
    if p0 is None:
        r["skipped"], r["skip_reason"] = True, "no expected win rate recorded"
        return r
    n = int(paper.get("n_trades", 0))
    x = int(paper.get("n_wins", 0))
    t = _s.binomial_test(x, n, float(p0))
    if t["skipped"]:
        r["skipped"], r["skip_reason"] = True, t["skip_reason"]
        return r
    obs = x / n
    r.update(observed=obs, expected=float(p0), statistic=t["z"],
             se=math.sqrt(p0 * (1 - p0) / n), p_value=t["p_value"],
             detail=(f"paper win rate {obs:.1%} ({x}/{n}) vs expected "
                     f"{p0:.1%} (z={t['z']:+.2f}, p={t['p_value']:.3f})"))
    return r


def test_payoff(paper: dict, expected: dict) -> dict:
    """Realized payoff (avg_win/avg_loss) vs expected: delta-method z-test.

    Needs ≥5 trades and at least one loss (same humility as
    trade-incubate): a payoff ratio with no losing trades is undefined,
    not infinite.
    """
    r = _base("payoff")
    exp_po = expected.get("expected_payoff")
    if exp_po is None:
        r["skipped"], r["skip_reason"] = True, "no expected payoff recorded"
        return r
    n = int(paper.get("n_trades", 0))
    if n < 5:
        r["skipped"], r["skip_reason"] = True, \
            f"only {n} trades — payoff ratio is luck-dominated below 5"
        return r
    if float(paper.get("avg_loss", 0.0)) <= 0:
        r["skipped"], r["skip_reason"] = True, \
            "no losing trades yet — payoff undefined, not infinite"
        return r
    se = _s.payoff_se(float(paper["avg_win"]), float(paper["avg_loss"]),
                      int(paper["n_wins"]), int(paper["n_losses"]),
                      float(paper["sd_win"]), float(paper["sd_loss"]))
    if se is None or not math.isfinite(se) or se <= 0:
        r["skipped"], r["skip_reason"] = True, \
            "payoff SE undefined (degenerate win/loss dispersion)"
        return r
    obs = float(paper["payoff"])
    z = (obs - float(exp_po)) / se
    r.update(observed=obs, expected=float(exp_po), statistic=z, se=se,
             p_value=_s.two_sided_p(z),
             detail=(f"paper payoff {obs:.2f} vs expected {exp_po:.2f} "
                     f"(delta-method z={z:+.2f}, p={_s.two_sided_p(z):.3f})"))
    return r


def test_drawdown(paper: dict, expected: dict) -> dict:
    """Realized max drawdown vs expected: one-sided exceedance ratio.

    Better-than-expected drawdown is never drift — the test is one-sided
    by construction.  The ratio bands (1.25 / 1.50) are conventions shared
    with trade-incubate, not theorems: there is no sampling distribution
    for "max drawdown of a different sample path".  Documented honestly
    in METHODOLOGY.md.
    """
    r = _base("max_drawdown")
    exp_dd = expected.get("expected_max_drawdown")
    if exp_dd is None:
        r["skipped"], r["skip_reason"] = True, "no expected drawdown recorded"
        return r
    exp_dd = abs(float(exp_dd)) or 1e-9
    obs = abs(float(paper.get("max_drawdown", 0.0)))
    ratio = obs / exp_dd
    r.update(observed=obs, expected=exp_dd, statistic=ratio,
             detail=(f"paper max DD {obs:.2%} vs backtest {exp_dd:.2%} "
                     f"(exceedance {ratio:.2f}x; one-sided, better is never drift)"))
    return r


def test_frequency(paper: dict, expected: dict) -> dict:
    """Trade pace vs expected: two-sided Poisson test on the count.

    Both directions matter: a strategy that stops trading and one that
    starts churning have both diverged.  Needs ≥5 days of record — day 1
    with zero trades is "the program just started", not drift.
    """
    r = _base("trade_frequency")
    exp_tpd = expected.get("expected_trades_per_day")
    if exp_tpd is None:
        r["skipped"], r["skip_reason"] = True, \
            "no expected trades/day recorded"
        return r
    n_days = int(paper.get("n_days", 0))
    if n_days < 5:
        r["skipped"], r["skip_reason"] = True, \
            f"only {n_days} days of record — too early to judge pace"
        return r
    lam = float(exp_tpd) * n_days
    k = int(paper.get("n_trades", 0))
    p = _s.poisson_two_sided_p(k, lam)
    # Report a z-like statistic for uniform grading: signed deviation in
    # units of Poisson SD (continuity-corrected).
    sd = math.sqrt(lam) if lam > 0 else float("inf")
    z = (abs(k - lam) - 0.5) / sd if math.isfinite(sd) and sd > 0 else 0.0
    z = max(z, 0.0)
    r.update(observed=k / n_days if n_days else 0.0, expected=float(exp_tpd),
             statistic=z, p_value=p,
             detail=(f"{k} trades in {n_days}d vs {lam:.1f} expected "
                     f"(z={z:+.2f}, Poisson two-sided p={p:.3f})"))
    return r


def cusum_scan(daily_equity: list[float], expected_daily_drift: float,
               ref_k: float | None = None, decision_h: float | None = None,
               min_days: int = 20) -> dict:
    """One-sided lower CUSUM on residuals vs the expected daily drift.

    S_t = max(0, S_{t-1} + (mu_exp − r_t) − k); signal when S_t > h.
    Detects a sustained downward shift in mean return — the signature of
    quiet decay.  HEURISTIC, not a formal change-point proof: k and h
    default to 0.5σ / 4σ of the residuals' own early history, a textbook
    starting point (Montgomery) whose false-alarm rate on real,
    nonstationary return series is uncalibrated.  Use it as early warning,
    never as sole evidence.
    """
    r = _base("cusum")
    rets = _s.returns_from_equity(daily_equity)
    if len(rets) < min_days:
        r["skipped"], r["skip_reason"] = True, \
            f"only {len(rets)} returns — need {min_days} for a CUSUM scan"
        return r
    resid = [mu - r_t for r_t, mu in
             zip(rets, [expected_daily_drift] * len(rets))]
    # Calibrate on the first half so a late break doesn't inflate k/h.
    half = max(len(resid) // 2, 5)
    sd = _s._sd(resid[:half]) or 1e-9
    k = ref_k if ref_k is not None else 0.5 * sd
    h = decision_h if decision_h is not None else 4.0 * sd
    s, signals, max_s = 0.0, [], 0.0
    for i, e in enumerate(resid):
        s = max(0.0, s + e - k)
        max_s = max(max_s, s)
        if s > h:
            signals.append(i + 1)  # 1-based day index into returns
            s = 0.0  # reset after signal (standard tabular CUSUM practice)
    r.update(observed=len(signals), expected=0, statistic=len(signals),
             detail=(f"CUSUM: {len(signals)} downward-shift signals "
                     f"(k={k:.5f}, h={h:.5f}); "
                     + (f"first at return-day {signals[0]}" if signals
                        else "no sustained degradation detected")),
             )
    r["signals"] = signals
    r["max_stat"] = max_s
    return r


def run_all(paper: dict, expected: dict,
            daily_equity: list[float] | None = None,
            expected_daily_drift: float | None = None) -> list[dict]:
    """Run every applicable test.  Returns the list of test dicts."""
    results = [test_sharpe(paper, expected), test_win_rate(paper, expected),
               test_payoff(paper, expected), test_drawdown(paper, expected),
               test_frequency(paper, expected)]
    if daily_equity is not None and expected_daily_drift is not None:
        results.append(cusum_scan(daily_equity, expected_daily_drift))
    return results
