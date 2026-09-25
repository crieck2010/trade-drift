"""Shared statistical primitives.  Pure functions, stdlib only.

Every formula here is derived in docs/METHODOLOGY.md.  Nothing in this
module knows about strategies, ledgers, or verdicts — it is arithmetic.
"""

from __future__ import annotations

import math


# ------------------------------------------------------------- basics

def returns_from_equity(equity: list[float]) -> list[float]:
    """Simple per-period returns, skipping zero/negative prior equity."""
    rets = []
    for prev, cur in zip(equity, equity[1:]):
        if prev and prev > 0:
            rets.append(cur / prev - 1.0)
    return rets


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs: list[float], mu: float | None = None) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mu if mu is not None else _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def sharpe_ratio(returns: list[float], periods_per_year: float = 252.0) -> float:
    """Annualized Sharpe of a return series (zero if no variance)."""
    if len(returns) < 2:
        return 0.0
    mu, sd = _mean(returns), _sd(returns)
    return (mu / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def sharpe_se(sharpe_ann: float, n: int, periods_per_year: float = 252.0,
              hac_lags: int = 0, returns: list[float] | None = None) -> float:
    """Standard error of an annualized Sharpe estimate.

    Default is Lo (2002) under iid returns:
        SE_ann = sqrt((1 + S_p^2 / 2) / N) * sqrt(ppy)
    where S_p is the per-period Sharpe.  With ``hac_lags > 0`` the SE is
    inflated by a Bartlett-kernel autocorrelation factor computed from
    ``returns`` (Newey–West spirit):  sqrt(1 + 2*Σ_{l=1..L} (1-l/(L+1)) ρ_l).
    This is an approximation, documented as such in METHODOLOGY.md — it
    widens bands for autocorrelated strategies (e.g. trend followers)
    instead of pretending their Sharpe is precisely estimated.
    """
    if n < 2:
        return float("inf")
    s_p = sharpe_ann / math.sqrt(periods_per_year)
    se = math.sqrt((1.0 + 0.5 * s_p * s_p) / n) * math.sqrt(periods_per_year)
    if hac_lags > 0 and returns and len(returns) > hac_lags + 1:
        mu = _mean(returns)
        var = sum((r - mu) ** 2 for r in returns)
        if var > 0:
            adj = 0.0
            for lag in range(1, hac_lags + 1):
                w = 1.0 - lag / (hac_lags + 1)
                cov = sum((returns[i] - mu) * (returns[i - lag] - mu)
                          for i in range(lag, len(returns)))
                adj += w * cov / var
            se *= math.sqrt(max(1.0 + 2.0 * adj, 1e-9))
    return se


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p(z: float) -> float:
    """Two-sided p-value for a standard-normal statistic."""
    if not math.isfinite(z):
        return 1.0
    return 2.0 * (1.0 - normal_cdf(abs(z)))


# ------------------------------------------------- binomial (win rate)

def binomial_test(x: int, n: int, p0: float) -> dict:
    """Continuity-corrected normal approximation to the binomial test.

    H0: true win rate = p0.  Returns {z, p_value, skipped, skip_reason}.
    Skips when n == 0 or p0 is degenerate (0 or 1): with no variation in
    the null there is nothing to test.
    """
    if n <= 0:
        return {"z": 0.0, "p_value": 1.0, "skipped": True,
                "skip_reason": "no trades yet"}
    if p0 <= 0.0 or p0 >= 1.0:
        return {"z": 0.0, "p_value": 1.0, "skipped": True,
                "skip_reason": f"expected win rate {p0} is degenerate"}
    var = n * p0 * (1.0 - p0)
    if var <= 0:
        return {"z": 0.0, "p_value": 1.0, "skipped": True,
                "skip_reason": "zero null variance"}
    z = (abs(x - n * p0) - 0.5) / math.sqrt(var)
    z = max(z, 0.0)  # continuity correction can overshoot at tiny n
    return {"z": z, "p_value": two_sided_p(z), "skipped": False,
            "skip_reason": ""}


# -------------------------------------------------- Poisson (frequency)

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # log-domain for stability, then exponentiate
    logp = -lam + k * math.log(lam) - math.lgamma(k + 1)
    return math.exp(logp)


def poisson_two_sided_p(k: int, lam: float) -> float:
    """Two-sided Poisson p-value: total mass of outcomes with pmf <= pmf(k).

    The "pmf-ordering" two-sided test.  Conservative and honest for small
    counts; documented in METHODOLOGY.md.
    """
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    p_obs = poisson_pmf(k, lam)
    total, j = 0.0, 0
    # Walk out until the tail mass beyond j is negligible.
    while True:
        p = poisson_pmf(j, lam)
        if p <= p_obs:
            total += p
        j += 1
        if j > lam + 12 * math.sqrt(lam) + 20 and p < 1e-12:
            break
        if j > 100000:  # pathological guard
            break
    return min(total, 1.0)


# ------------------------------------------- payoff ratio (delta method)

def payoff_se(avg_win: float, avg_loss: float, n_w: int, n_l: int,
              sd_w: float, sd_l: float) -> float | None:
    """Delta-method SE of payoff = avg_win / avg_loss.

    Works on log(payoff): Var(log R) ≈ (sd_w/(avg_win·√n_w))²
    + (sd_l/(avg_loss·√n_l))², then SE(R) = R·√Var(log R).
    Returns None when the inputs can't support it (too few trades or
    degenerate averages) — callers must treat None as "skip", never as 0.
    """
    if n_w < 2 or n_l < 2:
        return None
    if avg_win <= 0 or avg_loss <= 0:
        return None
    var_log = ((sd_w / (avg_win * math.sqrt(n_w))) ** 2
               + (sd_l / (avg_loss * math.sqrt(n_l))) ** 2)
    if not math.isfinite(var_log) or var_log < 0:
        return None
    return (avg_win / avg_loss) * math.sqrt(var_log)


# ------------------------------------------------------------- drawdown

def max_drawdown(equity: list[float]) -> dict:
    """Max drawdown and where it happened.  Fractions, not percents."""
    peak = float("-inf")
    max_dd, peak_i, trough_i, cur_peak_i = 0.0, 0, 0, 0
    for i, e in enumerate(equity):
        if e > peak:
            peak, cur_peak_i = e, i
        if peak > 0:
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd, peak_i, trough_i = dd, cur_peak_i, i
    return {"max_drawdown": max_dd, "peak_index": peak_i,
            "trough_index": trough_i}


# ------------------------------------------------- paper-side summary

def paper_stats(daily_equity: list[float], trades: list[dict],
                periods_per_year: float = 252.0,
                hac_lags: int = 0) -> dict:
    """Summarize realized paper/live history in backtest-comparable metrics.

    Same contract as trade-incubate's paper_stats (intentionally), plus the
    win/loss standard deviations the delta-method payoff test needs.
    ``trades``: closed trades, each with at least ``pnl`` (currency units).
    """
    stats: dict = {"n_days": len(daily_equity), "n_trades": len(trades)}
    rets = returns_from_equity(daily_equity)
    stats["n_returns"] = len(rets)
    stats["sharpe"] = sharpe_ratio(rets, periods_per_year)
    stats["sharpe_se"] = sharpe_se(stats["sharpe"], len(rets),
                                   periods_per_year, hac_lags, rets)
    stats.update(max_drawdown(daily_equity))
    wins = [float(t["pnl"]) for t in trades if float(t.get("pnl", 0.0)) > 0]
    losses = [-float(t["pnl"]) for t in trades if float(t.get("pnl", 0.0)) < 0]
    n = len(trades)
    stats["n_wins"], stats["n_losses"] = len(wins), len(losses)
    stats["win_rate"] = len(wins) / n if n else 0.0
    stats["avg_win"] = _mean(wins)
    stats["avg_loss"] = _mean(losses)
    stats["sd_win"] = _sd(wins)
    stats["sd_loss"] = _sd(losses)
    stats["payoff"] = (stats["avg_win"] / stats["avg_loss"]
                       if stats["avg_loss"] > 0 else 0.0)
    stats["trades_per_day"] = n / len(daily_equity) if daily_equity else 0.0
    return stats
