"""Tests for drift.py — each statistic on hand-computed synthetic data."""

import math

import pytest

from trade_drift import drift as d
from trade_drift import stats as s


def _paper(**kw):
    base = {"sharpe": 0.0, "sharpe_se": float("inf"), "n_trades": 0,
            "n_wins": 0, "n_losses": 0, "win_rate": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "sd_win": 0.0, "sd_loss": 0.0, "payoff": 0.0,
            "max_drawdown": 0.0, "trades_per_day": 0.0, "n_days": 0,
            "n_returns": 0}
    base.update(kw)
    return base


def test_sharpe_z_hand_computed():
    # (0.2 - 1.5) / 0.5 = -2.6
    r = d.test_sharpe(_paper(sharpe=0.2, sharpe_se=0.5),
                      {"expected_sharpe": 1.5})
    assert r["statistic"] == pytest.approx(-2.6)
    assert r["p_value"] == pytest.approx(2 * (1 - s.normal_cdf(2.6)))
    assert r["skipped"] is False


def test_sharpe_combines_backtest_se():
    # sqrt(0.4^2 + 0.3^2) = 0.5 → z = (1.0-1.5)/0.5 = -1.0
    r = d.test_sharpe(_paper(sharpe=1.0, sharpe_se=0.4),
                      {"expected_sharpe": 1.5, "sharpe_se": 0.3})
    assert r["se"] == pytest.approx(0.5)
    assert r["statistic"] == pytest.approx(-1.0)


def test_sharpe_skips_without_expectation_or_data():
    assert d.test_sharpe(_paper(), {})["skipped"] is True
    assert d.test_sharpe(_paper(sharpe=1.0), {"expected_sharpe": 1.5})["skipped"] is True


def test_win_rate_z_hand_computed():
    # 40/100 vs 0.55: (|40-55|-0.5)/sqrt(100*.55*.45) = 14.5/4.9749
    r = d.test_win_rate(_paper(n_trades=100, n_wins=40),
                        {"expected_win_rate": 0.55})
    assert r["statistic"] == pytest.approx(14.5 / math.sqrt(24.75))
    assert r["observed"] == pytest.approx(0.40)
    assert r["skipped"] is False


def test_win_rate_skips():
    assert d.test_win_rate(_paper(), {"expected_win_rate": 0.5})["skipped"] is True
    assert d.test_win_rate(_paper(n_trades=10, n_wins=5), {})["skipped"] is True


def test_payoff_z_and_skips():
    paper = _paper(n_trades=50, n_wins=30, n_losses=20, avg_win=130.0,
                   avg_loss=100.0, sd_win=25.0, sd_loss=18.0, payoff=1.3)
    r = d.test_payoff(paper, {"expected_payoff": 1.6})
    se = s.payoff_se(130.0, 100.0, 30, 20, 25.0, 18.0)
    assert r["statistic"] == pytest.approx((1.3 - 1.6) / se)
    assert r["skipped"] is False
    # too few trades → skip
    assert d.test_payoff(_paper(n_trades=4), {"expected_payoff": 1.6})["skipped"] is True
    # no losing trades → undefined, skip
    p2 = _paper(n_trades=10, n_wins=10, n_losses=0, avg_win=50.0, avg_loss=0.0)
    assert d.test_payoff(p2, {"expected_payoff": 1.6})["skipped"] is True
    # no expectation → skip
    assert d.test_payoff(paper, {})["skipped"] is True


def test_drawdown_ratio():
    r = d.test_drawdown(_paper(max_drawdown=0.12),
                        {"expected_max_drawdown": 0.06})
    assert r["statistic"] == pytest.approx(2.0)
    assert r["skipped"] is False
    # better-than-expected is measured but never drift (one-sided)
    r2 = d.test_drawdown(_paper(max_drawdown=0.03),
                         {"expected_max_drawdown": 0.06})
    assert r2["statistic"] == pytest.approx(0.5)
    assert d.test_drawdown(_paper(), {})["skipped"] is True


def test_frequency_poisson():
    # 54 trades in 60d vs 0.9/day → lam=54: not significant
    r = d.test_frequency(_paper(n_trades=54, n_days=60),
                         {"expected_trades_per_day": 0.9})
    assert r["p_value"] > 0.2
    # 10 trades in 60d vs 54 expected → significant
    r2 = d.test_frequency(_paper(n_trades=10, n_days=60),
                          {"expected_trades_per_day": 0.9})
    assert r2["p_value"] < 0.01
    # too early → skip
    assert d.test_frequency(_paper(n_trades=0, n_days=3),
                            {"expected_trades_per_day": 0.9})["skipped"] is True


def test_cusum_detects_planted_break():
    eq = [100.0]
    for _ in range(40):
        eq.append(eq[-1] * 1.001)
    for _ in range(20):
        eq.append(eq[-1] * 0.99)  # sustained -1%/day break
    r = d.cusum_scan(eq, expected_daily_drift=0.001)
    assert r["skipped"] is False
    assert r["observed"] >= 1  # at least one signal


def test_cusum_quiet_on_healthy():
    import random
    rng = random.Random(11)
    eq = [100.0]
    for _ in range(60):
        eq.append(eq[-1] * (1 + rng.gauss(0.001, 0.01)))
    r = d.cusum_scan(eq, expected_daily_drift=0.001)
    assert r["skipped"] is False
    assert r["observed"] == 0


def test_cusum_skips_when_short():
    r = d.cusum_scan([100.0, 101.0], expected_daily_drift=0.001)
    assert r["skipped"] is True


def test_run_all_counts():
    paper = _paper(n_days=60, n_trades=50, n_wins=28, sharpe=1.2, sharpe_se=0.8,
                   max_drawdown=0.05)
    exp = {"expected_sharpe": 1.5, "expected_win_rate": 0.55,
           "expected_payoff": 1.6, "expected_max_drawdown": 0.06,
           "expected_trades_per_day": 0.9}
    eq = [100.0 * (1.001 ** i) for i in range(61)]
    with_cusum = d.run_all(paper, exp, daily_equity=eq, expected_daily_drift=0.001)
    without = d.run_all(paper, exp)
    assert len(with_cusum) == 6 and len(without) == 5
    assert {t["metric"] for t in with_cusum} == {
        "sharpe", "win_rate", "payoff", "max_drawdown", "trade_frequency",
        "cusum"}


def test_sharpe_outperformance_also_flagged_two_sided():
    # Drift is divergence: a surprisingly GOOD Sharpe is still divergence
    # (expectations were wrong, or a regime tailwind).
    r = d.test_sharpe(_paper(sharpe=4.0, sharpe_se=1.0),
                      {"expected_sharpe": 1.0})
    assert r["statistic"] == pytest.approx(3.0)


def test_frequency_overtrading_flagged():
    r = d.test_frequency(_paper(n_trades=120, n_days=60),
                         {"expected_trades_per_day": 0.9})
    assert r["p_value"] < 0.01
