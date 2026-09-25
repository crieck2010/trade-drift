"""Tests for stats.py — every value hand-computable from the formula."""

import math

import pytest

from trade_drift import stats as s


def test_returns_from_equity():
    assert s.returns_from_equity([100.0, 110.0, 99.0]) == pytest.approx(
        [0.10, -0.10])
    # zero prior equity is skipped, not divided
    assert s.returns_from_equity([0.0, 100.0, 110.0]) == pytest.approx([0.10])


def test_sharpe_ratio_hand_computed():
    rets = [0.01, 0.02, -0.01, 0.015, -0.005]
    mu = sum(rets) / 5
    var = sum((r - mu) ** 2 for r in rets) / 4
    expected = mu / math.sqrt(var) * math.sqrt(252.0)
    assert s.sharpe_ratio(rets) == pytest.approx(expected)
    assert s.sharpe_ratio([0.01] * 10) == 0.0  # no variance
    assert s.sharpe_ratio([0.01]) == 0.0       # too short


def test_sharpe_se_lo_formula():
    # Lo (2002): SE = sqrt((1 + s_p^2/2)/N) * sqrt(ppy)
    sharpe_ann, n = 1.5, 252
    s_p = sharpe_ann / math.sqrt(252.0)
    expected = math.sqrt((1.0 + 0.5 * s_p * s_p) / n) * math.sqrt(252.0)
    assert s.sharpe_se(sharpe_ann, n) == pytest.approx(expected)
    assert s.sharpe_se(1.5, 252) == pytest.approx(1.0022, abs=1e-3)
    assert s.sharpe_se(1.5, 1) == float("inf")


def test_sharpe_se_hac_widens_for_autocorrelated():
    # AR(1)-ish returns: HAC SE must exceed the iid SE.
    rng_rets = [0.01 * (0.6 ** 0) ]  # placeholder replaced below
    rets = []
    x = 0.0
    import random
    rng = random.Random(3)
    for _ in range(300):
        x = 0.7 * x + rng.gauss(0, 0.01)
        rets.append(x)
    iid = s.sharpe_se(s.sharpe_ratio(rets), len(rets))
    hac = s.sharpe_se(s.sharpe_ratio(rets), len(rets), hac_lags=5, returns=rets)
    assert hac > iid


def test_normal_cdf_and_p():
    assert s.normal_cdf(0.0) == pytest.approx(0.5)
    assert s.normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert s.two_sided_p(0.0) == pytest.approx(1.0)
    assert s.two_sided_p(1.96) == pytest.approx(0.05, abs=1e-3)


def test_binomial_hand_computed():
    # 60 wins of 100 vs p0=0.5: z = (|60-50|-0.5)/5 = 1.9
    t = s.binomial_test(60, 100, 0.5)
    assert t["z"] == pytest.approx(1.9)
    assert t["p_value"] == pytest.approx(2 * (1 - s.normal_cdf(1.9)))
    assert t["skipped"] is False


def test_binomial_skips():
    assert s.binomial_test(0, 0, 0.5)["skipped"] is True
    assert s.binomial_test(5, 10, 1.0)["skipped"] is True
    assert s.binomial_test(5, 10, 0.0)["skipped"] is True


def test_poisson_pmf():
    assert s.poisson_pmf(0, 1.0) == pytest.approx(math.exp(-1))
    assert s.poisson_pmf(3, 2.0) == pytest.approx(math.exp(-2) * 8 / 6)


def test_poisson_two_sided():
    # observed == mean → not significant
    assert s.poisson_two_sided_p(10, 10.0) > 0.5
    # far tail → significant
    assert s.poisson_two_sided_p(0, 10.0) < 0.01
    assert s.poisson_two_sided_p(25, 10.0) < 0.01


def test_payoff_se_delta_method():
    # hand-computed via the log-delta formula
    se = s.payoff_se(160.0, 100.0, 30, 20, 30.0, 20.0)
    var_log = (30.0 / (160.0 * math.sqrt(30))) ** 2 + \
        (20.0 / (100.0 * math.sqrt(20))) ** 2
    assert se == pytest.approx(1.6 * math.sqrt(var_log))
    assert se == pytest.approx(0.0901, abs=1e-3)


def test_payoff_se_none_when_unsupported():
    assert s.payoff_se(160.0, 100.0, 1, 20, 30.0, 20.0) is None
    assert s.payoff_se(160.0, 0.0, 30, 20, 30.0, 20.0) is None


def test_max_drawdown():
    dd = s.max_drawdown([100.0, 110.0, 105.0, 95.0, 100.0])
    assert dd["max_drawdown"] == pytest.approx((110 - 95) / 110)
    assert dd["peak_index"] == 1 and dd["trough_index"] == 3


def test_paper_stats_contract():
    eq = [100.0 + i for i in range(21)]
    trades = [{"pnl": 10.0}] * 6 + [{"pnl": -5.0}] * 4
    st = s.paper_stats(eq, trades)
    assert st["n_days"] == 21 and st["n_trades"] == 10
    assert st["win_rate"] == pytest.approx(0.6)
    assert st["avg_win"] == pytest.approx(10.0)
    assert st["avg_loss"] == pytest.approx(5.0)
    assert st["payoff"] == pytest.approx(2.0)
    assert st["trades_per_day"] == pytest.approx(10 / 21)
    assert math.isfinite(st["sharpe_se"])
    for key in ("sharpe", "max_drawdown", "sd_win", "sd_loss"):
        assert key in st


def test_sharpe_se_hac_zero_lags_is_iid():
    rets = [0.01, -0.005, 0.02, -0.01, 0.008] * 20
    assert s.sharpe_se(1.2, len(rets), hac_lags=0, returns=rets) == pytest.approx(
        s.sharpe_se(1.2, len(rets)))


def test_poisson_degenerate_lambda():
    assert s.poisson_two_sided_p(0, 0.0) == 1.0
    assert s.poisson_two_sided_p(3, 0.0) == 0.0
    assert s.poisson_pmf(0, 0.0) == 1.0
