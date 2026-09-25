"""Backtest expectations: the "what should happen" side of drift detection.

An Expectation is plain data (schema_version 1): the backtest values each
drift test compares against, plus provenance pointing at the report they
came from.  Missing fields are legal — checks that need them are skipped
with stated reasons, never silently assumed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# Canonical field names and what they mean.
FIELDS = {
    "expected_sharpe": "backtest annualized Sharpe ratio",
    "sharpe_se": "standard error of the backtest Sharpe (optional)",
    "expected_win_rate": "backtest win rate, 0..1",
    "expected_payoff": "backtest avg_win / avg_loss",
    "expected_max_drawdown": "backtest max drawdown, fraction (positive)",
    "expected_trades_per_day": "backtest trades per trading day",
    "n_backtest_trades": "number of backtest trades behind the expectations",
    "horizon_days": "backtest length in trading days (optional)",
    "source_report_id": "which ValidationReport these came from",
    "as_of": "ISO date these expectations were recorded",
}


def new_expectations(strategy_id: str, source_report_id: str = "",
                     **fields) -> dict:
    """Build an Expectation dict, validating ranges.  Unknown fields rejected."""
    unknown = [k for k in fields if k not in FIELDS]
    if unknown:
        raise ValueError(f"unknown expectation fields: {unknown}")
    exp: dict = {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "source_report_id": source_report_id,
        "as_of": datetime.now(timezone.utc).date().isoformat(),
    }
    for key, desc in FIELDS.items():
        if key in ("source_report_id", "as_of"):
            continue
        exp[key] = fields.get(key)  # None = unknown → checks skip
    wr = exp.get("expected_win_rate")
    if wr is not None and not 0.0 <= wr <= 1.0:
        raise ValueError("expected_win_rate must be in [0, 1]")
    dd = exp.get("expected_max_drawdown")
    if dd is not None and dd < 0:
        raise ValueError("expected_max_drawdown must be >= 0")
    for key in ("expected_payoff", "expected_trades_per_day"):
        v = exp.get(key)
        if v is not None and v < 0:
            raise ValueError(f"{key} must be >= 0")
    exp["missing"] = sorted(k for k in FIELDS
                            if k not in ("source_report_id", "as_of")
                            and exp.get(k) is None)
    return exp


def _num(report: dict, *path, default=None):
    cur = report
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if isinstance(cur, (int, float)) else default


def from_validation_report(report: dict, strategy_id: str = "",
                           n_days: float | None = None) -> dict:
    """Derive expectations from a trade-swing ValidationReport (schema v1).

    Metric keys in trade-swing's ``metrics`` block: ``sharpe_ratio``,
    ``win_rate``, ``profit_factor``, ``max_drawdown``, ``num_trades``.
    Payoff (avg_win/avg_loss) is recovered from profit factor and win rate:
        payoff = profit_factor × (1 − p) / p
    because profit_factor = (n_w·avg_win)/(n_l·avg_loss) = (p/(1−p))·payoff.
    ``n_days`` (backtest trading days) is not in the report schema; pass it
    to get a trades-per-day expectation, otherwise that check skips.
    Fail-soft: unrecognized/absent values become None (skip), never guesses.
    """
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("expected a trade-swing ValidationReport schema v1")
    metrics = report.get("backtest", {}).get("metrics", {}) or {}
    n_trades = _num(report, "backtest", "n_trades")
    p = _num(metrics, "win_rate")
    pf = _num(metrics, "profit_factor")
    payoff = None
    if pf is not None and p is not None and 0.0 < p < 1.0:
        payoff = pf * (1.0 - p) / p
    tpd = None
    if n_trades is not None and n_days:
        tpd = n_trades / n_days
    dd = _num(metrics, "max_drawdown")
    return new_expectations(
        strategy_id or str(report.get("strategy", {}).get("name", "")),
        source_report_id=str(report.get("generated_utc", "")),
        expected_sharpe=_num(metrics, "sharpe_ratio"),
        sharpe_se=None,  # trade-swing reports don't carry it; recompute if needed
        expected_win_rate=p,
        expected_payoff=payoff,
        expected_max_drawdown=abs(dd) if dd is not None else None,
        expected_trades_per_day=tpd,
        n_backtest_trades=n_trades,
        horizon_days=n_days,
    )


def to_json(exp: dict, indent: int = 2) -> str:
    return json.dumps(exp, indent=indent, default=str)


def from_json(text: str) -> dict:
    exp = json.loads(text)
    if exp.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported expectations schema "
                         f"{exp.get('schema_version')}")
    return exp
