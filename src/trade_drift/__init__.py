"""trade-drift: canonical model-drift detection for the trading suite.

Compares realized paper/live behavior against backtest expectations —
Sharpe, win rate, payoff, drawdown depth, trade frequency, plus a CUSUM
degradation scan — and grades OK / WATCH / VIOLATION with named reasons.
"""

from . import alerts, drift, expectations, stats, verdict
from .demo import run_demo

__version__ = "0.1.0"
__all__ = ["alerts", "drift", "expectations", "stats", "verdict",
           "run_demo", "__version__"]
