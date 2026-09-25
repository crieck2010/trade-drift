"""Named verdict profiles: strict / balanced / lenient.

A profile only moves the deviation bands (see verdict.PROFILES).  The
composite rule and the consecutive-WATCH escalation are deliberately
profile-independent — tolerance for noise is configurable, tolerance for
persistent decay is not.
"""

from __future__ import annotations

from .verdict import PROFILES, DEFAULT_PROFILE

DESCRIPTIONS = {
    "strict": ("flags deviations early (|z|≥1.5 WATCH, ≥2.5 VIOLATION; "
               "drawdown 1.15x/1.35x). More false alarms, faster catches. "
               "For strategies with long, clean backtests behind them."),
    "balanced": ("the default: |z|≥2.0 WATCH, ≥3.0 VIOLATION; drawdown "
                 "1.25x/1.50x — the same bands trade-incubate v0.1.0 used."),
    "lenient": ("wide bands (|z|≥2.5/3.5; drawdown 1.40x/1.75x) for short "
                "or noisy histories where statistical power is low anyway."),
}


def describe() -> list[dict]:
    return [{"name": name,
             "z_watch": p["z_watch"], "z_violation": p["z_violation"],
             "dd_watch": p["dd_watch"], "dd_violation": p["dd_violation"],
             "description": DESCRIPTIONS[name]}
            for name, p in PROFILES.items()]
