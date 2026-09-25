"""Verdicts: turn test statistics into OK / WATCH / VIOLATION.

Grading is on deviation size, not p-values alone: a z-score maps to bands,
drawdown uses exceedance-ratio bands, CUSUM uses signal counts.  Profiles
(strict / balanced / lenient) move the bands; the composite rule and the
consecutive-WATCH escalation are profile-independent.

Composite rule:
  - any metric VIOLATION  → overall VIOLATION
  - ≥2 metrics WATCH      → overall VIOLATION (correlated smoke is fire)
  - else the worst metric verdict wins.

Consecutive-WATCH escalation (stateful): a metric graded WATCH on
``streak_threshold`` consecutive checks escalates to VIOLATION — quiet,
persistent underperformance is exactly the decay this engine exists to
catch.  Streaks live in a tiny JSON state file; OK or VIOLATION resets a
metric's streak, skipped checks leave it untouched (missing data neither
confirms nor breaks a streak).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

SCHEMA_VERSION = 1

LEVELS = ("OK", "WATCH", "VIOLATION", "SKIPPED")

PROFILES = {
    # z-bands for sampling tests, exceedance-ratio bands for drawdown.
    "strict":   {"z_watch": 1.5, "z_violation": 2.5,
                 "dd_watch": 1.15, "dd_violation": 1.35},
    "balanced": {"z_watch": 2.0, "z_violation": 3.0,
                 "dd_watch": 1.25, "dd_violation": 1.50},
    "lenient":  {"z_watch": 2.5, "z_violation": 3.5,
                 "dd_watch": 1.40, "dd_violation": 1.75},
}

DEFAULT_PROFILE = "balanced"
STREAK_THRESHOLD = 3


def grade_z(z: float, profile: str = DEFAULT_PROFILE) -> str:
    p = PROFILES[profile]
    az = abs(z)
    if az >= p["z_violation"]:
        return "VIOLATION"
    if az >= p["z_watch"]:
        return "WATCH"
    return "OK"


def grade_drawdown(ratio: float, profile: str = DEFAULT_PROFILE) -> str:
    p = PROFILES[profile]
    if ratio >= p["dd_violation"]:
        return "VIOLATION"
    if ratio >= p["dd_watch"]:
        return "WATCH"
    return "OK"


def grade_cusum(n_signals: int) -> str:
    # Heuristic scan: any sustained-degradation signal is worth a WATCH.
    # It is deliberately never a lone VIOLATION — CUSUM here is early
    # warning, not proof (see drift.cusum_scan).
    return "WATCH" if n_signals > 0 else "OK"


def grade_test(test: dict, profile: str = DEFAULT_PROFILE) -> str:
    """Grade one test-result dict from drift.run_all."""
    if test.get("skipped"):
        return "SKIPPED"
    metric, stat = test["metric"], test["statistic"]
    if metric == "max_drawdown":
        return grade_drawdown(stat, profile)
    if metric == "cusum":
        return grade_cusum(int(stat))
    return grade_z(stat, profile)


def composite(grades: dict[str, str]) -> tuple[str, list[str]]:
    """Overall verdict + reasons from per-metric grades (SKIPPED ignored)."""
    active = {m: g for m, g in grades.items() if g != "SKIPPED"}
    reasons: list[str] = []
    if not active:
        return "OK", ["no applicable tests — nothing to judge"]
    viol = sorted(m for m, g in active.items() if g == "VIOLATION")
    watch = sorted(m for m, g in active.items() if g == "WATCH")
    if viol:
        reasons.append(f"VIOLATION on: {', '.join(viol)}")
        return "VIOLATION", reasons
    if len(watch) >= 2:
        reasons.append(f"multiple WATCH metrics ({', '.join(watch)}): "
                       "correlated smoke is treated as fire")
        return "VIOLATION", reasons
    if watch:
        reasons.append(f"WATCH on: {', '.join(watch)}")
        return "WATCH", reasons
    return "OK", ["all tested metrics within bands"]


# ------------------------------------------------- consecutive-WATCH state

class WatchState:
    """Tiny JSON store of per-strategy, per-metric WATCH streaks."""

    def __init__(self, path: str | None = None,
                 streak_threshold: int = STREAK_THRESHOLD) -> None:
        self.path = path
        self.streak_threshold = streak_threshold
        self._state: dict = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self._state = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._state = {}

    def streaks(self, strategy_id: str) -> dict:
        return self._state.setdefault(strategy_id, {})

    def record(self, strategy_id: str, grades: dict[str, str]) -> list[dict]:
        """Update streaks; return escalations (metric, streak length)."""
        streak = self.streaks(strategy_id)
        escalated: list[dict] = []
        for metric, grade in grades.items():
            if grade == "SKIPPED":
                continue  # missing data neither confirms nor breaks a streak
            if grade == "WATCH":
                streak[metric] = int(streak.get(metric, 0)) + 1
                if streak[metric] >= self.streak_threshold:
                    escalated.append({"metric": metric,
                                      "streak": streak[metric]})
            else:
                streak[metric] = 0
        # Drop metrics no longer reported so stale streaks can't linger.
        for metric in list(streak):
            if metric not in grades:
                del streak[metric]
        self.save()
        return escalated

    def save(self) -> None:
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
            os.replace(tmp, self.path)


def verdict_report(strategy_id: str, test_results: list[dict],
                   profile: str = DEFAULT_PROFILE,
                   state: WatchState | None = None,
                   as_of: str | None = None) -> dict:
    """Full verdict document: per-metric grades, escalation, composite.

    schema_version 1.  Every grade carries its reason; escalations name
    the streak length.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; "
                         f"choose from {sorted(PROFILES)}")
    grades = {t["metric"]: grade_test(t, profile) for t in test_results}
    reasons = [f"{t['metric']}: {t['detail'] or t['skip_reason'] or t.get('verdict','')}"
               for t in test_results]
    escalated: list[dict] = []
    if state is not None:
        for esc in state.record(strategy_id, grades):
            grades[esc["metric"]] = "VIOLATION"
            escalated.append(esc)
            reasons.append(f"{esc['metric']}: WATCH on {esc['streak']} "
                           f"consecutive checks → escalated to VIOLATION")
    overall, comp_reasons = composite(grades)
    reasons.extend(comp_reasons)
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "as_of": as_of or datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "grades": grades,
        "escalated": escalated,
        "verdict": overall,
        "reasons": reasons,
        "tests": test_results,
    }
