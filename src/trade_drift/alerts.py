"""Alerts: records, a phone-readable digest, and a notification hook spec.

This engine *detects* drift and *records* alerts.  It never sends anything
itself — delivery (push, SMS, email) belongs to a notification channel the
operator configures.  ``notify()`` documents the contract and reports
``sent: False`` until a channel is registered, so "nobody was told" is
always visible, never silent.
"""

from __future__ import annotations

from datetime import datetime, timezone

_notifiers: dict = {}


def make_alert(strategy_id: str, level: str, metric: str, message: str,
               at: str | None = None) -> dict:
    if level not in ("WATCH", "VIOLATION"):
        raise ValueError("alerts are only raised for WATCH/VIOLATION")
    return {
        "schema_version": 1,
        "at": at or datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy_id,
        "level": level,
        "metric": metric,
        "message": message,
    }


def alerts_from_report(report: dict) -> list[dict]:
    """One alert per non-OK metric grade in a verdict report."""
    alerts = []
    grades = report.get("grades", {})
    tests = {t["metric"]: t for t in report.get("tests", [])}
    for metric, grade in grades.items():
        if grade in ("WATCH", "VIOLATION"):
            t = tests.get(metric, {})
            detail = t.get("detail") or t.get("skip_reason") or ""
            alerts.append(make_alert(report.get("strategy_id", ""), grade,
                                     metric, detail, at=report.get("as_of")))
    return alerts


def digest(alerts: list[dict], strategy_id: str | None = None,
           max_items: int = 12) -> str:
    """Phone-readable summary: worst first, one line per alert."""
    if strategy_id:
        alerts = [a for a in alerts if a.get("strategy_id") == strategy_id]
    if not alerts:
        return "DRIFT: all clear — no WATCH/VIOLATION alerts."
    order = {"VIOLATION": 0, "WATCH": 1}
    alerts = sorted(alerts, key=lambda a: (order.get(a["level"], 9),
                                           a.get("at", "")))
    lines = [f"DRIFT ALERT — {len(alerts)} active "
             f"({sum(1 for a in alerts if a['level'] == 'VIOLATION')} violation, "
             f"{sum(1 for a in alerts if a['level'] == 'WATCH')} watch)"]
    for a in alerts[:max_items]:
        lines.append(f"[{a['level']}] {a['strategy_id']} · {a['metric']}: "
                     f"{a['message']}")
    if len(alerts) > max_items:
        lines.append(f"… and {len(alerts) - max_items} more")
    return "\n".join(lines)


# ------------------------------------------------------- hook spec

def register_notifier(name: str, fn) -> None:
    """Register a delivery channel: fn(alerts) -> dict receipt.

    ``fn`` may do anything (push/SMS/email); the engine only requires it
    to return a plain-data receipt describing what happened.
    """
    _notifiers[name] = fn


def notify(alerts: list[dict], channel: str | None = None) -> dict:
    """Deliver alerts through a registered channel.

    With no channel (the default), returns ``sent: False`` and says so —
    undelivered alerts are a visible fact, not a silent default.
    """
    if not alerts:
        return {"sent": False, "reason": "no alerts to deliver"}
    if channel is None:
        return {"sent": False,
                "reason": ("no notification channel configured — "
                           "this engine never sends anything itself; "
                           "register one with alerts.register_notifier()")}
    fn = _notifiers.get(channel)
    if fn is None:
        return {"sent": False,
                "reason": f"unknown channel {channel!r}; registered: "
                          f"{sorted(_notifiers) or 'none'}"}
    return {"sent": True, "channel": channel, "receipt": fn(alerts)}
