"""Lazy, fail-soft bridges to sibling repos.  No hard imports.

- trade-swing ValidationReport  → expectations
- trade-incubate daily JSONL    → equity curve + closed trades
- trade-paper SQLite ledger     → equity snapshots + fills (FIFO-paired)
- trade-lifecycle pause gate    → kill-condition payload (informs only)

Every reader returns ``{"ok": bool, "reason": str, ...}``.  A missing
sibling, an unreadable file, or an unrecognized schema is a *skipped*
read with a reason — never an exception, never a silent zero.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3

from . import expectations as _exp


def sibling_available(name: str) -> bool:
    """Is the sibling package importable in this environment?"""
    return importlib.util.find_spec(name) is not None


# ------------------------------------------------- trade-swing reports

def expectations_from_trade_swing(report_or_path, strategy_id: str = "",
                                  n_days: float | None = None) -> dict:
    """Read a ValidationReport (dict or JSON path) → Expectation dict."""
    try:
        report = (json.load(open(report_or_path, encoding="utf-8"))
                  if isinstance(report_or_path, (str, os.PathLike))
                  else report_or_path)
        exp = _exp.from_validation_report(report, strategy_id, n_days)
        return {"ok": True, "reason": "", "expectations": exp}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"cannot read validation report: {exc}"}


# ------------------------------------------- trade-incubate daily JSONL

def ledger_from_trade_incubate(path: str) -> dict:
    """Read a trade-incubate daily JSONL ledger.

    Day records carry ``equity`` and ``trades`` (each with ``pnl``) —
    see trade-incubate's IncubationLedger.
    """
    try:
        days = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    days.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"cannot read ledger: {exc}"}
    equity = [float(d["equity"]) for d in days if "equity" in d]
    trades: list[dict] = []
    for d in days:
        for t in d.get("trades", []) or []:
            if isinstance(t, dict) and "pnl" in t:
                trades.append({"pnl": float(t["pnl"])})
    if not equity:
        return {"ok": False, "reason": "ledger has no equity records"}
    return {"ok": True, "reason": "", "equity": equity, "trades": trades,
            "n_days": len(days)}


# ------------------------------------------------ trade-paper SQLite

def _fifo_trade_pnls(fills: list[dict]) -> list[dict]:
    """Approximate closed-trade pnls from fills, FIFO per symbol.

    Paper fills are executions, not strategy trades: pairing buys with
    sells FIFO is an approximation.  Exact strategy-level pnl belongs to
    the strategy's own ledger; this is documented, not hidden.
    """
    lots: dict[str, list] = {}  # symbol -> [(qty, price)]
    pnls: list[dict] = []
    for f in fills:
        sym = f["symbol"]
        qty, price = float(f["quantity"]), float(f["price"])
        side = str(f["side"]).lower()
        book = lots.setdefault(sym, [])
        # Opening direction inferred from first fill side for the symbol.
        if side in ("buy", "buy_to_open", "bto"):
            book.append([qty, price])
        else:  # sell / close against long lots
            need = qty
            while need > 1e-12 and book:
                lot_qty, lot_px = book[0]
                take = min(need, lot_qty)
                pnls.append({"pnl": take * (price - lot_px),
                             "symbol": sym})
                lot_qty -= take
                need -= take
                if lot_qty <= 1e-12:
                    book.pop(0)
                else:
                    book[0][0] = lot_qty
    return pnls


def ledger_from_trade_paper(db_path: str, strategy: str | None = None) -> dict:
    """Read equity snapshots + fills from a trade-paper SQLite ledger."""
    if not os.path.exists(db_path):
        return {"ok": False, "reason": f"no such ledger file: {db_path}"}
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in
                  db.execute("SELECT name FROM sqlite_master "
                             "WHERE type='table'")}
        if not {"equity_snapshots", "fills"} <= tables:
            return {"ok": False,
                    "reason": "not a trade-paper ledger (missing "
                              "equity_snapshots/fills tables)"}
        snaps = db.execute("SELECT at, equity FROM equity_snapshots "
                           "ORDER BY at").fetchall()
        equity = [float(r["equity"]) for r in snaps]
        q = ("SELECT symbol, side, quantity, price FROM fills "
             "ORDER BY filled_at")
        fills = [dict(r) for r in db.execute(q).fetchall()]
        db.close()
    except sqlite3.Error as exc:
        return {"ok": False, "reason": f"sqlite read failed: {exc}"}
    if not equity:
        return {"ok": False, "reason": "ledger has no equity snapshots"}
    trades = _fifo_trade_pnls(fills)
    note = ("trade pnls are FIFO-paired fills (approximation); "
            "exact strategy pnl belongs to the strategy ledger")
    _ = strategy  # per-strategy filtering is a future v0.2.0 item
    return {"ok": True, "reason": "", "equity": equity, "trades": trades,
            "n_days": len(equity), "note": note}


# --------------------------------------- trade-lifecycle pause payload

def kill_condition_input(strategy_id: str, report: dict) -> dict:
    """Build the plain-data payload trade-lifecycle's pause gate consumes.

    This *informs* — it never pauses anything itself.  The lifecycle's
    ``gate_pause`` (and the human operating it) decides.  Separate failure
    domains by design: a broken drift engine must not be able to halt a
    healthy strategy, and a healthy drift engine must not be bypassable
    by a broken one — the pause decision always stays with the committee.
    """
    violated = sorted(m for m, g in report.get("grades", {}).items()
                      if g == "VIOLATION")
    return {
        "schema_version": 1,
        "strategy_id": strategy_id,
        "condition": "drift_violation" if violated else "drift_watch",
        "at": report.get("as_of", ""),
        "metrics": violated,
        "reasons": list(report.get("reasons", [])),
        "profile": report.get("profile", ""),
    }
