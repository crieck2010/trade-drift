"""trade-drift CLI: expect / check / watch / alerts / demo / presets."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import alerts as _alerts
from . import drift as _drift
from . import stats as _stats
from . import verdict as _verdict
from . import expectations as _exp
from . import adapters as _adapters
from . import presets as _presets
from . import licensing as _licensing
from .demo import run_demo, print_demo

EXIT = {"OK": 0, "WATCH": 1, "VIOLATION": 2}


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _ledger_from_path(path: str) -> dict:
    """Read a ledger from JSON/JSONL: {equity: [...], trades: [{pnl}]}."""
    if path.endswith(".jsonl"):
        rec = _adapters.ledger_from_trade_incubate(path)
        if not rec["ok"]:
            raise SystemExit(f"error: {rec['reason']}")
        return {"equity": rec["equity"], "trades": rec["trades"]}
    obj = _read_json(path)
    if isinstance(obj, dict) and "equity" in obj:
        return obj
    raise SystemExit("error: ledger JSON must be {equity: [...], trades: [{pnl}]}")


def _expect_from_path(path: str) -> dict:
    obj = _read_json(path)
    if obj.get("schema_version") == 1 and "expected_sharpe" in obj:
        return obj  # already an expectations file
    # otherwise assume a trade-swing ValidationReport
    rec = _adapters.expectations_from_trade_swing(obj)
    if not rec["ok"]:
        raise SystemExit(f"error: {rec['reason']}")
    return rec["expectations"]


def _print_report(rep: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(f"strategy: {rep['strategy_id']}  verdict: {rep['verdict']} "
              f"(profile: {rep['profile']})")
        for metric, grade in rep["grades"].items():
            print(f"  {metric:15s} {grade}")
        for reason in rep["reasons"]:
            print(f"  - {reason}")
        al = _alerts.alerts_from_report(rep)
        if al:
            print("--- digest ---")
            print(_alerts.digest(al))
    return EXIT.get(rep["verdict"], 0)


def cmd_expect(args) -> int:
    rec = _adapters.expectations_from_trade_swing(args.report,
                                                  args.strategy_id,
                                                  args.n_days)
    if not rec["ok"]:
        print(f"error: {rec['reason']}", file=sys.stderr)
        return 3
    exp = rec["expectations"]
    if args.json or not args.out:
        print(json.dumps(exp, indent=2, default=str))
    if args.out:
        _write_json(args.out, exp)
        print(f"wrote {args.out}")
        if exp.get("missing"):
            print(f"note: missing expectations (checks will skip): "
                  f"{', '.join(exp['missing'])}")
    return 0


def cmd_check(args) -> int:
    ledger = _ledger_from_path(args.ledger)
    exp = _expect_from_path(args.expect)
    paper = _stats.paper_stats(ledger["equity"], ledger.get("trades", []),
                               hac_lags=args.hac_lags)
    mu_daily = exp.get("expected_daily_drift")
    tests = _drift.run_all(paper, exp,
                           daily_equity=ledger["equity"] if mu_daily else None,
                           expected_daily_drift=mu_daily)
    state = _verdict.WatchState(args.state) if args.state else None
    rep = _verdict.verdict_report(args.strategy_id or exp.get("strategy_id", ""),
                                  tests, profile=args.profile, state=state)
    if args.out:
        _write_json(args.out, rep)
    return _print_report(rep, args.json)


def cmd_watch(args) -> int:
    """Rolling windows over one ledger: the practitioner's decay monitor."""
    ledger = _ledger_from_path(args.ledger)
    exp = _expect_from_path(args.expect)
    equity, trades = ledger["equity"], ledger.get("trades", [])
    n = len(equity) - 1
    mu_daily = exp.get("expected_daily_drift")
    state = _verdict.WatchState(args.state) if args.state else None
    rows = []
    end = n
    worst = "OK"
    while end >= args.window:
        win_eq = equity[end - args.window:end + 1]
        # Ledger schema here carries no trade timestamps, so take the
        # proportional slice (real ledgers filter trades by date).
        lo = int(len(trades) * (end - args.window) / n) if n else 0
        hi = int(len(trades) * end / n) if n else len(trades)
        win_tr = trades[max(lo, 0):hi] or trades[:1]
        paper = _stats.paper_stats(win_eq, win_tr)
        tests = _drift.run_all(paper, exp,
                               daily_equity=win_eq if mu_daily else None,
                               expected_daily_drift=mu_daily)
        rep = _verdict.verdict_report(
            args.strategy_id or exp.get("strategy_id", ""), tests,
            profile=args.profile, state=state,
            as_of=f"window-ending-day-{end}")
        rows.append({"end_day": end, "verdict": rep["verdict"],
                     "grades": rep["grades"]})
        if EXIT.get(rep["verdict"], 0) > EXIT.get(worst, 0):
            worst = rep["verdict"]
        end -= args.step
    rows.reverse()
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        metrics = sorted({m for r in rows for m in r["grades"]})
        header = f"{'end_day':>8}  {'verdict':>9}  " + "  ".join(f"{m[:6]:>6}" for m in metrics)
        print(header)
        for r in rows:
            cells = "  ".join(f"{r['grades'].get(m, '?')[:6]:>6}" for m in metrics)
            print(f"{r['end_day']:>8}  {r['verdict']:>9}  {cells}")
    return EXIT.get(worst, 0)


def cmd_alerts(args) -> int:
    rep = _read_json(args.report)
    al = _alerts.alerts_from_report(rep)
    if args.json:
        print(json.dumps(al, indent=2, default=str))
    else:
        print(_alerts.digest(al))
    if args.notify:
        print(json.dumps(_alerts.notify(al, channel=args.notify), indent=2))
    return 0


def cmd_demo(args) -> int:
    result = run_demo()
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_demo(result)
    return 0


def cmd_presets(args) -> int:
    rows = _presets.describe()
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['name']}: z WATCH ≥{r['z_watch']}, VIOLATION ≥{r['z_violation']}; "
                  f"drawdown {r['dd_watch']}x/{r['dd_violation']}x")
            print(f"  {r['description']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trade-drift",
                                description="Model-drift detection: realized vs backtest expectations.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("expect", help="derive expectations from a ValidationReport")
    e.add_argument("--report", required=True)
    e.add_argument("--strategy-id", default="")
    e.add_argument("--n-days", type=float, default=None,
                   help="backtest trading days (for trades/day expectation)")
    e.add_argument("--out", default="")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_expect)

    c = sub.add_parser("check", help="one drift check of a ledger vs expectations")
    c.add_argument("--ledger", required=True,
                   help="{equity,trades} JSON or trade-incubate JSONL")
    c.add_argument("--expect", required=True)
    c.add_argument("--strategy-id", default="")
    c.add_argument("--profile", default="balanced",
                   choices=sorted(_verdict.PROFILES))
    c.add_argument("--state", default="",
                   help="watch-state JSON for consecutive-WATCH escalation")
    c.add_argument("--hac-lags", type=int, default=0,
                   help="Newey-West-style lags for the Sharpe SE (0 = Lo iid)")
    c.add_argument("--out", default="")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    w = sub.add_parser("watch", help="rolling-window drift scan over a ledger")
    w.add_argument("--ledger", required=True)
    w.add_argument("--expect", required=True)
    w.add_argument("--strategy-id", default="")
    w.add_argument("--window", type=int, default=20)
    w.add_argument("--step", type=int, default=5)
    w.add_argument("--profile", default="balanced",
                   choices=sorted(_verdict.PROFILES))
    w.add_argument("--state", default="")
    w.add_argument("--json", action="store_true")
    w.set_defaults(func=cmd_watch)

    a = sub.add_parser("alerts", help="render alerts/digest from a verdict report")
    a.add_argument("--report", required=True)
    a.add_argument("--notify", default="",
                   help="registered notification channel (default: none — reports unsent)")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_alerts)

    d = sub.add_parser("demo", help="seeded drift theater")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_demo)

    pr = sub.add_parser("presets", help="list verdict profiles")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_presets)

    for name, fn, help_ in [
            ("license", lambda a: (print(json.dumps(_licensing.check_license(a.key), indent=2)), 0)[1],
             "show license status"),
            ("update-check", lambda a: (print(json.dumps(_licensing.check_update(), indent=2)), 0)[1],
             "check for a newer release")]:
        s = sub.add_parser(name, help=help_)
        if name == "license":
            s.add_argument("--key", default=None)
        s.add_argument("--json", action="store_true")
        s.set_defaults(func=fn)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
