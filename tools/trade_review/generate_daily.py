"""Build a self-contained, dated trade review from local inference logs."""
import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import html
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import serve_dashboard as source


IST = timezone(timedelta(hours=5, minutes=30))
OUTPUT = source.REPO / "runtime/trade_reviews"
DISPLAY_FIELDS = ("signal_time entry_time expiry strike side quantity entry_premium "
                  "horizon_pnl_inr status probability paper_skip_reason target_hit stop_hit").split()


def read_positions(directory, day):
    csv_file = directory / f"live_predictions_demo_positions_{day}.csv"
    ledger = directory / "live_predictions_demo_positions.json"
    if csv_file.is_file():
        with csv_file.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif ledger.is_file():
        rows = json.loads(ledger.read_text(encoding="utf-8-sig"))["positions"]
    else:
        return [], False
    return [{k: r.get(k) for k in DISPLAY_FIELDS} for r in rows
            if r["signal_time"][:10] == day], True


def render(day):
    """Return the review page for one session date and its summary counts."""
    page, _, _ = _render(day)
    return page


def _render(day):
    logs = source.PREDICTIONS.parent
    archive = logs / "prevDayLogs" / day
    directory = archive if (archive / source.PREDICTIONS.name).is_file() else logs
    predictions = directory / source.PREDICTIONS.name
    market = []
    trades = {"date": day, "market": [], "trades": []}
    if predictions.is_file():
        stat = predictions.stat()
        market = json.loads(source.market_history(stat.st_mtime_ns, stat.st_size, predictions, day))
        files = [predictions, *sorted(directory.glob("paper_trade_events*.jsonl"))]
        signature = tuple((p, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        trades = json.loads(source.execution_history(signature, day))
    positions, ledger_present = read_positions(directory, day)
    paths = directory / "fixed_strike_120m_paths.jsonl"
    marks = ([q for q in json.loads(source.CutoffMarks().read(paths)) if q[0] == day]
             if paths.is_file() else [])
    notes = ["Generated " + datetime.now(IST).strftime("%d %b %Y at %H:%M IST")]
    if not (market or positions or trades["trades"]):
        notes.append("No session data recorded for this date")
    else:
        if not market:
            notes.append("Market history unavailable")
        if not ledger_present:
            notes.append("Demo-position ledger unavailable")
        if not marks:
            notes.append("15:15 fixed-strike marks unavailable")
    values = {"market-snapshot": market, "outcome-snapshot": positions,
              "exec-snapshot": trades, "cutoff-data": {"marks": marks, "positions": positions}}
    page = (source.ROOT / "daily_snapshot_template.html").read_text(encoding="utf-8")
    for name, data in values.items():
        payload = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        page = page.replace("@@" + name + "@@", payload.replace("<", r"\u003c"))
    page = page.replace("@@day@@", date.fromisoformat(day).strftime("%A, %d %B %Y"))
    page = page.replace("@@report_note@@", html.escape(" · ".join(notes)))
    return page, positions, trades


def build(day, output):
    page, positions, trades = _render(day)
    output.mkdir(parents=True, exist_ok=True)
    dest = output / f"Trade review {day}.html"
    with NamedTemporaryFile(dir=output, delete=False, suffix=".tmp") as tmp:
        tmp.write(page.encode("utf-8"))
    Path(tmp.name).replace(dest)
    print(f"{dest} | {len(positions)} demo positions | {len(trades['trades'])} paper trades")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date())
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.date.isoformat(), args.output)
