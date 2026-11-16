"""Reprice only the entered IVRV 50% positions from raw NSE FO token ticks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "banknifty" / "2026-11-02" / "inference_replay" / "runs" / "final_20261102"
TRADE_FILE = ROOT / "ivrv" / "threshold_50" / "trades.jsonl"
SOURCE: Path | None = None
OUT = ROOT / "tick_reprice_50"


def entries_and_windows(trade_files=None):
    trade_files = trade_files or [TRADE_FILE]
    entries = []
    for trade_file in trade_files:
        entries.extend(row for line in trade_file.read_text(encoding="utf-8").splitlines()
                       if (row := json.loads(line))["event"] == "ENTRY")
    windows = defaultdict(list)
    cutoff = datetime.combine(datetime.fromisoformat(entries[0]["entry_time"]).date(), time(15, 15, 10))
    for entry in entries:
        start = datetime.fromisoformat(entry["entry_time"])
        end = min(start + timedelta(minutes=120), cutoff)
        for token in (entry["call_token"], entry["put_token"]):
            windows[str(token)].append(((start - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S"),
                                        end.strftime("%Y-%m-%d %H:%M:%S")))
    return entries, windows


def seek_near(source, first_timestamp: bytes):
    """Seek near a timestamp in the chronologically ordered source CSV."""
    size = SOURCE.stat().st_size
    lo, hi = 0, size
    while hi - lo > 4 * 1024 * 1024:
        mid = (lo + hi) // 2
        source.seek(mid)
        source.readline()
        row = source.readline()
        if not row:
            hi = mid
        elif row[:19] < first_timestamp:
            lo = mid
        else:
            hi = mid
    source.seek(lo)
    if lo:
        source.readline()
    return source.tell()


def extract_ticks(windows):
    earliest = min(a for ranges in windows.values() for a, _ in ranges).encode("ascii")
    latest = max(b for ranges in windows.values() for _, b in ranges).encode("ascii")
    selected = set(windows)
    path = OUT / "selected_token_ticks.csv"
    scanned = kept = 0
    per_token = defaultdict(int)
    with SOURCE.open("rb", buffering=8 * 1024 * 1024) as source, path.open("w", newline="", encoding="utf-8") as target:
        source_position = seek_near(source, earliest)
        writer = csv.writer(target)
        writer.writerow(["timestamp", "token", "last_traded_price", "source_line_order"])
        for line in source:
            scanned += 1
            stamp = line[:19]
            if stamp < earliest:
                continue
            if stamp > latest:
                break
            p1 = line.find(b",")
            p2 = line.find(b",", p1 + 1)
            p3 = line.find(b",", p2 + 1)
            if p3 < 0:
                continue
            token = line[p2 + 1:p3].decode("ascii")
            if token not in selected:
                continue
            current = stamp.decode("ascii")
            if not any(start <= current <= end for start, end in windows[token]):
                continue
            # LTP is column 9. Only decode selected rows; other feed fields are
            # irrelevant to fixed-token option price triggers.
            fields = line.split(b",", 9)
            if len(fields) < 9:
                continue
            try:
                price = float(fields[8])
            except ValueError:
                continue
            if price <= 0:
                continue
            writer.writerow([line[:23].decode("ascii"), token, price, scanned])
            per_token[token] += 1
            kept += 1
    metadata = dict(source=str(SOURCE), source_bytes=SOURCE.stat().st_size,
                    scan_start_byte=source_position, rows_scanned=scanned,
                    rows_kept=kept, per_token=dict(per_token), token_windows=windows)
    (OUT / "extraction.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("scan_start_byte", "rows_scanned", "rows_kept", "per_token")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, required=True, help="raw feed CSV for this replay")
    parser.add_argument("--thresholds", type=int, nargs="+", default=[50])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ROOT = args.run_root.resolve()
    SOURCE = args.source
    OUT = (args.output or ROOT / "tick_reprice").resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    trade_files = [ROOT / "ivrv" / f"threshold_{n}" / "trades.jsonl" for n in args.thresholds]
    _, token_windows = entries_and_windows(trade_files)
    extract_ticks(token_windows)
