"""Build one-minute BANKNIFTY option and futures snapshots from an NSE FO feed.

The feed's Timestamp is used as the trading date/time. Contract token IDs, expiry
ordering, and strike prices come from the supplied contract master. NSE contract
epoch fields in this dataset are seconds since 1980-01-01.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = ROOT / "datasets" / "contracts" / "NSE_FO_contract_30102026.csv.gz"
DEFAULT_FEED = Path(os.environ["VOLTR_FEED_CSV"]) if os.environ.get("VOLTR_FEED_CSV") else None
DEFAULT_OUTPUT = ROOT / "datasets" / "banknifty" / "2026-11-02"


def contracts(master: Path):
    options = defaultdict(dict)
    futures = []
    with gzip.open(master, "rt", newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row["TckrSymb"] != "BANKNIFTY":
                continue
            item = {
                "token": row["FinInstrmId"],
                "symbol": row["StockNm"],
                "expiry_epoch": int(row["XpryDt"]),
            }
            if row["FinInstrmNm"] == "FUTIDX":
                futures.append(item)
            elif row["FinInstrmNm"] == "OPTIDX" and row["OptnTp"] in ("CE", "PE"):
                item.update(strike=int(row["StrkPric"]) / 100, option_type=row["OptnTp"])
                options[item["expiry_epoch"]][(item["strike"], item["option_type"])] = item
    if not futures:
        raise ValueError("No BANKNIFTY futures found in the contract master")
    futures.sort(key=lambda x: x["expiry_epoch"])
    near = futures[0]
    return near, futures, options[near["expiry_epoch"]]


def read_feed(feed: Path, tokens: set[str], date: str):
    # Keep only the first quote for each token in each minute. A source row with
    # a zero LTP is retained: bid/ask and OI can still carry useful information.
    updates = defaultdict(dict)
    scanned = matched = 0
    with feed.open("rb", buffering=8 * 1024 * 1024) as source:
        header = source.readline().decode("utf-8-sig").rstrip("\r\n").split(",")
        required = ("Timestamp", "Token", "LastTradedPrice", "VolumeTradedToday",
                    "OpenInterest", "BestBuyPrice_1", "BestSellPrice_1")
        missing = set(required) - set(header)
        if missing:
            raise ValueError(f"Feed lacks columns: {sorted(missing)}")
        ix = {name: header.index(name) for name in required}
        if ix["Token"] != 2 or ix["Timestamp"] != 0:
            raise ValueError("Fast reader expects Timestamp and Token in columns 1 and 3")
        for line in source:
            scanned += 1
            if line[:10] != date.encode("ascii"):
                continue
            p1 = line.find(b",")
            p2 = line.find(b",", p1 + 1)
            p3 = line.find(b",", p2 + 1)
            if p3 < 0:
                continue
            token = line[p2 + 1:p3].decode("ascii")
            if token not in tokens:
                continue
            minute = line[:16].decode("ascii")
            if minute < date + " 09:15" or minute > date + " 15:29":
                continue
            if token in updates[minute]:
                continue
            fields = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
            if len(fields) != len(header):
                raise ValueError(f"Malformed feed row near line {scanned}")
            updates[minute][token] = {
                "tick_time": fields[ix["Timestamp"]],
                "ltp": fields[ix["LastTradedPrice"]],
                "volume": fields[ix["VolumeTradedToday"]],
                "oi": fields[ix["OpenInterest"]],
                "bid": fields[ix["BestBuyPrice_1"]],
                "ask": fields[ix["BestSellPrice_1"]],
            }
            matched += 1
    return updates, scanned, matched


def minute_range(date: str, last: str):
    t = datetime.fromisoformat(date + " 09:15:00")
    end = datetime.fromisoformat(date + " " + last)
    while t <= end:
        yield t
        t += timedelta(minutes=1)


def expiry_label(epoch: int) -> str:
    return (datetime(1980, 1, 1) + timedelta(seconds=epoch)).date().isoformat()


def output_csv(path: Path, fields: list[str], rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(master: Path, feed: Path, output: Path, date: str):
    near, futures, options = contracts(master)
    output.mkdir(parents=True, exist_ok=True)
    all_tokens = {item["token"] for item in futures} | {item["token"] for item in options.values()}
    updates, scanned, matched = read_feed(feed, all_tokens, date)

    # Each minute uses its first available source tick. If a contract has no tick
    # that minute, carry forward its last known quote and expose its tick_time.
    state = {}
    snapshots = {}
    for t in minute_range(date, "15:29:00"):
        key = t.strftime("%Y-%m-%d %H:%M")
        state.update(updates.get(key, {}))
        snapshots[key] = state.copy()

    future_fields = ["minute", "token", "symbol", "expiry_master", "is_near_expiry",
                     "source_tick_time", "last_traded_price", "volume_traded_today",
                     "open_interest", "best_buy_price", "best_sell_price"]
    future_rows = []
    for t in minute_range(date, "15:29:00"):
        minute = t.strftime("%Y-%m-%d %H:%M")
        for item in futures:
            q = snapshots[minute].get(item["token"], {})
            future_rows.append({
                "minute": t.isoformat(sep=" "), "token": item["token"],
                "symbol": item["symbol"], "expiry_master": expiry_label(item["expiry_epoch"]),
                "is_near_expiry": item is near, "source_tick_time": q.get("tick_time", ""),
                "last_traded_price": q.get("ltp", ""), "volume_traded_today": q.get("volume", ""),
                "open_interest": q.get("oi", ""), "best_buy_price": q.get("bid", ""),
                "best_sell_price": q.get("ask", ""),
            })
    output_csv(output / "banknifty_futures_1min.csv", future_fields, future_rows)

    chain_fields = ["minute", "future_token", "future_source_tick_time", "future_spot_proxy",
                    "expiry_master", "strike_offset", "strike", "option_type", "option_token",
                    "option_symbol", "source_tick_time", "last_traded_price", "volume_traded_today",
                    "open_interest", "best_buy_price", "best_sell_price", "target_minute",
                    "target_source_tick_time", "target_last_traded_price", "target_volume_traded_today",
                    "target_open_interest", "target_best_buy_price", "target_best_sell_price",
                    "price_change_to_target"]
    chain_rows = []
    strikes = sorted({key[0] for key in options})
    cutoff = datetime.fromisoformat(date + " 15:15:00")
    for t in minute_range(date, "15:15:00"):
        minute = t.strftime("%Y-%m-%d %H:%M")
        spot_quote = snapshots[minute].get(near["token"])
        if not spot_quote or not spot_quote["ltp"] or float(spot_quote["ltp"]) <= 0:
            continue
        spot = float(spot_quote["ltp"])
        # Tie goes to the lower strike.
        atm_index = min(range(len(strikes)), key=lambda i: (abs(strikes[i] - spot), strikes[i]))
        target = min(t + timedelta(minutes=120), cutoff)
        target_minute = target.strftime("%Y-%m-%d %H:%M")
        for offset in range(-2, 3):
            index = atm_index + offset
            if not 0 <= index < len(strikes):
                continue
            strike = strikes[index]
            for option_type in ("CE", "PE"):
                item = options.get((strike, option_type))
                if item is None:
                    continue
                current = snapshots[minute].get(item["token"], {})
                later = snapshots[target_minute].get(item["token"], {})
                a, b = current.get("ltp", ""), later.get("ltp", "")
                change = round(float(b) - float(a), 2) if a and b and float(a) > 0 and float(b) > 0 else ""
                chain_rows.append({
                    "minute": t.isoformat(sep=" "), "future_token": near["token"],
                    "future_source_tick_time": spot_quote["tick_time"], "future_spot_proxy": spot_quote["ltp"],
                    "expiry_master": expiry_label(near["expiry_epoch"]), "strike_offset": offset,
                    "strike": strike, "option_type": option_type, "option_token": item["token"],
                    "option_symbol": item["symbol"], "source_tick_time": current.get("tick_time", ""),
                    "last_traded_price": a, "volume_traded_today": current.get("volume", ""),
                    "open_interest": current.get("oi", ""), "best_buy_price": current.get("bid", ""),
                    "best_sell_price": current.get("ask", ""),
                    "target_minute": target.isoformat(sep=" "),
                    "target_source_tick_time": later.get("tick_time", ""),
                    "target_last_traded_price": b,
                    "target_volume_traded_today": later.get("volume", ""),
                    "target_open_interest": later.get("oi", ""),
                    "target_best_buy_price": later.get("bid", ""),
                    "target_best_sell_price": later.get("ask", ""),
                    "price_change_to_target": change,
                })
    output_csv(output / "banknifty_option_chain_1min.csv", chain_fields, chain_rows)
    print(f"Feed rows scanned: {scanned:,}; selected first ticks: {matched:,}")
    print(f"Near future: {near['symbol']} ({near['token']}); master expiry {expiry_label(near['expiry_epoch'])}")
    print(f"Futures rows: {len(future_rows):,}; option-chain rows: {len(chain_rows):,}")
    print(f"Output directory: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--feed", type=Path, default=DEFAULT_FEED, required=DEFAULT_FEED is None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", default="2026-11-02", help="Feed trading date, YYYY-MM-DD")
    args = parser.parse_args()
    build(args.master, args.feed, args.output, args.date)
