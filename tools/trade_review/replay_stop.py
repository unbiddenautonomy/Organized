"""Dashboard-only fixed-entry replay of a 1,500 INR loss limit."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DLL = REPO / "voltr/feed/native/lzo2.dll"
# Built from extract_replay_ticks.c; not tracked in git (see README.md).
EXTRACTOR = Path(os.environ.get("VOLTR_TICK_EXTRACTOR", REPO / "runtime/trade_review/extract_replay_ticks.exe"))
EPOCH = datetime(1980, 1, 1)
IST = timezone(timedelta(hours=5, minutes=30))
STOP = -1500.0


def records(path):
    with path.open("rb") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:
                if not line.endswith(b"\n"):
                    break
                raise


def journal_files(logs, day):
    folder = logs / "isolated_service" / "sessions" / day / "feed_journal_runs"
    return sorted(folder.glob("*/*.feed"), key=lambda p: (p.name, p.stat().st_mtime_ns))


def read_ticks(logs, files, tokens, windows):
    """Cache extracted quotes by file version; read only minutes relevant to continuations."""
    by_token = defaultdict(list)
    if not files or not tokens:
        return by_token
    binary = EXTRACTOR
    if not binary.is_file():
        # Without the extractor, continuations fall back to minute-close estimates.
        return by_token
    dll = DLL
    token_arg = ",".join(str(t) for t in sorted(tokens))
    cache = Path(tempfile.gettempdir()) / "dashboard-fixed-entry-ticks-v1"
    cache.mkdir(exist_ok=True)
    order = 0
    for file in files:
        when = datetime.fromtimestamp(int(file.stem)*60, IST).replace(tzinfo=None)
        if not any(a-timedelta(minutes=2)<=when<=b+timedelta(minutes=1) for a,b in windows):
            continue
        st = file.stat()
        signature = f"{file}|{st.st_mtime_ns}|{st.st_size}|{token_arg}"
        target = cache / (hashlib.sha256(signature.encode()).hexdigest()+".bin")
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=cache,delete=False) as temp:
                subprocess.run([binary,dll,token_arg,file],stdout=temp,check=True)
            os.replace(temp.name,target)
        for second,token,paise in struct.iter_unpack("<III",target.read_bytes()):
            order += 1
            by_token[token].append((second,order,paise/100))
    return by_token


def time_of(text):
    return datetime.fromisoformat(text)


def expiry_limit(pos):
    start = time_of(pos["entry_datetime"])
    return min(start+timedelta(minutes=pos["maximum_holding_minutes"]),
               time_of(str(start.date())+" "+pos["square_off_time"]))


def pair_path(pos,ticks,raw_start):
    """Pair fresh leg quotes; preserve order between tokens within an exchange second."""
    # ticks carries a global sequence order for identical timestamps.
    latest = {}
    events = [(sec,order,token,price) for token in (pos["call_token"],pos["put_token"])
              for sec,order,price in ticks[token]]
    events.sort(key=lambda x:x[1])
    deadline = expiry_limit(pos)
    for sec,_,token,price in events:
        now = EPOCH+timedelta(seconds=sec)
        latest[token] = (sec,price)
        if now < raw_start or len(latest)!=2:
            continue
        if any(s>sec or (now<deadline and sec-s>60) for s,_ in latest.values()):
            continue
        yield now,sum(p for _,p in latest.values()),'tick'

def continuation(pos, trade, minute_quotes, ticks, square_marks):
    old_end = time_of(pos["exit_datetime"])
    deadline = expiry_limit(pos)
    raw = list(pair_path(pos,ticks,old_end))
    old_premium = round(pos["exit_straddle_price"], 6)
    aligned = next((i for i,(ts,price,_) in enumerate(raw)
                    if ts == old_end and round(price,6)==old_premium),None)
    raw = raw[aligned+1:] if aligned is not None else [r for r in raw if r[0]>old_end]
    # Use a bar's closing premium only once its exchange minute has ended.
    minutes = [(t,p,"minute") for t,p in minute_quotes if old_end < t <= deadline+timedelta(minutes=1)]
    first_raw = raw[0][0] if raw else datetime.max
    stream = [v for v in minutes if v[0]<first_raw] + raw
    peak, drawdown = trade["peak"],trade["drawdown"]
    basis = "Tick continuation" if aligned is not None else "Estimated continuation"
    chart = dict(trade["marks"])
    previous = old_end
    last_price = old_premium
    for stamp,premium,kind in stream:
        if stamp < previous:
            continue
        if kind=="minute":
            basis = "Minute-close estimate"
        if stamp>deadline:
            # At a scheduled exit, use the last observed pair. Never use a future price.
            if previous<deadline-timedelta(seconds=60):
                break
            stamp,premium=deadline,last_price
            if kind=="minute":
                basis="Minute-close estimate"
        if stamp==deadline and stamp.time().isoformat()>=pos["square_off_time"]:
            premium=square_marks.get((pos["call_token"],pos["put_token"],stamp),premium)
        pnl=round((premium-pos["entry_straddle_price"])*pos["quantity"]*
                  (1 if pos["side"]=="LONG_STRADDLE" else -1),2)
        peak=max(peak,pnl)
        drawdown=min(drawdown,round(pnl-peak,2))
        when=stamp.isoformat(sep=" ")
        chart[when]=pnl
        previous,last_price=stamp,premium
        reason=None
        if stamp.time().isoformat()>=pos["square_off_time"]:
            reason="session_square_off_151430"
        elif pnl>=pos["target_inr"]:
            reason="profit_target"
        elif pnl<=STOP:
            reason="stop_loss"
        elif stamp>=deadline:
            reason="time_exit_120m"
        if reason:
            return dict(end=when,reason=reason,pnl=pnl,peak=peak,drawdown=drawdown,
                        basis=basis,marks=sorted(chart.items()))
    return dict(end=None,reason="Awaiting exit evidence",pnl=None,peak=peak,drawdown=drawdown,
                basis="Incomplete continuation",marks=sorted(chart.items()))


def apply(trades, events, folder, logs, day):
    stopped = [p for p in events.values() if p.get("exit_reason")=="stop_loss"
               and p.get("realized_pnl_inr",0)>STOP]
    needed={(p["expiry"],p["fixed_strike"]) for p in stopped}
    minute_data=defaultdict(dict)
    paths=folder/"fixed_strike_120m_paths.jsonl"
    if needed and paths.exists():
        for r in records(paths):
            key=(r.get("expiry"),r.get("entry_strike"))
            if key in needed and r["observation_datetime"].startswith(day) and r.get("observation_straddle_price") is not None:
                minute_data[key][r["observation_datetime"]]=r["observation_straddle_price"]
    minutes={k:sorted((time_of(t)+timedelta(minutes=1),p) for t,p in values.items())
             for k,values in minute_data.items()}
    windows=[(time_of(p["exit_datetime"]),expiry_limit(p)) for p in stopped]
    tokens={p[k] for p in stopped for k in ("call_token","put_token")}
    ticks=read_ticks(logs,journal_files(logs,day),tokens,windows)
    square_marks={(p["call_token"],p["put_token"],time_of(p["exit_datetime"])):p["exit_straddle_price"]
                  for p in events.values() if p.get("exit_reason")=="session_square_off_151430"}
    for trade in trades:
        p=events[(trade["side"],trade["start"],trade["strike"])]
        trade.update(actual_end=trade.get("end"),actual_pnl=trade.get("pnl"),
                     actual_reason=trade.get("reason"),actual_stop=p["stop_loss_inr"],
                     peak=round(p.get("peak_unrealized_pnl_inr",0),2),
                     drawdown=round(p.get("max_drawdown_inr",0),2),basis="Recorded path")
        if p.get("exit_reason")=="stop_loss" and p["realized_pnl_inr"]>STOP:
            trade.update(continuation(p,trade,minutes.get((p["expiry"],p["fixed_strike"]),[]),ticks,square_marks))
