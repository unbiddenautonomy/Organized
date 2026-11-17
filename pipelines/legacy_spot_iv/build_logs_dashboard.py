"""Build a self-contained historical spot/IV logs dashboard."""
import argparse
import json
import collections
import pathlib

here = pathlib.Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logs", type=pathlib.Path, default=here / "logs")
parser.add_argument("--date", help="optional exchange date, YYYY-MM-DD")
parser.add_argument("--output", type=pathlib.Path, help="destination HTML (default: reports/<date>/)")
args = parser.parse_args()
L = args.logs.resolve()


def rows(name):
    path = L / name
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


days = collections.defaultdict(
    lambda: {"spot": [], "iv": [], "rej": {}, "events": [], "retrain": {}, "manual": []}
)


def hm(m):
    return m[11:16]


def man(day, src, r):
    days[day]["manual"].append({
        "file": src,
        "time": r["exchange_minute"][11:19],
        "event": r["event_type"],
        "detail": {k: v for k, v in r.items() if k not in ("event_type", "exchange_minute", "source")},
    })


for r in rows("spot_predictions.jsonl"):
    d = r["exchange_minute"][:10]
    if r.get("source")=="long_vol_model":
        man(d, "spot_predictions", r)
        continue
    p = r["predictions"]
    days[d]["spot"].append([hm(r["exchange_minute"]), r["current_spot"],
                            p.get("5", {}).get("predicted_spot"), p.get("15", {}).get("predicted_spot")])

for r in rows("iv_predictions.jsonl"):
    d = r["exchange_minute"][:10]
    if r.get("source")=="long_vol_model":
        man(d, "iv_predictions", r)
        continue
    p = r["predictions"]
    days[d]["iv"].append([hm(r["exchange_minute"]), r["current_iv"],
                          p.get("5", {}).get("predicted_iv"), p.get("15", {}).get("predicted_iv")])

for r in rows("paper_decisions.jsonl"):
    d = r["exchange_minute"][:10]
    if r.get("source")=="long_vol_model":
        man(d, "paper_decisions", r)
        continue
    if r["event_type"] == "entry_rejected":
        for x in r["reasons"]:
            days[d]["rej"][x] = days[d]["rej"].get(x, 0) + 1
    else:
        days[d]["events"].append({"time": r["exchange_minute"][11:19], "event": r["event_type"],
                                  "decision": r.get("decision"), "reasons": r.get("reasons")})

for r in rows("paper_trades.jsonl"):
    t = r.get("exchange_minute") or r.get("entry_time")
    d = t[:10]
    if r.get("source")=="long_vol_model":
        man(d, "paper_trades", dict(r, exchange_minute=t))
        continue
    if r["event_type"] == "entry_rejected":
        continue
    days[d]["events"].append({
        "time": t[11:19], "event": r["event_type"], "decision": r.get("decision"),
        "reasons": r.get("reasons") or r.get("reason"), "strike": r.get("strike"),
        "horizon": r.get("horizon"), "net_pnl": r.get("net_pnl"), "trade_id": r.get("trade_id"),
    })

for r in rows("retraining_live_outputs.jsonl"):  # latest row per target/horizon per day
    d = r["exchange_minute"][:10]
    days[d]["retrain"]["%s-%sm" % (r["target"], r["horizon"])] = {
        k: r[k] for k in ("exchange_minute", "current", "base_prediction", "intraday_prediction", "eod_prediction")
    }

data = {k: days[k] for k in sorted(days) if not args.date or k == args.date}
tpl = (here / "logs_dashboard_template.html").read_text(encoding="utf-8")
html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
output = args.output or here / "reports" / (args.date or "all") / "logs_dashboard.html"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(html, encoding="utf-8")
print("wrote", output, len(html) // 1024, "KB; days:", list(data))
