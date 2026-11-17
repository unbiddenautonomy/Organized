"""Build a self-contained historical dashboard from selected IVRV logs."""
import argparse
import json
import collections
import pathlib

here = pathlib.Path(__file__).resolve().parents[1]  # pipelines/ivrv
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logs", type=pathlib.Path, default=here / "logs")
parser.add_argument("--date", help="optional exchange date, YYYY-MM-DD")
parser.add_argument("--output", type=pathlib.Path, help="destination HTML (default: reports/<date>/)")
args = parser.parse_args()
L = args.logs.resolve()


def matches_date(row):
    if not args.date:
        return True
    stamp = next((row.get(key) for key in ("exchange_minute", "entry_datetime", "entry_time", "exit_time", "time")
                  if row.get(key)), None)
    return str(stamp).startswith(args.date)


def jl(name):
    out = []
    p = L / name
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if matches_date(row):
                        out.append(row)
    return out


def js(name):
    p = L / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() and not args.date else {}


preds, lv_signals = [], []
for r in jl("live_predictions.jsonl"):
    if r.get("source") == "long_vol_model":
        lv_signals.append(r)
        continue
    m, s, sp = r["market"], r["strategy_signal"], r["spot_prediction"]
    preds.append({
        "t": r["exchange_minute"][11:16], "d": r["exchange_minute"][:10], "strike": r["fixed_entry_strike"],
        "spot": m["spot"], "straddle": m["straddle_price"], "civ": m["combined_iv"],
        "rv": s.get("current_rv"), "prob": s.get("convergence_probability"), "thr": s.get("convergence_probability_threshold"),
        "state": s.get("state"), "reason": s.get("entry_reason"), "taken": bool(s.get("entry_taken")),
        "l1": sp["one_sigma_low"], "h1": sp["one_sigma_high"], "l9": sp["approx_90pct_low"], "h9": sp["approx_90pct_high"],
    })

events = jl("paper_trade_events.jsonl")
paths = collections.Counter()
entries = set()
for r in jl("fixed_strike_120m_paths.jsonl"):
    entries.add(r["entry_id"])
    paths[r["entry_datetime"][11:13]] += 1

data = {
    "preds": preds,
    "longvol": lv_signals,
    "events": events,
    "latest": js("latest_prediction.json"),
    "position": js("paper_position_state.json"),
    "paths": {"entries": len(entries), "by_hour": dict(sorted(paths.items()))},
}
tpl = (here / "dashboard" / "inference_dashboard_template.html").read_text(encoding="utf-8")
html = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"), default=str))
output = args.output or here / "reports" / (args.date or "all") / "inference_dashboard.html"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(html, encoding="utf-8")
print("wrote", output, len(html) // 1024, "KB;", len(preds), "predictions,", len(events), "events")
