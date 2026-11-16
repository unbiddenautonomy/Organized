"""One shared, causal feature definition for training, replay and live inference.

Input rows are completed minute snapshots. No target, exit or future column is
read here. Lags allow at most one minute of backward age, same date and expiry.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import math

LAGS = (5, 15, 30, 60)
BASE = ["days_to_expiry", "four_week", "basis", "atm_premium", "otm_premium",
        "premium_balance", "otm_balance", "moneyness", "call_distance",
        "put_distance", "oi_ratio", "otm_oi_ratio", "time_sin", "time_cos"]
FEATURES = BASE + [f"{field}_{lag}" for lag in LAGS for field in
                    ("future_return", "abs_future_return", "spot_return",
                     "atm_return", "otm_return", "oi_change")]
FEATURES += ["range_60", "realized_vol_60", "trend_strength_60", "four_week_trend"]

def parse_time(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))

def feature_row(row: dict, history: dict) -> dict | None:
    t = parse_time(row["signal_time"])
    key = (t.date().isoformat(), str(row["expiry"])[:10])
    def at(lag, allow_age=True):
        target=t-timedelta(minutes=lag)
        p = history.get(target.isoformat(sep=" "))
        if p is None and allow_age:
            p=history.get((target-timedelta(minutes=1)).isoformat(sep=" "))
        if p is None or (parse_time(p["signal_time"]).date().isoformat(), str(p["expiry"])[:10]) != key:
            return None
        return p
    try:
        spot, fut, atm = (float(row[k]) for k in ("spot", "future", "atm"))
        ap = float(row["atm_call"]) + float(row["atm_put"])
        op = float(row["call"]) + float(row["put"])
        ratio = (float(row["atm_put_oi"]) + 1) / (float(row["atm_call_oi"]) + 1)
        elapsed = t.hour * 60 + t.minute - 555
        f = dict(zip(BASE, [float(row["days_to_expiry"]), float(row["days_to_expiry"] >= 28),
            fut / spot - 1, ap, op, (row["atm_call"] - row["atm_put"]) / ap,
            (row["call"] - row["put"]) / op, (spot - atm) / float(row["step"]),
            (row["call_strike"] - spot) / spot, (spot - row["put_strike"]) / spot,
            ratio, (row["put_oi"] + 1) / (row["call_oi"] + 1),
            math.sin(2 * math.pi * elapsed / 375), math.cos(2 * math.pi * elapsed / 375)]))
        for lag in LAGS:
            p = at(lag)
            if p is None: return None
            ret = math.log(fut / p["future"])
            f.update({f"future_return_{lag}": ret, f"abs_future_return_{lag}": abs(ret),
                f"spot_return_{lag}": math.log(spot / p["spot"]),
                f"atm_return_{lag}": math.log(ap / (p["atm_call"] + p["atm_put"])),
                f"otm_return_{lag}": math.log(op / (p["call"] + p["put"])),
                f"oi_change_{lag}": math.log(ratio / ((p["atm_put_oi"]+1)/(p["atm_call_oi"]+1)))})
        # Some historical early-expiry chains are sparse. Use observed intervals
        # only for RV; lag values have at most one minute of backward age.
        past = [(i, at(i,False)) for i in range(61)]
        past = [(i,p) for i,p in past if p is not None]
        if len(past)<13: return None
        values = [float(p["future"]) for _,p in past]
        ret = [math.log(values[i] / values[i+1])/math.sqrt(past[i+1][0]-past[i][0]) for i in range(len(values)-1)]
        mean = sum(ret) / len(ret)
        rv = math.sqrt(sum((v-mean)**2 for v in ret) / (len(ret)-1))
        f.update(range_60=(max(values)-min(values))/fut, realized_vol_60=rv,
                 trend_strength_60=abs(f["future_return_60"])/(rv*math.sqrt(60)+1e-8),
                 four_week_trend=f["four_week"]*f["future_return_60"])
        if not all(math.isfinite(v) for v in f.values()): return None
        return f
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None

def history_add(history, row):
    t = parse_time(row["signal_time"])
    history[t.isoformat(sep=" ")] = row
    cutoff = (t - timedelta(minutes=65)).isoformat(sep=" ")
    for k in list(history):
        if k < cutoff: del history[k]

def historical_quantity(expiry):
    # Source report's contract-lot schedule; live quantity comes from its contract file.
    e = str(expiry)[:10]
    return 30 if e >= "2025-02-01" else 15 if e >= "2023-07-01" else 25

def costs(entry, exit_price, quantity, config, trade_date=None):
    sell, buy = float(entry)*quantity, float(exit_price)*quantity
    stt, exchange = config["stt_sell_premium_rate"], config["nse_options_transaction_rate"]
    if trade_date is not None and config.get("historical_cost_mode") == "source_report_rates":
        day = str(trade_date)[:10]
        stt = .0005 if day < "2023-04-01" else .000625 if day < "2024-10-01" else .001
        exchange = .0005 if day < "2024-10-01" else .0003503
    exchange_fee = (sell+buy)*exchange
    sebi = (sell+buy)*config["sebi_turnover_rate"]
    brokerage = 4*config["brokerage_per_order_inr"]
    parts = {"brokerage": brokerage, "stt": sell*stt, "exchange": exchange_fee,
             "sebi": sebi, "stamp": buy*config["stamp_buy_premium_rate"],
             "gst": (brokerage+exchange_fee+sebi)*config["gst_rate"],
             "slippage": config["round_trip_slippage_points"]*quantity}
    parts["total"] = sum(parts.values())
    return parts

def pnl(entry, exit_price, quantity, config, trade_date=None):
    c = costs(entry, exit_price, quantity, config, trade_date)
    gross = (float(entry)-float(exit_price))*quantity
    return {"gross_pnl": gross, "costs": c, "net_pnl": gross-c["total"]}
