from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs" / "economic_evaluation" / "economic_trade_panel.parquet"
FIVE_STRIKE_PATH = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
EXIT_PATH = ROOT / "outputs" / "intraday_structure_attribution" / "_fixed_strike_exit_prices.parquet"
OUT = ROOT / "outputs" / "short_vol_expansion"
THRESHOLDS = (1.0, 1.25, 1.5, 1.75, 2.0)


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    threshold: float
    horizons: tuple[int, ...] = (120,)
    div_threshold: float | None = None
    require_div_negative: bool = False
    filter_col: str | None = None
    filter_value: str | None = None


def charges(entry: np.ndarray, exit_: np.ndarray, quantity: np.ndarray, dates: pd.Series,
            bps: float, current: bool) -> pd.DataFrame:
    entry, exit_, quantity = map(lambda x: np.asarray(x, float), (entry, exit_, quantity))
    buy, sell = exit_ * quantity, entry * quantity
    turnover = buy + sell
    d = pd.to_datetime(dates)
    if current:
        stt_rate, exchange_rate = np.full(len(entry), .0015), np.full(len(entry), .0003553)
    else:
        stt_rate = np.where(d < pd.Timestamp("2024-10-01"), .000625, .001)
        exchange_rate = np.where(d < pd.Timestamp("2024-04-01"), .0005,
                                 np.where(d < pd.Timestamp("2024-10-01"), .000495, .0003503))
    brokerage = np.full(len(entry), 80.0)
    stt = np.rint(sell * stt_rate)
    exchange = turnover * exchange_rate
    sebi, ipft = turnover * .000001, turnover * .000000001
    stamp = buy * .00003
    gst = .18 * (brokerage + exchange + sebi + ipft)
    slippage = turnover * bps / 10000.0
    return pd.DataFrame({"total_cost": brokerage + stt + exchange + sebi + ipft + stamp + gst + slippage})


def estimated_cost(frame: pd.DataFrame, bps: float = 10.0) -> np.ndarray:
    # Match the original rule-selection convention: entry premium on both sides
    # and the rate schedule applicable to each historical date.
    return charges(frame.entry_premium, frame.entry_premium, frame.lot_size, frame.date, bps, False).total_cost.to_numpy()


def candidates(frame: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    z = frame[frame.horizon_min.isin(spec.horizons)].copy()
    if spec.filter_col:
        z = z[z[spec.filter_col].astype(str).eq(str(spec.filter_value))]
    cost = estimated_cost(z)
    iv_expected = z.theta_decay_points.to_numpy(float) * z.iv_only_score.to_numpy(float) * z.lot_size.to_numpy(float)
    div_expected = z.vega_s_decimal.to_numpy(float) * z.pred_d_iv.to_numpy(float) * z.lot_size.to_numpy(float)
    iv_active = iv_expected < -spec.threshold * cost
    if spec.family == "iv":
        active = iv_active
        conviction = -iv_expected / cost
    elif spec.family == "iv_confirm_div":
        active = iv_active & (z.pred_d_iv.to_numpy(float) < 0)
        conviction = -iv_expected / cost
    elif spec.family == "union":
        assert spec.div_threshold is not None
        div_active = div_expected < -spec.div_threshold * cost
        active = iv_active | div_active
        conviction = np.maximum(-iv_expected / cost, -div_expected / cost)
    else:
        raise ValueError(spec.family)
    z = z.loc[active].copy()
    z["conviction"] = conviction[active]
    z["candidate_source"] = spec.name
    return z


def select_trades(frame: pd.DataFrame, max_per_day: int = 2) -> pd.DataFrame:
    if frame.empty:
        return frame
    # When horizons collide at one timestamp, retain the stronger normalized edge.
    z = frame.sort_values(["datetime", "conviction"], ascending=[True, False]).drop_duplicates("datetime")
    chosen: list[int] = []
    for _, day in z.groupby("date", sort=True):
        count, next_free = 0, pd.Timestamp.min
        for idx, row in day.sort_values("datetime").iterrows():
            if count >= max_per_day:
                break
            if row.datetime < next_free:
                continue
            chosen.append(idx)
            count += 1
            next_free = row.exit_datetime
    return z.loc[chosen].sort_values("datetime").copy()


def attach(trades: pd.DataFrame, bps: float) -> pd.DataFrame:
    z = trades.copy()
    if z.empty:
        return z.assign(gross_pnl_rupees=[], total_cost=[], net_pnl_rupees=[])
    z["gross_pnl_rupees"] = -z.long_pnl_points * z.lot_size
    z["total_cost"] = charges(z.entry_premium, z.exit_premium, z.lot_size, z.date, bps, True).total_cost.to_numpy()
    z["net_pnl_rupees"] = z.gross_pnl_rupees - z.total_cost
    return z


def metrics(trades: pd.DataFrame, spec: Spec, bps: float, total_days: int) -> dict:
    z = attach(trades, bps)
    daily = z.groupby("date").net_pnl_rupees.sum() if len(z) else pd.Series(dtype=float)
    by_year = z.groupby(z.datetime.dt.year).net_pnl_rupees.sum().to_dict() if len(z) else {}
    days = int(z.date.nunique()) if len(z) else 0
    profitable_days = int((daily > 0).sum())
    trade_hit = float((z.net_pnl_rupees > 0).mean()) if len(z) else np.nan
    day_success = profitable_days / days if days else np.nan
    result = {
        "name": spec.name, "family": spec.family, "threshold": spec.threshold,
        "horizons": "+".join(map(str, spec.horizons)), "cost_bps": bps,
        "opportunity_days": total_days, "trades": len(z), "signal_days": days,
        "signal_day_share": days / total_days if total_days else np.nan,
        "profitable_days": profitable_days, "losing_days": int((daily < 0).sum()),
        "day_success_rate": day_success, "winning_trades": int((z.net_pnl_rupees > 0).sum()),
        "trade_hit_rate": trade_hit, "total_net": float(z.net_pnl_rupees.sum()),
        "net_2024": float(by_year.get(2024, 0.0)), "net_2025": float(by_year.get(2025, 0.0)),
    }
    result["passes_required"] = bool(
        bps == 10 and days > 18 and day_success > .75 and trade_hit > .65
        and result["total_net"] > 0 and result["net_2024"] > 0 and result["net_2025"] > 0
    )
    return result


def load_strangle(base60: pd.DataFrame) -> pd.DataFrame:
    cache = OUT / "short_1_strangle_60m_2024_2025.parquet"
    if cache.exists():
        raw = pd.read_parquet(cache)
    else:
        use = ["source_index", "strike_position", "datetime", "expiry", "call_price", "put_price",
               "theta_call", "theta_put"]
        blocks = []
        for chunk in pd.read_csv(FIVE_STRIKE_PATH, usecols=use, chunksize=300_000,
                                 parse_dates=["datetime", "expiry"]):
            keep = chunk[(chunk.strike_position.isin([-1, 1]))
                         & (chunk.datetime >= pd.Timestamp("2024-01-01"))]
            if len(keep):
                blocks.append(keep)
        entries = pd.concat(blocks, ignore_index=True)
        exits = pd.read_parquet(
            EXIT_PATH,
            filters=[("horizon", "==", "60m"), ("entry_datetime", ">=", pd.Timestamp("2024-01-01"))],
        )
        legs = entries.merge(
            exits[["source_index", "entry_datetime", "exit_datetime", "exit_call_price", "exit_put_price"]],
            on="source_index", how="inner", validate="one_to_one",
        )
        call = legs[legs.strike_position.eq(1)][
            ["datetime", "expiry", "call_price", "exit_call_price", "theta_call", "exit_datetime"]
        ].rename(columns={"call_price": "entry_call", "exit_call_price": "exit_call", "theta_call": "call_theta"})
        put = legs[legs.strike_position.eq(-1)][
            ["datetime", "expiry", "put_price", "exit_put_price", "theta_put"]
        ].rename(columns={"put_price": "entry_put", "exit_put_price": "exit_put", "theta_put": "put_theta"})
        raw = call.merge(put, on=["datetime", "expiry"], how="inner", validate="one_to_one")
        raw["entry_premium"] = raw.entry_call + raw.entry_put
        raw["exit_premium"] = raw.exit_call + raw.exit_put
        raw["actual"] = raw.entry_premium - raw.exit_premium
        raw["theta_decay_points"] = (raw.call_theta + raw.put_theta).abs() * 60.0 / 1440.0
        raw["stale_any"] = raw.entry_call.eq(raw.exit_call) | raw.entry_put.eq(raw.exit_put)
        raw["tiny_any"] = raw.entry_call.lt(5) | raw.entry_put.lt(5)
        raw.to_parquet(cache, index=False)
    # ``stale_any`` depends on the exit quote and is not a causal entry filter.
    # Here ``tiny_any`` is defined solely from entry leg prices and is safe.
    raw = raw[~raw.tiny_any.astype(bool)].copy()
    meta = base60.drop_duplicates("datetime").copy()
    joined = meta.drop(columns=["entry_premium", "exit_premium", "long_pnl_points", "theta_decay_points", "exit_datetime"], errors="ignore").merge(
        raw, on="datetime", how="inner", suffixes=("", "_structure"), validate="one_to_one")
    joined["long_pnl_points"] = -joined.actual
    joined["date"] = joined.datetime.dt.normalize()
    return joined


def html_report(results: pd.DataFrame, recommendation: str) -> str:
    records = results.replace({np.nan: None}).to_dict("records")
    payload = json.dumps(records, separators=(",", ":"))
    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Short-vol expansion study</title><style>
:root{{--bg:#08111f;--card:#122139;--line:#2b3e59;--text:#eef5fd;--muted:#94a8bf;--good:#4fd2a9;--bad:#ff737d;--accent:#69aaff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}main{{width:min(1400px,calc(100% - 28px));margin:auto;padding:28px 0}}h1{{font-weight:500;margin:0}}p{{color:var(--muted)}}.note{{background:var(--card);border:1px solid var(--line);padding:16px;border-radius:14px;margin:18px 0}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}button{{background:transparent;color:var(--text);border:1px solid var(--line);padding:8px 12px;border-radius:9px}}button.active{{background:var(--accent);color:#06101d}}.wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--card)}}.pass{{color:var(--good)}}.fail{{color:var(--bad)}}
</style></head><body><main><h1>Short-vol signal-day expansion</h1><p>Offline only · live inference unchanged</p><div class='note'><b>Decision:</b> {recommendation}</div><div class='controls'><button class='active' data-cost='10'>10 bps</button><button data-cost='25'>25 bps</button></div><div class='wrap'><table><thead><tr><th>Candidate</th><th>Horizons</th><th>Days</th><th>Profitable days</th><th>Day success</th><th>Trades</th><th>Trade hit</th><th>Net</th><th>2024 net</th><th>2025 net</th><th>Pass</th></tr></thead><tbody id='rows'></tbody></table></div></main><script>
const data={payload};const fmt=n=>Number(n).toLocaleString('en-IN',{{maximumFractionDigits:0}});const pct=n=>(100*Number(n)).toFixed(1)+'%';function draw(cost){{document.getElementById('rows').innerHTML=data.filter(x=>x.cost_bps==cost).sort((a,b)=>b.signal_days-a.signal_days).map(x=>`<tr><td>${{x.name}}</td><td>${{x.horizons}}</td><td>${{x.signal_days}}</td><td>${{x.profitable_days}}</td><td>${{pct(x.day_success_rate)}}</td><td>${{x.trades}}</td><td>${{pct(x.trade_hit_rate)}}</td><td>${{fmt(x.total_net)}}</td><td>${{fmt(x.net_2024)}}</td><td>${{fmt(x.net_2025)}}</td><td class='${{x.passes_required?'pass':'fail'}}'>${{x.passes_required?'PASS':'REJECT'}}</td></tr>`).join('')}}document.querySelectorAll('button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');draw(Number(b.dataset.cost))}});draw(10);
</script></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    for col in ("datetime", "exit_datetime", "date"):
        panel[col] = pd.to_datetime(panel[col])
    panel = panel[(panel.scope == "intraday") & (panel.is_0915 == 0) & panel.horizon_min.isin([60, 120])].copy()
    total_days = int(panel.date.nunique())
    specs = [Spec(f"120m threshold {h:.2f}x", "iv", h) for h in THRESHOLDS]
    specs += [Spec(f"60m threshold {h:.2f}x", "iv", h, (60,)) for h in THRESHOLDS]
    specs += [Spec(f"60m+120m threshold {h:.2f}x", "iv", h, (60, 120)) for h in (1.5, 1.75, 2.0)]
    specs += [
        Spec("120m 1.50x + negative dIV confirmation", "iv_confirm_div", 1.5),
        Spec("120m RVIV 2x OR dIV 0.5x", "union", 2.0, div_threshold=.5),
        Spec("120m RVIV 1.5x OR dIV 0.5x", "union", 1.5, div_threshold=.5),
    ]
    for value in ("Open 09:15-09:45", "Morning 09:45-11:30", "Midday 11:30-13:30", "Afternoon 13:30-15:00"):
        specs.append(Spec(f"120m 1.5x · {value}", "iv", 1.5, filter_col="tod_bucket", filter_value=value))
    for value in ("High IV", "Medium IV", "Low IV"):
        specs.append(Spec(f"120m 1.5x · {value}", "iv", 1.5, filter_col="iv_regime", filter_value=value))
    for value in ("0DTE", "1-3 DTE", "4-7 DTE", ">7 DTE"):
        specs.append(Spec(f"120m 1.5x · {value}", "iv", 1.5, filter_col="dte_group", filter_value=value))

    result_rows, saved_trades = [], []
    for spec in specs:
        selected = select_trades(candidates(panel, spec), 2)
        for bps in (10.0, 25.0):
            result_rows.append(metrics(selected, spec, bps, total_days))
        if len(selected):
            out = attach(selected, 10.0)
            out["candidate"] = spec.name
            saved_trades.append(out)

    # Five-strike structure experiment: 60-minute short +/-1 strangle.
    strangle = load_strangle(panel[panel.horizon_min == 60].copy())
    for h in (1.5, 2.0):
        spec = Spec(f"60m short +/-1 strangle {h:.2f}x", "iv", h, (60,))
        selected = select_trades(candidates(strangle, spec), 2)
        for bps in (10.0, 25.0):
            result_rows.append(metrics(selected, spec, bps, total_days))
        if len(selected):
            out = attach(selected, 10.0); out["candidate"] = spec.name; saved_trades.append(out)

    results = pd.DataFrame(result_rows)
    ten = results[results.cost_bps == 10].copy()
    accepted = ten[ten.passes_required].sort_values(["signal_days", "day_success_rate", "total_net"], ascending=False)
    recommendation = (f"Accept {accepted.iloc[0]['name']} for further paper testing; it is the strongest candidate meeting every predeclared criterion."
                      if len(accepted) else "Reject all expansions: none met every predeclared criterion. Keep the original two-trade 120-minute rule.")
    results.to_csv(OUT / "candidate_results.csv", index=False)
    if saved_trades:
        pd.concat(saved_trades, ignore_index=True).to_parquet(OUT / "candidate_trades_10bps.parquet", index=False)
    (OUT / "acceptance_dashboard.html").write_text(html_report(results, recommendation), encoding="utf-8")
    lines = ["# Short-vol signal-day expansion", "", "Live inference was not modified or restarted.", "", f"**Decision:** {recommendation}", "",
             f"Dataset: {total_days} distinct trading days. Costs: current 2026 Zerodha schedule plus modeled execution friction.", "",
             "## Candidates passing all required criteria", ""]
    if len(accepted):
        lines.append(accepted[["name", "signal_days", "profitable_days", "day_success_rate", "trades", "trade_hit_rate", "total_net", "net_2024", "net_2025"]].to_string(index=False))
    else:
        lines.append("None.")
    lines += ["", "## Best candidates by signal days", "",
              ten.sort_values(["signal_days", "day_success_rate"], ascending=False).head(12)[["name", "signal_days", "profitable_days", "day_success_rate", "trades", "trade_hit_rate", "total_net", "net_2024", "net_2025", "passes_required"]].to_string(index=False), "",
              "## Limitations", "", "- The 2024 period is development evidence and 2025 is the held-out period, but the underlying forecast models had already been studied.",
              "- The structure file has modeled prices rather than executable bid/ask quotes; 10/25 bps scenarios are stress assumptions.",
              "- No event strategy was tested because no point-in-time event calendar was available.",
              "- Passing candidates still require forward paper trading before changing live rules."]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(recommendation)
    print(ten.sort_values(["passes_required", "signal_days", "day_success_rate"], ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
