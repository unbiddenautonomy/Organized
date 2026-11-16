from __future__ import annotations

"""Walk-forward DTE 4-14 directional + IV matrix shadow backtest."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly_dte4_research as research
import monthly_dte4_structure_hybrid as structures
import strategy_backtest_dashboard as dashboard


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "directional_iv_matrix_dashboard"
PRIMARY_BPS = 25.0
BRANCHES = [
    "spot_up_iv_up_long_call_put",
    "spot_up_iv_down_short_predspot",
    "spot_down_iv_up_long_put_call",
    "spot_down_iv_down_short_predspot",
]


def leg(row: pd.Series, kind: str, pos: int) -> tuple[float, ...]:
    return (
        float(row[f"price_{kind}_{pos:+d}"]),
        float(row[f"exit_price_{kind}_{pos:+d}"]),
        float(row[f"delta_{kind}_{pos:+d}"]),
        float(row[f"gamma_{kind}_{pos:+d}"]),
        float(row[f"theta_{kind}_{pos:+d}"]),
        float(row[f"vega_{kind}_{pos:+d}"]),
    )


def predicted_position(row: pd.Series) -> int:
    positions = np.array([-2, -1, 0, 1, 2])
    strikes = np.array([row[f"strike_{p:+d}"] for p in positions], float)
    return int(positions[np.nanargmin(np.abs(strikes - float(row["predicted_spot"])))])


def branch_definition(row: pd.Series) -> tuple[str, list[tuple[str, int, int]]]:
    spot_up = float(row["pred_session_return"]) > 0
    iv_up = float(row["pred_d_iv"]) > 0
    if spot_up and iv_up:
        # Correct OTM convention: a put below ATM is OTM.
        return BRANCHES[0], [("call", 0, 1), ("put", -1, 1)]
    if spot_up and not iv_up:
        pos = predicted_position(row)
        return BRANCHES[1], [("call", pos, -1), ("put", pos, -1)]
    if not spot_up and iv_up:
        # Correct OTM convention: a call above ATM is OTM.
        return BRANCHES[2], [("put", 0, 1), ("call", 1, 1)]
    pos = predicted_position(row)
    return BRANCHES[3], [("call", pos, -1), ("put", pos, -1)]


def build_trades(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_exit = [f"valid_close_exit_{p:+d}" for p in (-2, -1, 0, 1, 2)]
    eligible = frame[frame[valid_exit].fillna(False).all(axis=1)].copy()
    signal_rows, trades = [], []
    for _, row in eligible.iterrows():
        if not np.isfinite(row.get("pred_session_return", np.nan)) or not np.isfinite(row.get("pred_d_iv", np.nan)):
            continue
        name, specs = branch_definition(row)
        signal_rows.append({
            "date": row["date"], "datetime": row["datetime"], "expiry": row["expiry"],
            "strategy": name, "spot_signal": int(np.sign(row["pred_session_return"])),
            "iv_signal": int(np.sign(row["pred_d_iv"])),
        })
        values = [leg(row, kind, pos) for kind, pos, _ in specs]
        arr = np.asarray(values, float)
        directions = np.array([sign for _, _, sign in specs], float)
        if not np.isfinite(arr).all() or (arr[:, 0] < 5).any() or (arr[:, 1] < 0).any():
            continue
        entry, exit_ = arr[:, 0], arr[:, 1]
        delta, gamma, theta, vega = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        d_spot = float(row["predicted_spot"] - row["spot_price"])
        expected_delta = delta * d_spot
        expected_gamma = 0.5 * gamma * d_spot**2
        expected_theta = theta * float(row["elapsed_days"])
        expected_vega = vega * 100.0 * float(row["pred_d_iv"])
        expected_leg_long = expected_delta + expected_gamma + expected_theta + expected_vega
        expected_gross = float(np.sum(directions * expected_leg_long) * row["lot_size"])
        selection_cost = structures.leg_level_cost(
            entry[None, :], entry[None, :], directions,
            np.array([row["lot_size"]]), pd.Series([row["date"]]), PRIMARY_BPS,
        ).iloc[0]
        predicted_net = expected_gross - float(selection_cost["total_cost"])
        if predicted_net <= 0:
            continue
        realized_cost = structures.leg_level_cost(
            entry[None, :], exit_[None, :], directions,
            np.array([row["lot_size"]]), pd.Series([row["date"]]), PRIMARY_BPS,
        ).iloc[0]
        gross = float(np.sum(directions * (exit_ - entry)) * row["lot_size"])
        actual_d_spot = float(row["session_end_spot"] - row["spot_price"])
        actual_d_iv = float(row["end_iv_atm"] - row["iv_atm"])
        actual_components = {
            "actual_delta_attribution": float(np.sum(directions * delta * actual_d_spot) * row["lot_size"]),
            "actual_gamma_attribution": float(np.sum(directions * 0.5 * gamma * actual_d_spot**2) * row["lot_size"]),
            "actual_theta_attribution": float(np.sum(directions * theta * float(row["elapsed_days"])) * row["lot_size"]),
            "actual_vega_attribution": float(np.sum(directions * vega * 100.0 * actual_d_iv) * row["lot_size"]),
        }
        record = row.to_dict()
        record.update({
            "strategy": name, "spot_signal": int(np.sign(row["pred_session_return"])),
            "iv_signal": int(np.sign(row["pred_d_iv"])), "leg_definition": repr(specs),
            "predicted_strike_position": predicted_position(row),
            "expected_gross_rupees": expected_gross,
            "expected_delta_rupees": float(np.sum(directions * expected_delta) * row["lot_size"]),
            "expected_gamma_rupees": float(np.sum(directions * expected_gamma) * row["lot_size"]),
            "expected_theta_rupees": float(np.sum(directions * expected_theta) * row["lot_size"]),
            "expected_vega_rupees": float(np.sum(directions * expected_vega) * row["lot_size"]),
            "selection_cost_rupees": float(selection_cost["total_cost"]),
            "predicted_net_rupees": predicted_net, "gross_pnl_rupees": gross,
            "net_pnl_rupees": gross - float(realized_cost["total_cost"]),
        })
        record.update(realized_cost.to_dict())
        record.update(actual_components)
        record["greek_attribution_residual"] = gross - sum(actual_components.values())
        record["profitable"] = record["net_pnl_rupees"] > 0
        trades.append(record)
    return pd.DataFrame(trades), pd.DataFrame(signal_rows)


def reprice(frame: pd.DataFrame, bps: float) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    turnover = out["buy_turnover"] + out["sell_turnover"]
    out["total_cost"] = out["total_cost"] - out["slippage"] + turnover * bps / 10000.0
    out["slippage"] = turnover * bps / 10000.0
    out["net_pnl_rupees"] = out["gross_pnl_rupees"] - out["total_cost"]
    out["profitable"] = out["net_pnl_rupees"].gt(0)
    return out


def safe_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "trades": 0, "days": 0, "total_gross": 0.0, "total_cost": 0.0,
            "total_net": 0.0, "mean_net": np.nan, "median_net": np.nan,
            "hit_rate": np.nan, "profit_factor": np.nan, "daily_sharpe": np.nan,
            "max_drawdown": np.nan, "cycles": 0, "mean_cycle_net": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "one_sided_p": np.nan,
            "mde_80pct": np.nan,
        }
    return {**research.performance(frame), **research.cycle_inference(frame)}


def summarize(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ledgers = [], []
    for bps in (10.0, 25.0, 50.0):
        priced = reprice(trades, bps)
        priced["cost_bps"] = bps
        ledgers.append(priced)
        for branch in BRANCHES:
            rows.append({"strategy": branch, "cost_bps": bps, **safe_metrics(priced[priced["strategy"].eq(branch)])})
        rows.append({"strategy": "combined_matrix", "cost_bps": bps, **safe_metrics(priced)})
    result = pd.DataFrame(rows)
    result["holm_p"] = result.groupby("cost_bps")["one_sided_p"].transform(research.holm_adjust)
    at_50 = result[result["cost_bps"].eq(50)].set_index("strategy")["total_net"]
    result["promotion_gate"] = (
        result["cycles"].ge(30) & result["ci_low"].gt(0) & result["holm_p"].lt(0.05)
        & result["strategy"].map(at_50).gt(0)
    )
    return result, pd.concat(ledgers, ignore_index=True)


def subperiods(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    z = ledger[ledger["cost_bps"].eq(PRIMARY_BPS)].copy()
    z["dte_group"] = pd.cut(z["days_to_expiry"], [4, 8, 15], right=False, labels=["4-7", "8-14"])
    z["year"] = pd.to_datetime(z["date"]).dt.year
    return z.groupby(["strategy", "year", "dte_group"], observed=True).agg(
        trades=("net_pnl_rupees", "size"), net=("net_pnl_rupees", "sum"),
        hit_rate=("profitable", "mean"), mean_net=("net_pnl_rupees", "mean"),
    ).reset_index()


def report_html(summary: pd.DataFrame, signal_counts: dict, diag: dict) -> str:
    data = json.dumps(summary.replace({np.nan: None, np.inf: None}).to_dict("records"), separators=(",", ":"))
    names = {
        BRANCHES[0]: "Spot up + IV up: long ATM call + OTM put",
        BRANCHES[1]: "Spot up + IV down: short predicted-spot straddle",
        BRANCHES[2]: "Spot down + IV up: long ATM put + OTM call",
        BRANCHES[3]: "Spot down + IV down: short predicted-spot straddle",
        "combined_matrix": "Combined 2x2 matrix",
    }
    name_json = json.dumps(names)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Directional + IV Matrix</title>
<style>:root{{--bg:#08111f;--p:#111f34;--line:#2b405e;--t:#eef5fc;--m:#9bb0c8;--g:#4fd1a5;--r:#ff7480;--a:#69a8ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--t);font:14px/1.55 system-ui}}main{{max-width:1400px;margin:auto;padding:28px}}h1{{font-size:26px}}h2{{font-size:18px;margin-top:28px}}p{{color:var(--m);max-width:1100px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card,.box{{background:var(--p);border:1px solid var(--line);border-radius:13px;padding:15px}}.v{{font-size:22px}}button{{padding:8px 13px;border-radius:9px;border:1px solid var(--line);background:transparent;color:var(--t);margin-right:8px}}button.active{{background:var(--a);color:#07111f}}.wrap{{overflow:auto;border:1px solid var(--line);border-radius:13px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--m)}}.pos{{color:var(--g)}}.neg{{color:var(--r)}}code{{color:var(--t)}}</style></head><body><main>
<h1>DTE 4&ndash;14 directional + IV matrix</h1><p>Shadow backtest at approximately 09:45 with verified same-day exits after 15:15, one lot, no overnight carry, and no overlapping daily positions. The incumbent models are unchanged.</p>
<div class="grid"><div class="card"><div>Spot-direction hit rate</div><div class="v">{diag['direction_hit_rate']:.1%}</div></div><div class="card"><div>Spot model R&sup2;</div><div class="v">{diag['r2_vs_zero']:.3f}</div></div><div class="card"><div>IV-up forecast days</div><div class="v">{signal_counts.get('iv_up',0)}</div></div><div class="card"><div>IV-down forecast days</div><div class="v">{signal_counts.get('iv_down',0)}</div></div></div>
<h2>Cost sensitivity</h2><p><button class="active" data-bps="25">25 bps</button><button data-bps="10">10 bps</button><button data-bps="50">50 bps</button></p>
<div class="wrap"><table><thead><tr><th>Matrix branch</th><th>Trades</th><th>Net &#8377;</th><th>Hit</th><th>Profit factor</th><th>Sharpe</th><th>Cycles</th><th>95% CI/cycle</th><th>Holm p</th><th>Gate</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Mathematical specification</h2><div class="box"><code>spotSignal = sign(r_hat)</code>; <code>ivSignal = sign(deltaIV_hat)</code>.<br><code>S_hat = S exp(r_hat)</code> and the predicted-spot strike minimizes <code>|K-S_hat|</code>.<br><br><code>E[dV] = Delta*dS_hat + 0.5*Gamma*dS_hat^2 + Theta*dt + Vega*100*deltaIV_hat</code>. A trade is entered only if summed expected leg P&amp;L minus complete modeled costs is positive.<br><br>Corrected strike convention: bullish long-vol uses ATM call + put one strike <em>below</em> ATM; bearish long-vol uses ATM put + call one strike <em>above</em> ATM.</div>
<p>The forecast-cost hurdle is the economic signal-strength threshold; no separately optimized threshold was added. This avoids choosing a cutoff on the test outcomes. A finite backtest does not prove future profitability. Promotion requires at least 30 expiry cycles, positive bootstrap lower bound, Holm p &lt; 0.05, and positive P&amp;L at 50 bps.</p>
<p><b>Data limitations:</b> historical bid/ask, volume, and event-calendar fields requested in the proposal are not present. They were not invented; execution is stressed at 10/25/50 bps. Because every IV forecast in this horizon was negative, the two IV-up long-vol branches received no opportunities.</p>
<script>const data={data},names={name_json};const f=x=>x==null?'--':Number(x).toLocaleString('en-IN',{{maximumFractionDigits:1}});function draw(b){{document.getElementById('rows').innerHTML=data.filter(x=>x.cost_bps===b).map(x=>`<tr><td>${{names[x.strategy]}}</td><td>${{f(x.trades)}}</td><td class="${{x.total_net>=0?'pos':'neg'}}">${{f(x.total_net)}}</td><td>${{x.hit_rate==null?'--':(100*x.hit_rate).toFixed(1)+'%'}}</td><td>${{f(x.profit_factor)}}</td><td>${{f(x.daily_sharpe)}}</td><td>${{f(x.cycles)}}</td><td>${{x.ci_low==null?'--':f(x.ci_low)+' to '+f(x.ci_high)}}</td><td>${{x.holm_p==null?'--':Number(x.holm_p).toFixed(3)}}</td><td>${{x.promotion_gate?'PASS':'NO'}}</td></tr>`).join('')}}document.querySelectorAll('button').forEach(x=>x.onclick=()=>{{document.querySelectorAll('button').forEach(y=>y.classList.remove('active'));x.classList.add('active');draw(Number(x.dataset.bps))}});draw(25);</script></main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, _ = research.load_daily_panel()
    days, frame, audit = dashboard.prepare_frame()
    frame = frame.merge(
        full[["datetime", "session_end_spot"]], on="datetime", how="left", validate="one_to_one"
    )
    trades, signals = build_trades(frame)
    summary, ledger = summarize(trades)
    diag = dashboard.signed_diagnostics(full, days)
    counts = {"iv_up": int(signals["iv_signal"].gt(0).sum()), "iv_down": int(signals["iv_signal"].lt(0).sum())}
    summary.to_csv(OUT / "matrix_results.csv", index=False)
    ledger.to_csv(OUT / "matrix_trade_ledger.csv", index=False)
    signals.to_csv(OUT / "matrix_signal_opportunities.csv", index=False)
    subperiods(ledger).to_csv(OUT / "subperiod_results.csv", index=False)
    pd.DataFrame([{**audit, **diag, **counts}]).to_csv(OUT / "audit.csv", index=False)
    (OUT / "index.html").write_text(report_html(summary, counts, diag), encoding="utf-8")
    print(summary[summary["cost_bps"].eq(PRIMARY_BPS)].to_string(index=False))
    print("\nSignal opportunities:\n", signals["strategy"].value_counts().to_string())


if __name__ == "__main__":
    main()
