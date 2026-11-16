from __future__ import annotations

"""Full 2x2x2 spot-direction, realized-variance, and IV-change shadow backtest."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import monthly_dte4_research as research
import monthly_dte4_structure_hybrid as structures
import strategy_backtest_dashboard as dashboard
from directional_iv_matrix_backtest import leg, predicted_position, reprice, safe_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spot_rv_iv_matrix_dashboard"
PRIMARY_BPS = 25.0
COST_MULTIPLIER = 2.0

MATRIX: dict[tuple[int, int, int], tuple[str, str]] = {
    (1, 1, 1): ("up_highrv_ivup_long_atmcall_otmput", "long ATM call + OTM put"),
    (1, 1, -1): ("up_highrv_ivdown_long_strangle", "long OTM strangle"),
    (1, -1, 1): ("up_lowrv_ivup_short_atm_long_call", "short ATM straddle + long OTM call"),
    (1, -1, -1): ("up_lowrv_ivdown_short_predspot", "short predicted-spot straddle"),
    (-1, 1, 1): ("down_highrv_ivup_long_atmput_otmcall", "long ATM put + OTM call"),
    (-1, 1, -1): ("down_highrv_ivdown_long_strangle", "long OTM strangle"),
    (-1, -1, 1): ("down_lowrv_ivup_short_atm_long_put", "short ATM straddle + long OTM put"),
    (-1, -1, -1): ("down_lowrv_ivdown_short_predspot", "short predicted-spot straddle"),
}
BRANCHES = [value[0] for value in MATRIX.values()]


def rv_edge(row: pd.Series) -> float:
    """Predicted ATM long-straddle gamma P&L plus theta, before IV and costs."""
    gamma = float(row["gamma_call_+0"] + row["gamma_put_+0"])
    theta = float(row["theta_call_+0"] + row["theta_put_+0"])
    return (
        0.5 * gamma * float(row["spot_price"]) ** 2 * float(row["pred_future_variance"])
        + theta * float(row["elapsed_days"])
    )


def state_and_legs(row: pd.Series) -> tuple[tuple[int, int, int], str, list[tuple[str, int, int]]]:
    spot = 1 if float(row["pred_session_return"]) > 0 else -1
    rv = 1 if rv_edge(row) > 0 else -1
    iv = 1 if float(row["pred_d_iv"]) > 0 else -1
    state = (spot, rv, iv)
    name = MATRIX[state][0]
    pos = predicted_position(row)
    definitions = {
        (1, 1, 1): [("call", 0, 1), ("put", -1, 1)],
        (1, 1, -1): [("call", 1, 1), ("put", -1, 1)],
        (1, -1, 1): [("call", 0, -1), ("put", 0, -1), ("call", 1, 1)],
        (1, -1, -1): [("call", pos, -1), ("put", pos, -1)],
        (-1, 1, 1): [("put", 0, 1), ("call", 1, 1)],
        (-1, 1, -1): [("put", -1, 1), ("call", 1, 1)],
        (-1, -1, 1): [("call", 0, -1), ("put", 0, -1), ("put", -1, 1)],
        (-1, -1, -1): [("call", pos, -1), ("put", pos, -1)],
    }
    return state, name, definitions[state]


def build_trades(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_exit = [f"valid_close_exit_{p:+d}" for p in (-2, -1, 0, 1, 2)]
    eligible = frame[frame[valid_exit].fillna(False).all(axis=1)].copy()
    opportunities, trades = [], []
    for _, row in eligible.iterrows():
        required = ["pred_session_return", "pred_future_variance", "pred_d_iv", "session_end_spot"]
        if not all(np.isfinite(row.get(x, np.nan)) for x in required):
            continue
        state, name, specs = state_and_legs(row)
        opportunities.append({
            "date": row["date"], "datetime": row["datetime"], "expiry": row["expiry"],
            "strategy": name, "spot_signal": state[0], "rv_signal": state[1], "iv_signal": state[2],
            "rv_gamma_theta_edge_points": rv_edge(row),
        })
        arr = np.asarray([leg(row, kind, pos) for kind, pos, _ in specs], float)
        directions = np.array([sign for _, _, sign in specs], float)
        if not np.isfinite(arr).all() or (arr[:, 0] < 5).any() or (arr[:, 1] < 0).any():
            opportunities[-1].update({"eligible_trade": False, "rejection_reason": "invalid_leg_or_price"})
            continue
        entry, exit_ = arr[:, 0], arr[:, 1]
        delta, gamma, theta, vega = arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        d_spot = float(row["predicted_spot"] - row["spot_price"])
        components = {
            "expected_delta_rupees": float(np.sum(directions * delta * d_spot) * row["lot_size"]),
            "expected_gamma_rupees": float(np.sum(directions * 0.5 * gamma * d_spot**2) * row["lot_size"]),
            "expected_theta_rupees": float(np.sum(directions * theta * row["elapsed_days"]) * row["lot_size"]),
            "expected_vega_rupees": float(np.sum(directions * vega * 100.0 * row["pred_d_iv"]) * row["lot_size"]),
        }
        expected_gross = sum(components.values())
        selection = structures.leg_level_cost(
            entry[None, :], entry[None, :], directions, np.array([row["lot_size"]]),
            pd.Series([row["date"]]), PRIMARY_BPS,
        ).iloc[0]
        hurdle = COST_MULTIPLIER * float(selection["total_cost"])
        opportunities[-1].update({
            "expected_gross_rupees": expected_gross,
            "one_x_cost_rupees": float(selection["total_cost"]),
            "two_x_cost_hurdle_rupees": hurdle,
            "eligible_trade": bool(expected_gross > hurdle),
            "rejection_reason": "" if expected_gross > hurdle else "expected_pnl_below_2x_cost",
            **components,
        })
        if expected_gross <= hurdle:
            continue
        costs = structures.leg_level_cost(
            entry[None, :], exit_[None, :], directions, np.array([row["lot_size"]]),
            pd.Series([row["date"]]), PRIMARY_BPS,
        ).iloc[0]
        gross = float(np.sum(directions * (exit_ - entry)) * row["lot_size"])
        actual_ds = float(row["session_end_spot"] - row["spot_price"])
        actual_div = float(row["end_iv_atm"] - row["iv_atm"])
        attribution = {
            "actual_delta_attribution": float(np.sum(directions * delta * actual_ds) * row["lot_size"]),
            "actual_gamma_attribution": float(np.sum(directions * 0.5 * gamma * actual_ds**2) * row["lot_size"]),
            "actual_theta_attribution": float(np.sum(directions * theta * row["elapsed_days"]) * row["lot_size"]),
            "actual_vega_attribution": float(np.sum(directions * vega * 100.0 * actual_div) * row["lot_size"]),
        }
        record = row.to_dict()
        record.update({
            "strategy": name, "spot_signal": state[0], "rv_signal": state[1], "iv_signal": state[2],
            "rv_gamma_theta_edge_points": rv_edge(row), "leg_definition": repr(specs),
            "predicted_strike_position": predicted_position(row), "expected_gross_rupees": expected_gross,
            "selection_cost_rupees": float(selection["total_cost"]), "cost_hurdle_rupees": hurdle,
            "predicted_net_after_hurdle": expected_gross - hurdle,
            "gross_pnl_rupees": gross, "net_pnl_rupees": gross - float(costs["total_cost"]),
        })
        record.update(components)
        record.update(costs.to_dict())
        record.update(attribution)
        record["greek_attribution_residual"] = gross - sum(attribution.values())
        record["profitable"] = record["net_pnl_rupees"] > 0
        trades.append(record)
    return pd.DataFrame(trades), pd.DataFrame(opportunities)


def summarize(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ledgers = [], []
    for bps in (10.0, 25.0, 50.0):
        priced = reprice(trades, bps)
        priced["cost_bps"] = bps
        ledgers.append(priced)
        for branch in BRANCHES:
            subset = priced[priced["strategy"].eq(branch)] if "strategy" in priced else priced
            rows.append({"strategy": branch, "cost_bps": bps, **safe_metrics(subset)})
        rows.append({"strategy": "combined_8_state_matrix", "cost_bps": bps, **safe_metrics(priced)})
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


def html(summary: pd.DataFrame, opportunities: pd.DataFrame, diag: dict) -> str:
    payload = json.dumps(summary.replace({np.nan: None, np.inf: None}).to_dict("records"), separators=(",", ":"))
    names = {key: f"{state}: {label}" for state, (key, label) in MATRIX.items()}
    names["combined_8_state_matrix"] = "Combined 8-state matrix"
    labels = json.dumps(names)
    counts = opportunities["strategy"].value_counts().to_dict()
    count_text = "; ".join(f"{names.get(k,k)} = {v}" for k, v in counts.items())
    expected = opportunities["expected_gross_rupees"].dropna() if "expected_gross_rupees" in opportunities else pd.Series(dtype=float)
    economics = (
        f"median expected gross = {expected.median():,.0f} rupees; maximum = {expected.max():,.0f} rupees"
        if len(expected) else "no valid candidate economics"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spot + RV + IV Backtest</title><style>:root{{--bg:#08111f;--p:#111f34;--l:#2b405e;--t:#eef5fc;--m:#9bb0c8;--g:#4fd1a5;--r:#ff7480;--a:#69a8ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--t);font:14px/1.55 system-ui}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{font-size:26px}}h2{{font-size:18px;margin-top:28px}}p{{color:var(--m);max-width:1150px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card,.box{{background:var(--p);border:1px solid var(--l);border-radius:13px;padding:15px}}.v{{font-size:22px}}button{{padding:8px 13px;border-radius:9px;border:1px solid var(--l);background:transparent;color:var(--t);margin-right:8px}}button.active{{background:var(--a);color:#07111f}}.wrap{{overflow:auto;border:1px solid var(--l);border-radius:13px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid var(--l);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--m)}}.pos{{color:var(--g)}}.neg{{color:var(--r)}}code{{color:var(--t)}}</style></head><body><main>
<h1>DTE 4&ndash;14 spot + RV + IV matrix</h1><p>Eight-state walk-forward shadow backtest, entered around 09:45 and closed on the same day after 15:15. One lot, one trade per day, no overnight exposure. Incumbent models are unchanged.</p>
<div class="grid"><div class="card"><div>Spot hit rate</div><div class="v">{diag['direction_hit_rate']:.1%}</div></div><div class="card"><div>Spot R&sup2;</div><div class="v">{diag['r2_vs_zero']:.3f}</div></div><div class="card"><div>Signal opportunities</div><div class="v">{len(opportunities)}</div></div><div class="card"><div>Cost hurdle</div><div class="v">2.0&times;</div></div></div>
<h2>Cost sensitivity</h2><p><button class="active" data-bps="25">25 bps</button><button data-bps="10">10 bps</button><button data-bps="50">50 bps</button></p><div class="wrap"><table><thead><tr><th>State and structure</th><th>Trades</th><th>Net &#8377;</th><th>Hit</th><th>PF</th><th>Sharpe</th><th>Drawdown</th><th>Cycles</th><th>95% CI/cycle</th><th>Holm p</th><th>Gate</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Mathematical rules</h2><div class="box"><code>spot = sign(r_hat)</code>.<br><code>RV-high iff 0.5*(Gamma_call+Gamma_put)*S^2*RV_hat + (Theta_call+Theta_put)*dt &gt; 0</code> at ATM.<br><code>IV = sign(deltaIV_hat)</code>.<br><br><code>E[dV] = Delta*dS_hat + 0.5*Gamma*dS_hat^2 + Theta*dt + Vega*100*deltaIV_hat</code>.<br>A trade is taken only when expected gross P&amp;L exceeds <b>2 times</b> complete modeled transaction costs.</div>
<p><b>Opportunity distribution:</b> {count_text or 'none'}. Candidate economics: {economics}. None passed the 2&times; cost hurdle.</p><p>OTM conventions were corrected: OTM puts are below ATM and OTM calls are above ATM. Bid/ask, volume, and event-calendar history are absent and were not fabricated. Execution is stressed at 10/25/50 bps. No finite backtest mathematically proves future profit.</p>
<script>const data={payload},names={labels};const f=x=>x==null?'--':Number(x).toLocaleString('en-IN',{{maximumFractionDigits:1}});function draw(b){{document.getElementById('rows').innerHTML=data.filter(x=>x.cost_bps===b).map(x=>`<tr><td>${{names[x.strategy]||x.strategy}}</td><td>${{f(x.trades)}}</td><td class="${{x.total_net>=0?'pos':'neg'}}">${{f(x.total_net)}}</td><td>${{x.hit_rate==null?'--':(100*x.hit_rate).toFixed(1)+'%'}}</td><td>${{f(x.profit_factor)}}</td><td>${{f(x.daily_sharpe)}}</td><td>${{f(x.max_drawdown)}}</td><td>${{f(x.cycles)}}</td><td>${{x.ci_low==null?'--':f(x.ci_low)+' to '+f(x.ci_high)}}</td><td>${{x.holm_p==null?'--':Number(x.holm_p).toFixed(3)}}</td><td>${{x.promotion_gate?'PASS':'NO'}}</td></tr>`).join('')}}document.querySelectorAll('button').forEach(x=>x.onclick=()=>{{document.querySelectorAll('button').forEach(y=>y.classList.remove('active'));x.classList.add('active');draw(Number(x.dataset.bps))}});draw(25);</script></main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, _ = research.load_daily_panel()
    days, frame, audit = dashboard.prepare_frame()
    frame = frame.merge(full[["datetime", "session_end_spot"]], on="datetime", how="left", validate="one_to_one")
    trades, opportunities = build_trades(frame)
    summary, ledger = summarize(trades)
    diag = dashboard.signed_diagnostics(full, days)
    summary.to_csv(OUT / "matrix_results.csv", index=False)
    ledger.to_csv(OUT / "matrix_trade_ledger.csv", index=False)
    opportunities.to_csv(OUT / "matrix_opportunities.csv", index=False)
    subperiods(ledger).to_csv(OUT / "subperiod_results.csv", index=False)
    baseline_path = ROOT / "outputs" / "strategy_backtest_dashboard" / "strategy_results.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        baseline[baseline["cost_bps"].eq(PRIMARY_BPS)].to_csv(OUT / "baseline_comparison_25bps.csv", index=False)
    pd.DataFrame([{**audit, **diag, "opportunities": len(opportunities), "trades": len(trades)}]).to_csv(OUT / "audit.csv", index=False)
    (OUT / "index.html").write_text(html(summary, opportunities, diag), encoding="utf-8")
    print(summary[summary["cost_bps"].eq(PRIMARY_BPS)].to_string(index=False))
    print("\nOpportunities:\n", opportunities["strategy"].value_counts().to_string())


if __name__ == "__main__":
    main()
