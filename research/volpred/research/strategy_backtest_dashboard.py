from __future__ import annotations

"""Build the requested multi-strategy backtest and self-contained HTML report."""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import monthly_dte4_research as research
import monthly_dte4_structure_hybrid as structure_tools
import monthly_dte4_multi_strategy_engine as engine


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "strategy_backtest_dashboard"
PRIMARY_BPS = 25.0
ALPHAS = (0.1, 1.0, 10.0, 100.0)
SIGNED_FEATURES = [
    # Spot-derived information only: no option IV, Greeks, OI, or forward basis.
    "return_past_5", "return_past_15", "return_past_60", "overnight_return",
    "abs_return_past_5", "abs_return_past_15", "abs_return_past_60",
    "rv_past_5", "rv_past_15", "rv_past_60",
]


def signed_spot_walk_forward(full: pd.DataFrame, test_cycles: list[pd.Timestamp]) -> pd.DataFrame:
    cycles = list(pd.Series(full["expiry_cycle"].unique()).sort_values())
    rows = []
    for cycle in test_cycles:
        pos = cycles.index(cycle)
        if pos < research.MIN_TRAIN_CYCLES + research.VALIDATION_CYCLES:
            continue
        train_cycles = cycles[:pos - research.VALIDATION_CYCLES]
        validation_cycles = cycles[pos - research.VALIDATION_CYCLES:pos]
        train = full[full["expiry_cycle"].isin(train_cycles)].replace([np.inf, -np.inf], np.nan)
        validation = full[full["expiry_cycle"].isin(validation_cycles)].replace([np.inf, -np.inf], np.nan)
        test = full[full["expiry_cycle"].eq(cycle)].replace([np.inf, -np.inf], np.nan)
        best = None
        for alpha in ALPHAS:
            tr = train.dropna(subset=SIGNED_FEATURES + ["target_session_return"])
            va = validation.dropna(subset=SIGNED_FEATURES + ["target_session_return"])
            if len(tr) < 100 or va.empty:
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(tr[SIGNED_FEATURES], tr["target_session_return"])
            loss = float(np.mean((va["target_session_return"] - model.predict(va[SIGNED_FEATURES])) ** 2))
            if best is None or loss < best[0]:
                best = (loss, alpha)
        if best is None:
            continue
        refit = pd.concat([train, validation], ignore_index=True).dropna(
            subset=SIGNED_FEATURES + ["target_session_return"]
        )
        score = test.dropna(subset=SIGNED_FEATURES).copy()
        model = make_pipeline(StandardScaler(), Ridge(alpha=best[1]))
        model.fit(refit[SIGNED_FEATURES], refit["target_session_return"])
        score["pred_session_return"] = model.predict(score[SIGNED_FEATURES])
        score["signed_model_alpha"] = best[1]
        rows.append(score[["datetime", "pred_session_return", "signed_model_alpha"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    full, _ = research.load_daily_panel()
    con = duckdb.connect()
    days = con.execute(
        "SELECT * FROM read_parquet(?) WHERE days_to_expiry>=4 AND days_to_expiry<15",
        [str(ROOT / "outputs" / "monthly_dte4_research" / "oos_predictions.parquet")],
    ).fetchdf()
    con.close()
    cycles = list(pd.Series(days["outer_test_cycle"].unique()).sort_values())
    signed = signed_spot_walk_forward(full, cycles)
    days = days.merge(signed, on="datetime", how="left", validate="one_to_one")
    days["predicted_spot"] = days["spot_price"] * np.exp(days["pred_session_return"])
    legs, audit = structure_tools.load_legs(days)
    wide = engine.normalize_wide(structure_tools.wide_legs(legs))
    frame = days.merge(wide, on=["datetime", "expiry"], how="left", suffixes=("", "_structure"), validate="one_to_one")
    return days, frame, audit


def score_centered_short(frame: pd.DataFrame) -> pd.DataFrame:
    strike_cols = [f"strike_{p:+d}" for p in (-2, -1, 0, 1, 2)]
    strikes = frame[strike_cols].to_numpy(float)
    chosen_idx = np.abs(strikes - frame["predicted_spot"].to_numpy(float)[:, None]).argmin(axis=1)
    positions = np.array([-2, -1, 0, 1, 2])[chosen_idx]
    n = len(frame)
    entry = np.empty((n, 2)); exit_ = np.empty((n, 2)); gamma = np.empty((n, 2)); theta = np.empty((n, 2)); vega = np.empty((n, 2))
    for i, pos in enumerate(positions):
        entry[i] = [frame.iloc[i][f"price_call_{pos:+d}"], frame.iloc[i][f"price_put_{pos:+d}"]]
        exit_[i] = [frame.iloc[i][f"exit_price_call_{pos:+d}"], frame.iloc[i][f"exit_price_put_{pos:+d}"]]
        gamma[i] = [frame.iloc[i][f"gamma_call_{pos:+d}"], frame.iloc[i][f"gamma_put_{pos:+d}"]]
        theta[i] = [frame.iloc[i][f"theta_call_{pos:+d}"], frame.iloc[i][f"theta_put_{pos:+d}"]]
        vega[i] = [frame.iloc[i][f"vega_call_{pos:+d}"], frame.iloc[i][f"vega_put_{pos:+d}"]]
    direction = np.array([-1.0, -1.0])
    valid = (
        np.isfinite(entry).all(axis=1) & np.isfinite(exit_).all(axis=1) & np.isfinite(gamma).all(axis=1)
        & np.isfinite(theta).all(axis=1) & np.isfinite(vega).all(axis=1) & (entry >= 5).all(axis=1)
    )
    z = frame.loc[valid].copy().reset_index(drop=True)
    ent, ex, ga, th, ve, pos = entry[valid], exit_[valid], gamma[valid], theta[valid], vega[valid], positions[valid]
    expected_long = (
        0.5 * ga * z["spot_price"].to_numpy(float)[:, None] ** 2 * z["pred_future_variance"].to_numpy(float)[:, None]
        + th * z["elapsed_days"].to_numpy(float)[:, None]
        + ve * (100.0 * z["pred_d_iv"].to_numpy(float)[:, None])
    )
    expected = (expected_long * direction[None, :]).sum(axis=1) * z["lot_size"].to_numpy(float)
    selection = structure_tools.leg_level_cost(ent, ent, direction, z["lot_size"].to_numpy(), z["date"], PRIMARY_BPS)["total_cost"].to_numpy()
    z["predicted_strike_position"] = pos
    z["predicted_net_rupees"] = expected - selection
    active = z["predicted_net_rupees"].gt(0)
    z, ent, ex = z[active].copy().reset_index(drop=True), ent[active.to_numpy()], ex[active.to_numpy()]
    costs = structure_tools.leg_level_cost(ent, ex, direction, z["lot_size"].to_numpy(), z["date"], PRIMARY_BPS)
    for col in costs:
        z[col] = costs[col].to_numpy()
    z["strategy"] = "predicted_spot_short_straddle"
    z["gross_pnl_rupees"] = ((ex - ent) * direction[None, :]).sum(axis=1) * z["lot_size"].to_numpy(float)
    z["net_pnl_rupees"] = z["gross_pnl_rupees"] - z["total_cost"]
    z["profitable"] = z["net_pnl_rupees"].gt(0)
    return z


def strategy_set(days: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    base = engine.score_all(days, structure_tools.load_legs(days)[0])
    keep = ["short_atm_straddle", "long_atm_straddle", "short_1_strangle", "long_1_strangle"]
    fixed = base[base["strategy"].isin(keep)].copy()
    centered = score_centered_short(frame)
    short_atm = fixed[fixed["strategy"].eq("short_atm_straddle") & fixed["pred_d_iv"].le(0)].copy()
    long_atm = fixed[fixed["strategy"].eq("long_atm_straddle") & fixed["pred_d_iv"].gt(0)].copy()
    switch = pd.concat([short_atm, long_atm], ignore_index=True)
    switch["source_strategy"] = switch["strategy"]
    switch["strategy"] = "iv_regime_switch_atm"
    return pd.concat([fixed, centered, switch], ignore_index=True, sort=False)


def reprice(frame: pd.DataFrame, bps: float) -> pd.DataFrame:
    z = frame.copy()
    turnover = z["buy_turnover"] + z["sell_turnover"]
    z["total_cost"] = z["total_cost"] - z["slippage"] + turnover * bps / 10000.0
    z["slippage"] = turnover * bps / 10000.0
    z["net_pnl_rupees"] = z["gross_pnl_rupees"] - z["total_cost"]
    z["profitable"] = z["net_pnl_rupees"].gt(0)
    return z


def summarize(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, ledgers = [], []
    strategies = trades["strategy"].unique().tolist()
    # A one-position-per-day hybrid across every cost-qualified alternative.
    hybrid = trades.sort_values(["date", "predicted_net_rupees"], ascending=[True, False]).drop_duplicates("date").copy()
    hybrid["source_strategy"] = hybrid["strategy"]
    hybrid["strategy"] = "hybrid_max_expected_net"
    for bps in (10.0, 25.0, 50.0):
        costed = reprice(trades, bps)
        hybrid_costed = reprice(hybrid, bps)
        for strategy in strategies:
            group = costed[costed["strategy"].eq(strategy)]
            results.append({"strategy": strategy, "cost_bps": bps, **research.performance(group), **research.cycle_inference(group)})
        results.append({"strategy": "hybrid_max_expected_net", "cost_bps": bps,
                        **research.performance(hybrid_costed), **research.cycle_inference(hybrid_costed)})
        costed["cost_bps"] = bps; hybrid_costed["cost_bps"] = bps
        ledgers.extend([costed, hybrid_costed])
    summary = pd.DataFrame(results)
    summary["holm_p"] = summary.groupby("cost_bps")["one_sided_p"].transform(research.holm_adjust)
    at_50 = summary[summary["cost_bps"].eq(50.0)].set_index("strategy")["total_net"]
    summary["promotion_gate"] = (
        summary["cycles"].ge(30) & summary["ci_low"].gt(0) & summary["holm_p"].lt(0.05)
        & summary["strategy"].map(at_50).gt(0)
    )
    return summary, pd.concat(ledgers, ignore_index=True)


def signed_diagnostics(full: pd.DataFrame, days: pd.DataFrame) -> dict:
    target = full[["datetime", "target_session_return", "spot_price", "session_end_spot"]]
    z = days.merge(target, on="datetime", how="left", suffixes=("", "_target"))
    ok = z[["pred_session_return", "target_session_return"]].notna().all(axis=1)
    y, pred = z.loc[ok, "target_session_return"].to_numpy(), z.loc[ok, "pred_session_return"].to_numpy()
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum(y ** 2) if np.sum(y ** 2) > 0 else np.nan
    hit = float((np.sign(y) == np.sign(pred)).mean()) if len(y) else np.nan
    mae_points = float(np.mean(np.abs(z.loc[ok, "session_end_spot"] - z.loc[ok, "predicted_spot"]))) if len(y) else np.nan
    return {"n": int(len(y)), "r2_vs_zero": float(r2), "direction_hit_rate": hit, "mae_spot_points": mae_points}


def _html_report_legacy(results: pd.DataFrame, diag: dict, audit: dict, iv_audit: dict) -> str:
    payload = results.replace({np.nan: None, np.inf: None}).to_dict("records")
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DTE 4-14 Strategy Backtest</title><style>
:root{{--bg:#08111f;--panel:#111f34;--panel2:#172841;--line:#2b405e;--text:#eef5fc;--muted:#9bb0c8;--good:#4fd1a5;--bad:#ff7480;--accent:#69a8ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:28px}}h1{{font-size:25px;font-weight:500;margin:0}}h2{{font-size:17px;font-weight:500;margin:28px 0 10px}}p{{color:var(--muted);max-width:1000px}}.controls{{display:flex;gap:10px;align-items:center;margin:20px 0;flex-wrap:wrap}}button{{border:1px solid var(--line);background:transparent;color:var(--text);padding:8px 13px;border-radius:9px}}button.active{{background:var(--accent);color:#06101d}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}}.value{{font-size:22px;font-variant-numeric:tabular-nums}}.label{{color:var(--muted)}}.wrap{{overflow:auto;border:1px solid var(--line);border-radius:13px;background:var(--panel)}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--muted);font-weight:500}}.pos{{color:var(--good)}}.neg{{color:var(--bad)}}.bars{{display:grid;gap:9px}}.barrow{{display:grid;grid-template-columns:minmax(190px,280px) 1fr 100px;gap:12px;align-items:center}}.track{{height:12px;background:var(--panel2);border-radius:8px;position:relative;overflow:hidden}}.bar{{height:100%;position:absolute}}.bar.pos{{left:50%;background:var(--good)}}.bar.neg{{right:50%;background:var(--bad)}}.zero{{position:absolute;left:50%;width:1px;height:100%;background:var(--muted)}}.formula{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px;color:var(--muted)}}code{{color:var(--text)}}@media(max-width:650px){{main{{padding:16px}}.barrow{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>DTE 4–14 non-directional strategy backtest</h1><p>Fixed-strike positions entered around 09:45 and exited on the same date after 15:15. All entries require predicted rupee edge to exceed complete leg-level costs. No overnight or overlapping hybrid positions.</p>
<div class="controls"><span>Execution friction:</span><button class="active" data-bps="25">25 bps</button><button data-bps="10">10 bps</button><button data-bps="50">50 bps</button></div>
<div class="stats"><div class="card"><div class="label">Signed spot model R² vs zero</div><div class="value">{diag['r2_vs_zero']:.3f}</div></div><div class="card"><div class="label">Direction hit rate</div><div class="value">{diag['direction_hit_rate']:.1%}</div></div><div class="card"><div class="label">Predicted-spot MAE</div><div class="value">{diag['mae_spot_points']:.0f} pts</div></div><div class="card"><div class="label">Verified close-exit coverage</div><div class="value">{audit['signal_days_with_all_valid_close_exits']}/{audit['signal_days']}</div></div></div>
<h2>Net profit comparison</h2><div id="bars" class="bars"></div><h2>Full results</h2><div class="wrap"><table><thead><tr><th>Strategy</th><th>Trades</th><th>Net ₹</th><th>Hit rate</th><th>Profit factor</th><th>Sharpe</th><th>Max drawdown ₹</th><th>Cycles</th><th>95% CI/cycle</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Mathematical hypotheses</h2><div class="formula"><code>E[P&amp;L] = lot × Σ positionᵢ(½ΓᵢS²·RV̂ + ΘᵢΔt + Vegaᵢ·100·ΔÎV) − full costs</code><br><br><b>Predicted-spot straddle:</b> <code>Ŝ = S·exp(r̂)</code>; select the available strike minimizing <code>|K−Ŝ|</code>, then require positive predicted net edge.<br><b>IV switch:</b> choose short ATM straddle when <code>ÎV_future ≤ IV_current</code>; otherwise choose long ATM straddle. The chosen side must still clear full costs.<br><b>Hybrid:</b> among all cost-qualified alternatives at the timestamp, take the single structure with maximum predicted net rupee edge.</div>
<p>The signed-spot forecast is a separately validated Ridge model because the existing magnitude model cannot supply direction. Negative or weak signed-model R² means the predicted-strike strategy is not supported even if an isolated P&amp;L happens to be positive. Confidence intervals are expiry-cycle bootstraps, not guarantees.</p>
<script>const data={data};const fmt=x=>x==null?'—':Number(x).toLocaleString('en-IN',{{maximumFractionDigits:1}});const names={{short_atm_straddle:'Short ATM straddle',long_atm_straddle:'Long ATM straddle',short_1_strangle:'Short ±1 strangle',long_1_strangle:'Long ±1 strangle',predicted_spot_short_straddle:'Predicted-spot short straddle',iv_regime_switch_atm:'IV regime switch',hybrid_max_expected_net:'Non-overlapping hybrid'}};function draw(bps){{const d=data.filter(x=>x.cost_bps===bps).sort((a,b)=>b.total_net-a.total_net);const mx=Math.max(...d.map(x=>Math.abs(x.total_net||0)),1);document.getElementById('rows').innerHTML=d.map(x=>`<tr><td>${{names[x.strategy]||x.strategy}}</td><td>${{fmt(x.trades)}}</td><td class="${{x.total_net>=0?'pos':'neg'}}">${{fmt(x.total_net)}}</td><td>${{x.hit_rate==null?'—':(100*x.hit_rate).toFixed(1)+'%'}}</td><td>${{fmt(x.profit_factor)}}</td><td>${{fmt(x.daily_sharpe)}}</td><td>${{fmt(x.max_drawdown)}}</td><td>${{fmt(x.cycles)}}</td><td>${{x.ci_low==null?'—':fmt(x.ci_low)+' to '+fmt(x.ci_high)}}</td></tr>`).join('');document.getElementById('bars').innerHTML=d.map(x=>{{const w=50*Math.abs(x.total_net||0)/mx;return `<div class="barrow"><span>${{names[x.strategy]||x.strategy}}</span><div class="track"><i class="zero"></i><i class="bar ${{x.total_net>=0?'pos':'neg'}}" style="width:${{w}}%"></i></div><b class="${{x.total_net>=0?'pos':'neg'}}">₹${{fmt(x.total_net)}}</b></div>`}}).join('')}}document.querySelectorAll('button[data-bps]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');draw(Number(b.dataset.bps))}});draw(25);</script></main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, _ = research.load_daily_panel()
    days, frame, audit = prepare_frame()
    trades = strategy_set(days, frame)
    results, ledgers = summarize(trades)
    diag = signed_diagnostics(full, days)
    iv_audit = {
        "pred_iv_up_days": int(days["pred_d_iv"].gt(0).sum()),
        "pred_iv_nonup_days": int(days["pred_d_iv"].le(0).sum()),
    }
    switch_mix = trades[trades["strategy"].eq("iv_regime_switch_atm")]["source_strategy"].value_counts()
    primary_hybrid = ledgers[
        ledgers["strategy"].eq("hybrid_max_expected_net") & ledgers["cost_bps"].eq(PRIMARY_BPS)
    ]["source_strategy"].value_counts()
    iv_audit["switch_short_trades"] = int(switch_mix.get("short_atm_straddle", 0))
    iv_audit["switch_long_trades"] = int(switch_mix.get("long_atm_straddle", 0))
    iv_audit["hybrid_mix"] = ", ".join(f"{name}: {count}" for name, count in primary_hybrid.items())
    results.to_csv(OUT / "strategy_results.csv", index=False)
    ledgers.to_csv(OUT / "trade_ledger_all_costs.csv", index=False)
    pd.DataFrame([diag]).to_csv(OUT / "signed_spot_model_diagnostics.csv", index=False)
    pd.DataFrame([iv_audit]).to_csv(OUT / "strategy_selection_audit.csv", index=False)
    (OUT / "index.html").write_text(html_report(results, diag, audit, iv_audit), encoding="utf-8")
    print(results[results["cost_bps"].eq(PRIMARY_BPS)].sort_values("total_net", ascending=False).to_string(index=False))
    print("\nSigned spot diagnostics:", diag)


def html_report(results: pd.DataFrame, diag: dict, audit: dict, iv_audit: dict) -> str:
    payload = results.replace({np.nan: None, np.inf: None}).to_dict("records")
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DTE 4-14 Strategy Backtest</title>
<style>
:root{{--bg:#08111f;--panel:#111f34;--panel2:#172841;--line:#2b405e;--text:#eef5fc;--muted:#9bb0c8;--good:#4fd1a5;--bad:#ff7480;--accent:#69a8ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,sans-serif}} main{{max-width:1400px;margin:auto;padding:28px}}
h1{{font-size:26px;font-weight:550;margin:0}} h2{{font-size:18px;font-weight:550;margin:28px 0 10px}} p{{color:var(--muted);max-width:1100px}}
a{{color:#8fc0ff}} .controls{{display:flex;gap:10px;align-items:center;margin:20px 0;flex-wrap:wrap}} button{{border:1px solid var(--line);background:transparent;color:var(--text);padding:8px 13px;border-radius:9px;cursor:pointer}} button.active{{background:var(--accent);color:#06101d}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}} .card,.formula{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}} .value{{font-size:22px;font-variant-numeric:tabular-nums}} .label{{color:var(--muted)}}
.wrap{{overflow:auto;border:1px solid var(--line);border-radius:13px;background:var(--panel)}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-weight:500}} .pos{{color:var(--good)}} .neg{{color:var(--bad)}}
.bars{{display:grid;gap:9px}} .barrow{{display:grid;grid-template-columns:minmax(190px,280px) 1fr 110px;gap:12px;align-items:center}} .track{{height:12px;background:var(--panel2);border-radius:8px;position:relative;overflow:hidden}} .bar{{height:100%;position:absolute}} .bar.pos{{left:50%;background:var(--good)}} .bar.neg{{right:50%;background:var(--bad)}} .zero{{position:absolute;left:50%;width:1px;height:100%;background:var(--muted)}} code{{color:var(--text)}} .tag{{display:inline-block;padding:2px 7px;border-radius:8px;background:var(--panel2)}}
@media(max-width:650px){{main{{padding:16px}}.barrow{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>DTE 4&ndash;14 non-directional strategy backtest</h1>
<p>Fixed-strike positions entered around 09:45 and exited on the same date after 15:15. Entries require forecast rupee edge to exceed complete leg-level costs. There are no overnight positions, and the hybrid holds at most one structure per trading day.</p>
<div class="controls"><span>Execution friction:</span><button class="active" data-bps="25">25 bps</button><button data-bps="10">10 bps</button><button data-bps="50">50 bps</button></div>
<div class="stats">
  <div class="card"><div class="label">Signed spot model R&sup2; vs zero</div><div class="value">{diag['r2_vs_zero']:.3f}</div></div>
  <div class="card"><div class="label">Direction hit rate</div><div class="value">{diag['direction_hit_rate']:.1%}</div></div>
  <div class="card"><div class="label">Predicted-spot MAE</div><div class="value">{diag['mae_spot_points']:.0f} pts</div></div>
  <div class="card"><div class="label">Verified close exits</div><div class="value">{audit['signal_days_with_all_valid_close_exits']}/{audit['signal_days']}</div></div>
</div>
<h2>Net profit comparison</h2><div id="bars" class="bars"></div>
<h2>Full results</h2><div class="wrap"><table><thead><tr><th>Strategy</th><th>Trades</th><th>Net &#8377;</th><th>Hit rate</th><th>Profit factor</th><th>Sharpe</th><th>Max drawdown &#8377;</th><th>Cycles</th><th>95% CI/cycle</th><th>Holm p</th><th>Gate</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Mathematically specified hypotheses</h2>
<div class="formula"><b>Primary test:</b> <code>H0: E[net P&amp;L per expiry cycle] &le; 0</code> against <code>H1: E[net P&amp;L per expiry cycle] &gt; 0</code>.<br><br>
<code>E[P&amp;L] = lot &times; &Sigma; position_i(0.5&Gamma;_i S&sup2; RV_hat + &Theta;_i dt + Vega_i &times; 100 &times; deltaIV_hat) - full costs</code><br><br>
<b>Predicted-spot straddle:</b> <code>S_hat = S &times; exp(r_hat)</code>. The separate Ridge forecast uses only lagged spot returns, absolute returns, realized variance, and the overnight spot return. It selects the available strike minimizing <code>|K-S_hat|</code>, subject to positive forecast net edge.<br>
<b>IV switch:</b> short ATM when <code>IV_hat_future &le; IV_current</code>; otherwise long ATM. Both sides must clear full costs. Here, IV-up forecasts = <b>{iv_audit['pred_iv_up_days']}</b> and IV-non-up forecasts = <b>{iv_audit['pred_iv_nonup_days']}</b>, so the long fallback had no forecast opportunity.<br>
<b>Hybrid:</b> select the one cost-qualified structure with maximum forecast net rupee edge at that timestamp.</div>
<p>At the 25 bps primary setting, the IV switch executed {iv_audit['switch_short_trades']} short and {iv_audit['switch_long_trades']} long straddles. The non-overlapping hybrid selected: {iv_audit['hybrid_mix']}.</p>
<p><b>Promotion gate:</b> at least 30 independent expiry cycles, positive 95% cycle-bootstrap lower bound, Holm-adjusted one-sided p &lt; 0.05, and positive net P&amp;L at 50 bps. A finite backtest can reject a statistical null under stated assumptions; it cannot mathematically prove future profit.</p>
<p>The existing volatility model and artifacts were not changed. The signed-spot forecast is a separate walk-forward layer. Historical bid/ask and volume are unavailable, so 10/25/50 bps are modeled execution stresses. The history has also influenced earlier research, so this is nested retrospective evidence, not an untouched final holdout.</p>
<h2>Research basis</h2>
<p>The hypotheses follow the variance-risk-premium and delta-hedged option-return literature and use a HAR-style realized-volatility forecast: <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1359527">Carr &amp; Wu (2009)</a>, <a href="https://doi.org/10.1093/rfs/hhg002">Bakshi &amp; Kapadia (2003)</a>, <a href="https://doi.org/10.1093/jjfinec/nbp001">Corsi (2009)</a>, and India-specific evidence from <a href="https://doi.org/10.1016/j.iref.2020.06.010">Sankar et al. (2020)</a>.</p>
<script>
const data={data};
const fmt=x=>x==null?'--':Number(x).toLocaleString('en-IN',{{maximumFractionDigits:1}});
const names={{short_atm_straddle:'Short ATM straddle',long_atm_straddle:'Long ATM straddle',short_1_strangle:'Short +/-1 strangle',long_1_strangle:'Long +/-1 strangle',predicted_spot_short_straddle:'Predicted-spot short straddle',iv_regime_switch_atm:'IV regime switch',hybrid_max_expected_net:'Non-overlapping hybrid'}};
function draw(bps){{
 const d=data.filter(x=>x.cost_bps===bps).sort((a,b)=>b.total_net-a.total_net), mx=Math.max(...d.map(x=>Math.abs(x.total_net||0)),1);
 document.getElementById('rows').innerHTML=d.map(x=>`<tr><td>${{names[x.strategy]||x.strategy}}</td><td>${{fmt(x.trades)}}</td><td class="${{x.total_net>=0?'pos':'neg'}}">${{fmt(x.total_net)}}</td><td>${{x.hit_rate==null?'--':(100*x.hit_rate).toFixed(1)+'%'}}</td><td>${{fmt(x.profit_factor)}}</td><td>${{fmt(x.daily_sharpe)}}</td><td>${{fmt(x.max_drawdown)}}</td><td>${{fmt(x.cycles)}}</td><td>${{x.ci_low==null?'--':fmt(x.ci_low)+' to '+fmt(x.ci_high)}}</td><td>${{x.holm_p==null?'--':Number(x.holm_p).toFixed(3)}}</td><td><span class="tag">${{x.promotion_gate?'PASS':'NO'}}</span></td></tr>`).join('');
 document.getElementById('bars').innerHTML=d.map(x=>{{const w=50*Math.abs(x.total_net||0)/mx;return `<div class="barrow"><span>${{names[x.strategy]||x.strategy}}</span><div class="track"><i class="zero"></i><i class="bar ${{x.total_net>=0?'pos':'neg'}}" style="width:${{w}}%"></i></div><b class="${{x.total_net>=0?'pos':'neg'}}">&#8377;${{fmt(x.total_net)}}</b></div>`}}).join('');
}}
document.querySelectorAll('button[data-bps]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');draw(Number(b.dataset.bps))}}); draw(25);
</script></main></body></html>"""


if __name__ == "__main__":
    main()
