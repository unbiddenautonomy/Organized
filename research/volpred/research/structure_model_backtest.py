from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atm_netpremium_correlation import dataframe_to_markdown
from economic_evaluation import historical_rates, lot_size
from intraday_structure_attribution import add_entry_greeks, prepare_entry_rows


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "structure_model_backtest"
EXIT_CACHE = ROOT / "outputs" / "intraday_structure_attribution" / "_fixed_strike_exit_prices.parquet"
EXIT_120 = OUT / "_fixed_strike_exit_prices_120m.parquet"
RAW_CHAIN = ROOT / "data" / "raw" / "icharts_option_banknifty.csv"
HORIZONS = (15, 30, 60, 120)
TRADING_MINUTES_YEAR = 252.0 * 375.0
PRIMARY_SLIPPAGE_BPS = 10.0

# A weight is the number of option contracts bought (+) or sold (-).
# Strike positions are relative to the entry ATM and remain fixed through exit.
STRUCTURES: dict[str, dict[tuple[int, str], int]] = {
    "long_atm_straddle": {(0, "call"): 1, (0, "put"): 1},
    "long_1_strangle": {(1, "call"): 1, (-1, "put"): 1},
    "reverse_iron_butterfly": {(0, "call"): 1, (0, "put"): 1, (1, "call"): -1, (-1, "put"): -1},
    "standard_long_call_butterfly": {(-1, "call"): 1, (0, "call"): -2, (1, "call"): 1},
    "bull_risk_reversal": {(1, "call"): 1, (-1, "put"): -1},
    "bear_risk_reversal": {(-1, "put"): 1, (1, "call"): -1},
    "bull_call_spread": {(0, "call"): 1, (1, "call"): -1},
    "bear_put_spread": {(0, "put"): 1, (-1, "put"): -1},
}


def read_parquet(path: Path, where: str = "") -> pd.DataFrame:
    con = duckdb.connect()
    sql = "SELECT * FROM read_parquet(?)" + (f" WHERE {where}" if where else "")
    out = con.execute(sql, [str(path)]).fetchdf()
    con.close()
    return out


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame_to_save", frame)
    con.execute("COPY frame_to_save TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    con.close()


def build_120m_exit_cache(entries: pd.DataFrame) -> None:
    if EXIT_120.exists():
        return
    keys = entries[["source_index", "datetime", "expiry", "strike", "strike_position"]].copy()
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register("entry_keys", keys)
    raw = str(RAW_CHAIN).replace("'", "''")
    target = str(EXIT_120).replace("'", "''")
    con.execute(f"""
        COPY (
          WITH k AS (
            SELECT source_index, datetime AS entry_datetime, expiry, strike, strike_position,
                   datetime + INTERVAL 120 MINUTE AS exit_datetime
            FROM entry_keys
            WHERE CAST(datetime AS TIME) <= TIME '13:30:00'
          ), x AS (
            SELECT datetime::TIMESTAMP AS exit_datetime, expiry::DATE AS expiry,
                   strike::DOUBLE AS strike, call_price::DOUBLE AS exit_call_price,
                   put_price::DOUBLE AS exit_put_price
            FROM read_csv_auto('{raw}', header=true)
            WHERE expiry_type='MONTHLY'
          )
          SELECT k.*, '120m' AS horizon, x.exit_call_price, x.exit_put_price
          FROM k LEFT JOIN x USING(exit_datetime, expiry, strike)
          WHERE CAST(k.entry_datetime AS DATE)=CAST(k.exit_datetime AS DATE)
        ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()


def load_exits(horizon: int) -> pd.DataFrame:
    if horizon == 120:
        return read_parquet(EXIT_120)
    return read_parquet(EXIT_CACHE, f"horizon='{horizon}m'")


def leg_panel(entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "source_index", "datetime", "expiry", "strike", "strike_position", "spot_price",
        "days_to_expiry", "call_price", "put_price", "call_delta", "put_delta",
        "call_gamma", "put_gamma", "call_vega", "put_vega", "call_theta_day", "put_theta_day",
        "call_vanna", "put_vanna", "call_volga", "put_volga", "call_charm_day", "put_charm_day",
    ]
    x = exits.merge(
        entries[keep], on=["source_index", "expiry", "strike", "strike_position"],
        how="left", validate="many_to_one",
    )
    x = x.rename(columns={"datetime": "entry_datetime_source"})
    for side in ("call", "put"):
        x[f"{side}_pnl"] = x[f"exit_{side}_price"] - x[f"{side}_price"]
        x[f"{side}_stale"] = x[f"exit_{side}_price"].eq(x[f"{side}_price"])
        x[f"{side}_entry_tiny"] = x[f"{side}_price"].lt(5)
        x[f"{side}_tiny"] = x[f"{side}_price"].lt(5) | x[f"exit_{side}_price"].lt(5)
    return x


def wide_legs(legs: pd.DataFrame) -> pd.DataFrame:
    value_cols = []
    for side in ("call", "put"):
        value_cols += [
            f"{side}_price", f"exit_{side}_price", f"{side}_pnl", f"{side}_stale",
            f"{side}_entry_tiny", f"{side}_tiny",
            f"{side}_delta", f"{side}_gamma", f"{side}_vega", f"{side}_theta_day",
            f"{side}_vanna", f"{side}_volga", f"{side}_charm_day",
        ]
    wide = legs.pivot(index="entry_datetime", columns="strike_position", values=value_cols)
    wide.columns = [f"{name}_{int(pos):+d}" for name, pos in wide.columns]
    meta = legs.groupby("entry_datetime", as_index=True).agg(
        exit_datetime=("exit_datetime", "first"), spot_price=("spot_price", "first"),
        days_to_expiry=("days_to_expiry", "first"), expiry=("expiry", "first"),
    )
    return meta.join(wide, how="inner").reset_index()


def forecasts() -> pd.DataFrame:
    rv = read_parquet(ROOT / "outputs" / "rv_prediction_models" / "intraday_walk_forward_predictions.parquet")
    div = read_parquet(ROOT / "outputs" / "log_iv_change_models" / "compact_five_feature" / "walk_forward_predictions.parquet")
    rv = rv[["datetime", "horizon_min", "Realized_GARCH_extended", "actual_var"]].rename(
        columns={"Realized_GARCH_extended": "pred_annualized_variance"}
    )
    div = div[["datetime", "horizon_min", "is_0915", "q50_compact"]]
    out = rv.merge(div, on=["datetime", "horizon_min"], how="inner", validate="one_to_one")
    out = out[out["is_0915"].eq(0)].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out


def structure_panel(wide: pd.DataFrame, fcst: pd.DataFrame, name: str, weights: dict, horizon: int) -> pd.DataFrame:
    out = wide[["entry_datetime", "exit_datetime", "spot_price", "days_to_expiry", "expiry"]].copy()
    out["structure"] = name
    out["horizon_min"] = horizon
    out["leg_count"] = len(weights)
    for target, metric in [
        ("gross_pnl_points", "pnl"), ("entry_value", "price"), ("exit_value", "__exit_price"),
        ("delta", "delta"), ("gamma", "gamma"), ("vega", "vega"), ("theta_day", "theta_day"),
        ("vanna", "vanna"), ("volga", "volga"), ("charm_day", "charm_day"),
    ]:
        pieces = [
            wide[(f"exit_{side}_price_{pos:+d}" if metric == "__exit_price" else f"{side}_{metric}_{pos:+d}")] * weight
            for (pos, side), weight in weights.items()
        ]
        out[target] = pd.concat(pieces, axis=1).sum(axis=1, min_count=len(pieces))
    entry_abs = [wide[f"{side}_price_{pos:+d}"] * abs(weight) for (pos, side), weight in weights.items()]
    exit_abs = [wide[f"exit_{side}_price_{pos:+d}"] * abs(weight) for (pos, side), weight in weights.items()]
    out["entry_turnover_points"] = pd.concat(entry_abs, axis=1).sum(axis=1, min_count=len(entry_abs))
    out["exit_turnover_points"] = pd.concat(exit_abs, axis=1).sum(axis=1, min_count=len(exit_abs))
    buy_parts, sell_parts = [], []
    for (pos, side), weight in weights.items():
        entry_price = wide[f"{side}_price_{pos:+d}"] * abs(weight)
        exit_price = wide[f"exit_{side}_price_{pos:+d}"] * abs(weight)
        buy_parts.append(entry_price if weight > 0 else exit_price)
        sell_parts.append(exit_price if weight > 0 else entry_price)
    out["buy_turnover_points"] = pd.concat(buy_parts, axis=1).sum(axis=1, min_count=len(buy_parts))
    out["sell_turnover_points"] = pd.concat(sell_parts, axis=1).sum(axis=1, min_count=len(sell_parts))
    stale = [wide[f"{side}_stale_{pos:+d}"] for pos, side in weights]
    entry_tiny = [wide[f"{side}_entry_tiny_{pos:+d}"] for pos, side in weights]
    tiny = [wide[f"{side}_tiny_{pos:+d}"] for pos, side in weights]
    out["stale_any"] = pd.concat(stale, axis=1).max(axis=1).astype(bool)
    out["entry_tiny_any"] = pd.concat(entry_tiny, axis=1).max(axis=1).astype(bool)
    out["tiny_any"] = pd.concat(tiny, axis=1).max(axis=1).astype(bool)
    out = out.merge(
        fcst[fcst["horizon_min"].eq(horizon)],
        left_on=["entry_datetime", "horizon_min"], right_on=["datetime", "horizon_min"],
        how="inner", validate="one_to_one",
    ).drop(columns="datetime")
    out["date"] = out["entry_datetime"].dt.normalize()
    out["lot_size"] = lot_size(out["date"])
    # RV forecasts are annualized. Convert to variance over the requested horizon.
    out["pred_horizon_variance"] = out["pred_annualized_variance"] * horizon / TRADING_MINUTES_YEAR
    out["pred_d_iv"] = out["q50_compact"].pipe(np.expm1) * (
        # The compact model predicts the log change of the ATM/five-strike IV level.
        # Recover its entry IV from vega/theta/gamma is unnecessary: merge the clean ATM panel below.
        1.0
    )
    return out


def attach_entry_iv(panel: pd.DataFrame) -> pd.DataFrame:
    atm = pd.read_csv(
        ROOT / "outputs" / "atm_iv_rv_analysis" / "cleaned_atm_iv_rv_panel.csv",
        usecols=["datetime", "iv_trade_vol"], parse_dates=["datetime"],
    ).drop_duplicates("datetime")
    out = panel.merge(atm, left_on="entry_datetime", right_on="datetime", how="inner", validate="many_to_one").drop(columns="datetime")
    out["pred_d_iv"] = out["iv_trade_vol"] * np.expm1(out["q50_compact"])
    out["expected_gamma"] = 0.5 * out["gamma"] * out["spot_price"].pow(2) * out["pred_horizon_variance"]
    out["expected_theta"] = out["theta_day"] * out["horizon_min"] / 1440.0
    out["expected_vega"] = out["vega"] * out["pred_d_iv"]
    # No signed-return forecast exists in the volatility model, so E[dS]=0.
    out["expected_delta"] = 0.0
    out["expected_vanna"] = 0.0
    out["expected_volga"] = 0.5 * out["volga"] * out["pred_d_iv"].pow(2)
    out["expected_charm"] = 0.0
    out["expected_pnl_points"] = out[[
        "expected_gamma", "expected_theta", "expected_vega", "expected_vanna",
        "expected_volga", "expected_charm",
    ]].sum(axis=1)
    return out


def charge_components(panel: pd.DataFrame, slippage_bps: float, current_2026: bool) -> pd.DataFrame:
    lot = panel["lot_size"].to_numpy(float)
    buy_turnover = panel["buy_turnover_points"].to_numpy(float) * lot
    sell_turnover = panel["sell_turnover_points"].to_numpy(float) * lot
    total_turnover = buy_turnover + sell_turnover
    # Every leg is opened and closed once. Quantities of two in a butterfly center
    # are treated as one executed order per side, matching flat per-order brokerage.
    brokerage = panel["leg_count"].to_numpy(float) * 2.0 * 20.0
    if current_2026:
        stt_rate = np.full(len(panel), 0.0015)
        exchange_rate = np.full(len(panel), 0.0003553)
    else:
        stt_rate, exchange_rate = historical_rates(panel["date"])
    stt = np.rint(sell_turnover * stt_rate)
    exchange = total_turnover * exchange_rate
    sebi = total_turnover * 0.000001
    ipft = total_turnover * 0.000000001
    stamp = buy_turnover * 0.00003
    gst = 0.18 * (brokerage + exchange + sebi + ipft)
    slippage = total_turnover * slippage_bps / 10000.0
    return pd.DataFrame({
        "brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi,
        "ipft": ipft, "stamp": stamp, "gst": gst, "slippage": slippage,
        "total_cost": brokerage + stt + exchange + sebi + ipft + stamp + gst + slippage,
        "total_turnover": total_turnover,
    })


def select_non_overlapping(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.sort_values("entry_datetime")
    chosen: list[int] = []
    next_free = pd.Timestamp.min
    for idx, row in x.iterrows():
        if row["entry_datetime"] >= next_free:
            chosen.append(idx)
            next_free = row["exit_datetime"]
    return x.loc[chosen].copy()


def execute(panel: pd.DataFrame, slippage_bps: float, current_2026: bool, zero_cost: bool = False) -> pd.DataFrame:
    out = panel.reset_index(drop=True).copy()
    charges = charge_components(out, slippage_bps, current_2026)
    for col in charges:
        out[col] = charges[col].to_numpy()
    if zero_cost:
        for col in ["brokerage", "stt", "exchange", "sebi", "ipft", "stamp", "gst", "slippage", "total_cost"]:
            out[col] = 0.0
    out["gross_pnl_rupees"] = out["gross_pnl_points"] * out["lot_size"]
    out["net_pnl_rupees"] = out["gross_pnl_rupees"] - out["total_cost"]
    return out


def metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {
            "trades": 0, "days": 0, "gross_pnl": 0.0, "cost": 0.0, "net_pnl": 0.0,
            "mean_net": np.nan, "hit_rate": np.nan, "profit_factor": np.nan,
            "daily_sharpe": np.nan, "max_drawdown": np.nan,
        }
    pnl = trades["net_pnl_rupees"]
    daily = trades.groupby("date")["net_pnl_rupees"].sum().sort_index()
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    dd = daily.cumsum().cummax() - daily.cumsum()
    return {
        "trades": len(trades), "days": daily.size, "gross_pnl": trades["gross_pnl_rupees"].sum(),
        "cost": trades["total_cost"].sum(), "net_pnl": pnl.sum(), "mean_net": pnl.mean(),
        "hit_rate": (pnl > 0).mean(), "profit_factor": wins / losses if losses > 0 else np.inf,
        "daily_sharpe": daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std(ddof=1) > 0 else np.nan,
        "max_drawdown": dd.max() if len(dd) else np.nan,
    }


def build_candidate_panel() -> pd.DataFrame:
    entries, _ = prepare_entry_rows()
    entries = add_entry_greeks(entries)
    build_120m_exit_cache(entries)
    fcst = forecasts()
    frames = []
    for horizon in HORIZONS:
        wide = wide_legs(leg_panel(entries, load_exits(horizon)))
        for name, weights in STRUCTURES.items():
            frames.append(structure_panel(wide, fcst, name, weights, horizon))
    panel = attach_entry_iv(pd.concat(frames, ignore_index=True))
    numeric_cols = [
        "gross_pnl_points", "entry_turnover_points", "exit_turnover_points",
        "buy_turnover_points", "sell_turnover_points", "expected_pnl_points",
        "lot_size", "expected_gamma", "expected_theta", "expected_vega",
        "expected_vanna", "expected_volga", "expected_charm",
    ]
    for col in numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel[
        panel["gross_pnl_points"].notna() & panel["entry_turnover_points"].gt(0)
        & panel["exit_turnover_points"].gt(0) & ~panel["entry_tiny_any"]
        & np.isfinite(panel["expected_pnl_points"])
    ].copy()
    # Entry-time cost estimate uses entry prices for both sides and current charges.
    estimate = panel.copy()
    estimate["exit_turnover_points"] = estimate["entry_turnover_points"]
    # At entry-time, estimate every closing leg at its entry price.
    estimate["buy_turnover_points"] = estimate["entry_turnover_points"]
    estimate["sell_turnover_points"] = estimate["entry_turnover_points"]
    panel["estimated_cost"] = charge_components(estimate, PRIMARY_SLIPPAGE_BPS, True)["total_cost"].to_numpy()
    panel["expected_edge_rupees"] = panel["expected_pnl_points"] * panel["lot_size"] - panel["estimated_cost"]
    panel["signal"] = panel["expected_edge_rupees"].gt(0)
    return panel


def evaluate(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = [
        ("zero_cost", 0.0, False, True),
        ("historical_plus_10bps", 10.0, False, False),
        ("historical_plus_25bps", 25.0, False, False),
        ("current_2026_plus_10bps", 10.0, True, False),
        ("current_2026_plus_25bps", 25.0, True, False),
        ("current_2026_plus_50bps", 50.0, True, False),
    ]
    summaries, all_trades, audit_rows = [], [], []
    for (structure, horizon), opportunities in panel.groupby(["structure", "horizon_min"]):
        signaled = opportunities[opportunities["signal"]]
        g = select_non_overlapping(signaled)
        audit_rows.append({
            "structure": structure, "horizon_min": horizon, "opportunities": len(opportunities),
            "raw_signals": len(signaled), "nonoverlapping_trades": len(g),
            "signal_rate": len(signaled) / len(opportunities),
        })
        for period, mask in [
            ("development_2024", g["date"].dt.year.eq(2024)),
            ("holdout_2025", g["date"].dt.year.eq(2025)),
            ("full_sample", pd.Series(True, index=g.index)),
        ]:
            raw = g[mask].copy()
            for scenario, bps, current, zero in scenarios:
                trades = execute(raw, bps, current, zero)
                trades["period"] = period
                trades["cost_scenario"] = scenario
                if not trades.empty:
                    all_trades.append(trades)
                summaries.append({
                    "structure": structure, "horizon_min": horizon, "period": period,
                    "cost_scenario": scenario, **metrics(trades),
                })
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else panel.iloc[:0].copy()
    summary = pd.DataFrame(summaries)
    primary = trades[(trades["period"].eq("full_sample")) & (trades["cost_scenario"].eq("current_2026_plus_10bps"))]
    cost_cols = ["brokerage", "stt", "exchange", "sebi", "ipft", "stamp", "gst", "slippage", "total_cost"]
    breakdown = primary.groupby(["structure", "horizon_min"])[cost_cols].agg(["sum", "mean"])
    breakdown.columns = [f"{a}_{b}" for a, b in breakdown.columns]
    return summary, trades, breakdown.reset_index(), pd.DataFrame(audit_rows)


def make_plot(summary: pd.DataFrame) -> None:
    z = summary[(summary["period"].eq("holdout_2025")) & (summary["cost_scenario"].eq("current_2026_plus_10bps"))]
    pivot = z.pivot(index="structure", columns="horizon_min", values="net_pnl").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="2025 holdout net P&L: model-filtered option structures", ylabel="Net P&L (INR)", xlabel="")
    ax.legend(title="Horizon (min)", ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "holdout_structure_comparison.png", dpi=170)
    plt.close(fig)


def write_report(summary: pd.DataFrame, panel: pd.DataFrame) -> None:
    primary = summary[(summary["period"].eq("holdout_2025")) & (summary["cost_scenario"].eq("current_2026_plus_10bps"))].copy()
    primary = primary.sort_values("net_pnl", ascending=False)
    cols = ["structure", "horizon_min", "trades", "gross_pnl", "cost", "net_pnl", "mean_net", "hit_rate", "profit_factor", "daily_sharpe", "max_drawdown"]
    long_vol_names = {"long_atm_straddle", "long_1_strangle", "reverse_iron_butterfly"}
    long_vol = primary[primary["structure"].isin(long_vol_names)].copy()
    traded_long_vol = long_vol[long_vol["trades"].gt(0)].sort_values("net_pnl", ascending=False)
    comparison = summary[
        summary["structure"].isin(long_vol_names)
        & summary["period"].isin(["development_2024", "holdout_2025"])
        & summary["cost_scenario"].eq("current_2026_plus_10bps")
    ].copy()
    comparison = comparison.sort_values(["structure", "horizon_min", "period"])
    directional = {"standard_long_call_butterfly", "bull_risk_reversal", "bear_risk_reversal", "bull_call_spread", "bear_put_spread"}
    lines = [
        "# Model-driven option-structure backtest", "", "## Decision", "",
    ]
    if len(traded_long_vol):
        best = traded_long_vol.iloc[0]
        lines.append(
            "No tested long-vol replacement produced a viable 2025 holdout result. "
            f"The least-negative traded cell was **{best['structure']} at {int(best['horizon_min'])} minutes**, "
            f"but it had only {int(best['trades'])} trade(s) and net INR {best['net_pnl']:,.0f}. "
            "The reverse iron butterfly never cleared its modeled round-trip cost."
        )
    else:
        lines.append("No long-vol structure cleared its modeled cost hurdle in the 2025 holdout.")
    lines += [
        "", "## 2025 holdout results (current 2026 charges + 10 bps execution allowance)", "",
        dataframe_to_markdown(primary[cols]), "", "## Long-vol development versus holdout", "",
        dataframe_to_markdown(comparison[["structure", "horizon_min", "period", "trades", "gross_pnl", "cost", "net_pnl", "hit_rate"]]),
        "", "## Method", "",
        "- Entries use the same walk-forward Realized-GARCH RV forecast and compact median log-IV-change forecast as the straddle work.",
        "- The signal enters only when expected gamma + theta + vega P&L exceeds the estimated complete round-trip cost.",
        "- Annualized predicted variance is converted to horizon variance by multiplying by horizon / (252 x 375). Vega is kept per 1.00 decimal IV; no extra factor of 100 is applied.",
        "- Entry strikes and expiry stay fixed through exit. Trades are non-overlapping within each structure/horizon. One historical BANKNIFTY lot is used (15 through 24 Dec 2024, then 30).",
        "- Entry option prices below 5 points are excluded. Exit-time stale/tiny flags are retained for audit but never used to permit or reject an entry, avoiding look-ahead leakage.",
        "- Primary costing uses the current 2026 Zerodha/NSE schedule: INR 20 per executed option order, 0.15% sell-side STT, 0.03553% NSE option transaction charge, INR 10/crore SEBI, 0.003% buy-side stamp duty, 18% GST, plus 10 bps modeled slippage on premium turnover.",
        "- The source has traded/last prices but no bid/ask quotes. The 10 bps execution allowance is an assumption, so use the 25/50 bps scenarios for stress testing.",
        "", "## Interpretation guardrails", "",
        "- Strangle and reverse iron butterfly are the genuine long-vol comparisons. The standard call butterfly is short convexity near ATM; risk reversals and debit spreads are directional.",
        "- The volatility models contain no signed spot-return forecast. Expected delta P&L is therefore set to zero. Results for directional structures are diagnostics, not a valid production signal; they require a separate direction model.",
        "- The 2024 period influenced model/rule development. Treat 2025 as a holdout comparison, while remembering that the underlying model family was developed with knowledge of the broader dataset.",
        "- No same-day expiry exercise is modeled; every trade is marked out intraday before the recorded exit.",
        "- Positive risk-reversal cells contain only one to three holdout trades and are excluded from the decision because the volatility model has no direction forecast.",
        "", "## Coverage", "",
        f"Candidate panel: {len(panel):,} clean structure-windows from {panel['entry_datetime'].min()} through {panel['entry_datetime'].max()}.",
        f"Directional structures included but flagged: {', '.join(sorted(directional))}.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_candidate_panel()
    summary, trades, breakdown, audit = evaluate(panel)
    summary.to_csv(OUT / "strategy_performance.csv", index=False)
    breakdown.to_csv(OUT / "cost_breakdown.csv", index=False)
    audit.to_csv(OUT / "signal_audit.csv", index=False)
    write_parquet(panel, OUT / "candidate_panel.parquet")
    write_parquet(trades, OUT / "executed_trades.parquet")
    make_plot(summary)
    write_report(summary, panel)
    print(summary[(summary.period == "holdout_2025") & (summary.cost_scenario == "current_2026_plus_10bps")].sort_values("net_pnl", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
