from __future__ import annotations

from pathlib import Path
import math

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis, t as student_t

from five_strike_spot_hypothesis import load_nearest_expiry, add_fixed_forward_iv
from intraday_structure_attribution import add_entry_greeks
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "economic_evaluation"
HORIZONS = (15, 30, 60, 120)
PRIMARY_SLIPPAGE_BPS = 25.0
SEED = 7319


def read_parquet(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
    con.close()
    return frame


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame_to_save", frame)
    con.execute("COPY frame_to_save TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    con.close()


def load_atm_greeks() -> pd.DataFrame:
    rows = add_entry_greeks(add_fixed_forward_iv(load_nearest_expiry()))
    atm = rows[rows["strike_position"].eq(0)].copy()
    atm["iv_atm"] = atm[["iv_call", "iv_put"]].mean(axis=1)
    atm["gamma_s"] = atm["call_gamma"] + atm["put_gamma"]
    # Stored vega is per one volatility point. Convert to per 1.00 decimal IV.
    atm["vega_s_decimal"] = 100.0 * (atm["call_vega"] + atm["put_vega"])
    atm["theta_s_day"] = atm["call_theta_day"] + atm["put_theta_day"]
    keep = [
        "datetime", "expiry", "strike", "days_to_expiry", "spot_price", "iv_atm",
        "gamma_s", "vega_s_decimal", "theta_s_day",
    ]
    atm = atm[keep].sort_values(["datetime", "days_to_expiry"]).drop_duplicates("datetime")
    return atm


def lot_size(date: pd.Series) -> np.ndarray:
    # BANKNIFTY old monthly contracts remained at 15; the revised 30 lot became
    # operational after the December 2024 monthly transition.
    return np.where(pd.to_datetime(date) <= pd.Timestamp("2024-12-24"), 15, 30)


def historical_rates(date: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    d = pd.to_datetime(date)
    stt = np.where(d < pd.Timestamp("2024-10-01"), 0.000625, 0.001)
    exchange = np.where(
        d < pd.Timestamp("2024-04-01"), 0.0005,
        np.where(d < pd.Timestamp("2024-10-01"), 0.000495, 0.0003503),
    )
    return stt, exchange


def charge_components(
    entry_premium: np.ndarray,
    exit_premium: np.ndarray,
    action: np.ndarray,
    lots: np.ndarray,
    dates: pd.Series,
    slippage_bps: float = 0.0,
    current_2026: bool = False,
) -> pd.DataFrame:
    entry_premium = np.asarray(entry_premium, float)
    exit_premium = np.asarray(exit_premium, float)
    action = np.asarray(action, int)
    lots = np.asarray(lots, float)
    buy_points = np.where(action > 0, entry_premium, exit_premium)
    sell_points = np.where(action > 0, exit_premium, entry_premium)
    buy_turnover = buy_points * lots
    sell_turnover = sell_points * lots
    total_turnover = buy_turnover + sell_turnover
    if current_2026:
        stt_rate = np.full(len(action), 0.0015)
        exchange_rate = np.full(len(action), 0.0003553)
    else:
        stt_rate, exchange_rate = historical_rates(dates)
    brokerage = np.full(len(action), 80.0)  # four executed orders: CE/PE entry and exit
    stt = np.rint(sell_turnover * stt_rate)
    exchange = total_turnover * exchange_rate
    sebi = total_turnover * 0.000001
    ipft = total_turnover * 0.000000001
    stamp = buy_turnover * 0.00003
    gst = 0.18 * (brokerage + exchange + sebi + ipft)
    slippage = total_turnover * slippage_bps / 10000.0
    total = brokerage + stt + exchange + sebi + ipft + stamp + gst + slippage
    return pd.DataFrame({
        "brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi,
        "ipft": ipft, "stamp": stamp, "gst": gst, "slippage": slippage, "total_cost": total,
        "buy_turnover": buy_turnover, "sell_turnover": sell_turnover,
    })


def build_trade_panel() -> pd.DataFrame:
    rv = read_parquet(OUT.parent / "rv_prediction_models" / "intraday_walk_forward_predictions.parquet")
    rv_open = read_parquet(OUT.parent / "rv_prediction_models" / "opening_0915_predictions.parquet")
    div = read_parquet(OUT.parent / "log_iv_change_models" / "compact_five_feature" / "walk_forward_predictions.parquet")
    pnl = read_parquet(OUT.parent / "short_straddle_extended_analysis" / "short_straddle_windows.parquet")
    atm = load_atm_greeks()

    rv["datetime"] = pd.to_datetime(rv["datetime"])
    rv_open["datetime"] = pd.to_datetime(rv_open["datetime"])
    div["datetime"] = pd.to_datetime(div["datetime"])
    pnl["entry_datetime"] = pd.to_datetime(pnl["entry_datetime"])
    pnl["exit_datetime"] = pd.to_datetime(pnl["exit_datetime"])
    atm["datetime"] = pd.to_datetime(atm["datetime"])
    pnl["horizon_min"] = pnl["horizon"].str.replace("m", "", regex=False)
    pnl["horizon_min"] = pd.to_numeric(pnl["horizon_min"], errors="coerce")
    pnl = pnl[pnl["horizon_min"].isin(HORIZONS)].copy()
    pnl["horizon_min"] = pnl["horizon_min"].astype(int)
    pnl = pnl.rename(columns={"entry_datetime": "datetime", "actual": "short_pnl_points"})
    pnl["entry_premium"] = pnl["entry_value"].abs()
    pnl["long_pnl_points"] = -pnl["short_pnl_points"]
    pnl["exit_premium"] = pnl["entry_premium"] + pnl["long_pnl_points"]

    div = div[["datetime", "horizon_min", "is_0915", "q50_compact"]].copy()
    general = rv[["datetime", "horizon_min", "Realized_GARCH_extended", "actual_var"]].rename(
        columns={"Realized_GARCH_extended": "pred_total_variance"}
    )
    general["scope"] = "intraday"
    opening = rv_open[["datetime", "horizon_min", "opening_extended", "actual_var"]].rename(
        columns={"opening_extended": "pred_total_variance"}
    )
    opening["scope"] = "09:15"
    forecasts = pd.concat([general, opening], ignore_index=True)
    panel = forecasts.merge(div, on=["datetime", "horizon_min"], how="inner", validate="many_to_one")
    panel = panel.merge(
        pnl[[
            "datetime", "exit_datetime", "horizon_min", "entry_premium", "exit_premium",
            "long_pnl_points", "spot_price", "exit_spot", "days_to_expiry", "time_bucket",
            "stale_any", "tiny_any",
        ]],
        on=["datetime", "horizon_min"], how="inner", validate="many_to_one",
    )
    panel = panel.merge(atm, on="datetime", how="inner", suffixes=("", "_greek"), validate="many_to_one")
    panel = panel[
        panel["pred_total_variance"].gt(0)
        & panel["entry_premium"].gt(0)
        & panel["exit_premium"].gt(0)
        & panel["gamma_s"].gt(0)
        & panel["theta_s_day"].lt(0)
        & panel["iv_atm"].gt(0)
        & np.isfinite(panel["iv_atm"])
        & np.isfinite(panel["gamma_s"])
        & np.isfinite(panel["vega_s_decimal"])
        & np.isfinite(panel["theta_s_day"])
    ].copy()
    panel["date"] = panel["datetime"].dt.normalize()
    panel["lot_size"] = lot_size(panel["date"])
    panel["theta_decay_points"] = panel["theta_s_day"].abs() * panel["horizon_min"] / 1440.0
    panel["break_even_move_sq"] = 2.0 * panel["theta_decay_points"] / panel["gamma_s"]
    panel["break_even_total_variance"] = panel["break_even_move_sq"] / panel["spot_price"].pow(2)
    panel["R_RV"] = panel["pred_total_variance"] / panel["break_even_total_variance"]
    panel["pred_d_iv"] = panel["iv_atm"] * np.expm1(panel["q50_compact"])
    panel["gamma_theta_score"] = panel["R_RV"] - 1.0
    panel["vega_score"] = (
        panel["vega_s_decimal"] * panel["pred_d_iv"] / panel["theta_decay_points"]
    )
    panel["combined_score"] = panel["gamma_theta_score"] + panel["vega_score"]
    panel["expected_long_pnl_points"] = panel["theta_decay_points"] * panel["combined_score"]
    panel["iv_implied_total_variance"] = (
        panel["iv_atm"].pow(2) * panel["horizon_min"] / (252.0 * 375.0)
    )
    panel["iv_only_score"] = panel["pred_total_variance"] / panel["iv_implied_total_variance"] - 1.0
    panel["hour"] = panel["datetime"].dt.hour
    panel["minute"] = panel["datetime"].dt.minute
    mod = panel["hour"] * 60 + panel["minute"]
    panel["tod_bucket"] = pd.cut(
        mod, [-np.inf, 585, 690, 810, 900, np.inf],
        labels=["Open 09:15-09:45", "Morning 09:45-11:30", "Midday 11:30-13:30",
                "Afternoon 13:30-15:00", "Close 15:00-15:30"], right=False,
    ).astype(str)
    panel["dte_group"] = pd.cut(
        panel["days_to_expiry"], [-np.inf, 1, 4, 8, np.inf],
        labels=["0DTE", "1-3 DTE", "4-7 DTE", ">7 DTE"], right=False,
    ).astype(str)

    # Causal regimes based only on earlier observations in the joined evaluation stream.
    panel = panel.sort_values(["scope", "horizon_min", "datetime"]).reset_index(drop=True)
    regimes = []
    for (_, _), g in panel.groupby(["scope", "horizon_min"], sort=False):
        iv_q1 = g["iv_atm"].shift().expanding(min_periods=250).quantile(0.25)
        iv_q3 = g["iv_atm"].shift().expanding(min_periods=250).quantile(0.75)
        rv_q1 = g["pred_total_variance"].shift().expanding(min_periods=250).quantile(0.25)
        rv_q3 = g["pred_total_variance"].shift().expanding(min_periods=250).quantile(0.75)
        z = g.copy()
        z["iv_regime"] = np.where(z["iv_atm"] < iv_q1, "Low IV", np.where(z["iv_atm"] > iv_q3, "High IV", "Medium IV"))
        z["rv_regime"] = np.where(z["pred_total_variance"] < rv_q1, "Low predicted RV", np.where(z["pred_total_variance"] > rv_q3, "High predicted RV", "Medium predicted RV"))
        z.loc[iv_q1.isna(), "iv_regime"] = "Insufficient history"
        z.loc[rv_q1.isna(), "rv_regime"] = "Insufficient history"
        regimes.append(z)
    panel = pd.concat(regimes, ignore_index=True)
    return panel


def estimated_roundtrip_cost(panel: pd.DataFrame, action: int, slippage_bps: float, current_2026: bool = False) -> np.ndarray:
    # Entry-time estimate: use entry premium for both legs of the round trip; no future-price leakage.
    components = charge_components(
        panel["entry_premium"].to_numpy(), panel["entry_premium"].to_numpy(),
        np.full(len(panel), action), panel["lot_size"].to_numpy(), panel["date"],
        slippage_bps=slippage_bps, current_2026=current_2026,
    )
    return components["total_cost"].to_numpy()


def strategy_actions(panel: pd.DataFrame, strategy: str, slippage_bps: float, cost_multiplier: float = 1.0) -> np.ndarray:
    lot = panel["lot_size"].to_numpy()
    long_cost = estimated_roundtrip_cost(panel, 1, slippage_bps) * cost_multiplier
    short_cost = estimated_roundtrip_cost(panel, -1, slippage_bps) * cost_multiplier
    if strategy == "combined":
        expected_points = panel["expected_long_pnl_points"].to_numpy()
    elif strategy == "rv_only":
        expected_points = (panel["theta_decay_points"] * panel["gamma_theta_score"]).to_numpy()
    elif strategy == "div_only":
        expected_points = (panel["vega_s_decimal"] * panel["pred_d_iv"]).to_numpy()
    elif strategy == "iv_only":
        expected_points = (panel["theta_decay_points"] * panel["iv_only_score"]).to_numpy()
    elif strategy == "always_long":
        return np.ones(len(panel), dtype=int)
    elif strategy == "always_short":
        return -np.ones(len(panel), dtype=int)
    else:
        raise ValueError(strategy)
    expected_rupees = expected_points * lot
    return np.where(expected_rupees > long_cost, 1, np.where(expected_rupees < -short_cost, -1, 0))


def non_overlapping(panel: pd.DataFrame, action: np.ndarray) -> pd.DataFrame:
    x = panel.copy()
    x["action"] = action
    x = x[x["action"].ne(0)].sort_values("datetime")
    chosen = []
    next_free = pd.Timestamp.min
    for idx, row in x.iterrows():
        if row["datetime"] >= next_free:
            chosen.append(idx)
            next_free = row["exit_datetime"]
    return x.loc[chosen].copy()


def attach_pnl(trades: pd.DataFrame, slippage_bps: float, current_2026: bool = False) -> pd.DataFrame:
    if trades.empty:
        return trades.assign(gross_pnl_rupees=[], net_pnl_rupees=[], total_cost=[])
    c = charge_components(
        trades["entry_premium"].to_numpy(), trades["exit_premium"].to_numpy(),
        trades["action"].to_numpy(), trades["lot_size"].to_numpy(), trades["date"],
        slippage_bps=slippage_bps, current_2026=current_2026,
    )
    out = trades.reset_index(drop=True).copy()
    for col in c.columns:
        out[col] = c[col].to_numpy()
    out["gross_pnl_rupees"] = out["action"] * out["long_pnl_points"] * out["lot_size"]
    out["net_pnl_rupees"] = out["gross_pnl_rupees"] - out["total_cost"]
    out["action_label"] = np.where(out["action"].gt(0), "LONG", "SHORT")
    return out


def performance(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {k: np.nan for k in ["trades", "days", "total_gross", "total_cost", "total_net", "mean_net",
                                             "median_net", "hit_rate", "profit_factor", "daily_sharpe", "max_drawdown"]}
    pnl = trades["net_pnl_rupees"].to_numpy(float)
    daily = trades.groupby("date")["net_pnl_rupees"].sum().sort_index()
    daily_std = daily.std(ddof=1)
    sharpe = daily.mean() / daily_std * np.sqrt(252.0) if daily_std > 0 else np.nan
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    cumulative = daily.cumsum()
    drawdown = cumulative.cummax() - cumulative
    return {
        "trades": len(trades), "days": daily.size,
        "total_gross": trades["gross_pnl_rupees"].sum(), "total_cost": trades["total_cost"].sum(),
        "total_net": pnl.sum(), "mean_net": pnl.mean(), "median_net": np.median(pnl),
        "hit_rate": np.mean(pnl > 0), "profit_factor": wins / losses if losses > 0 else np.inf,
        "daily_sharpe": sharpe, "max_drawdown": drawdown.max() if len(drawdown) else np.nan,
    }


def bootstrap_daily(trades: pd.DataFrame, n_boot: int = 1000) -> dict[str, float]:
    daily = trades.groupby("date")["net_pnl_rupees"].sum().to_numpy(float)
    if len(daily) < 10 or np.std(daily, ddof=1) == 0:
        return {"sharpe_ci_low": np.nan, "sharpe_ci_high": np.nan, "mean_t": np.nan,
                "mean_pvalue": np.nan, "probabilistic_sharpe_gt_zero": np.nan, "deflated_sharpe_probability": np.nan}
    rng = np.random.default_rng(SEED)
    draws = rng.choice(daily, size=(n_boot, len(daily)), replace=True)
    std = draws.std(axis=1, ddof=1)
    sharpes = np.divide(draws.mean(axis=1), std, out=np.full(n_boot, np.nan), where=std > 0) * np.sqrt(252)
    t_stat = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
    pvalue = 2 * student_t.sf(abs(t_stat), df=len(daily) - 1)
    sr_daily = daily.mean() / daily.std(ddof=1)
    sk = skew(daily, bias=False)
    ku = kurtosis(daily, fisher=False, bias=False)
    denom = np.sqrt(max(1e-12, 1 - sk * sr_daily + ((ku - 1) / 4) * sr_daily * sr_daily))
    psr = norm.cdf(sr_daily * np.sqrt(len(daily) - 1) / denom)
    # Conservative deflation for 48 examined scope/horizon/strategy combinations.
    trials = 48
    sr0 = norm.ppf(1 - 1 / trials) / np.sqrt(max(1, len(daily) - 1))
    dsr = norm.cdf((sr_daily - sr0) * np.sqrt(len(daily) - 1) / denom)
    return {
        "sharpe_ci_low": np.nanpercentile(sharpes, 2.5), "sharpe_ci_high": np.nanpercentile(sharpes, 97.5),
        "mean_t": t_stat, "mean_pvalue": pvalue,
        "probabilistic_sharpe_gt_zero": psr, "deflated_sharpe_probability": dsr,
    }


def evaluate(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategies = ["combined", "rv_only", "div_only", "iv_only", "always_long", "always_short"]
    perf_rows, boot_rows, trade_frames = [], [], []
    rng = np.random.default_rng(SEED)
    for scope in ["intraday", "09:15"]:
        for h in HORIZONS:
            part = panel[(panel["scope"].eq(scope)) & panel["horizon_min"].eq(h)].copy()
            if scope == "intraday":
                part = part[part["is_0915"].eq(0)]
            else:
                part = part[part["is_0915"].eq(1)]
            if part.empty:
                continue
            combined_action = strategy_actions(part, "combined", PRIMARY_SLIPPAGE_BPS)
            probs = pd.Series(combined_action).value_counts(normalize=True)
            for strategy in strategies + ["random"]:
                if strategy == "random":
                    action = rng.choice([-1, 0, 1], size=len(part), p=[probs.get(-1, 0), probs.get(0, 0), probs.get(1, 0)])
                else:
                    action = strategy_actions(part, strategy, PRIMARY_SLIPPAGE_BPS)
                selected = non_overlapping(part, action)
                trades = attach_pnl(selected, PRIMARY_SLIPPAGE_BPS)
                trades["strategy"] = strategy
                trades["cost_scenario"] = "historical_zerodha_plus_25bps"
                trade_frames.append(trades)
                perf_rows.append({"scope": scope, "horizon_min": h, "strategy": strategy,
                                  "opportunities": len(part), "trade_rate": len(trades) / len(part), **performance(trades)})
                boot_rows.append({"scope": scope, "horizon_min": h, "strategy": strategy,
                                  **bootstrap_daily(trades)})

    all_trades = pd.concat(trade_frames, ignore_index=True)
    perf = pd.DataFrame(perf_rows)
    boot = pd.DataFrame(boot_rows)

    # Cost sensitivity uses the primary combined trade set and keeps trades fixed.
    cost_rows = []
    scenarios = [
        ("zero_cost", 0.0, False, True),
        ("historical_zerodha_only", 0.0, False, False),
        ("historical_plus_10bps", 10.0, False, False),
        ("historical_plus_25bps", 25.0, False, False),
        ("historical_plus_50bps", 50.0, False, False),
        ("current_2026_zerodha_only", 0.0, True, False),
        ("current_2026_plus_25bps", 25.0, True, False),
    ]
    primary = all_trades[all_trades["strategy"].eq("combined")].copy()
    base_cols = [c for c in panel.columns if c in primary.columns]
    for (scope, h), base in primary.groupby(["scope", "horizon_min"]):
        raw = base[base_cols + ["action"]].copy()
        for name, bps, current, zero in scenarios:
            if zero:
                z = raw.copy()
                z["gross_pnl_rupees"] = z["action"] * z["long_pnl_points"] * z["lot_size"]
                z["total_cost"] = 0.0
                z["net_pnl_rupees"] = z["gross_pnl_rupees"]
            else:
                z = attach_pnl(raw, bps, current_2026=current)
            cost_rows.append({"scope": scope, "horizon_min": h, "cost_scenario": name,
                              "slippage_bps_each_turnover": bps, **performance(z)})
    cost_sens = pd.DataFrame(cost_rows)

    # Cost breakdown for the actual primary trade set.
    cost_cols = ["brokerage", "stt", "exchange", "sebi", "ipft", "stamp", "gst", "slippage", "total_cost"]
    cost_breakdown = primary.groupby(["scope", "horizon_min"])[cost_cols].agg(["sum", "mean"])
    cost_breakdown.columns = [f"{a}_{b}" for a, b in cost_breakdown.columns]
    cost_breakdown = cost_breakdown.reset_index()

    # Subperiod evaluation on primary combined trades.
    subgroup_rows = []
    for (scope, h), g in primary.groupby(["scope", "horizon_min"]):
        for dimension in ["tod_bucket", "iv_regime", "rv_regime", "dte_group", "action_label"]:
            for value, z in g.groupby(dimension):
                subgroup_rows.append({"scope": scope, "horizon_min": h, "dimension": dimension,
                                      "bucket": value, **performance(z)})
    subperiod = pd.DataFrame(subgroup_rows)

    # Threshold/cost-buffer multiplier sensitivity, with new non-overlapping selections.
    threshold_rows = []
    for scope in ["intraday", "09:15"]:
        for h in HORIZONS:
            part = panel[(panel["scope"].eq(scope)) & panel["horizon_min"].eq(h)].copy()
            part = part[part["is_0915"].eq(0 if scope == "intraday" else 1)]
            for multiplier in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
                action = strategy_actions(part, "combined", PRIMARY_SLIPPAGE_BPS, multiplier)
                z = attach_pnl(non_overlapping(part, action), PRIMARY_SLIPPAGE_BPS)
                threshold_rows.append({"scope": scope, "horizon_min": h, "cost_buffer_multiplier": multiplier,
                                       "opportunities": len(part), **performance(z)})
    thresholds = pd.DataFrame(threshold_rows)
    return perf, boot, cost_sens, cost_breakdown, subperiod, thresholds, all_trades


def create_chart(perf: pd.DataFrame, cost: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    intraday = perf[(perf["scope"].eq("intraday")) & perf["strategy"].isin(["combined", "rv_only", "div_only", "always_short"])]
    for name, g in intraday.groupby("strategy"):
        axes[0].plot(g["horizon_min"], g["daily_sharpe"], marker="o", label=name)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Intraday daily Sharpe after Zerodha costs + 25 bps")
    axes[0].set_xlabel("Horizon (minutes)"); axes[0].set_ylabel("Daily P&L Sharpe"); axes[0].legend()
    c = cost[(cost["scope"].eq("intraday")) & cost["cost_scenario"].isin([
        "zero_cost", "historical_zerodha_only", "historical_plus_25bps", "current_2026_plus_25bps"
    ])]
    for name, g in c.groupby("cost_scenario"):
        axes[1].plot(g["horizon_min"], g["total_net"], marker="o", label=name)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Combined signal: cost sensitivity")
    axes[1].set_xlabel("Horizon (minutes)"); axes[1].set_ylabel("Net P&L per lot (₹)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "economic_evaluation.png", dpi=180)
    plt.close(fig)


def write_report(perf: pd.DataFrame, boot: pd.DataFrame, cost: pd.DataFrame, panel: pd.DataFrame) -> None:
    primary = perf[(perf["strategy"].eq("combined"))].copy()
    eligible = perf[perf["trades"].ge(50) & perf["days"].ge(30)].copy()
    baselines = eligible.sort_values(["scope", "horizon_min", "daily_sharpe"], ascending=[True, True, False]).groupby(
        ["scope", "horizon_min"], as_index=False
    ).first()
    lines = [
        "# ATM straddle economic evaluation", "",
        "Primary execution assumption: historical Zerodha/statutory charges plus 25 bps of execution friction on premium turnover. "
        "The 25 bps is a transparent model assumption because the source has no bid/ask quotes.", "",
        "P&L uses observed fixed-strike ATM call-plus-put premium changes. Signals use the Realized-GARCH extended RV forecast, "
        "compact median log-IV forecast, and forward-consistent entry Greeks. Only one position can be open per horizon.", "",
        "## Combined signal", "",
        dataframe_to_markdown(primary[["scope", "horizon_min", "trades", "total_net", "mean_net", "hit_rate", "profit_factor", "daily_sharpe", "max_drawdown"]]),
        "", "## Best strategy with at least 50 trades and 30 trading days", "",
        dataframe_to_markdown(baselines[["scope", "horizon_min", "strategy", "trades", "total_net", "hit_rate", "daily_sharpe"]]),
        "", "## Statistical significance", "", dataframe_to_markdown(boot),
        "", "## Cost sensitivity", "",
        dataframe_to_markdown(cost[["scope", "horizon_min", "cost_scenario", "trades", "total_cost", "total_net", "daily_sharpe"]]),
        "", "## Data and interpretation limits", "",
        f"- Joined clean forecast/trade opportunities: {len(panel):,}.",
        "- Historical bid/ask was unavailable; reported execution friction is sensitivity, not observed spread.",
        "- Short-option margin and return on capital are not reported because historical SPAN/exposure margin snapshots are unavailable.",
        "- Brokerage assumes adequate cash collateral and self-placed orders: ₹20 per executed option order, four orders per round-trip straddle.",
        "- Intraday Sharpe uses daily aggregated P&L and sqrt(252), avoiding the inflated minute-window annualization.",
        "- The 2024-2025 period has already influenced model development; results are design evidence, not an untouched performance claim.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_trade_panel()
    perf, boot, cost, breakdown, subperiod, thresholds, trades = evaluate(panel)
    write_parquet(panel, OUT / "economic_trade_panel.parquet")
    write_parquet(trades, OUT / "executed_trades.parquet")
    perf.to_csv(OUT / "strategy_performance.csv", index=False)
    boot.to_csv(OUT / "bootstrap_significance.csv", index=False)
    cost.to_csv(OUT / "cost_sensitivity.csv", index=False)
    breakdown.to_csv(OUT / "cost_breakdown.csv", index=False)
    subperiod.to_csv(OUT / "subperiod_analysis.csv", index=False)
    thresholds.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    create_chart(perf, cost)
    write_report(perf, boot, cost, panel)
    print("PANEL", len(panel), panel.groupby(["scope", "horizon_min"]).size().to_dict())
    print("\nPRIMARY COMBINED")
    print(perf[perf["strategy"].eq("combined")].to_string(index=False))
    print("\nBEST BY SHARPE")
    print(perf.loc[perf.groupby(["scope", "horizon_min"])["daily_sharpe"].idxmax(),
                   ["scope", "horizon_min", "strategy", "trades", "total_net", "hit_rate", "daily_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
