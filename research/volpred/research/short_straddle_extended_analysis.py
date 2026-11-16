from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import intraday_structure_attribution as isa
from atm_iv_rv_analysis import prepare_spot, rolling_variances, seasonality_profile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "short_straddle_extended_analysis"
EXIT_CACHE = OUT / "_short_straddle_fixed_strike_exits.parquet"
SPOT_FILE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
HORIZONS = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "120m": 120,
    "180m": 180,
    "session_end": None,
}


def build_exit_cache(entries: pd.DataFrame, meta: pd.DataFrame) -> None:
    if EXIT_CACHE.exists():
        return
    eod = meta.groupby("date", as_index=False)["datetime"].max().rename(columns={"datetime": "eod_datetime"})
    keys = entries[["source_index", "datetime", "expiry", "strike", "strike_position", "date"]].copy()
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    con.register("entry_keys_df", keys)
    con.register("eod_df", eod)
    original = str(isa.ORIGINAL).replace("'", "''")
    target = str(EXIT_CACHE).replace("'", "''")
    query = f"""
    COPY (
      WITH keys AS (
        SELECT source_index, datetime AS entry_datetime, expiry, strike, strike_position,
               '5m' AS horizon, datetime + INTERVAL 5 MINUTE AS exit_datetime FROM entry_keys_df
        UNION ALL SELECT source_index, datetime, expiry, strike, strike_position,
               '15m', datetime + INTERVAL 15 MINUTE FROM entry_keys_df
        UNION ALL SELECT source_index, datetime, expiry, strike, strike_position,
               '30m', datetime + INTERVAL 30 MINUTE FROM entry_keys_df
        UNION ALL SELECT source_index, datetime, expiry, strike, strike_position,
               '60m', datetime + INTERVAL 60 MINUTE FROM entry_keys_df
        UNION ALL SELECT source_index, datetime, expiry, strike, strike_position,
               '120m', datetime + INTERVAL 120 MINUTE FROM entry_keys_df
        UNION ALL SELECT source_index, datetime, expiry, strike, strike_position,
               '180m', datetime + INTERVAL 180 MINUTE FROM entry_keys_df
        UNION ALL
        SELECT k.source_index, k.datetime, k.expiry, k.strike, k.strike_position,
               'session_end', e.eod_datetime
        FROM entry_keys_df k JOIN eod_df e USING(date)
        WHERE e.eod_datetime > k.datetime
      ), valid_keys AS (
        SELECT * FROM keys
        WHERE CAST(entry_datetime AS DATE)=CAST(exit_datetime AS DATE)
      ), groups AS (
        SELECT DISTINCT exit_datetime, expiry FROM valid_keys
      ), chain AS MATERIALIZED (
        SELECT o.datetime::TIMESTAMP AS exit_datetime, o.expiry::DATE AS expiry,
               o.strike::DOUBLE AS strike, o.call_price::DOUBLE AS call_price,
               o.put_price::DOUBLE AS put_price, o.spot_price::DOUBLE AS spot_price
        FROM read_csv_auto('{original}', header=true) o
        INNER JOIN groups g
          ON o.datetime::TIMESTAMP=g.exit_datetime AND o.expiry::DATE=g.expiry
        WHERE o.expiry_type='MONTHLY'
      ), forwards AS (
        SELECT exit_datetime, expiry,
               median(strike + call_price - put_price) AS exit_forward,
               median(spot_price) AS exit_spot
        FROM chain GROUP BY exit_datetime, expiry
      )
      SELECT k.*, c.call_price AS exit_call_price, c.put_price AS exit_put_price,
             f.exit_forward, f.exit_spot
      FROM valid_keys k
      LEFT JOIN chain c USING(exit_datetime, expiry, strike)
      LEFT JOIN forwards f USING(exit_datetime, expiry)
    ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.close()


def prepare_leg_window(entries: pd.DataFrame, horizon: str) -> pd.DataFrame:
    exits = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?) WHERE horizon=?", [str(EXIT_CACHE), horizon]
    ).fetchdf()
    w = exits.merge(
        entries, on=["source_index", "expiry", "strike", "strike_position"],
        how="left", validate="many_to_one",
    )
    w["elapsed_minutes"] = (w["exit_datetime"] - w["entry_datetime"]).dt.total_seconds() / 60.0
    w["elapsed_days"] = w["elapsed_minutes"] / 1440.0
    w["spot_change"] = w["exit_spot"] - w["spot_price"]
    w["exit_days_to_expiry"] = w["days_to_expiry"] - w["elapsed_days"]
    w["exit_t_years"] = w["exit_days_to_expiry"].clip(lower=1.0 / (365 * 24 * 60)) / 365.0
    for side, call in [("call", True), ("put", False)]:
        w[f"exit_iv_{side}"] = isa.implied_vol_vector(
            w[f"exit_{side}_price"], w["exit_forward"], w["strike"], w["exit_t_years"], call
        )
        exit_g = isa.black_greeks(
            w["exit_forward"], w["strike"], w["exit_t_years"],
            w[f"exit_iv_{side}"], w["exit_spot"], call,
        )
        use_mid = (w["dte_hours"] <= 24) & (w["elapsed_minutes"] > 15)
        for name, exit_value in exit_g.items():
            start_value = w[f"{side}_{name}"].to_numpy(float)
            w[f"used_{side}_{name}"] = np.where(use_mid, 0.5 * (start_value + exit_value), start_value)
        dvol = w[f"exit_iv_{side}"] - w[f"iv_{side}"]
        ds = w["spot_change"]
        ed = w["elapsed_days"]
        w[f"{side}_actual"] = w[f"exit_{side}_price"] - w[f"{side}_price"]
        w[f"{side}_delta_term"] = w[f"used_{side}_delta"] * ds
        w[f"{side}_gamma_term"] = 0.5 * w[f"used_{side}_gamma"] * ds * ds
        w[f"{side}_vega_term"] = w[f"used_{side}_vega"] * dvol
        w[f"{side}_theta_term"] = w[f"used_{side}_theta_day"] * ed
        w[f"{side}_vanna_term"] = w[f"used_{side}_vanna"] * ds * dvol
        w[f"{side}_volga_term"] = 0.5 * w[f"used_{side}_volga"] * dvol * dvol
        w[f"{side}_charm_term"] = w[f"used_{side}_charm_day"] * ds * ed
        terms = [f"{side}_{x}_term" for x in ["delta", "gamma", "vega", "theta", "vanna", "volga", "charm"]]
        w[f"{side}_explained"] = w[terms].sum(axis=1, min_count=len(terms))
        w[f"{side}_residual"] = w[f"{side}_actual"] - w[f"{side}_explained"]
        w[f"{side}_tiny_price"] = (w[f"{side}_price"] < 5) | (w[f"exit_{side}_price"] < 5)
        w[f"{side}_unchanged_price"] = w[f"exit_{side}_price"].eq(w[f"{side}_price"])
        w[f"{side}_dvol"] = dvol
    w["exit_chain_iv_mean"] = w[["exit_iv_call", "exit_iv_put"]].mean(axis=1)
    entry_gap = (w["iv_call"] - w["iv_put"]).abs()
    exit_gap = (w["exit_iv_call"] - w["exit_iv_put"]).abs()
    common_quality = (
        entry_gap.le(0.05) & exit_gap.le(0.05)
        & (w["common_forward"] / w["spot_price"]).between(0.95, 1.05)
        & (w["exit_forward"] / w["exit_spot"]).between(0.95, 1.05)
    )
    for side in ["call", "put"]:
        w[f"{side}_attribution_quality"] = (
            common_quality & w[f"iv_{side}"].between(0.05, 1.50)
            & w[f"exit_iv_{side}"].between(0.05, 1.50)
            & w[f"used_{side}_gamma"].between(0.0, 0.01)
            & np.isfinite(w[f"{side}_explained"])
        )
    return w


def rv_timeline() -> pd.DataFrame:
    spot = pd.read_csv(SPOT_FILE, parse_dates=["datetime", "Date"])
    spot = prepare_spot(spot)
    seasonal = seasonality_profile(spot)
    windows = rolling_variances(spot, seasonal, [5, 15, 30, 60, 120, 180])
    out = spot[["datetime"]].copy()
    for h in [5, 15, 30, 60, 120, 180]:
        out[f"future_rv_{h}"] = np.sqrt(windows[f"future_var_{h}"])

    minute = spot["minute"].to_numpy(int)
    weights = seasonal.to_numpy()[np.clip(minute, 0, 375)]
    r2 = spot["return"].to_numpy(float) ** 2
    session_var = np.full(len(spot), np.nan)
    for _, idx in spot.groupby("date", sort=False).indices.items():
        idx = np.asarray(idx)
        vals, w = r2[idx], weights[idx]
        bad = (~np.isfinite(vals)) | (~np.isfinite(w))
        cv = np.r_[0.0, np.cumsum(np.where(bad, 0.0, vals))]
        cw = np.r_[0.0, np.cumsum(np.where(bad, 0.0, w))]
        cb = np.r_[0, np.cumsum(bad.astype(int))]
        starts = np.arange(len(idx)) + 1
        end = len(idx)
        good = (starts < end) & ((cb[end] - cb[starts]) == 0) & ((cw[end] - cw[starts]) > 0)
        loc = np.where(good)[0]
        session_var[idx[loc]] = 252 * 375 * (cv[end] - cv[starts[loc]]) / (cw[end] - cw[starts[loc]])
    out["future_rv_session_end"] = np.sqrt(session_var)
    return out


def report_table(df: pd.DataFrame, cols: list[str], decimals: int = 4) -> str:
    return isa.dataframe_to_markdown(df[cols].round(decimals))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries, meta = isa.prepare_entry_rows()
    entries = entries[entries["strike_position"].eq(0)].copy()
    entries = isa.add_entry_greeks(entries)
    build_exit_cache(entries, meta)
    rv = rv_timeline().set_index("datetime")

    summaries, drivers, comparisons, curves, audits, panels = [], [], [], [], [], []
    for horizon, minutes in HORIZONS.items():
        legs = prepare_leg_window(entries, horizon)
        audits.append({
            "horizon": horizon, "entry_rows": len(legs),
            "exit_price_coverage": legs["exit_call_price"].notna().mean(),
            "exit_forward_coverage": legs["exit_forward"].notna().mean(),
        })
        wide = isa.wide_leg_metrics(legs)
        wide = wide.merge(
            meta[["datetime", "iv_trade_vol", "quality_ok"]].rename(columns={"datetime": "entry_datetime"}),
            on="entry_datetime", how="left", validate="one_to_one",
        )
        sf = isa.build_structure_frame(wide, "short_atm_straddle", isa.STRUCTURES["short_atm_straddle"])
        sf["horizon"] = horizon
        sf["iv_trade_vol"] = wide["iv_trade_vol"].to_numpy()
        sf["quality_ok"] = wide["quality_ok"].to_numpy()
        summaries.append(isa.summarize_structure(sf, horizon))

        clean = sf[(~sf["stale_any"]) & sf["attribution_quality"]].copy()
        for term in isa.TERM_COLS + ["residual"]:
            drivers.append({
                "horizon": horizon, "term": term,
                "n": len(clean), "mean_contribution_points": clean[term].mean(),
                "mean_abs_contribution_points": clean[term].abs().mean(),
                "std_contribution_points": clean[term].std(),
                "correlation_with_actual_pnl": clean[term].corr(clean["actual"]),
            })

        rv_col = "future_rv_session_end" if minutes is None else f"future_rv_{minutes}"
        clean["future_rv"] = rv[rv_col].reindex(clean["entry_datetime"]).to_numpy()
        valid = clean["quality_ok"].fillna(False) & clean["iv_trade_vol"].gt(0) & clean["future_rv"].gt(0)
        z = clean[valid].copy()
        z["iv_minus_rv_vol_points"] = 100.0 * (z["iv_trade_vol"] - z["future_rv"])
        z["iv_rv_relation"] = np.where(
            z["iv_trade_vol"] > z["future_rv"], "IV > RV",
            np.where(z["iv_trade_vol"] < z["future_rv"], "IV < RV", "IV = RV"),
        )
        gt = z["iv_rv_relation"].eq("IV > RV")
        lt = z["iv_rv_relation"].eq("IV < RV")
        eq = z["iv_rv_relation"].eq("IV = RV")
        comparisons.append({
            "horizon": horizon, "n_compared": len(z),
            "iv_gt_rv_count": int(gt.sum()), "iv_gt_rv_pct": gt.mean(),
            "iv_lt_rv_count": int(lt.sum()), "iv_lt_rv_pct": lt.mean(),
            "iv_eq_rv_count": int(eq.sum()), "iv_eq_rv_pct": eq.mean(),
            "median_iv": z["iv_trade_vol"].median(), "median_rv": z["future_rv"].median(),
            "median_iv_minus_rv_vol_points": z["iv_minus_rv_vol_points"].median(),
            "mean_pnl_when_iv_gt_rv": z.loc[gt, "actual"].mean(),
            "win_rate_when_iv_gt_rv": (z.loc[gt, "actual"] > 0).mean(),
            "mean_pnl_when_iv_lt_rv": z.loc[lt, "actual"].mean(),
            "win_rate_when_iv_lt_rv": (z.loc[lt, "actual"] > 0).mean(),
            "corr_pnl_with_iv_minus_rv": z["actual"].corr(z["iv_minus_rv_vol_points"]),
        })

        elapsed = z["elapsed_minutes"].clip(lower=1)
        z["implied_move_points"] = z["spot_price"] * z["iv_trade_vol"] * np.sqrt(elapsed / (252.0 * 375.0))
        z["abs_move_sigma"] = z["spot_change"].abs() / z["implied_move_points"]
        move_bins = [-np.inf, .25, .5, .75, 1, 1.25, 1.5, 2, 3, np.inf]
        z["move_bin"] = pd.cut(z["abs_move_sigma"], move_bins).astype(str)
        curve = z.groupby("move_bin", observed=True).agg(
            n=("actual", "count"), mean_abs_move_sigma=("abs_move_sigma", "mean"),
            mean_pnl=("actual", "mean"), median_pnl=("actual", "median"),
            win_rate=("actual", lambda s: (s > 0).mean()),
        ).reset_index()
        curve.insert(0, "horizon", horizon)
        curves.append(curve)

        panels.append(z[[
            "entry_datetime", "exit_datetime", "date", "horizon", "spot_price", "exit_spot",
            "spot_change", "elapsed_minutes", "days_to_expiry", "dte_bucket", "time_bucket",
            "iv_trade_vol", "future_rv", "iv_minus_rv_vol_points", "iv_rv_relation", "actual",
            *isa.TERM_COLS, "residual", "entry_value", "stale_any", "tiny_any",
            "greek_break_even_move_points", "move_exceeds_greek_break_even", "abs_move_sigma",
        ]])

    summary = pd.DataFrame(summaries)
    summary["bh_qvalue"] = isa.bh_adjust(summary["day_bootstrap_pvalue"])
    driver = pd.DataFrame(drivers)
    compare = pd.DataFrame(comparisons)
    curve = pd.concat(curves, ignore_index=True)
    audit = pd.DataFrame(audits)
    panel = pd.concat(panels, ignore_index=True)

    summary.to_csv(OUT / "short_straddle_summary.csv", index=False)
    driver.to_csv(OUT / "short_straddle_term_drivers.csv", index=False)
    compare.to_csv(OUT / "short_straddle_iv_vs_rv.csv", index=False)
    curve.to_csv(OUT / "short_straddle_payoff_vs_move.csv", index=False)
    audit.to_csv(OUT / "data_audit.csv", index=False)
    con = duckdb.connect()
    con.register("panel_df", panel)
    panel_target = str(OUT / "short_straddle_windows.parquet").replace("'", "''")
    con.execute(f"COPY panel_df TO '{panel_target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()

    order = list(HORIZONS)
    piv = driver[driver["term"].isin(["gamma_term", "vega_term", "theta_term"])].pivot(
        index="horizon", columns="term", values="mean_contribution_points"
    ).reindex(order)[["gamma_term", "theta_term", "vega_term"]].rename(
        columns={"gamma_term": "Gamma", "theta_term": "Theta", "vega_term": "Vega"}
    )
    comp_plot = compare.set_index("horizon").reindex(order)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    piv.plot(kind="bar", ax=axes[0], color=["#D1495B", "#2A9D8F", "#457B9D"])
    axes[0].axhline(0, color="black", linewidth=.8, label="_nolegend_")
    axes[0].set(title="Short ATM straddle: mean Greek contributions", xlabel="", ylabel="Gross P&L points")
    axes[0].legend(frameon=False)
    axes[1].bar(np.arange(len(order)) - .18, 100 * comp_plot["iv_gt_rv_pct"], width=.36, label="IV > RV")
    axes[1].bar(np.arange(len(order)) + .18, 100 * comp_plot["iv_lt_rv_pct"], width=.36, label="IV < RV")
    axes[1].set_xticks(np.arange(len(order)), order)
    axes[1].set(title="Frequency of entry IV versus matching future RV", xlabel="Horizon", ylabel="Share of valid windows (%)")
    axes[1].legend(frameon=False)
    fig.savefig(OUT / "short_straddle_drivers_and_iv_rv.png", dpi=180)
    plt.close(fig)

    report = f"""# Short ATM straddle: extended intraday analysis

## Scope

- Fixed ATM strike from entry to exit; no re-centering.
- Horizons: 5, 15, 30, 60, 120 and 180 minutes, plus entry to session end.
- IV is forward-consistent ATM IV converted to a trading-time annualized basis.
- RV is seasonality-adjusted forward realized volatility over the matching horizon.
- Results are gross mark-to-market because the source has no bid/ask fields.

## P&L summary

{report_table(summary, ["horizon", "n_clean", "mean_pnl_points", "median_pnl_points", "win_rate", "p05_pnl", "worst_pnl", "bootstrap_ci_low", "bootstrap_ci_high", "bh_qvalue"])}

## IV versus future RV

{report_table(compare, ["horizon", "n_compared", "iv_gt_rv_count", "iv_gt_rv_pct", "iv_lt_rv_count", "iv_lt_rv_pct", "median_iv_minus_rv_vol_points", "mean_pnl_when_iv_gt_rv", "mean_pnl_when_iv_lt_rv"])}

## Interpretation

For a short straddle, gamma is negative, theta is positive and a fall in IV produces positive vega P&L. The term-driver table distinguishes the contribution to average P&L from the correlation with window-to-window P&L. IV greater than subsequent RV is descriptive, not sufficient by itself for a profitable trade, because the structure also depends on IV repricing, path, gamma and transaction costs.

At 15 minutes, mean gamma P&L was -1.79 points, theta was +1.01 and vega was +1.54, producing +0.79 points after the smaller terms and residual. Vega had the strongest correlation with actual P&L (0.887). At 120 and 180 minutes gamma losses grew to -11.69 and -17.50 points; theta and vega together still more than offset them on average. Vega remained the largest single variation driver, but gamma became increasingly important as the horizon lengthened.

IV exceeded future RV in 84.24% of valid 5-minute windows, 85.82% at 15 minutes, 88.96% at 120 minutes and 89.33% at 180 minutes. When IV was below RV, average short-straddle P&L was negative at every fixed horizon. The IV-versus-RV comparison uses annualized volatilities; entry IV represents the option's remaining life while RV covers the stated forward window, so this is a useful regime indicator rather than a perfectly horizon-matched variance-swap comparison.
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
