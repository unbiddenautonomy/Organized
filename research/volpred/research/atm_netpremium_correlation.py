from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "atm_iv_rv_analysis"
ATM_FILE = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm.csv"
PANEL_FILE = OUT / "cleaned_atm_iv_rv_panel.csv"


def corr_pair(x: pd.Series, y: pd.Series, method: str = "pearson") -> tuple[float, int]:
    z = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan, len(z)
    return float(z.iloc[:, 0].corr(z.iloc[:, 1], method=method)), len(z)


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    good = np.isfinite(y) & np.isfinite(x).all(axis=1)
    out = np.full(len(y), np.nan)
    beta, *_ = np.linalg.lstsq(x[good], y[good], rcond=None)
    out[good] = y[good] - x[good] @ beta
    return out


def time_bucket_for_end(ts: pd.Series, horizon_min: int = 15) -> pd.Series:
    end = ts + pd.to_timedelta(horizon_min, unit="m")
    mins = end.dt.hour * 60 + end.dt.minute
    return pd.Series(
        np.select(
            [mins <= 600, mins <= 720, mins <= 870],
            ["open_to_1000", "1000_to_1200", "1200_to_1430"],
            default="after_1430",
        ),
        index=ts.index,
    )


def dataframe_to_markdown(df: pd.DataFrame, decimals: int = 4, include_index: bool = False) -> str:
    shown = df.reset_index() if include_index else df.copy()
    cells = []
    for row in shown.itertuples(index=False, name=None):
        cells.append([
            (f"{value:.{decimals}f}" if isinstance(value, (float, np.floating)) and np.isfinite(value)
             else ("" if pd.isna(value) else str(value)))
            for value in row
        ])
    headers = [str(c) for c in shown.columns]
    widths = [max(len(headers[j]), *(len(r[j]) for r in cells)) for j in range(len(headers))]
    line1 = "| " + " | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))) + " |"
    line2 = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(r[j].ljust(widths[j]) for j in range(len(headers))) + " |" for r in cells]
    return "\n".join([line1, line2, *body])


def daily_bootstrap_corr(df: pd.DataFrame, xcol: str, ycol: str, reps: int = 500) -> tuple[float, float]:
    z = df[["date", xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = [g[[xcol, ycol]].to_numpy() for _, g in z.groupby("date")]
    rng = np.random.default_rng(20261007)
    vals = []
    for _ in range(reps):
        chosen = rng.integers(0, len(groups), len(groups))
        a = np.concatenate([groups[i] for i in chosen], axis=0)
        if np.std(a[:, 0]) > 0 and np.std(a[:, 1]) > 0:
            vals.append(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    return tuple(np.quantile(vals, [0.025, 0.975]))


def build_panel() -> pd.DataFrame:
    source_cols = [
        "datetime", "expiry", "strike", "call_price", "put_price",
        "call_oi", "put_oi", "spot_price", "days_to_expiry",
    ]
    src = pd.read_csv(ATM_FILE, usecols=source_cols, parse_dates=["datetime", "expiry"])
    panel = pd.read_csv(PANEL_FILE, parse_dates=["datetime", "expiry"])

    # Join on the exact expiry-specific ATM contract used by the IV/RV panel.
    src["strike"] = src["strike"].astype(float)
    panel["strike"] = panel["strike"].astype(float)
    if src.duplicated(["datetime", "expiry", "strike"]).any():
        raise ValueError("ATM source has duplicate datetime-expiry-strike keys")
    merged = panel.merge(
        src.drop(columns=["spot_price", "days_to_expiry"]),
        on=["datetime", "expiry", "strike"],
        how="left",
        validate="one_to_one",
    )
    if merged[["call_price", "put_price"]].isna().any().any():
        raise ValueError("Some IV/RV panel rows did not match ATM option prices")

    merged["net_premium"] = merged["call_price"] + merged["put_price"]
    merged["net_premium_spot_pct"] = 100.0 * merged["net_premium"] / merged["spot_price"]
    t_years = merged["days_to_expiry"].clip(lower=1.0 / (365.0 * 24.0 * 60.0)) / 365.0
    merged["net_premium_time_normalized"] = (merged["net_premium"] / merged["spot_price"]) / np.sqrt(t_years)
    merged["date"] = merged["datetime"].dt.date.astype(str)
    merged["year"] = merged["datetime"].dt.year
    merged["minute_of_day"] = merged["datetime"].dt.hour * 60 + merged["datetime"].dt.minute
    merged["time_bucket_15m_end"] = time_bucket_for_end(merged["datetime"], 15)
    return merged


def save_premium_time_result(panel: pd.DataFrame) -> None:
    z = panel.loc[
        panel["quality_ok"].fillna(False)
        & panel["split"].eq("evaluation")
        & panel["future_rv_15"].gt(0)
        & panel["iv_trade_vol"].gt(0)
    ].copy()
    z["rv_iv_ratio"] = z["future_rv_15"] / z["iv_trade_vol"]

    rows = []
    for dimension, col in [("DTE", "dte_bucket"), ("time_of_day", "time_bucket_15m_end")]:
        for key, g in z.groupby(col, observed=True):
            ratio = float(g["rv_iv_ratio"].median())
            rows.append({
                "dimension": dimension,
                "bucket": key,
                "n": len(g),
                "median_rv_iv_ratio": ratio,
                "median_iv_premium_pct": 100.0 * (1.0 - ratio),
                "median_iv_minus_rv_vol_points": 100.0 * (g["iv_trade_vol"] - g["future_rv_15"]).median(),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "iv_premium_dte_time_summary.csv", index=False)

    dte_order = ["over_336h", "168_336h", "72_168h", "24_72h", "0_24h"]
    time_order = ["open_to_1000", "1000_to_1200", "1200_to_1430", "after_1430"]
    cell = z.pivot_table(
        index="time_bucket_15m_end", columns="dte_bucket", values="rv_iv_ratio", aggfunc="median"
    ).reindex(index=time_order, columns=dte_order)

    def md_table(df: pd.DataFrame) -> str:
        return dataframe_to_markdown(df, decimals=4)

    dte = summary[summary.dimension.eq("DTE")].set_index("bucket").reindex(dte_order).reset_index()
    tod = summary[summary.dimension.eq("time_of_day")].set_index("bucket").reindex(time_order).reset_index()
    report = f"""# How the 15-minute ATM IV premium changes

## Scope

This is the untouched 2023 evaluation sample only. Premium is defined as `1 - median(future 15-minute RV / trading-time ATM IV)`. Positive values mean IV exceeds subsequent RV. Time buckets use the end of the 15-minute target window, preventing windows from crossing bucket boundaries.

## As monthly expiry approaches

{md_table(dte[["bucket", "n", "median_rv_iv_ratio", "median_iv_premium_pct", "median_iv_minus_rv_vol_points"]])}

The premium is broadly 19%-28% until the last day, then falls to about 3%. The final 24 hours are therefore structurally different; a single unconditional IV-to-RV multiplier is inappropriate there.

## As the trading day progresses

{md_table(tod[["bucket", "n", "median_rv_iv_ratio", "median_iv_premium_pct", "median_iv_minus_rv_vol_points"]])}

The premium is U-shaped: it widens from the open into midday and narrows after 14:30. This is mainly the familiar intraday RV seasonality. In the last 24 hours before expiry, some late-day cells reverse sign, meaning realized volatility can exceed IV.

## Median RV/IV by time-of-day and DTE

{dataframe_to_markdown(cell, decimals=4, include_index=True)}

## Trading interpretation

Do not treat the unconditional premium as a standalone sell signal. Use DTE- and time-specific calibration, and require execution-cost data before concluding that the statistical premium is tradable.
"""
    (OUT / "iv_premium_dte_time_report.md").write_text(report, encoding="utf-8")


def correlation_analysis(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = panel.loc[
        panel["quality_ok"].fillna(False)
        & panel["split"].isin(["train", "evaluation"])
        & panel["net_premium"].gt(0)
        & panel["spot_price"].gt(0)
        & panel["days_to_expiry"].gt(0)
    ].copy()

    premium_cols = ["net_premium", "net_premium_spot_pct", "net_premium_time_normalized"]
    vol_cols = [
        "iv_trade_vol", "atm_iv_calendar", "past_rv_5", "past_rv_15", "past_rv_60",
        "future_rv_5", "future_rv_15", "future_rv_30", "future_rv_60",
    ]
    label = {
        "net_premium": "raw_rupees",
        "net_premium_spot_pct": "percent_of_spot",
        "net_premium_time_normalized": "spot_and_sqrt_time_normalized",
    }
    split_label = {"evaluation": "2023_evaluation", "development": "2019_2023_development"}
    rows = []
    for sample_name, sample in [
        ("evaluation", valid[valid["split"].eq("evaluation")]),
        ("development", valid),
    ]:
        for p in premium_cols:
            lp = np.log(sample[p])
            for v in vol_cols:
                lv = np.log(sample[v].where(sample[v] > 0))
                pearson, n = corr_pair(lp, lv, "pearson")
                spearman, _ = corr_pair(sample[p], sample[v], "spearman")
                day_lp = lp - lp.groupby(sample["date"]).transform("mean")
                day_lv = lv - lv.groupby(sample["date"]).transform("mean")
                within, _ = corr_pair(day_lp, day_lv, "pearson")
                rows.append({
                    "sample": split_label[sample_name], "premium_measure": label[p],
                    "volatility_measure": v, "n": n, "pearson_log_levels": pearson,
                    "spearman_levels": spearman, "within_day_pearson_log": within,
                })
    corr = pd.DataFrame(rows)

    # Partial log correlation after removing spot, time-to-expiry, and clock-time effects.
    partial_rows = []
    for sample_name, sample in [
        ("evaluation", valid[valid["split"].eq("evaluation")].copy()),
        ("development", valid.copy()),
    ]:
        angle = 2.0 * np.pi * (sample["minute_of_day"] - 555.0) / 375.0
        controls = np.column_stack([
            np.ones(len(sample)), np.log(sample["spot_price"]), np.log(sample["days_to_expiry"]),
            np.sin(angle), np.cos(angle),
        ])
        for p in premium_cols:
            rp = residualize(np.log(sample[p]).to_numpy(), controls)
            for v in vol_cols:
                rv = residualize(np.log(sample[v].where(sample[v] > 0)).to_numpy(), controls)
                val, n = corr_pair(pd.Series(rp), pd.Series(rv), "pearson")
                partial_rows.append({
                    "sample": split_label[sample_name], "premium_measure": label[p],
                    "volatility_measure": v, "n": n,
                    "partial_pearson_log": val,
                    "controls": "log_spot + log_DTE + intraday_sine_cosine",
                })
    partial = pd.DataFrame(partial_rows)

    yearly_rows = []
    for year, g in valid.groupby("year"):
        for p in premium_cols:
            for v in ["iv_trade_vol", "past_rv_15", "future_rv_15"]:
                value, n = corr_pair(np.log(g[p]), np.log(g[v].where(g[v] > 0)), "pearson")
                yearly_rows.append({
                    "year": year, "premium_measure": label[p], "volatility_measure": v,
                    "n": n, "pearson_log_levels": value,
                })
    yearly = pd.DataFrame(yearly_rows)
    return corr, partial, yearly


def save_correlation_outputs(panel: pd.DataFrame, corr: pd.DataFrame, partial: pd.DataFrame, yearly: pd.DataFrame) -> None:
    panel_cols = [
        "datetime", "expiry", "strike", "spot_price", "days_to_expiry", "dte_bucket",
        "call_price", "put_price", "call_oi", "put_oi", "net_premium",
        "net_premium_spot_pct", "net_premium_time_normalized", "iv_trade_vol",
        "atm_iv_calendar", "past_rv_5", "past_rv_15", "past_rv_60", "future_rv_5",
        "future_rv_15", "future_rv_30", "future_rv_60", "quality_ok", "regime", "split",
    ]
    panel[panel_cols].to_csv(OUT / "atm_netpremium_iv_rv_panel.csv", index=False)
    corr.to_csv(OUT / "netpremium_correlations.csv", index=False)
    partial.to_csv(OUT / "netpremium_partial_correlations.csv", index=False)
    yearly.to_csv(OUT / "netpremium_correlation_by_year.csv", index=False)

    ev = corr[corr["sample"].eq("2023_evaluation")]
    evp = partial[partial["sample"].eq("2023_evaluation")]
    focus_vars = ["iv_trade_vol", "past_rv_15", "future_rv_15"]
    focus = ev[ev["volatility_measure"].isin(focus_vars)].merge(
        evp[["premium_measure", "volatility_measure", "partial_pearson_log"]],
        on=["premium_measure", "volatility_measure"], how="left"
    )

    # Cluster-by-day bootstrap intervals for the primary, economically comparable premium measure.
    boot_base = panel.loc[
        panel["quality_ok"].fillna(False) & panel["split"].eq("evaluation")
        & panel["net_premium_time_normalized"].gt(0)
    ].copy()
    boot_base["log_premium_tn"] = np.log(boot_base["net_premium_time_normalized"])
    boot_rows = []
    for v in focus_vars:
        boot_base[f"log_{v}"] = np.log(boot_base[v].where(boot_base[v] > 0))
        point, n = corr_pair(boot_base["log_premium_tn"], boot_base[f"log_{v}"])
        lo, hi = daily_bootstrap_corr(boot_base, "log_premium_tn", f"log_{v}")
        boot_rows.append({"volatility_measure": v, "n": n, "correlation": point, "ci_low": lo, "ci_high": hi})
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "netpremium_primary_bootstrap_ci.csv", index=False)

    heat = focus[focus["premium_measure"].eq("spot_and_sqrt_time_normalized")].set_index("volatility_measure")
    heat = heat.reindex(index=focus_vars)[
        ["pearson_log_levels", "spearman_levels", "within_day_pearson_log", "partial_pearson_log"]
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(heat.columns)), ["Pearson\n(log levels)", "Spearman", "Within-day\nPearson", "Partial\nPearson"])
    ax.set_yticks(range(len(heat.index)), ["ATM IV", "Trailing RV (15m)", "Future RV (15m)"])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", color="black")
    ax.set_title("2023 correlation with spot- and sqrt(time)-normalized ATM net premium")
    ax.set_xlabel("Partial correlation controls for spot, DTE, and time of day")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Correlation")
    fig.tight_layout()
    fig.savefig(OUT / "netpremium_partial_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    fmt_focus = focus.copy()
    numeric = ["pearson_log_levels", "spearman_levels", "within_day_pearson_log", "partial_pearson_log"]
    fmt_focus[numeric] = fmt_focus[numeric].round(4)
    report = f"""# ATM net premium correlation with IV and RV

## Decision

`netPremium = call_price + put_price` is strongly linked to ATM IV after normalizing for spot and remaining time, as option-pricing mechanics imply. Its link to trailing and future realized volatility is weaker. Therefore raw net premium should not be treated as an independent volatility signal: use the spot- and sqrt(time)-normalized version, and test incremental predictive value against IV and trailing RV before adding it to a trading model.

## Scope and hygiene

- Contract: exact ATM strike, nearest monthly expiry at each timestamp.
- Main test: untouched 2023 evaluation sample. 2019-2022 is training; 2024-2025 remains excluded holdout.
- Correlations are reported in log levels, ranks, within-day demeaned logs, and partial logs.
- Partial correlations remove log spot, log DTE, and intraday sine/cosine effects.
- Raw rupee premium is included for completeness but is mechanically affected by index level and time to expiry.

## Main 2023 correlations

{dataframe_to_markdown(fmt_focus[["premium_measure", "volatility_measure", "n", *numeric]])}

## Day-clustered bootstrap, primary normalized premium

{dataframe_to_markdown(boot.round(4))}

## Interpretation

The IV correlation is partly mechanical because both IV and net premium come from the same call and put prices. Future-RV correlation is descriptive, not proof that net premium adds forecast information. The next valid test is an out-of-sample encompassing model: HAR/trailing RV + IV versus HAR/trailing RV + IV + normalized net premium, evaluated with QLIKE and a day-level Diebold-Mariano test.
"""
    (OUT / "netpremium_correlation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    save_premium_time_result(panel)
    corr, partial, yearly = correlation_analysis(panel)
    save_correlation_outputs(panel, corr, partial, yearly)
    print(f"rows={len(panel):,}; unique_timestamps={panel['datetime'].nunique():,}")
    print("split counts:")
    print(panel["split"].value_counts(dropna=False).to_string())
    print("\n2023 focus correlations:")
    focus = corr[(corr["sample"] == "2023_evaluation") & corr["volatility_measure"].isin(["iv_trade_vol", "past_rv_15", "future_rv_15"])]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
