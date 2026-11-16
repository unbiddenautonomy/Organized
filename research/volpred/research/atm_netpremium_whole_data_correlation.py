from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atm_netpremium_correlation import (
    build_panel,
    corr_pair,
    daily_bootstrap_corr,
    dataframe_to_markdown,
    residualize,
)


OUT = Path(__file__).resolve().parents[1] / "outputs" / "atm_iv_rv_analysis"
PREMIUMS = {
    "net_premium": "raw_rupees",
    "net_premium_spot_pct": "percent_of_spot",
    "net_premium_time_normalized": "spot_and_sqrt_time_normalized",
}
VOLATILITY = [
    "iv_trade_vol", "atm_iv_calendar", "past_rv_5", "past_rv_15", "past_rv_60",
    "future_rv_5", "future_rv_15", "future_rv_30", "future_rv_60",
]


def valid_whole_sample(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.loc[
        panel["quality_ok"].fillna(False)
        & panel["net_premium"].gt(0)
        & panel["spot_price"].gt(0)
        & panel["days_to_expiry"].gt(0)
    ].copy()


def correlations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for premium, premium_label in PREMIUMS.items():
        log_premium = np.log(data[premium])
        for vol in VOLATILITY:
            log_vol = np.log(data[vol].where(data[vol] > 0))
            pearson, n = corr_pair(log_premium, log_vol, "pearson")
            spearman, _ = corr_pair(data[premium], data[vol], "spearman")
            within_p = log_premium - log_premium.groupby(data["date"]).transform("mean")
            within_v = log_vol - log_vol.groupby(data["date"]).transform("mean")
            within, _ = corr_pair(within_p, within_v, "pearson")
            rows.append({
                "sample": "all_available_data",
                "premium_measure": premium_label,
                "volatility_measure": vol,
                "n": n,
                "pearson_log_levels": pearson,
                "spearman_levels": spearman,
                "within_day_pearson_log": within,
            })
    return pd.DataFrame(rows)


def partial_correlations(data: pd.DataFrame) -> pd.DataFrame:
    angle = 2.0 * np.pi * (data["minute_of_day"] - 555.0) / 375.0
    controls = np.column_stack([
        np.ones(len(data)),
        np.log(data["spot_price"]),
        np.log(data["days_to_expiry"]),
        np.sin(angle),
        np.cos(angle),
    ])
    rows = []
    for premium, premium_label in PREMIUMS.items():
        premium_resid = residualize(np.log(data[premium]).to_numpy(), controls)
        for vol in VOLATILITY:
            vol_resid = residualize(np.log(data[vol].where(data[vol] > 0)).to_numpy(), controls)
            value, n = corr_pair(pd.Series(premium_resid), pd.Series(vol_resid), "pearson")
            rows.append({
                "sample": "all_available_data",
                "premium_measure": premium_label,
                "volatility_measure": vol,
                "n": n,
                "partial_pearson_log": value,
                "controls": "log_spot + log_DTE + intraday_sine_cosine",
            })
    return pd.DataFrame(rows)


def yearly_correlations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in data.groupby("year"):
        for vol in ["iv_trade_vol", "past_rv_15", "future_rv_15"]:
            value, n = corr_pair(
                np.log(group["net_premium_time_normalized"]),
                np.log(group[vol].where(group[vol] > 0)),
                "pearson",
            )
            rows.append({
                "year": int(year),
                "premium_measure": "spot_and_sqrt_time_normalized",
                "volatility_measure": vol,
                "n": n,
                "pearson_log_levels": value,
            })
    return pd.DataFrame(rows)


def bootstrap_primary(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["log_premium"] = np.log(work["net_premium_time_normalized"])
    rows = []
    for vol in ["iv_trade_vol", "past_rv_15", "future_rv_15"]:
        name = f"log_{vol}"
        work[name] = np.log(work[vol].where(work[vol] > 0))
        point, n = corr_pair(work["log_premium"], work[name])
        low, high = daily_bootstrap_corr(work, "log_premium", name, reps=500)
        rows.append({
            "volatility_measure": vol,
            "n": n,
            "pearson_log_levels": point,
            "day_bootstrap_ci_low": low,
            "day_bootstrap_ci_high": high,
        })
    return pd.DataFrame(rows)


def main() -> None:
    panel = build_panel()
    data = valid_whole_sample(panel)
    corr = correlations(data)
    partial = partial_correlations(data)
    yearly = yearly_correlations(data)
    bootstrap = bootstrap_primary(data)

    corr.to_csv(OUT / "netpremium_correlations_whole_data.csv", index=False)
    partial.to_csv(OUT / "netpremium_partial_correlations_whole_data.csv", index=False)
    yearly.to_csv(OUT / "netpremium_correlation_by_year_whole_data.csv", index=False)
    bootstrap.to_csv(OUT / "netpremium_primary_bootstrap_whole_data.csv", index=False)

    focus_vars = ["iv_trade_vol", "past_rv_15", "future_rv_15"]
    focus = corr[
        corr["premium_measure"].eq("spot_and_sqrt_time_normalized")
        & corr["volatility_measure"].isin(focus_vars)
    ].merge(
        partial[["premium_measure", "volatility_measure", "partial_pearson_log"]],
        on=["premium_measure", "volatility_measure"],
        how="left",
    )
    focus = focus.set_index("volatility_measure").reindex(focus_vars).reset_index()

    future_horizons = corr[
        corr["premium_measure"].eq("spot_and_sqrt_time_normalized")
        & corr["volatility_measure"].isin(["future_rv_5", "future_rv_15", "future_rv_30", "future_rv_60"])
    ].copy()
    date_min = data["datetime"].min()
    date_max = data["datetime"].max()
    years = ", ".join(str(x) for x in sorted(data["year"].unique()))

    report = f"""# ATM net premium correlation with IV and RV: whole dataset

## Scope

- Available period: {date_min:%Y-%m-%d %H:%M:%S} to {date_max:%Y-%m-%d %H:%M:%S}.
- Years included: {years}.
- Quality-valid ATM observations: {len(data):,} from {data['date'].nunique():,} trading days.
- `netPremium = call_price + put_price` for the exact ATM strike and nearest monthly expiry.
- The primary premium measure divides net premium by spot and `sqrt(calendar time to expiry)`.
- This is a retrospective full-sample analysis. The former 2024-2025 holdout is now included and must not be described as untouched in later tests.

## Main whole-sample result

{dataframe_to_markdown(focus[["volatility_measure", "n", "pearson_log_levels", "spearman_levels", "within_day_pearson_log", "partial_pearson_log"]].round(4))}

## Correlation with future RV by horizon

{dataframe_to_markdown(future_horizons[["volatility_measure", "n", "pearson_log_levels", "spearman_levels", "within_day_pearson_log"]].round(4))}

## Day-clustered bootstrap intervals

{dataframe_to_markdown(bootstrap.round(4))}

## Interpretation

The spot- and time-normalized ATM straddle premium has a strong contemporaneous relationship with ATM IV. Its relationship with future RV is positive in full-sample levels, but much weaker after removing each day's mean. This indicates that much of the RV relationship comes from persistent high- and low-volatility regimes across days rather than minute-to-minute predictive variation within a trading day.

The IV relationship is partly mechanical because IV and net premium use the same option prices. Correlation alone does not show that net premium improves a forecast already containing IV and trailing RV. An encompassing out-of-sample model is still required for that decision.
"""
    (OUT / "netpremium_correlation_whole_data_report.md").write_text(report, encoding="utf-8")

    print(f"period={date_min} to {date_max}")
    print(f"quality_valid_rows={len(data):,}; trading_days={data['date'].nunique():,}")
    print("\nWhole-data primary results:")
    print(focus.to_string(index=False))
    print("\nBootstrap intervals:")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
