from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import pearsonr, spearmanr

from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
SPOT = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
OUT = ROOT / "outputs" / "five_strike_spot_analysis"
POSITIONS = {-2: "m2", -1: "m1", 0: "atm", 1: "p1", 2: "p2"}
HORIZONS = (5, 15, 30, 60)
MIN_OPTION_PRICE = 5.0


def load_nearest_expiry() -> pd.DataFrame:
    query = """
    WITH base AS (
      SELECT source_index, datetime::TIMESTAMP AS datetime, expiry::DATE AS expiry,
             strike_position::INTEGER AS strike_position,
             available_strike_count::INTEGER AS available_strike_count,
             strike::DOUBLE AS strike, spot_price::DOUBLE AS spot_price,
             days_to_expiry::DOUBLE AS days_to_expiry,
             call_price::DOUBLE AS call_price, put_price::DOUBLE AS put_price,
             call_oi::DOUBLE AS call_oi, put_oi::DOUBLE AS put_oi,
             iv_call::DOUBLE AS iv_call_raw, iv_put::DOUBLE AS iv_put_raw
      FROM read_csv_auto(?, header=true)
    ), ranked AS (
      SELECT *, dense_rank() OVER (PARTITION BY datetime ORDER BY expiry) AS expiry_rank
      FROM base
    )
    SELECT * EXCLUDE(expiry_rank)
    FROM ranked
    WHERE expiry_rank=1
    ORDER BY datetime, strike_position
    """
    return duckdb.connect().execute(query, [str(SOURCE)]).fetchdf()


def black_price(forward: np.ndarray, strike: np.ndarray, t: np.ndarray, vol: np.ndarray, call: bool) -> np.ndarray:
    root_t = np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(forward / strike) + 0.5 * vol * vol * t) / (vol * root_t)
    d2 = d1 - vol * root_t
    if call:
        return forward * ndtr(d1) - strike * ndtr(d2)
    return strike * ndtr(-d2) - forward * ndtr(-d1)


def implied_vol_vector(price: pd.Series, forward: pd.Series, strike: pd.Series, t: pd.Series, call: bool) -> np.ndarray:
    p = price.to_numpy(float)
    f = forward.to_numpy(float)
    k = strike.to_numpy(float)
    tau = t.to_numpy(float)
    intrinsic = np.maximum(f - k, 0.0) if call else np.maximum(k - f, 0.0)
    upper_price = f if call else k
    valid = (
        np.isfinite(p) & np.isfinite(f) & np.isfinite(k) & np.isfinite(tau)
        & (p >= MIN_OPTION_PRICE) & (tau > 0) & (f > 0) & (k > 0)
        & (p >= intrinsic - 1e-7) & (p < upper_price)
    )
    lo = np.full(len(p), 1e-5)
    hi = np.full(len(p), 5.0)
    for _ in range(48):
        mid = (lo + hi) / 2.0
        model = black_price(f, k, tau, mid, call)
        lo = np.where(model < p, mid, lo)
        hi = np.where(model >= p, mid, hi)
    result = (lo + hi) / 2.0
    result[~valid] = np.nan
    return result


def add_fixed_forward_iv(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["parity_forward_by_strike"] = rows["strike"] + rows["call_price"] - rows["put_price"]
    rows["common_forward"] = rows.groupby(["datetime", "expiry"])["parity_forward_by_strike"].transform("median")
    tau = rows["days_to_expiry"].clip(lower=1.0 / (365.0 * 24.0 * 60.0)) / 365.0
    rows["iv_call"] = implied_vol_vector(rows["call_price"], rows["common_forward"], rows["strike"], tau, True)
    rows["iv_put"] = implied_vol_vector(rows["put_price"], rows["common_forward"], rows["strike"], tau, False)
    rows["parity_gap"] = (
        rows["call_price"] - rows["put_price"] - (rows["common_forward"] - rows["strike"])
    )
    return rows


def add_same_contract_changes(rows: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["iv_call", "iv_put", "call_oi", "put_oi"]
    previous = rows[["datetime", "expiry", "strike", *value_cols]].copy()
    previous["datetime"] = previous["datetime"] + pd.Timedelta(minutes=5)
    previous = previous.rename(columns={c: f"prev_{c}" for c in value_cols})
    rows = rows.merge(previous, on=["datetime", "expiry", "strike"], how="left", validate="one_to_one")
    rows["d_iv_call_5m"] = rows["iv_call"] - rows["prev_iv_call"]
    rows["d_iv_put_5m"] = rows["iv_put"] - rows["prev_iv_put"]
    rows["log_call_oi"] = np.log1p(rows["call_oi"].clip(lower=0))
    rows["log_put_oi"] = np.log1p(rows["put_oi"].clip(lower=0))
    rows["d_log_call_oi_5m"] = rows["log_call_oi"] - np.log1p(rows["prev_call_oi"].clip(lower=0))
    rows["d_log_put_oi_5m"] = rows["log_put_oi"] - np.log1p(rows["prev_put_oi"].clip(lower=0))
    return rows


def make_wide_panel(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    values = [
        "strike", "call_oi", "put_oi", "iv_call", "iv_put", "log_call_oi", "log_put_oi",
        "d_iv_call_5m", "d_iv_put_5m", "d_log_call_oi_5m", "d_log_put_oi_5m",
    ]
    wide = rows.pivot(index="datetime", columns="strike_position", values=values)
    wide.columns = [f"{metric}_{POSITIONS[int(position)]}" for metric, position in wide.columns]
    wide = wide.reset_index()
    meta = rows.groupby("datetime", as_index=False).agg(
        expiry=("expiry", "first"), spot_price=("spot_price", "first"),
        days_to_expiry=("days_to_expiry", "first"),
        available_strike_count=("strike_position", "count"),
        common_forward=("common_forward", "first"),
        median_abs_parity_gap=("parity_gap", lambda x: float(np.nanmedian(np.abs(x)))),
    )
    panel = meta.merge(wide, on="datetime", how="left", validate="one_to_one")

    position_suffixes = list(POSITIONS.values())
    for metric in ["iv_call", "iv_put", "log_call_oi", "log_put_oi", "d_iv_call_5m", "d_iv_put_5m", "d_log_call_oi_5m", "d_log_put_oi_5m"]:
        cols = [f"{metric}_{s}" for s in position_suffixes]
        panel[f"{metric}_mean"] = panel[cols].mean(axis=1, skipna=True)

    call_oi_cols = [f"call_oi_{s}" for s in position_suffixes]
    put_oi_cols = [f"put_oi_{s}" for s in position_suffixes]
    panel["total_call_oi"] = panel[call_oi_cols].sum(axis=1, min_count=1)
    panel["total_put_oi"] = panel[put_oi_cols].sum(axis=1, min_count=1)
    panel["log_total_call_oi"] = np.log1p(panel["total_call_oi"])
    panel["log_total_put_oi"] = np.log1p(panel["total_put_oi"])
    denom = panel["total_put_oi"] + panel["total_call_oi"]
    panel["oi_imbalance_put_minus_call"] = (panel["total_put_oi"] - panel["total_call_oi"]) / denom.where(denom > 0)
    panel["log_put_call_oi_ratio"] = np.log1p(panel["total_put_oi"]) - np.log1p(panel["total_call_oi"])
    panel["iv_put_minus_call_mean"] = panel["iv_put_mean"] - panel["iv_call_mean"]
    panel["otm_skew_put_minus_call"] = panel[["iv_put_m2", "iv_put_m1"]].mean(axis=1) - panel[["iv_call_p1", "iv_call_p2"]].mean(axis=1)

    # Exact five-minute aggregate changes. This avoids treating gaps or overnight moves as five-minute changes.
    aggregate_for_change = [
        "iv_call_mean", "iv_put_mean", "iv_put_minus_call_mean", "otm_skew_put_minus_call",
        "log_total_call_oi", "log_total_put_oi", "oi_imbalance_put_minus_call", "log_put_call_oi_ratio",
    ]
    previous = panel[["datetime", *aggregate_for_change]].copy()
    previous["datetime"] = previous["datetime"] + pd.Timedelta(minutes=5)
    previous = previous.rename(columns={c: f"prev_{c}" for c in aggregate_for_change})
    panel = panel.merge(previous, on="datetime", how="left", validate="one_to_one")
    for c in aggregate_for_change:
        panel[f"d_{c}_5m"] = panel[c] - panel[f"prev_{c}"]

    individual_levels = [
        f"{metric}_{suffix}"
        for metric in ["iv_call", "iv_put", "log_call_oi", "log_put_oi"]
        for suffix in position_suffixes
    ]
    individual_changes = [
        f"{metric}_{suffix}"
        for metric in ["d_iv_call_5m", "d_iv_put_5m", "d_log_call_oi_5m", "d_log_put_oi_5m"]
        for suffix in position_suffixes
    ]
    aggregate_levels = aggregate_for_change
    aggregate_changes = [f"d_{c}_5m" for c in aggregate_for_change]
    return panel, individual_levels + aggregate_levels, individual_changes + aggregate_changes


def add_spot_targets(panel: pd.DataFrame) -> pd.DataFrame:
    spot = pd.read_csv(SPOT, usecols=["datetime", "spot_price"], parse_dates=["datetime"])
    spot = spot.drop_duplicates("datetime").sort_values("datetime")
    result = panel.copy()
    for minutes, label in [(-5, "past_5")]:
        lookup = spot.copy()
        lookup["datetime"] = lookup["datetime"] - pd.Timedelta(minutes=minutes)
        lookup = lookup.rename(columns={"spot_price": f"spot_{label}"})
        result = result.merge(lookup, on="datetime", how="left", validate="one_to_one")
        r = np.log(result["spot_price"] / result[f"spot_{label}"])
        result[f"return_{label}"] = r
        result[f"direction_{label}"] = np.where(r > 0, 1.0, np.where(r < 0, 0.0, np.nan))
    for minutes in HORIZONS:
        lookup = spot.copy()
        lookup["datetime"] = lookup["datetime"] - pd.Timedelta(minutes=minutes)
        lookup = lookup.rename(columns={"spot_price": f"spot_future_{minutes}"})
        result = result.merge(lookup, on="datetime", how="left", validate="one_to_one")
        r = np.log(result[f"spot_future_{minutes}"] / result["spot_price"])
        result[f"return_future_{minutes}"] = r
        result[f"direction_future_{minutes}"] = np.where(r > 0, 1.0, np.where(r < 0, 0.0, np.nan))
    result["date"] = result["datetime"].dt.date.astype(str)
    result["year"] = result["datetime"].dt.year
    result["minute_of_day"] = result["datetime"].dt.hour * 60 + result["datetime"].dt.minute
    result["split"] = np.where(result["datetime"] < pd.Timestamp("2023-01-01"), "train_2019_2022", "test_2023_2025")
    return result


def add_causal_adjustments(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Create live-available feature deviations and training-only clock-time adjustments."""
    panel = panel.sort_values("datetime").copy()
    train_mask = panel["split"].eq("train_2019_2022")
    for feature in features:
        expanding_mean = (
            panel.groupby("date", sort=False)[feature]
            .expanding(min_periods=5).mean()
            .reset_index(level=0, drop=True)
            .reindex(panel.index)
        )
        deviation = panel[feature] - expanding_mean
        profile = deviation[train_mask].groupby(panel.loc[train_mask, "minute_of_day"]).mean()
        panel[f"causal_adj__{feature}"] = deviation - panel["minute_of_day"].map(profile)

    target_pairs = [("return_past_5", "direction_past_5")]
    target_pairs += [(f"return_future_{h}", f"direction_future_{h}") for h in HORIZONS]
    for return_col, direction_col in target_pairs:
        return_profile = panel.loc[train_mask].groupby("minute_of_day")[return_col].mean()
        direction_profile = panel.loc[train_mask].groupby("minute_of_day")[direction_col].mean()
        panel[f"tod_adj__{return_col}"] = panel[return_col] - panel["minute_of_day"].map(return_profile)
        panel[f"tod_adj__{direction_col}"] = panel[direction_col] - panel["minute_of_day"].map(direction_profile)
    return panel


def pair_stats(frame: pd.DataFrame, feature: str, return_col: str, direction_col: str) -> dict:
    causal_feature = f"causal_adj__{feature}"
    causal_return = f"tod_adj__{return_col}"
    causal_direction = f"tod_adj__{direction_col}"
    z = frame[["date", "minute_of_day", feature, return_col, direction_col, causal_feature, causal_return, causal_direction]].replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, return_col])
    if len(z) < 100 or z[feature].nunique() < 3:
        return {"n": len(z)}
    x = z[feature].to_numpy(float)
    y = z[return_col].to_numpy(float)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(spearmanr(x, y).statistic)
    x_dm = x - z.groupby("date")[feature].transform("mean").to_numpy()
    y_dm = y - z.groupby("date")[return_col].transform("mean").to_numpy()
    within = float(np.corrcoef(x_dm, y_dm)[0, 1]) if np.std(x_dm) > 0 and np.std(y_dm) > 0 else np.nan
    x_adj = x_dm - pd.Series(x_dm, index=z.index).groupby(z["minute_of_day"]).transform("mean").to_numpy()
    y_adj = y_dm - pd.Series(y_dm, index=z.index).groupby(z["minute_of_day"]).transform("mean").to_numpy()
    adjusted = float(np.corrcoef(x_adj, y_adj)[0, 1]) if np.std(x_adj) > 0 and np.std(y_adj) > 0 else np.nan
    causal = z.dropna(subset=[causal_feature, causal_return])
    causal_corr = float(causal[causal_feature].corr(causal[causal_return])) if len(causal) > 100 else np.nan
    causal_dir_frame = z.dropna(subset=[causal_feature, causal_direction])
    causal_direction_corr = float(causal_dir_frame[causal_feature].corr(causal_dir_frame[causal_direction])) if len(causal_dir_frame) > 100 else np.nan
    q20, q80 = np.nanquantile(causal[causal_feature], [0.2, 0.8])
    low = causal[causal[causal_feature] <= q20]
    high = causal[causal[causal_feature] >= q80]
    direction = z.dropna(subset=[direction_col])
    direction_corr = float(np.corrcoef(direction[feature], direction[direction_col])[0, 1]) if len(direction) > 100 else np.nan
    direction_day = direction[direction_col] - direction.groupby("date")[direction_col].transform("mean")
    direction_adj = direction_day - direction_day.groupby(direction["minute_of_day"]).transform("mean")
    feature_day = direction[feature] - direction.groupby("date")[feature].transform("mean")
    feature_adj = feature_day - feature_day.groupby(direction["minute_of_day"]).transform("mean")
    adjusted_direction = float(np.corrcoef(feature_adj, direction_adj)[0, 1]) if len(direction) > 100 else np.nan
    return {
        "n": len(z), "pearson_return": pearson, "spearman_return": spearman,
        "within_day_pearson_return": within, "day_time_adjusted_pearson_return": adjusted,
        "causal_adjusted_pearson_return": causal_corr,
        "direction_point_biserial": direction_corr,
        "day_time_adjusted_direction_corr": adjusted_direction,
        "causal_adjusted_direction_corr": causal_direction_corr,
        "low_quintile_up_rate": float(low[direction_col].mean()),
        "high_quintile_up_rate": float(high[direction_col].mean()),
        "high_minus_low_up_rate": float(high[direction_col].mean() - low[direction_col].mean()),
        "low_quintile_mean_return_bps": float(low[return_col].mean() * 10000.0),
        "high_quintile_mean_return_bps": float(high[return_col].mean() * 10000.0),
        "high_minus_low_return_bps": float((high[return_col].mean() - low[return_col].mean()) * 10000.0),
    }


def analyze_features(panel: pd.DataFrame, levels: list[str], changes: list[str]) -> pd.DataFrame:
    target_specs = [("past_5m", "return_past_5", "direction_past_5")]
    target_specs += [(f"future_{h}m", f"return_future_{h}", f"direction_future_{h}") for h in HORIZONS]
    rows = []
    for sample, sample_df in [
        ("full_2019_2025", panel),
        ("train_2019_2022", panel[panel["split"].eq("train_2019_2022")]),
        ("test_2023_2025", panel[panel["split"].eq("test_2023_2025")]),
    ]:
        for target, return_col, direction_col in target_specs:
            for feature in [*levels, *changes]:
                result = pair_stats(sample_df, feature, return_col, direction_col)
                rows.append({
                    "sample": sample, "target": target, "feature": feature,
                    "feature_type": "change_5m" if feature in changes else "level", **result,
                })
    return pd.DataFrame(rows)


def spot_level_correlations(panel: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    rows = []
    log_spot = np.log(panel["spot_price"])
    spot_dm = log_spot - log_spot.groupby(panel["date"]).transform("mean")
    for feature in levels:
        z = pd.DataFrame({"x": panel[feature], "spot": log_spot, "x_dm": panel[feature] - panel[feature].groupby(panel["date"]).transform("mean"), "spot_dm": spot_dm}).dropna()
        if len(z) < 100:
            continue
        rows.append({
            "feature": feature, "n": len(z),
            "pearson_with_log_spot_level": float(z["x"].corr(z["spot"])),
            "spearman_with_spot_level": float(z["x"].corr(z["spot"], method="spearman")),
            "within_day_pearson_with_log_spot": float(z["x_dm"].corr(z["spot_dm"])),
        })
    return pd.DataFrame(rows)


def make_train_test_table(results: pd.DataFrame) -> pd.DataFrame:
    train = results[(results["sample"] == "train_2019_2022") & results["target"].str.startswith("future")].copy()
    test = results[results["sample"] == "test_2023_2025"].copy()
    selected = train.sort_values(["target", "causal_adjusted_pearson_return"], key=lambda s: s.abs() if s.name == "causal_adjusted_pearson_return" else s)
    selected = selected.groupby("target", group_keys=False).tail(10)
    cols = ["target", "feature", "feature_type", "n", "pearson_return", "within_day_pearson_return", "day_time_adjusted_pearson_return", "causal_adjusted_pearson_return", "direction_point_biserial", "day_time_adjusted_direction_corr", "causal_adjusted_direction_corr", "high_minus_low_up_rate", "high_minus_low_return_bps"]
    joined = selected[cols].merge(test[cols], on=["target", "feature", "feature_type"], suffixes=("_train", "_test"), validate="one_to_one")
    joined["same_sign_train_test"] = np.sign(joined["causal_adjusted_pearson_return_train"]) == np.sign(joined["causal_adjusted_pearson_return_test"])
    joined["absolute_test_correlation"] = joined["causal_adjusted_pearson_return_test"].abs()
    return joined.sort_values(["target", "absolute_test_correlation"], ascending=[True, False])


def yearly_for_selected(panel: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    pairs = selected.groupby("target", group_keys=False).head(3)[["target", "feature"]].drop_duplicates()
    rows = []
    for record in pairs.itertuples(index=False):
        h = int(record.target.split("_")[1][:-1])
        return_col = f"return_future_{h}"
        direction_col = f"direction_future_{h}"
        for year, group in panel.groupby("year"):
            s = pair_stats(group, record.feature, return_col, direction_col)
            rows.append({"year": int(year), "target": record.target, "feature": record.feature, **s})
    return pd.DataFrame(rows)


def data_audit(source_rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    before_gap = np.nanmedian(np.abs(source_rows["iv_call_raw"] - source_rows["iv_put_raw"]))
    after_gap = np.nanmedian(np.abs(source_rows["iv_call"] - source_rows["iv_put"]))
    return pd.DataFrame([
        ["source_file", SOURCE.name],
        ["nearest_expiry_option_rows", len(source_rows)],
        ["unique_timestamps", panel["datetime"].nunique()],
        ["first_timestamp", panel["datetime"].min()],
        ["last_timestamp", panel["datetime"].max()],
        ["timestamps_with_fewer_than_5_strikes", int((panel["available_strike_count"] < 5).sum())],
        ["minimum_available_strikes", int(panel["available_strike_count"].min())],
        ["median_abs_raw_call_put_iv_gap", before_gap],
        ["median_abs_forward_fixed_call_put_iv_gap", after_gap],
        ["rows_valid_future_15m", int(panel["return_future_15"].notna().sum())],
    ], columns=["metric", "value"])


def generate_report(panel: pd.DataFrame, results: pd.DataFrame, selected: pd.DataFrame, audit: pd.DataFrame) -> str:
    full = results[(results["sample"] == "full_2019_2025") & results["target"].str.startswith("future")]
    top_full = full.assign(abs_corr=full["causal_adjusted_pearson_return"].abs()).sort_values(["target", "abs_corr"], ascending=[True, False]).groupby("target", group_keys=False).head(5)
    stable = selected[selected["same_sign_train_test"]].copy()
    stable_top = stable.sort_values(["target", "absolute_test_correlation"], ascending=[True, False]).groupby("target", group_keys=False).head(5)
    display_cols = ["target", "feature", "feature_type", "causal_adjusted_pearson_return_train", "causal_adjusted_pearson_return_test", "causal_adjusted_direction_corr_test", "high_minus_low_up_rate_test", "high_minus_low_return_bps_test", "same_sign_train_test"]
    test = results[results["sample"].eq("test_2023_2025")].set_index(["target", "feature"])

    def metric(target: str, feature: str, column: str) -> float:
        return float(test.loc[(target, feature), column])

    oi_past = metric("past_5m", "oi_imbalance_put_minus_call", "causal_adjusted_pearson_return")
    oi_future = [metric(f"future_{h}m", "oi_imbalance_put_minus_call", "causal_adjusted_pearson_return") for h in HORIZONS]
    call_iv_future = [metric(f"future_{h}m", "iv_call_mean", "causal_adjusted_pearson_return") for h in HORIZONS]
    put_iv_future = [metric(f"future_{h}m", "iv_put_mean", "causal_adjusted_pearson_return") for h in HORIZONS]
    call_p2_60 = metric("future_60m", "iv_call_p2", "causal_adjusted_pearson_return")
    put_p2_60 = metric("future_60m", "iv_put_p2", "causal_adjusted_pearson_return")
    up_spread_p2_60 = metric("future_60m", "iv_put_p2", "high_minus_low_up_rate")
    report = f"""# Five-strike IV/OI relationship with Bank Nifty spot direction

## Decision

The OI hypothesis is not supported as a forward directional signal. Put-minus-call OI imbalance has a strong relationship with the preceding five-minute spot return ({oi_past:.3f}), but its causal correlation with later returns is only {oi_future[0]:.3f}, {oi_future[1]:.3f}, {oi_future[2]:.3f}, and {oi_future[3]:.3f} at 5, 15, 30, and 60 minutes. OI is mainly reacting to the move already observed.

IV contains a weak bearish relationship. Mean call-IV correlations are {call_iv_future[0]:.3f}, {call_iv_future[1]:.3f}, {call_iv_future[2]:.3f}, and {call_iv_future[3]:.3f}; mean put-IV correlations are {put_iv_future[0]:.3f}, {put_iv_future[1]:.3f}, {put_iv_future[2]:.3f}, and {put_iv_future[3]:.3f}. The strongest split-level 60-minute relationship is at relative strike +2: call IV {call_p2_60:.3f}, put IV {put_p2_60:.3f}. Moving from the bottom to top quintile of +2 put IV reduces the observed up-rate by {abs(up_spread_p2_60) * 100:.1f} percentage points. The sign reverses in some individual years, so this is regime-dependent and too weak to use alone.

Spot-level correlations are descriptive rather than predictive. IV is negatively correlated with the spot level over the long sample, while total OI is positively correlated with it; both relationships mix index-level growth, volatility regimes, DTE, and intraday accumulation. The causal return tests above are the relevant results for trading.

## Question tested

Do call/put IV and open interest at relative strikes -2, -1, ATM, +1 and +2 contain information about the current spot level, the trailing five-minute spot move, and future spot returns/direction over 5, 15, 30 and 60 minutes?

## Method

- Uses the nearest monthly expiry at each timestamp from the up-to-five-strike dataset; timestamps with fewer than five strikes are retained.
- Call and put IV are re-solved using one parity-implied forward per timestamp rather than the inconsistent strike-level forwards in the source.
- Five-minute IV/OI changes compare the same absolute strike and expiry. This prevents ATM re-centering from creating false changes.
- Predictors at time t are compared with later spot prices from the independent one-minute spot timeline. Windows do not cross missing intervals or overnight gaps.
- OI levels use log transforms. Results include raw correlation, descriptive day/time fixed-effect correlation, and a causal intraday correlation. The causal feature uses only observations available up to time t; its clock-time profile is learned on 2019-2022 and applied unchanged to 2023-2025.
- 2019-2022 is used for discovery and 2023-2025 for chronological confirmation.

## Data audit

{dataframe_to_markdown(audit)}

## Strongest full-sample relationships by horizon

{dataframe_to_markdown(top_full[["target", "feature", "feature_type", "n", "pearson_return", "within_day_pearson_return", "causal_adjusted_pearson_return", "causal_adjusted_direction_corr", "high_minus_low_up_rate", "high_minus_low_return_bps"]].round(5))}

## Signals retaining the same sign in 2019-2022 and 2023-2025

{dataframe_to_markdown(stable_top[display_cols].round(5))}

## Interpretation rule

A feature is potentially useful only if its sign survives the chronological split, its within-day relationship remains meaningful, and its high-versus-low quintile direction spread is economically large enough to exceed trading costs. Spot-level correlations are descriptive and are not evidence of predictability. Call-put IV differences should be treated as quote-quality or relative-pricing information because both IVs are derived from the same option market.
"""
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_nearest_expiry()
    rows = add_fixed_forward_iv(rows)
    rows = add_same_contract_changes(rows)
    panel, levels, changes = make_wide_panel(rows)
    panel = add_spot_targets(panel)
    panel = add_causal_adjustments(panel, [*levels, *changes])

    results = analyze_features(panel, levels, changes)
    spot_corr = spot_level_correlations(panel, levels)
    selected = make_train_test_table(results)
    yearly = yearly_for_selected(panel, selected)
    audit = data_audit(rows, panel)

    keep_panel = [
        "datetime", "expiry", "spot_price", "days_to_expiry", "available_strike_count", "common_forward",
        *[c for c in panel.columns if c.startswith(("strike_", "iv_", "log_", "d_iv_", "d_log_", "total_", "oi_", "otm_"))],
        "return_past_5", "direction_past_5",
        *[f"return_future_{h}" for h in HORIZONS], *[f"direction_future_{h}" for h in HORIZONS],
        "date", "year", "minute_of_day", "split",
    ]
    keep_panel = list(dict.fromkeys(c for c in keep_panel if c in panel.columns))
    panel[keep_panel].to_csv(OUT / "five_strike_feature_panel.csv", index=False)
    results.to_csv(OUT / "feature_spot_return_direction_correlations.csv", index=False)
    spot_corr.to_csv(OUT / "feature_spot_level_correlations.csv", index=False)
    selected.to_csv(OUT / "train_test_top_signals.csv", index=False)
    yearly.to_csv(OUT / "top_signals_by_year.csv", index=False)
    audit.to_csv(OUT / "data_audit.csv", index=False)
    report = generate_report(panel, results, selected, audit)
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")

    print(audit.to_string(index=False))
    print("\nBest chronological-test correlations by horizon:")
    print(selected.groupby("target", group_keys=False).head(5)[[
        "target", "feature", "causal_adjusted_pearson_return_train", "causal_adjusted_pearson_return_test",
        "causal_adjusted_direction_corr_test", "high_minus_low_up_rate_test",
        "high_minus_low_return_bps_test", "same_sign_train_test",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
