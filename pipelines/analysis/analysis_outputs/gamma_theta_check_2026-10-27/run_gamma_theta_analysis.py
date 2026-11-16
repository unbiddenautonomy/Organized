from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.stats import kurtosis, mannwhitneyu, norm, pearsonr, skew, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "gamma_theta_check_2026-10-27"
TRADES_PATH = ROOT / "analysis_outputs" / "test_period_final_routing_above75_event0" / "executed_trades.csv"
DAILY_PATH = ROOT / "analysis_outputs" / "test_period_final_routing_above75_event0" / "daily_pnl.csv"
PRICES_PATH = ROOT / "analysis_outputs" / "stress_test_2026-10-27" / "replay_prices_exact.csv.gz"
SPOT_PATH = Path(r"E:\v2\data\icharts_banknifty_spot_1min.csv")
PREDICTIONS_PATH = ROOT / "analysis_outputs" / "three_iv_rv_conditions" / "three_rule_predictions.csv.gz"
LIVE_DIR = ROOT.parent / "families" / "ivrv" / "Inference_full_2026-10-24" / "Inference" / "live"

LOT_SIZE = 30
RISK_FREE_RATE = 0.065
EXPIRY_HOUR = 15
EXPIRY_MINUTE = 30
YEAR_DAYS = 365.0
ANNUALIZATION_DAYS = 252
BOOTSTRAP_DRAWS = 10_000
PLACEBO_DRAWS = 1_000
SEED = 20261027
FILTER_PERCENTILES = [0.80, 0.90, 0.95]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def live_hashes() -> dict[str, str]:
    return {
        str(path): sha256(path)
        for path in sorted(LIVE_DIR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def black76_price(forward: float, strike: float, maturity: float, rate: float, sigma: float, is_call: bool) -> float:
    discount = math.exp(-rate * maturity)
    root_t = math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * maturity) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    if is_call:
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol_black76(price: float, forward: float, strike: float, maturity: float, rate: float, is_call: bool) -> float:
    discount = math.exp(-rate * maturity)
    intrinsic = discount * max(forward - strike, 0.0) if is_call else discount * max(strike - forward, 0.0)
    upper = discount * (forward if is_call else strike)
    if not all(math.isfinite(value) for value in [price, forward, strike, maturity, rate]):
        raise ValueError("non_finite_input")
    if forward <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("non_positive_forward_strike_or_maturity")
    tolerance = 1e-7
    if price < intrinsic - tolerance:
        raise ValueError("price_below_discounted_intrinsic")
    if price > upper + tolerance:
        raise ValueError("price_above_black76_upper_bound")
    if price <= intrinsic + tolerance:
        raise ValueError("price_has_no_resolvable_time_value")

    def objective(sigma: float) -> float:
        return black76_price(forward, strike, maturity, rate, sigma, is_call) - price

    low, high = 1e-6, 5.0
    if objective(low) * objective(high) > 0:
        raise ValueError("iv_root_not_bracketed")
    return float(brentq(objective, low, high, xtol=1e-12, rtol=1e-12, maxiter=200))


def black76_greeks(price: float, forward: float, strike: float, maturity: float, rate: float, sigma: float) -> tuple[float, float]:
    discount = math.exp(-rate * maturity)
    root_t = math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * maturity) / (sigma * root_t)
    density = norm.pdf(d1)
    gamma = discount * density / (forward * sigma * root_t)
    theta_annual = rate * price - discount * forward * density * sigma / (2.0 * root_t)
    return float(gamma), float(theta_annual)


def compute_snapshot(stamp: pd.Timestamp, expiry: pd.Timestamp, strike: float, call_price: float, put_price: float, spot: float) -> dict:
    expiry_stamp = pd.Timestamp(expiry).normalize() + pd.Timedelta(hours=EXPIRY_HOUR, minutes=EXPIRY_MINUTE)
    seconds = (expiry_stamp - stamp).total_seconds()
    if seconds <= 0:
        raise ValueError("entry_at_or_after_expiry")
    maturity = seconds / (YEAR_DAYS * 86400.0)
    discount = math.exp(-RISK_FREE_RATE * maturity)
    forward = strike + (call_price - put_price) / discount
    if forward <= 0:
        raise ValueError("synthetic_forward_non_positive")
    call_iv = implied_vol_black76(call_price, forward, strike, maturity, RISK_FREE_RATE, True)
    put_iv = implied_vol_black76(put_price, forward, strike, maturity, RISK_FREE_RATE, False)
    call_gamma, call_theta_annual = black76_greeks(call_price, forward, strike, maturity, RISK_FREE_RATE, call_iv)
    put_gamma, put_theta_annual = black76_greeks(put_price, forward, strike, maturity, RISK_FREE_RATE, put_iv)

    call_gamma_lot = -call_gamma * LOT_SIZE
    put_gamma_lot = -put_gamma * LOT_SIZE
    call_theta_day_lot = -call_theta_annual / YEAR_DAYS * LOT_SIZE
    put_theta_day_lot = -put_theta_annual / YEAR_DAYS * LOT_SIZE
    position_gamma = call_gamma_lot + put_gamma_lot
    position_theta_day = call_theta_day_lot + put_theta_day_lot
    if abs(position_theta_day) <= 1e-14:
        raise ValueError("position_theta_too_close_to_zero")
    ratio = abs(position_gamma) / abs(position_theta_day)
    mean_iv = (call_iv + put_iv) / 2.0
    sanity_lhs = abs(position_theta_day / position_gamma) * YEAR_DAYS
    sanity_rhs = 0.5 * mean_iv * mean_iv * forward * forward
    sanity_relative_error = (sanity_lhs - sanity_rhs) / sanity_rhs
    return {
        "expiry_timestamp": expiry_stamp,
        "calendar_days_to_expiry": seconds / 86400.0,
        "maturity_years": maturity,
        "spot_price": spot,
        "synthetic_forward": forward,
        "forward_minus_spot": forward - spot,
        "call_price": call_price,
        "put_price": put_price,
        "call_iv": call_iv,
        "put_iv": put_iv,
        "entry_iv_mean": mean_iv,
        "call_short_gamma_rupees_per_point2_per_lot": call_gamma_lot,
        "put_short_gamma_rupees_per_point2_per_lot": put_gamma_lot,
        "position_short_gamma_rupees_per_point2_per_lot": position_gamma,
        "call_short_theta_rupees_per_calendar_day_per_lot": call_theta_day_lot,
        "put_short_theta_rupees_per_calendar_day_per_lot": put_theta_day_lot,
        "position_short_theta_rupees_per_calendar_day_per_lot": position_theta_day,
        "gamma_theta_ratio": ratio,
        "theta_gamma_sanity_lhs_annualized": sanity_lhs,
        "theta_gamma_sanity_rhs_diffusion": sanity_rhs,
        "theta_gamma_sanity_relative_error": sanity_relative_error,
    }


def sharpe_from_daily(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    standard_deviation = values.std(ddof=1)
    return float(values.mean() / standard_deviation * math.sqrt(ANNUALIZATION_DAYS)) if standard_deviation > 0 else math.nan


def max_drawdown(values: np.ndarray) -> float:
    equity = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=float))))
    return float(np.max(np.maximum.accumulate(equity) - equity))


def strategy_metrics(frame: pd.DataFrame, eligible_dates: pd.DatetimeIndex, pnl_column: str = "net_pnl_inr", weights: np.ndarray | None = None) -> dict:
    if weights is None:
        pnl = frame[pnl_column].to_numpy(dtype=float)
    else:
        pnl = frame[pnl_column].to_numpy(dtype=float) * np.asarray(weights, dtype=float)
    daily = pd.Series(pnl, index=frame["trade_date"]).groupby(level=0).sum().reindex(eligible_dates, fill_value=0.0).to_numpy(dtype=float)
    return {
        "trades": int(len(frame)),
        "effective_lots": float(np.sum(weights)) if weights is not None else float(len(frame)),
        "net_pnl_inr": float(pnl.sum()),
        "average_pnl_per_trade_inr": float(pnl.mean()) if len(pnl) else None,
        "sharpe_daily": sharpe_from_daily(daily),
        "max_drawdown_inr": max_drawdown(pnl),
    }


def bucket_table(frame: pd.DataFrame, bucket_column: str, eligible_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for bucket, group in frame.groupby(bucket_column, observed=True, sort=True):
        pnl = group["net_pnl_inr"].to_numpy(dtype=float)
        stops = group["exit_reason"].eq("stop_loss")
        metrics = strategy_metrics(group, eligible_dates)
        rows.append({
            "bucket": str(bucket),
            "ratio_min": float(group["gamma_theta_ratio"].min()),
            "ratio_max": float(group["gamma_theta_ratio"].max()),
            "trades": int(len(group)),
            "win_rate_pct": float(np.mean(pnl > 0) * 100),
            "net_pnl_inr": metrics["net_pnl_inr"],
            "average_pnl_per_trade_inr": metrics["average_pnl_per_trade_inr"],
            "stop_out_rate_pct": float(stops.mean() * 100),
            "target_hit_rate_pct": float(group["exit_reason"].eq("profit_target").mean() * 100),
            "timeout_120m_rate_pct": float(group["exit_reason"].eq("time_exit_120m").mean() * 100),
            "session_square_off_rate_pct": float(group["exit_reason"].eq("session_square_off_151430").mean() * 100),
            "average_stop_out_loss_inr": float(group.loc[stops, "net_pnl_inr"].mean()) if stops.any() else None,
            "sharpe_daily": metrics["sharpe_daily"],
            "max_drawdown_inr": metrics["max_drawdown_inr"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes_before = live_hashes()
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_datetime", "exit_datetime", "datetime", "trade_date"], low_memory=False)
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        usecols=["row_id", "datetime", "expiry", "dte_flag", "entry_strike"],
        parse_dates=["datetime", "expiry"],
        low_memory=False,
    ).rename(columns={"datetime": "prediction_datetime"})
    predictions = predictions.drop_duplicates("row_id", keep="last")
    trades = trades.merge(predictions, on="row_id", how="left", validate="one_to_one")
    timestamp_mismatch = trades["prediction_datetime"].notna() & trades["signal_datetime"].ne(trades["prediction_datetime"])
    if timestamp_mismatch.any():
        raise ValueError(f"row_id timestamp mismatch for {int(timestamp_mismatch.sum())} executed trades")
    prices = pd.read_csv(PRICES_PATH, parse_dates=["datetime", "expiry"], low_memory=False)
    spot = pd.read_csv(SPOT_PATH, usecols=["datetime", "spot_price"], parse_dates=["datetime"], low_memory=False)
    eligible_dates = pd.DatetimeIndex(pd.read_csv(DAILY_PATH, parse_dates=["date"])["date"])

    entry_prices = prices.merge(spot, on="datetime", how="left", validate="many_to_one")
    entry_prices = entry_prices.rename(columns={"strike": "entry_strike"})
    entry_columns = ["datetime", "expiry", "entry_strike", "call_price", "put_price", "spot_price"]
    entry_prices = entry_prices[entry_columns].drop_duplicates(["datetime", "expiry", "entry_strike"], keep="last")
    work = trades.merge(
        entry_prices,
        left_on=["signal_datetime", "expiry", "entry_strike"],
        right_on=["datetime", "expiry", "entry_strike"],
        how="left",
        suffixes=("", "_entry"),
        validate="one_to_one",
    )
    work["trade_date"] = work["signal_datetime"].dt.normalize()
    work["entry_minutes_after_0915"] = (
        work["signal_datetime"].dt.hour * 60 + work["signal_datetime"].dt.minute - (9 * 60 + 15)
    )
    greek_records = []
    failures = []
    for row in work.itertuples():
        trade_id = int(row.row_id)
        missing = [name for name in ["expiry", "call_price", "put_price", "spot_price"] if pd.isna(getattr(row, name, np.nan))]
        if missing:
            failures.append({"row_id": trade_id, "signal_datetime": row.signal_datetime, "reason": "missing_" + "_".join(missing)})
            continue
        try:
            result = compute_snapshot(
                pd.Timestamp(row.signal_datetime), pd.Timestamp(row.expiry), float(row.entry_strike),
                float(row.call_price), float(row.put_price), float(row.spot_price),
            )
            greek_records.append({"row_id": trade_id, **result})
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            failures.append({"row_id": trade_id, "signal_datetime": row.signal_datetime, "reason": str(error)})
    greeks = pd.DataFrame(greek_records)
    failures_frame = pd.DataFrame(failures, columns=["row_id", "signal_datetime", "reason"])
    failures_frame.to_csv(OUT / "greek_failures.csv", index=False)
    work = work.merge(greeks, on="row_id", how="left", validate="one_to_one")
    analyzable = work.loc[work["gamma_theta_ratio"].notna()].copy()
    analyzable.to_csv(OUT / "trade_entry_greeks_and_outcomes.csv", index=False)

    sanity_valid = analyzable[["theta_gamma_sanity_lhs_annualized", "theta_gamma_sanity_rhs_diffusion"]].dropna()
    sanity_pearson = pearsonr(sanity_valid.iloc[:, 0], sanity_valid.iloc[:, 1])
    sanity_spearman = spearmanr(sanity_valid.iloc[:, 0], sanity_valid.iloc[:, 1])
    sanity_outliers = analyzable.loc[
        analyzable["theta_gamma_sanity_relative_error"].abs() > 0.10,
        ["row_id", "signal_datetime", "route", "entry_iv_mean", "calendar_days_to_expiry", "theta_gamma_sanity_lhs_annualized", "theta_gamma_sanity_rhs_diffusion", "theta_gamma_sanity_relative_error"],
    ].sort_values("theta_gamma_sanity_relative_error", key=lambda series: series.abs(), ascending=False)
    sanity_outliers.to_csv(OUT / "sanity_check_outliers_over_10pct.csv", index=False)

    continuous_correlations = []
    for column in ["calendar_days_to_expiry", "entry_iv_mean", "entry_minutes_after_0915"]:
        valid = analyzable[["gamma_theta_ratio", column]].dropna()
        p = pearsonr(valid["gamma_theta_ratio"], valid[column])
        s = spearmanr(valid["gamma_theta_ratio"], valid[column])
        continuous_correlations.append({
            "variable": column,
            "pearson_correlation": float(p.statistic),
            "pearson_p_value": float(p.pvalue),
            "spearman_correlation": float(s.statistic),
            "spearman_p_value": float(s.pvalue),
            "sample_size": int(len(valid)),
        })
    route_mean = analyzable.groupby("route")["gamma_theta_ratio"].mean()
    grand_mean = analyzable["gamma_theta_ratio"].mean()
    between = sum(len(group) * (group["gamma_theta_ratio"].mean() - grand_mean) ** 2 for _, group in analyzable.groupby("route"))
    total = float(((analyzable["gamma_theta_ratio"] - grand_mean) ** 2).sum())
    route_eta_squared = float(between / total) if total else math.nan
    route_rows = []
    for route in sorted(analyzable["route"].unique()):
        indicator = analyzable["route"].eq(route).astype(float)
        correlation = pearsonr(analyzable["gamma_theta_ratio"], indicator)
        route_rows.append({
            "route": route,
            "trades": int(indicator.sum()),
            "mean_ratio": float(analyzable.loc[indicator.astype(bool), "gamma_theta_ratio"].mean()),
            "median_ratio": float(analyzable.loc[indicator.astype(bool), "gamma_theta_ratio"].median()),
            "one_hot_pearson_correlation": float(correlation.statistic),
            "one_hot_p_value": float(correlation.pvalue),
            "overall_eta_squared": route_eta_squared,
        })
    pd.DataFrame(continuous_correlations).to_csv(OUT / "ratio_continuous_correlations.csv", index=False)
    pd.DataFrame(route_rows).to_csv(OUT / "ratio_route_correlations.csv", index=False)

    analyzable["ratio_tercile"] = pd.qcut(analyzable["gamma_theta_ratio"], 3, labels=["T1_low", "T2_mid", "T3_high"])
    analyzable["ratio_quintile"] = pd.qcut(analyzable["gamma_theta_ratio"], 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    terciles = bucket_table(analyzable, "ratio_tercile", eligible_dates)
    quintiles = bucket_table(analyzable, "ratio_quintile", eligible_dates)
    terciles.to_csv(OUT / "bucket_terciles.csv", index=False)
    quintiles.to_csv(OUT / "bucket_quintiles.csv", index=False)

    regression = analyzable[["exit_reason", "net_pnl_inr", "gamma_theta_ratio", "calendar_days_to_expiry", "entry_iv_mean", "entry_minutes_after_0915"]].dropna().copy()
    regression["stop_out"] = regression["exit_reason"].eq("stop_loss").astype(int)
    predictors = ["gamma_theta_ratio", "calendar_days_to_expiry", "entry_iv_mean", "entry_minutes_after_0915"]
    standardized_names = []
    for column in predictors:
        name = column + "_z"
        regression[name] = (regression[column] - regression[column].mean()) / regression[column].std(ddof=1)
        standardized_names.append(name)
    design = sm.add_constant(regression[standardized_names], has_constant="add")
    logit = sm.GLM(regression["stop_out"], design, family=sm.families.Binomial()).fit(cov_type="HC3")
    ols = sm.OLS(regression["net_pnl_inr"], design).fit(cov_type="HC3")
    regression_rows = []
    for model_name, model in [("stop_out_logit", logit), ("net_pnl_ols", ols)]:
        for name in model.params.index:
            regression_rows.append({
                "model": model_name,
                "term": name,
                "coefficient": float(model.params[name]),
                "robust_standard_error": float(model.bse[name]),
                "p_value": float(model.pvalues[name]),
                "odds_ratio": float(math.exp(model.params[name])) if model_name == "stop_out_logit" and name != "const" else None,
                "sample_size": int(model.nobs),
                "r_squared_or_pseudo": float(ols.rsquared) if model_name == "net_pnl_ols" else None,
            })
    regression_frame = pd.DataFrame(regression_rows)
    regression_frame.to_csv(OUT / "regression_results.csv", index=False)

    day_group = analyzable.groupby(["trade_date", "ratio_tercile"], observed=True)["net_pnl_inr"].agg(["sum", "count"]).reset_index()
    days = np.array(sorted(analyzable["trade_date"].unique()))
    day_lookup = {}
    for day in days:
        subset = day_group.loc[day_group["trade_date"].eq(day)].set_index("ratio_tercile")
        day_lookup[day] = {
            "low_sum": float(subset.loc["T1_low", "sum"]) if "T1_low" in subset.index else 0.0,
            "low_count": int(subset.loc["T1_low", "count"]) if "T1_low" in subset.index else 0,
            "high_sum": float(subset.loc["T3_high", "sum"]) if "T3_high" in subset.index else 0.0,
            "high_count": int(subset.loc["T3_high", "count"]) if "T3_high" in subset.index else 0,
        }
    low_sums = np.array([day_lookup[day]["low_sum"] for day in days])
    low_counts = np.array([day_lookup[day]["low_count"] for day in days])
    high_sums = np.array([day_lookup[day]["high_sum"] for day in days])
    high_counts = np.array([day_lookup[day]["high_count"] for day in days])
    observed_difference = float(high_sums.sum() / high_counts.sum() - low_sums.sum() / low_counts.sum())
    rng = np.random.default_rng(SEED)
    bootstrap_differences = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        selected = rng.integers(0, len(days), len(days))
        bootstrap_differences[draw] = (
            high_sums[selected].sum() / high_counts[selected].sum()
            - low_sums[selected].sum() / low_counts[selected].sum()
        )
    bootstrap_ci = np.percentile(bootstrap_differences, [2.5, 97.5])
    bootstrap_result = {
        "definition": "top tercile minus bottom tercile mean net P&L per trade; cluster bootstrap resamples complete trade days",
        "observed_difference_inr_per_trade": observed_difference,
        "ci_95_low": float(bootstrap_ci[0]),
        "ci_95_high": float(bootstrap_ci[1]),
        "draws": BOOTSTRAP_DRAWS,
        "days_resampled": int(len(days)),
        "ci_excludes_zero": bool(bootstrap_ci[0] > 0 or bootstrap_ci[1] < 0),
    }

    baseline_metrics = strategy_metrics(analyzable, eligible_dates)
    rule_a_rows = []
    configuration_sharpes = [baseline_metrics["sharpe_daily"]]
    for percentile in FILTER_PERCENTILES:
        cutoff = float(analyzable["gamma_theta_ratio"].quantile(percentile))
        kept = analyzable.loc[analyzable["gamma_theta_ratio"] <= cutoff]
        skipped = analyzable.loc[analyzable["gamma_theta_ratio"] > cutoff]
        result = strategy_metrics(kept, eligible_dates)
        configuration_sharpes.append(result["sharpe_daily"])
        rule_a_rows.append({
            "percentile_cutoff": percentile,
            "ratio_cutoff": cutoff,
            "trades_kept": int(len(kept)),
            "trades_skipped": int(len(skipped)),
            "net_pnl_inr": result["net_pnl_inr"],
            "sharpe_daily": result["sharpe_daily"],
            "skipped_trades_net_pnl_inr": float(skipped["net_pnl_inr"].sum()),
            "in_sample": True,
        })
    rule_a = pd.DataFrame(rule_a_rows)
    rule_a.to_csv(OUT / "filter_rule_a_percentiles.csv", index=False)

    median_ratio = float(analyzable["gamma_theta_ratio"].median())
    inverse_weights = np.minimum(1.0, median_ratio / analyzable["gamma_theta_ratio"].to_numpy(dtype=float))
    rule_b_metrics = strategy_metrics(analyzable, eligible_dates, weights=inverse_weights)
    configuration_sharpes.append(rule_b_metrics["sharpe_daily"])
    rule_b = pd.DataFrame([{
        "rule": "fractional_lot_weight=min(1, median_ratio/trade_ratio)",
        "median_ratio": median_ratio,
        **rule_b_metrics,
        "minimum_weight": float(inverse_weights.min()),
        "median_weight": float(np.median(inverse_weights)),
        "in_sample": True,
        "execution_note": "Fractional-lot research sizing; not directly executable with integer lots.",
    }])
    rule_b.to_csv(OUT / "sizing_rule_b_inverse_ratio.csv", index=False)

    analyzable["month"] = analyzable["trade_date"].dt.to_period("M").astype(str)
    months = sorted(analyzable["month"].unique())
    lomo_rows = []
    lomo_heldout_parts = []
    lomo_training_trials = 0
    for heldout_month in months:
        train = analyzable.loc[analyzable["month"].ne(heldout_month)]
        test = analyzable.loc[analyzable["month"].eq(heldout_month)]
        train_dates = eligible_dates[eligible_dates.to_period("M").astype(str) != heldout_month]
        test_dates = eligible_dates[eligible_dates.to_period("M").astype(str) == heldout_month]
        candidates = []
        for percentile in FILTER_PERCENTILES:
            cutoff = float(train["gamma_theta_ratio"].quantile(percentile))
            train_kept = train.loc[train["gamma_theta_ratio"] <= cutoff]
            train_result = strategy_metrics(train_kept, train_dates)
            configuration_sharpes.append(train_result["sharpe_daily"])
            lomo_training_trials += 1
            candidates.append((train_result["sharpe_daily"], percentile, cutoff))
        _, selected_percentile, selected_cutoff = max(candidates, key=lambda item: item[0])
        test_kept = test.loc[test["gamma_theta_ratio"] <= selected_cutoff].copy()
        test_kept["selected_percentile"] = selected_percentile
        lomo_heldout_parts.append(test_kept)
        baseline_test = strategy_metrics(test, test_dates)
        filtered_test = strategy_metrics(test_kept, test_dates)
        lomo_rows.append({
            "heldout_month": heldout_month,
            "selected_percentile_from_training": selected_percentile,
            "selected_ratio_cutoff": selected_cutoff,
            "heldout_trades_baseline": int(len(test)),
            "heldout_trades_kept": int(len(test_kept)),
            "heldout_baseline_net_pnl_inr": baseline_test["net_pnl_inr"],
            "heldout_filtered_net_pnl_inr": filtered_test["net_pnl_inr"],
            "heldout_baseline_sharpe": baseline_test["sharpe_daily"],
            "heldout_filtered_sharpe": filtered_test["sharpe_daily"],
        })
    lomo = pd.DataFrame(lomo_rows)
    lomo.to_csv(OUT / "leave_one_month_out_filter.csv", index=False)
    lomo_combined = pd.concat(lomo_heldout_parts, ignore_index=True).sort_values("signal_datetime", kind="stable")
    lomo_baseline_metrics = strategy_metrics(analyzable, eligible_dates)
    lomo_filtered_metrics = strategy_metrics(lomo_combined, eligible_dates)
    lomo_improves = lomo_filtered_metrics["sharpe_daily"] > lomo_baseline_metrics["sharpe_daily"]

    real_best_row = rule_a.loc[rule_a["sharpe_daily"].idxmax()]
    real_best_improvement = float(real_best_row["sharpe_daily"] - baseline_metrics["sharpe_daily"])
    placebo_best_improvements = np.empty(PLACEBO_DRAWS)
    placebo_best_net = np.empty(PLACEBO_DRAWS)
    grouped_indices = [group.index.to_numpy() for _, group in analyzable.groupby("trade_date")]
    ratios = analyzable["gamma_theta_ratio"].copy()
    for draw in range(PLACEBO_DRAWS):
        permuted = ratios.copy()
        for indices in grouped_indices:
            permuted.loc[indices] = rng.permutation(ratios.loc[indices].to_numpy())
        draw_results = []
        for percentile in FILTER_PERCENTILES:
            cutoff = float(permuted.quantile(percentile))
            kept = analyzable.loc[permuted <= cutoff]
            draw_results.append(strategy_metrics(kept, eligible_dates))
        best = max(draw_results, key=lambda result: result["sharpe_daily"])
        placebo_best_improvements[draw] = best["sharpe_daily"] - baseline_metrics["sharpe_daily"]
        placebo_best_net[draw] = best["net_pnl_inr"]
    placebo_p_value = float((1 + np.sum(placebo_best_improvements >= real_best_improvement)) / (PLACEBO_DRAWS + 1))
    placebo_result = {
        "method": "Permute gamma/theta ratio within each trade day; for each draw take the best Sharpe across 80th/90th/95th percentile Rule A filters.",
        "real_best_percentile": float(real_best_row["percentile_cutoff"]),
        "real_best_sharpe": float(real_best_row["sharpe_daily"]),
        "real_improvement_over_baseline": real_best_improvement,
        "real_net_pnl_inr": float(real_best_row["net_pnl_inr"]),
        "placebo_improvement_p05_p50_p95": np.percentile(placebo_best_improvements, [5, 50, 95]).tolist(),
        "real_result_percentile": float(np.mean(placebo_best_improvements <= real_best_improvement) * 100),
        "one_sided_p_value_multiple_cutoff_adjusted": placebo_p_value,
        "draws": PLACEBO_DRAWS,
    }

    price_lookup = prices.set_index(["datetime", "expiry", "strike"])[["call_price", "put_price"]]
    spot_lookup = spot.set_index("datetime")["spot_price"]
    book_rows = []
    book_failures = []
    for entry_time in sorted(analyzable["signal_datetime"].unique()):
        active = analyzable.loc[
            (analyzable["signal_datetime"] <= entry_time) & (analyzable["exit_datetime"] > entry_time)
        ]
        gamma_sum = 0.0
        theta_sum = 0.0
        computed = 0
        for position in active.itertuples():
            key = (pd.Timestamp(entry_time), pd.Timestamp(position.expiry), float(position.entry_strike))
            try:
                price_row = price_lookup.loc[key]
                if isinstance(price_row, pd.DataFrame):
                    price_row = price_row.iloc[-1]
                current_spot = float(spot_lookup.loc[pd.Timestamp(entry_time)])
                snapshot = compute_snapshot(
                    pd.Timestamp(entry_time), pd.Timestamp(position.expiry), float(position.entry_strike),
                    float(price_row.call_price), float(price_row.put_price), current_spot,
                )
                gamma_sum += snapshot["position_short_gamma_rupees_per_point2_per_lot"]
                theta_sum += snapshot["position_short_theta_rupees_per_calendar_day_per_lot"]
                computed += 1
            except (KeyError, ValueError, OverflowError, ZeroDivisionError) as error:
                book_failures.append({
                    "entry_time": entry_time, "position_row_id": int(position.row_id), "reason": str(error)
                })
        book_rows.append({
            "entry_time": entry_time,
            "trade_date": pd.Timestamp(entry_time).normalize(),
            "active_positions": int(len(active)),
            "computed_positions": computed,
            "book_short_gamma_rupees_per_point2_per_lot": gamma_sum,
            "book_short_theta_rupees_per_calendar_day_per_lot": theta_sum,
            "book_gamma_theta_ratio": abs(gamma_sum) / abs(theta_sum) if computed == len(active) and abs(theta_sum) > 1e-14 else np.nan,
        })
    book = pd.DataFrame(book_rows)
    book_failures_frame = pd.DataFrame(book_failures, columns=["entry_time", "position_row_id", "reason"])
    book_failures_frame.to_csv(OUT / "book_snapshot_failures.csv", index=False)
    book.to_csv(OUT / "book_entry_snapshots.csv", index=False)
    complete_book_snapshots = int(book["book_gamma_theta_ratio"].notna().sum())
    multi_book = book.loc[book["active_positions"].between(2, 3) & book["book_gamma_theta_ratio"].notna()].copy()
    daily_book = multi_book.groupby("trade_date").agg(
        max_book_gamma_theta_ratio=("book_gamma_theta_ratio", "max"),
        mean_book_gamma_theta_ratio=("book_gamma_theta_ratio", "mean"),
        entry_snapshots=("book_gamma_theta_ratio", "size"),
        maximum_active_positions=("active_positions", "max"),
    )
    daily_pnl = analyzable.groupby("trade_date")["net_pnl_inr"].sum()
    daily_book = daily_book.join(daily_pnl.rename("daily_net_pnl_inr"), how="left").reset_index()
    daily_book["tail_group"] = "rest"
    worst_days = daily_book.nsmallest(min(10, len(daily_book)), "daily_net_pnl_inr")["trade_date"]
    daily_book.loc[daily_book["trade_date"].isin(worst_days), "tail_group"] = "worst_10"
    daily_book.to_csv(OUT / "book_tail_days.csv", index=False)
    worst_values = daily_book.loc[daily_book["tail_group"].eq("worst_10"), "max_book_gamma_theta_ratio"]
    rest_values = daily_book.loc[daily_book["tail_group"].eq("rest"), "max_book_gamma_theta_ratio"]
    tail_test = mannwhitneyu(worst_values, rest_values, alternative="two-sided") if len(rest_values) else None
    book_tail_summary = {
        "days_with_2_or_3_positions": int(len(daily_book)),
        "worst_10_mean_max_book_ratio": float(worst_values.mean()),
        "rest_mean_max_book_ratio": float(rest_values.mean()) if len(rest_values) else None,
        "worst_10_median_max_book_ratio": float(worst_values.median()),
        "rest_median_max_book_ratio": float(rest_values.median()) if len(rest_values) else None,
        "mann_whitney_p_value": float(tail_test.pvalue) if tail_test is not None else None,
    }
    special = analyzable.loc[
        (analyzable["trade_date"].eq(pd.Timestamp("2024-09-23")))
        | (analyzable["trade_date"].dt.to_period("M").eq(pd.Period("2025-04"))),
        ["row_id", "signal_datetime", "exit_datetime", "trade_date", "route", "entry_strike", "expiry", "gamma_theta_ratio", "position_short_gamma_rupees_per_point2_per_lot", "position_short_theta_rupees_per_calendar_day_per_lot", "entry_iv_mean", "calendar_days_to_expiry", "net_pnl_inr", "exit_reason"],
    ]
    special.to_csv(OUT / "special_days_position_ratios.csv", index=False)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row_index, (title, table) in enumerate([("Terciles", terciles), ("Quintiles", quintiles)]):
        x = np.arange(len(table))
        axes[row_index, 0].bar(x, table["net_pnl_inr"], color=["#2E7D32" if value >= 0 else "#C62828" for value in table["net_pnl_inr"]])
        axes[row_index, 0].axhline(0, color="black", linewidth=0.8)
        axes[row_index, 0].set_xticks(x, table["bucket"], rotation=20)
        axes[row_index, 0].set_title(f"{title}: net P&L")
        axes[row_index, 0].set_ylabel("INR after costs")
        axes[row_index, 1].bar(x, table["stop_out_rate_pct"], color="#D97706")
        axes[row_index, 1].set_xticks(x, table["bucket"], rotation=20)
        axes[row_index, 1].set_title(f"{title}: stop-out rate")
        axes[row_index, 1].set_ylabel("Percent")
    figure.suptitle("Trade outcomes by entry gamma/theta ratio")
    figure.savefig(OUT / "bucket_pnl_stopout.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    n_configurations = 1 + len(FILTER_PERCENTILES) + 1 + lomo_training_trials
    trial_sharpes = np.array(configuration_sharpes, dtype=float) / math.sqrt(ANNUALIZATION_DAYS)
    trial_std = float(np.nanstd(trial_sharpes, ddof=1))
    gamma_constant = 0.5772156649015329
    expected_max_sr = trial_std * (
        (1 - gamma_constant) * norm.ppf(1 - 1 / n_configurations)
        + gamma_constant * norm.ppf(1 - 1 / (n_configurations * math.e))
    )
    daily_baseline = analyzable.groupby("trade_date")["net_pnl_inr"].sum().reindex(eligible_dates, fill_value=0.0).to_numpy(dtype=float)
    observed_sr = sharpe_from_daily(daily_baseline) / math.sqrt(ANNUALIZATION_DAYS)
    return_skew = float(skew(daily_baseline, bias=False))
    return_kurtosis = float(kurtosis(daily_baseline, fisher=False, bias=False))
    dsr_denominator = math.sqrt(max(1e-12, 1 - return_skew * observed_sr + ((return_kurtosis - 1) / 4) * observed_sr**2))
    dsr_probability = float(norm.cdf((observed_sr - expected_max_sr) * math.sqrt(len(daily_baseline) - 1) / dsr_denominator))
    deflated_sharpe = {
        "configurations_tried": n_configurations,
        "observed_annualized_sharpe": baseline_metrics["sharpe_daily"],
        "expected_max_annualized_sharpe": expected_max_sr * math.sqrt(ANNUALIZATION_DAYS),
        "deflated_sharpe_probability": dsr_probability,
    }

    bootstrap_pass = bool(bootstrap_result["ci_excludes_zero"])
    lomo_pass = bool(lomo_improves)
    placebo_pass = bool(placebo_p_value < 0.05)
    overall_predictive = bootstrap_pass and lomo_pass and placebo_pass
    summary_rows = [
        {"test": "Greek computation coverage", "result": f"{len(analyzable)}/{len(trades)} trades computed; {len(failures_frame)} failures", "pass_fail": "PASS" if len(analyzable) == len(trades) else "PARTIAL", "sample_size": len(trades)},
        {"test": "Book snapshot coverage", "result": f"{complete_book_snapshots}/{len(book)} entry-time books complete; {len(book_failures_frame)} missing position-minute prices across {len(book) - complete_book_snapshots} books", "pass_fail": "PARTIAL" if len(book_failures_frame) else "PASS", "sample_size": len(book)},
        {"test": "Theta/gamma consistency", "result": f"Pearson r={sanity_pearson.statistic:.4f}; >10% outliers={len(sanity_outliers)}", "pass_fail": "SANITY", "sample_size": len(sanity_valid)},
        {"test": "Top-bottom tercile bootstrap", "result": f"Difference ₹{observed_difference:.2f}; 95% CI [{bootstrap_ci[0]:.2f}, {bootstrap_ci[1]:.2f}]", "pass_fail": "PASS" if bootstrap_pass else "FAIL", "sample_size": len(analyzable)},
        {"test": "LOMO percentile filter", "result": f"Baseline Sharpe {lomo_baseline_metrics['sharpe_daily']:.3f}; held-out filtered {lomo_filtered_metrics['sharpe_daily']:.3f}", "pass_fail": "PASS" if lomo_pass else "FAIL", "sample_size": len(analyzable)},
        {"test": "Within-day ratio placebo", "result": f"Adjusted p={placebo_p_value:.4f}; real percentile {placebo_result['real_result_percentile']:.1f}%", "pass_fail": "PASS" if placebo_pass else "FAIL", "sample_size": PLACEBO_DRAWS},
        {"test": "Book-level tail comparison", "result": f"Worst/rest mean max ratio {book_tail_summary['worst_10_mean_max_book_ratio']:.6g}/{book_tail_summary['rest_mean_max_book_ratio']:.6g}; p={book_tail_summary['mann_whitney_p_value']:.4f}", "pass_fail": "DIAGNOSTIC", "sample_size": book_tail_summary["days_with_2_or_3_positions"]},
        {"test": "Deflated Sharpe", "result": f"Probability {dsr_probability:.4f}; N={n_configurations}", "pass_fail": "PASS" if dsr_probability > 0.95 else "FAIL", "sample_size": len(eligible_dates)},
        {"test": "Pre-declared overall criterion", "result": "predictive" if overall_predictive else "no evidence", "pass_fail": "PASS" if overall_predictive else "FAIL", "sample_size": len(analyzable)},
    ]
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(OUT / "one_page_summary.csv", index=False)

    assumptions = {
        "pricing_model": "Black-76 on synthetic forward derived from call-put parity F=K+exp(rT)*(C-P)",
        "risk_free_rate_continuously_compounded": RISK_FREE_RATE,
        "expiry_time_local": "15:30:00",
        "calendar_basis_days": YEAR_DAYS,
        "lot_size": LOT_SIZE,
        "option_style": "European approximation",
        "dividend_or_carry_handling": "Embedded in the synthetic forward; no separate dividend yield assumed",
        "gamma_unit": "rupees per one index-point squared per lot; standard gamma coefficient, before the 0.5 Taylor factor",
        "theta_unit": "rupees per calendar day per lot",
        "position": "short one call plus short one put at the fixed strike",
        "entry_iv": "simple mean of independently solved call and put Black-76 IVs",
        "book_snapshot": "At every executed entry timestamp, all positions still open are repriced from that minute's call/put prices and spot; exposures are summed after including the new entry",
        "sanity_check": "Because theta is reported daily, |theta/gamma| is multiplied by 365 before comparison with 0.5*IV^2*F^2; rate terms cause expected residual error",
        "rule_b": "Fractional research weights min(1, median ratio / trade ratio); this is not integer-lot executable",
    }
    limitations = [
        "Option and spot observations are one-minute values, not ticks.",
        "No historical bid/ask quotes are available, so IV and Greeks use recorded option prices without spread modelling.",
        "Black-76 uses a synthetic forward derived from same-strike call-put parity because a matched futures series was not supplied.",
        "Some tercile/quintile, route, month and tail-day subsets are small; their Sharpe and stop rates are unstable.",
        "Rule A and Rule B full-sample results are in-sample. The leave-one-month-out filter is the less-biased filter estimate.",
        "Greeks are model estimates and do not include discrete strike-surface interpolation or intraminute movement.",
        f"Book-level analysis stops rather than imputes at {len(book) - complete_book_snapshots} entry timestamps ({len(book_failures_frame)} position-minute prices missing); those book ratios are blank and excluded.",
    ]
    payload = {
        "verdict": "predictive" if overall_predictive else "no evidence",
        "pass_criteria": {
            "bootstrap_ci_excludes_zero": bootstrap_pass,
            "leave_one_month_out_filter_improves_heldout_sharpe": lomo_pass,
            "placebo_p_value_below_0_05": placebo_pass,
            "all_three_required": True,
        },
        "coverage": {"input_trades": len(trades), "analyzable_trades": len(analyzable), "failed_trades": len(failures_frame)},
        "assumptions": assumptions,
        "sanity": {
            "pearson_correlation": float(sanity_pearson.statistic),
            "pearson_p_value": float(sanity_pearson.pvalue),
            "spearman_correlation": float(sanity_spearman.statistic),
            "outliers_over_10pct": len(sanity_outliers),
        },
        "continuous_correlations": continuous_correlations,
        "route_eta_squared": route_eta_squared,
        "bootstrap_top_minus_bottom_tercile": bootstrap_result,
        "baseline": baseline_metrics,
        "rule_a": rule_a_rows,
        "rule_b": rule_b.iloc[0].to_dict(),
        "lomo": {"baseline": lomo_baseline_metrics, "filtered": lomo_filtered_metrics, "improves_sharpe": lomo_improves},
        "placebo": placebo_result,
        "book_tail": book_tail_summary,
        "book_snapshot_coverage": {
            "entry_timestamps": int(len(book)),
            "complete_entry_timestamps": complete_book_snapshots,
            "incomplete_entry_timestamps": int(len(book) - complete_book_snapshots),
            "missing_position_minute_prices": int(len(book_failures_frame)),
        },
        "deflated_sharpe": deflated_sharpe,
        "limitations": limitations,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUT / "assumptions_and_limitations.json").write_text(json.dumps({"assumptions": assumptions, "limitations": limitations}, indent=2), encoding="utf-8")

    hashes_after = live_hashes()
    integrity = {
        "unchanged": hashes_before == hashes_after,
        "files_checked": len(hashes_before),
        "before": hashes_before,
        "after": hashes_after,
    }
    (OUT / "live_hash_verification.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"],
        "coverage": payload["coverage"],
        "pass_criteria": payload["pass_criteria"],
        "bootstrap": bootstrap_result,
        "lomo": payload["lomo"],
        "placebo": placebo_result,
        "live_files_unchanged": integrity["unchanged"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
