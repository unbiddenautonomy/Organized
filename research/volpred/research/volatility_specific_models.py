from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atm_iv_rv_analysis import prepare_spot
from intraday_structure_attribution import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
BASE_PANEL = ROOT / "outputs" / "spot_magnitude_models" / "spot_magnitude_feature_panel.parquet"
SPOT_FILE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
OUT = ROOT / "outputs" / "volatility_specific_models"
HORIZONS = (15, 30, 60, 120)
EPS = 1e-12
TRAIN_DAYS = 60

FEATURE_SETS = {
    "HAR": ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60"],
    "HARQ": ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60", "log_rq_5", "log_rq_15", "log_rq_60"],
    "HAR_RS": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_rs_up_5", "log_rs_down_5", "log_rs_up_15", "log_rs_down_15",
        "log_rs_up_60", "log_rs_down_60",
    ],
    "HAR_J": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_bv_5", "log_jump_5", "log_bv_15", "log_jump_15", "log_bv_60", "log_jump_60",
    ],
    "HAR_RS_J": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_rs_up_5", "log_rs_down_5", "log_rs_up_15", "log_rs_down_15",
        "log_rs_up_60", "log_rs_down_60", "log_jump_5", "log_jump_15", "log_jump_60",
    ],
    "HEAVY_style": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_current_r2", "log_ewma_r2", "return_past_5", "abs_return_past_5",
    ],
    "Realized_GARCH_style": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_current_r2", "log_ewma_r2", "log_iv_trade",
    ],
    "MIDAS_style": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_daily_rv_lag1", "log_midas_5", "log_midas_10", "log_midas_20",
    ],
}


def map_valid_metric(values: pd.Series, valid_positions: np.ndarray, n_rows: int) -> np.ndarray:
    holder = pd.Series(np.nan, index=np.arange(n_rows), dtype=float)
    holder.iloc[valid_positions] = values.to_numpy()
    return holder.ffill().to_numpy()


def volatility_features() -> pd.DataFrame:
    spot = prepare_spot(pd.read_csv(SPOT_FILE, parse_dates=["datetime", "Date"]))
    valid = spot["return"].notna().to_numpy()
    pos = np.where(valid)[0]
    r = spot.loc[valid, "return"].reset_index(drop=True)
    extra = spot[["datetime", "date"]].copy()
    for h in (5, 15, 60):
        rv = r.pow(2).rolling(h, min_periods=h).sum()
        rq = (h / 3.0) * r.pow(4).rolling(h, min_periods=h).sum()
        up = r.pow(2).where(r > 0, 0.0).rolling(h, min_periods=h).sum()
        down = r.pow(2).where(r < 0, 0.0).rolling(h, min_periods=h).sum()
        products = r.abs() * r.abs().shift(1)
        bv = (np.pi / 2.0) * products.rolling(max(h - 1, 1), min_periods=max(h - 1, 1)).sum()
        jump = (rv - bv).clip(lower=EPS)
        for name, series in [("rq", rq), ("rs_up", up), ("rs_down", down), ("bv", bv), ("jump", jump)]:
            extra[f"log_{name}_{h}"] = np.log(np.clip(map_valid_metric(series, pos, len(spot)), EPS, None))

    current_r2 = r.pow(2)
    ewma = current_r2.ewm(alpha=0.06, adjust=False).mean()
    extra["log_current_r2"] = np.log(np.clip(map_valid_metric(current_r2, pos, len(spot)), EPS, None))
    extra["log_ewma_r2"] = np.log(np.clip(map_valid_metric(ewma, pos, len(spot)), EPS, None))

    daily = spot.groupby("date", sort=False)["return"].apply(lambda x: np.nansum(x.to_numpy(float) ** 2)).rename("daily_rv").reset_index()
    daily["daily_rv_lag1"] = daily["daily_rv"].shift(1)
    for window in (5, 10, 20):
        weights = np.exp(-0.15 * np.arange(window))
        weights = weights / weights.sum()
        lagged = daily["daily_rv"].shift(1)
        daily[f"midas_{window}"] = lagged.rolling(window, min_periods=window).apply(
            lambda x, w=weights: float(np.dot(x[::-1], w)), raw=True
        )
    for c in ["daily_rv_lag1", "midas_5", "midas_10", "midas_20"]:
        daily[f"log_{c}"] = np.log(daily[c].clip(lower=EPS))
    extra = extra.merge(
        daily[["date", "log_daily_rv_lag1", "log_midas_5", "log_midas_10", "log_midas_20"]],
        on="date", how="left", validate="many_to_one",
    )
    return extra.drop(columns="date")


def qlike_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    ratio = np.clip(y, EPS, None) / np.clip(p, EPS, None)
    return ratio - np.log(ratio) - 1


def fit_model(x: pd.DataFrame, y: np.ndarray, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    log_y = np.log(np.clip(y, EPS, None))
    model.fit(x, log_y)
    correction = float(0.5 * np.var(log_y - model.predict(x), ddof=1))
    return model, correction


def predict(model, correction: float, x: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(x) + correction)


def bootstrap_loss_difference(frame: pd.DataFrame, model: str, reps: int = 1000) -> tuple[float, float, float]:
    y = frame["actual_var"].to_numpy()
    daily = pd.DataFrame({
        "date": frame["date"],
        "difference": qlike_vector(y, frame[model].to_numpy()) - qlike_vector(y, frame["HAR"].to_numpy()),
    }).groupby("date")["difference"].mean().to_numpy()
    rng = np.random.default_rng(20261008)
    boot = np.array([rng.choice(daily, len(daily), replace=True).mean() for _ in range(reps)])
    return float(daily.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(BASE_PANEL)]).fetchdf()
    feature_file = OUT / "volatility_feature_extensions.parquet"
    if feature_file.exists():
        extra = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(feature_file)]).fetchdf()
    else:
        extra = volatility_features()
        con = duckdb.connect(); con.register("extra_df", extra)
        target = str(feature_file).replace("'", "''")
        con.execute(f"COPY extra_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"); con.close()
    panel = base.merge(extra, on="datetime", how="left", validate="one_to_one")

    all_features = sorted(set(sum(FEATURE_SETS.values(), [])))
    comparison_rows, tuning_rows, prediction_frames, manifest_rows = [], [], [], []
    best_params: dict[tuple[int, str], float] = {}

    for h in HORIZONS:
        required = all_features + [f"future_var_{h}", f"rv_past_{h}", f"iv_only_var_{h}"]
        data = panel[panel["quality_ok"].fillna(False) & panel[f"future_var_{h}"].gt(0)].copy()
        data = data[np.isfinite(data[required]).all(axis=1)]
        dev, val = data[data["year"].le(2022)], data[data["year"].eq(2023)]
        for name, features in FEATURE_SETS.items():
            best_alpha, best_loss = None, np.inf
            for alpha in (0.1, 1.0, 10.0, 100.0):
                model, correction = fit_model(dev[features], dev[f"future_var_{h}"].to_numpy(), alpha)
                loss = qlike_vector(val[f"future_var_{h}"].to_numpy(), predict(model, correction, val[features])).mean()
                tuning_rows.append({"horizon_min": h, "model": name, "alpha": alpha, "validation_qlike": loss})
                if loss < best_loss:
                    best_alpha, best_loss = alpha, loss
            best_params[(h, name)] = best_alpha

        evaluation = data[data["year"].ge(2024)].copy()
        month_key = evaluation["datetime"].dt.to_period("M")
        monthly = []
        for month in month_key.drop_duplicates().sort_values():
            test = evaluation[month_key.eq(month)].copy()
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)]
            if len(train) < 1500 or test.empty:
                continue
            pred = test[["datetime", "date", "spot_price", f"future_var_{h}"]].copy()
            pred = pred.rename(columns={f"future_var_{h}": "actual_var"})
            pred["horizon_min"] = h
            pred["random_walk"] = test[f"rv_past_{h}"].to_numpy()
            pred["iv_only"] = test[f"iv_only_var_{h}"].to_numpy()
            for name, features in FEATURE_SETS.items():
                model, correction = fit_model(train[features], train[f"future_var_{h}"].to_numpy(), best_params[(h, name)])
                pred[name] = predict(model, correction, test[features])
            monthly.append(pred)
        pred_h = pd.concat(monthly, ignore_index=True)
        prediction_frames.append(pred_h)
        models = ["random_walk", "iv_only", *FEATURE_SETS]
        har_loss = qlike_vector(pred_h["actual_var"].to_numpy(), pred_h["HAR"].to_numpy()).mean()
        for name in models:
            y, p = pred_h["actual_var"].to_numpy(), pred_h[name].to_numpy()
            log_y, log_p = np.log(y), np.log(p)
            diff, ci_low, ci_high = bootstrap_loss_difference(pred_h, name) if name not in ("random_walk", "iv_only", "HAR") else (np.nan, np.nan, np.nan)
            comparison_rows.append({
                "horizon_min": h, "model": name, "n": len(y), "days": pred_h["date"].nunique(),
                "qlike": qlike_vector(y, p).mean(),
                "qlike_improvement_vs_har_pct": 100 * (har_loss - qlike_vector(y, p).mean()) / har_loss,
                "log_r2": 1 - np.sum((log_y - log_p) ** 2) / np.sum((log_y - log_y.mean()) ** 2),
                "mean_daily_qlike_difference_vs_har": diff,
                "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
            })

        extension_names = [name for name in FEATURE_SETS if name != "HAR"]
        losses = {name: qlike_vector(pred_h["actual_var"].to_numpy(), pred_h[name].to_numpy()).mean() for name in extension_names}
        best_extension = min(losses, key=losses.get)
        latest_days = data["date"].drop_duplicates().sort_values().iloc[-TRAIN_DAYS:]
        final_train = data[data["date"].isin(latest_days)]
        features = FEATURE_SETS[best_extension]
        final_model, correction = fit_model(final_train[features], final_train[f"future_var_{h}"].to_numpy(), best_params[(h, best_extension)])
        model_file = OUT / f"volatility_extension_{h}m_model.joblib"
        joblib.dump({
            "model": final_model, "features": features, "log_bias_correction": correction,
            "horizon_min": h, "model_name": best_extension,
        }, model_file)
        best_row = next(r for r in comparison_rows if r["horizon_min"] == h and r["model"] == best_extension)
        manifest_rows.append({
            "horizon_min": h, "best_extension": best_extension, "model_file": model_file.name,
            "alpha": best_params[(h, best_extension)], "training_rows": len(final_train),
            "train_start": final_train["date"].min(), "train_end": final_train["date"].max(),
            "qlike_improvement_vs_har_pct": best_row["qlike_improvement_vs_har_pct"],
            "promote_over_har": bool(best_row["qlike_improvement_vs_har_pct"] > 0 and best_row["bootstrap_ci_high"] < 0),
        })

    comparison = pd.DataFrame(comparison_rows)
    manifest = pd.DataFrame(manifest_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "validation_tuning.csv", index=False)
    manifest.to_csv(OUT / "model_manifest.csv", index=False)
    feasibility = pd.DataFrame([
        {"model": "HARQ", "status": "implemented", "definition": "HAR plus realized quarticity at 5/15/60 minutes"},
        {"model": "HAR_RS", "status": "implemented", "definition": "HAR plus positive and negative realized semivariance"},
        {"model": "HAR_J", "status": "implemented", "definition": "HAR plus bipower variation and positive jump variation"},
        {"model": "HEAVY", "status": "adapted", "definition": "HEAVY-style linear log-variance model using return shocks and realized measures; not full QML HEAVY"},
        {"model": "Realized GARCH", "status": "adapted", "definition": "Realized-GARCH-style log model using EWMA return variance, RV and IV; not full joint MLE"},
        {"model": "MIDAS", "status": "adapted", "definition": "Intraday HAR plus lagged daily RV and exponentially weighted 5/10/20-session components"},
        {"model": "GARCH-MIDAS", "status": "not run", "definition": "No causal macro series was supplied; no macro values were fabricated"},
    ])
    feasibility.to_csv(OUT / "model_feasibility.csv", index=False)
    con = duckdb.connect(); con.register("pred_df", predictions)
    target = str(OUT / "walk_forward_predictions.parquet").replace("'", "''")
    con.execute(f"COPY pred_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"); con.close()

    order = ["HAR", "HARQ", "HAR_RS", "HAR_J", "HAR_RS_J", "HEAVY_style", "Realized_GARCH_style", "MIDAS_style"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True, gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(HORIZONS)); width = .1
    core_order = [name for name in order if name != "MIDAS_style"]
    for i, name in enumerate(core_order):
        z = comparison[comparison["model"].eq(name)].set_index("horizon_min").reindex(HORIZONS)
        axes[0].bar(x + (i - 3) * width, z["qlike_improvement_vs_har_pct"], width, label=name)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_xticks(x, [str(h) for h in HORIZONS])
    axes[0].set(title="Realized-measure extensions", xlabel="Horizon (minutes)", ylabel="QLIKE improvement vs HAR (%)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    midas = comparison[comparison["model"].eq("MIDAS_style")].set_index("horizon_min").reindex(HORIZONS)
    axes[1].bar(x, midas["qlike_improvement_vs_har_pct"], color="#888888")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set_xticks(x, [str(h) for h in HORIZONS])
    axes[1].set(title="MIDAS-style", xlabel="Horizon (minutes)", ylabel="Improvement (%)")
    fig.savefig(OUT / "qlike_improvement.png", dpi=180); plt.close(fig)

    best_table = comparison[comparison["model"].isin(order)].sort_values(["horizon_min", "qlike"]).groupby("horizon_min").head(3)
    report = f"""# Volatility-specific model comparison

## Decision

The models were evaluated on identical timestamps with a 60-session rolling window and monthly refits. Every linear specification uses a StandardScaler fitted only on its training window. QLIKE is the primary loss.

{dataframe_to_markdown(manifest.round(6))}

`promote_over_har` requires both positive QLIKE improvement and a day-block bootstrap confidence interval entirely below zero. A saved extension with `promote_over_har = false` is a research artifact, not a replacement for HAR.

## Three best specifications per horizon

{dataframe_to_markdown(best_table[["horizon_min", "model", "qlike", "qlike_improvement_vs_har_pct", "log_r2", "bootstrap_ci_low", "bootstrap_ci_high"]].round(6))}

## Scope limitations

- HARQ, HAR-RS and HAR-J are direct realized-measure extensions.
- HEAVY and Realized GARCH are regression adaptations, not full joint-QML estimators.
- MIDAS uses lagged daily realized variance because daily and intraday data are available.
- GARCH-MIDAS was not run because no causal macro series was supplied.
- The 2024-2025 period is design evidence already examined in earlier work, not an untouched performance claim.
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")
    print(manifest.to_string(index=False))
    print(best_table[["horizon_min", "model", "qlike", "qlike_improvement_vs_har_pct", "log_r2", "bootstrap_ci_low", "bootstrap_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
