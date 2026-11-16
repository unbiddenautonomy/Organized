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

import intraday_structure_attribution as isa


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rv_prediction_models"
BASE = ROOT / "outputs" / "spot_magnitude_models" / "spot_magnitude_feature_panel.parquet"
VOL_FEATURES = ROOT / "outputs" / "volatility_specific_models" / "volatility_feature_extensions.parquet"
HORIZONS = (15, 30, 60, 120)
EPS = 1e-12

MODEL_FEATURES = {
    "HAR": ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60"],
    "Realized_GARCH_core": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_current_r2", "log_ewma_r2", "log_iv_trade",
    ],
    "Realized_GARCH_extended": [
        "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
        "log_current_r2", "log_ewma_r2", "log_iv_trade", "log_rq_15",
        "iv_change_5", "smile_skew", "smile_butterfly", "rv_iv_gap_15",
        "time_sin", "time_cos", "days_to_expiry", "oi_ratio", "oi_imbalance",
    ],
}

OPEN_FEATURES = {
    "opening_previous_rv": [
        "prevday_log_rv_5", "prevday_log_rv_15", "prevday_log_rv_60", "overnight_return",
    ],
    "opening_extended": [
        "prevday_log_rv_5", "prevday_log_rv_15", "prevday_log_rv_60", "overnight_return",
        "log_iv_trade", "smile_skew", "smile_butterfly", "days_to_expiry",
        "oi_ratio", "oi_imbalance", "forward_log_basis",
    ],
}


def smile_panel() -> pd.DataFrame:
    cache = OUT / "five_strike_smile_features.parquet"
    if cache.exists():
        return duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(cache)]).fetchdf()
    entries, _ = isa.prepare_entry_rows()
    value = np.where(
        entries["strike_position"].lt(0), entries["iv_put"],
        np.where(entries["strike_position"].gt(0), entries["iv_call"], entries[["iv_call", "iv_put"]].mean(axis=1)),
    )
    temp = entries[["datetime", "strike_position"]].copy()
    temp["smile_iv"] = value
    wide = temp.pivot(index="datetime", columns="strike_position", values="smile_iv")
    wide["smile_skew"] = wide[2] - wide[-2]
    wide["smile_butterfly"] = wide[2] + wide[-2] - 2 * wide[0]
    result = wide[["smile_skew", "smile_butterfly"]].reset_index()
    con = duckdb.connect(); con.register("smile_df", result)
    target = str(cache).replace("'", "''")
    con.execute(f"COPY smile_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"); con.close()
    return result


def qlike_vector(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ratio = np.clip(y, EPS, None) / np.clip(pred, EPS, None)
    return ratio - np.log(ratio) - 1


def fit_model(x: pd.DataFrame, y_var: np.ndarray, alpha: float):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    log_y = np.log(np.clip(y_var, EPS, None))
    model.fit(x, log_y)
    correction = float(0.5 * np.var(log_y - model.predict(x), ddof=1))
    return model, correction


def predict_var(model, correction: float, x: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(x) + correction)


def clipped_pair(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    low, high = train.quantile(0.001), train.quantile(0.999)
    return train.clip(lower=low, upper=high, axis=1), test.clip(lower=low, upper=high, axis=1), low, high


def tune_alpha(dev: pd.DataFrame, val: pd.DataFrame, features: list[str], target: str) -> tuple[float, list[dict]]:
    rows, best_alpha, best_loss = [], None, np.inf
    dev_x, val_x, _, _ = clipped_pair(dev[features], val[features])
    for alpha in (0.1, 1.0, 10.0, 100.0, 300.0):
        model, correction = fit_model(dev_x, dev[target].to_numpy(), alpha)
        loss = qlike_vector(val[target].to_numpy(), predict_var(model, correction, val_x)).mean()
        rows.append({"alpha": alpha, "validation_qlike": loss})
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    return best_alpha, rows


def daily_walk_forward(data: pd.DataFrame, h: int, tuned: dict[str, float]) -> pd.DataFrame:
    evaluation = data[data["year"].ge(2024)].copy()
    frames = []
    for day in evaluation["date"].drop_duplicates().sort_values():
        test = evaluation[evaluation["date"].eq(day)].copy()
        prior = data.loc[data["date"].lt(day), "date"].drop_duplicates().sort_values()
        train_days = prior.iloc[-60:]
        train = data[data["date"].isin(train_days)]
        if len(train) < 1500:
            continue
        pred = test[["datetime", "date", "spot_price", f"future_var_{h}", f"future_abs_move_{h}"]].copy()
        pred = pred.rename(columns={f"future_var_{h}": "actual_var", f"future_abs_move_{h}": "actual_abs_move_points"})
        pred["horizon_min"] = h
        pred["random_walk"] = test[f"rv_past_{h}"].to_numpy()
        pred["iv_only"] = test[f"iv_only_var_{h}"].to_numpy()
        for name, features in MODEL_FEATURES.items():
            train_x, test_x, _, _ = clipped_pair(train[features], test[features])
            model, correction = fit_model(train_x, train[f"future_var_{h}"].to_numpy(), tuned[name])
            pred[name] = predict_var(model, correction, test_x)
        frames.append(pred)
    return pd.concat(frames, ignore_index=True)


def bootstrap_diff(frame: pd.DataFrame, model: str, baseline: str = "HAR", reps: int = 1000):
    y = frame["actual_var"].to_numpy()
    daily = pd.DataFrame({
        "date": frame["date"],
        "diff": qlike_vector(y, frame[model].to_numpy()) - qlike_vector(y, frame[baseline].to_numpy()),
    }).groupby("date")["diff"].mean().to_numpy()
    rng = np.random.default_rng(20261008)
    sample = np.array([rng.choice(daily, len(daily), replace=True).mean() for _ in range(reps)])
    return float(daily.mean()), float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    base = con.execute("SELECT * FROM read_parquet(?)", [str(BASE)]).fetchdf()
    vol = con.execute("SELECT * FROM read_parquet(?)", [str(VOL_FEATURES)]).fetchdf()
    con.close()
    panel = base.merge(vol, on="datetime", how="left", validate="one_to_one").merge(smile_panel(), on="datetime", how="left", validate="one_to_one")
    panel["rv_iv_gap_15"] = panel["log_rv_past_15"] - np.log(panel["iv_only_var_15"].clip(lower=EPS))

    all_features = sorted(set(sum(MODEL_FEATURES.values(), []) + sum(OPEN_FEATURES.values(), [])))
    metrics_rows, tuning_rows, predictions, manifest_rows = [], [], [], []
    for h in HORIZONS:
        required = all_features + [f"future_var_{h}", f"future_abs_move_{h}", f"rv_past_{h}", f"iv_only_var_{h}"]
        data = panel[panel["quality_ok"].fillna(False) & panel[f"future_var_{h}"].gt(0)].copy()
        data = data[np.isfinite(data[required]).all(axis=1)]
        dev, val = data[data["year"].le(2022)], data[data["year"].eq(2023)]
        tuned = {}
        for name, features in MODEL_FEATURES.items():
            alpha, rows = tune_alpha(dev, val, features, f"future_var_{h}")
            tuned[name] = alpha
            tuning_rows.extend({"horizon_min": h, "model": name, **row} for row in rows)
        pred = daily_walk_forward(data, h, tuned)
        predictions.append(pred)
        har_loss = qlike_vector(pred["actual_var"].to_numpy(), pred["HAR"].to_numpy()).mean()
        for name in ["random_walk", "iv_only", *MODEL_FEATURES]:
            y, p = pred["actual_var"].to_numpy(), pred[name].to_numpy()
            log_y, log_p = np.log(y), np.log(p)
            diff, low, high = bootstrap_diff(pred, name) if name not in ("HAR", "random_walk", "iv_only") else (np.nan, np.nan, np.nan)
            annual_y, annual_p = np.sqrt(y * 252 * 375 / h), np.sqrt(p * 252 * 375 / h)
            metrics_rows.append({
                "horizon_min": h, "model": name, "n": len(y), "days": pred["date"].nunique(),
                "qlike": qlike_vector(y, p).mean(), "qlike_improvement_vs_har_pct": 100 * (har_loss - qlike_vector(y, p).mean()) / har_loss,
                "log_r2": 1 - np.sum((log_y - log_p) ** 2) / np.sum((log_y - log_y.mean()) ** 2),
                "mae_annualized_rv": np.mean(np.abs(annual_y - annual_p)),
                "mean_daily_loss_diff_vs_har": diff, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            })
        candidate_losses = {name: qlike_vector(pred["actual_var"].to_numpy(), pred[name].to_numpy()).mean() for name in MODEL_FEATURES if name != "HAR"}
        winner = min(candidate_losses, key=candidate_losses.get)
        latest_days = data["date"].drop_duplicates().sort_values().iloc[-60:]
        final_train = data[data["date"].isin(latest_days)]
        final_x, _, final_low, final_high = clipped_pair(final_train[MODEL_FEATURES[winner]], final_train[MODEL_FEATURES[winner]])
        model, correction = fit_model(final_x, final_train[f"future_var_{h}"].to_numpy(), tuned[winner])
        model_file = OUT / f"rv_intraday_{h}m_model.joblib"
        joblib.dump({
            "model": model, "features": MODEL_FEATURES[winner], "log_bias_correction": correction,
            "horizon_min": h, "model_name": winner, "clip_lower": final_low.to_dict(), "clip_upper": final_high.to_dict(),
        }, model_file)
        winner_row = next(r for r in metrics_rows if r["horizon_min"] == h and r["model"] == winner)
        manifest_rows.append({
            "model_scope": "intraday", "horizon_min": h, "model_name": winner, "model_file": model_file.name,
            "alpha": tuned[winner], "train_start": final_train["date"].min(), "train_end": final_train["date"].max(),
            "training_rows": len(final_train), "qlike_improvement_vs_har_pct": winner_row["qlike_improvement_vs_har_pct"],
        })

    combined_pred = pd.concat(predictions, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)

    opening_predictions, opening_metrics, opening_manifest = [], [], []
    for h in HORIZONS:
        required = sorted(set(sum(OPEN_FEATURES.values(), []))) + [f"future_var_{h}", f"future_abs_move_{h}"]
        opening = panel[panel["is_0915"].eq(1) & panel[f"future_var_{h}"].gt(0)].copy()
        opening = opening[np.isfinite(opening[required]).all(axis=1)]
        dev, val = opening[opening["year"].le(2022)], opening[opening["year"].eq(2023)]
        tuned, validation_loss = {}, {}
        for name, features in OPEN_FEATURES.items():
            alpha, rows = tune_alpha(dev, val, features, f"future_var_{h}")
            tuned[name] = alpha
            validation_loss[name] = min(r["validation_qlike"] for r in rows)
            tuning_rows.extend({"horizon_min": h, "model": name, **row} for row in rows)
        evaluation = opening[opening["year"].ge(2024)].copy()
        day_frames = []
        for day in evaluation["date"].drop_duplicates().sort_values():
            test = evaluation[evaluation["date"].eq(day)]
            train = opening[opening["date"].lt(day)].tail(400)
            pred = test[["datetime", "date", "spot_price", f"future_var_{h}", f"future_abs_move_{h}"]].copy()
            pred = pred.rename(columns={f"future_var_{h}": "actual_var", f"future_abs_move_{h}": "actual_abs_move_points"})
            pred["horizon_min"] = h
            for name, features in OPEN_FEATURES.items():
                train_x, test_x, _, _ = clipped_pair(train[features], test[features])
                model, correction = fit_model(train_x, train[f"future_var_{h}"].to_numpy(), tuned[name])
                pred[name] = predict_var(model, correction, test_x)
            day_frames.append(pred)
        pred = pd.concat(day_frames, ignore_index=True)
        opening_predictions.append(pred)
        losses = {}
        for name in OPEN_FEATURES:
            y, p = pred["actual_var"].to_numpy(), pred[name].to_numpy()
            losses[name] = qlike_vector(y, p).mean()
            log_y, log_p = np.log(y), np.log(p)
            opening_metrics.append({
                "horizon_min": h, "model": name, "n": len(y), "qlike": losses[name],
                "log_r2": 1 - np.sum((log_y - log_p) ** 2) / np.sum((log_y - log_y.mean()) ** 2),
                "mae_expected_abs_move_points": np.mean(np.abs(pred["actual_abs_move_points"] - np.sqrt(2 / np.pi) * pred["spot_price"] * np.sqrt(p))),
            })
        winner = min(losses, key=losses.get)
        final_train = opening.tail(400)
        final_x, _, final_low, final_high = clipped_pair(final_train[OPEN_FEATURES[winner]], final_train[OPEN_FEATURES[winner]])
        model, correction = fit_model(final_x, final_train[f"future_var_{h}"].to_numpy(), tuned[winner])
        model_file = OUT / f"rv_0915_{h}m_model.joblib"
        joblib.dump({
            "model": model, "features": OPEN_FEATURES[winner], "log_bias_correction": correction,
            "horizon_min": h, "model_name": winner, "clip_lower": final_low.to_dict(), "clip_upper": final_high.to_dict(),
        }, model_file)
        opening_manifest.append({
            "model_scope": "09:15", "horizon_min": h, "model_name": winner, "model_file": model_file.name,
            "alpha": tuned[winner], "train_start": final_train["date"].min(), "train_end": final_train["date"].max(),
            "training_rows": len(final_train), "evaluation_qlike": losses[winner],
        })

    opening_pred = pd.concat(opening_predictions, ignore_index=True)
    open_metrics = pd.DataFrame(opening_metrics)
    manifest = pd.concat([pd.DataFrame(manifest_rows), pd.DataFrame(opening_manifest)], ignore_index=True)
    metrics.to_csv(OUT / "intraday_model_metrics.csv", index=False)
    open_metrics.to_csv(OUT / "opening_0915_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "validation_tuning.csv", index=False)
    manifest.to_csv(OUT / "model_manifest.csv", index=False)

    con = duckdb.connect(); con.register("pred_df", combined_pred); con.register("open_df", opening_pred)
    target = str(OUT / "intraday_walk_forward_predictions.parquet").replace("'", "''")
    con.execute(f"COPY pred_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    target = str(OUT / "opening_0915_predictions.parquet").replace("'", "''")
    con.execute(f"COPY open_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"); con.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in ["HAR", "Realized_GARCH_core", "Realized_GARCH_extended"]:
        z = metrics[metrics["model"].eq(name)].sort_values("horizon_min")
        axes[0].plot(z["horizon_min"], z["qlike"], marker="o", label=name)
    axes[0].set(title="Daily walk-forward QLIKE", xlabel="Horizon (minutes)", ylabel="Lower is better")
    axes[0].legend(frameon=False, fontsize=8)
    for name in OPEN_FEATURES:
        z = open_metrics[open_metrics["model"].eq(name)].sort_values("horizon_min")
        axes[1].plot(z["horizon_min"], z["qlike"], marker="o", label=name)
    axes[1].set(title="09:15 model QLIKE", xlabel="Horizon (minutes)", ylabel="Lower is better")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "rv_model_performance.png", dpi=180); plt.close(fig)

    best_intraday = metrics[metrics["model"].isin(["Realized_GARCH_core", "Realized_GARCH_extended"])].sort_values(["horizon_min", "qlike"]).groupby("horizon_min").head(1)
    best_open = open_metrics.sort_values(["horizon_min", "qlike"]).groupby("horizon_min").head(1)
    availability = pd.DataFrame([
        {"feature": "volume", "status": "unavailable", "reason": "not present in source"},
        {"feature": "bid_ask_spread", "status": "unavailable", "reason": "no bid/ask columns in source"},
        {"feature": "term_structure", "status": "unavailable", "reason": "ATM panel uses nearest monthly expiry only"},
        {"feature": "five_strike_skew_butterfly", "status": "implemented", "reason": "computed from fixed-forward five-strike IVs"},
    ])
    availability.to_csv(OUT / "feature_availability.csv", index=False)
    report = f"""# Future realized-variance prediction

## Intraday models

{isa.dataframe_to_markdown(best_intraday[["horizon_min", "model", "n", "qlike", "qlike_improvement_vs_har_pct", "log_r2", "mae_annualized_rv", "bootstrap_ci_low", "bootstrap_ci_high"]].round(6))}

Each horizon has its own model. Targets are total future variance from same-session one-minute log returns. Training uses a 60-session rolling window with daily refits. StandardScaler is fitted only on the training window, and log predictions receive the training-residual retransformation correction.

## 09:15 models

{isa.dataframe_to_markdown(best_open.round(6))}

The opening models use previous-session 5/15/60-minute RV windows and the known overnight return. The extended candidate additionally uses the current 09:15 IV, five-strike smile, DTE, OI and forward basis. Only one opening observation exists per day, so opening results have much lower statistical power than the intraday models.

## Data limitations

{isa.dataframe_to_markdown(availability)}

These results use 2024-2025 as a rolling design period already examined in prior work. They are not an untouched performance claim.
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")
    print(best_intraday.to_string(index=False))
    print(best_open.to_string(index=False))
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
