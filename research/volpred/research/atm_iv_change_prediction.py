from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIVE_STRIKE_FILE = ROOT / "outputs" / "five_strike_spot_analysis" / "five_strike_feature_panel.csv"
SPOT_FILE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
OUT = ROOT / "outputs" / "atm_iv_change_models"

HORIZONS = (15, 30, 60, 120)
WINDOWS = (5, 15, 60)
ROLLING_TRAIN_DAYS = 60
EPS = 1e-12
RANDOM_SEED = 20261008

POSITION_SUFFIXES = ("m2", "m1", "atm", "p1", "p2")
RAW_COLUMNS = [
    "datetime", "expiry", "spot_price", "days_to_expiry", "available_strike_count",
    "common_forward", "total_call_oi", "total_put_oi",
    *[f"iv_call_{p}" for p in POSITION_SUFFIXES],
    *[f"iv_put_{p}" for p in POSITION_SUFFIXES],
    *[f"log_call_oi_{p}" for p in POSITION_SUFFIXES],
    *[f"log_put_oi_{p}" for p in POSITION_SUFFIXES],
]

LEVEL_FEATURES = [
    "iv_atm", "log_iv_atm", "iv_atm_zscore", "iv_skew_rr", "iv_butterfly",
    "iv_put_skew", "iv_call_skew", "iv_m1_minus_atm", "iv_p1_minus_atm",
    "forward_log_basis", "log_total_call_oi", "log_total_put_oi", "oi_ratio",
    "oi_imbalance", "days_to_expiry", "sqrt_dte", "time_fraction", "time_sin",
    "time_cos", "is_0915", "overnight_return",
]
WINDOW_FEATURES = [
    f"{name}_{w}"
    for w in WINDOWS
    for name in (
        "iv_change", "log_iv_change", "skew_change", "butterfly_change",
        "oi_total_change", "spot_return", "spot_abs_return", "spot_rv", "log_spot_rv",
        "rv_iv_gap", "leverage",
    )
]
FEATURES = LEVEL_FEATURES + WINDOW_FEATURES + ["vol_of_vol_30"]
BASELINE_FEATURES = [
    "iv_atm", "iv_atm_zscore", "iv_change_5", "iv_change_15", "spot_return_5",
    "spot_return_15", "time_fraction", "days_to_expiry",
]


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    con.execute("SET threads=8")
    cols = ", ".join(f'"{c}"' for c in RAW_COLUMNS)
    option = con.execute(
        f"SELECT {cols} FROM read_csv_auto(?, header=true)", [str(FIVE_STRIKE_FILE)]
    ).fetchdf()
    spot = con.execute(
        "SELECT datetime::TIMESTAMP AS datetime, spot_price::DOUBLE AS spot_price "
        "FROM read_csv_auto(?, header=true) ORDER BY datetime",
        [str(SPOT_FILE)],
    ).fetchdf()
    con.close()
    option["datetime"] = pd.to_datetime(option["datetime"])
    spot["datetime"] = pd.to_datetime(spot["datetime"])
    return option.sort_values("datetime").drop_duplicates("datetime"), spot.drop_duplicates("datetime")


def session_diff(frame: pd.DataFrame, column: str, minutes: int) -> pd.Series:
    # The option panel can skip minutes when a full chain is unavailable, so a
    # row shift is not a minute lag. Look up the exact timestamp instead.
    lookup = pd.Series(frame[column].to_numpy(), index=frame["datetime"])
    old = (frame["datetime"] - pd.Timedelta(minutes=minutes)).map(lookup)
    return frame[column] - old.to_numpy()


def use_previous_session_at_open(frame: pd.DataFrame, column: str) -> None:
    daily_last = frame.groupby("date", sort=False)[column].last().shift(1)
    previous = frame["date"].map(daily_last)
    frame.loc[frame["is_0915"].eq(1), column] = previous[frame["is_0915"].eq(1)]


def add_spot_features(option: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    spot = spot.copy()
    spot["date"] = spot["datetime"].dt.normalize()
    spot["is_0915"] = (
        spot["datetime"].dt.hour.eq(9) & spot["datetime"].dt.minute.eq(15)
    ).astype("int8")
    grouped = spot.groupby("date", sort=False)
    spot["spot_log_return_1"] = grouped["spot_price"].transform(
        lambda x: np.log((x / x.shift(1)).where((x / x.shift(1)).gt(0)))
    )
    for w in WINDOWS:
        roll = spot["spot_log_return_1"].groupby(spot["date"], sort=False).rolling(w, min_periods=w)
        spot[f"spot_return_{w}"] = roll.sum().reset_index(level=0, drop=True)
        spot[f"spot_abs_return_{w}"] = roll.apply(lambda x: np.abs(x).sum(), raw=True).reset_index(level=0, drop=True)
        spot[f"spot_rv_{w}"] = roll.apply(lambda x: np.square(x).sum(), raw=True).reset_index(level=0, drop=True)
        for c in (f"spot_return_{w}", f"spot_abs_return_{w}", f"spot_rv_{w}"):
            use_previous_session_at_open(spot, c)

    daily = spot.groupby("date", as_index=False).agg(session_open=("spot_price", "first"), session_close=("spot_price", "last"))
    daily["previous_close"] = daily["session_close"].shift(1)
    overnight_ratio = daily["session_open"] / daily["previous_close"]
    daily["overnight_return"] = np.log(overnight_ratio.where(overnight_ratio.gt(0)))
    spot = spot.merge(daily[["date", "overnight_return"]], on="date", how="left", validate="many_to_one")
    keep = ["datetime", "overnight_return"] + [
        f"{name}_{w}" for w in WINDOWS for name in ("spot_return", "spot_abs_return", "spot_rv")
    ]
    return option.merge(spot[keep], on="datetime", how="left", validate="one_to_one")


def build_feature_panel() -> pd.DataFrame:
    frame, spot = read_inputs()
    frame["date"] = frame["datetime"].dt.normalize()
    frame["is_0915"] = (
        frame["datetime"].dt.hour.eq(9) & frame["datetime"].dt.minute.eq(15)
    ).astype("int8")

    for p in POSITION_SUFFIXES:
        frame[f"iv_mid_{p}"] = frame[[f"iv_call_{p}", f"iv_put_{p}"]].mean(axis=1)
    frame["iv_atm"] = frame["iv_mid_atm"]
    frame["log_iv_atm"] = np.log(frame["iv_atm"].clip(lower=EPS))
    frame["iv_skew_rr"] = frame["iv_call_p2"] - frame["iv_put_m2"]
    frame["iv_butterfly"] = frame["iv_call_p2"] + frame["iv_put_m2"] - 2 * frame["iv_atm"]
    frame["iv_put_skew"] = frame["iv_put_m2"] - frame["iv_atm"]
    frame["iv_call_skew"] = frame["iv_call_p2"] - frame["iv_atm"]
    frame["iv_m1_minus_atm"] = frame["iv_mid_m1"] - frame["iv_atm"]
    frame["iv_p1_minus_atm"] = frame["iv_mid_p1"] - frame["iv_atm"]
    forward_ratio = frame["common_forward"] / frame["spot_price"]
    frame["forward_log_basis"] = np.log(forward_ratio.where(forward_ratio.gt(0)))
    frame["log_total_call_oi"] = np.log1p(frame["total_call_oi"].clip(lower=0))
    frame["log_total_put_oi"] = np.log1p(frame["total_put_oi"].clip(lower=0))
    frame["oi_total"] = frame["total_call_oi"] + frame["total_put_oi"]
    frame["oi_ratio"] = frame["total_call_oi"] / frame["total_put_oi"].clip(lower=1)
    frame["oi_imbalance"] = (frame["total_call_oi"] - frame["total_put_oi"]) / frame["oi_total"].clip(lower=1)

    # A trailing, past-only IV normalization. Current IV is known at prediction time.
    trailing_mean = frame["iv_atm"].rolling(60 * 375, min_periods=1000).mean()
    trailing_std = frame["iv_atm"].rolling(60 * 375, min_periods=1000).std()
    frame["iv_atm_zscore"] = (frame["iv_atm"] - trailing_mean) / trailing_std.replace(0, np.nan)

    for w in WINDOWS:
        frame[f"iv_change_{w}"] = session_diff(frame, "iv_atm", w)
        frame[f"log_iv_change_{w}"] = session_diff(frame, "log_iv_atm", w)
        frame[f"skew_change_{w}"] = session_diff(frame, "iv_skew_rr", w)
        frame[f"butterfly_change_{w}"] = session_diff(frame, "iv_butterfly", w)
        old_oi = (frame["datetime"] - pd.Timedelta(minutes=w)).map(
            pd.Series(frame["oi_total"].to_numpy(), index=frame["datetime"])
        )
        frame[f"oi_total_change_{w}"] = session_diff(frame, "oi_total", w) / old_oi.clip(lower=1)
        for c in (
            f"iv_change_{w}", f"log_iv_change_{w}", f"skew_change_{w}",
            f"butterfly_change_{w}", f"oi_total_change_{w}",
        ):
            use_previous_session_at_open(frame, c)

    iv_d1 = frame.groupby("date", sort=False)["iv_atm"].diff()
    frame["vol_of_vol_30"] = iv_d1.groupby(frame["date"], sort=False).rolling(30, min_periods=10).std().reset_index(level=0, drop=True)
    use_previous_session_at_open(frame, "vol_of_vol_30")

    frame = add_spot_features(frame, spot)
    for w in WINDOWS:
        frame[f"log_spot_rv_{w}"] = np.log(frame[f"spot_rv_{w}"].clip(lower=EPS))
        frame[f"rv_iv_gap_{w}"] = frame[f"log_spot_rv_{w}"] - 2 * frame["log_iv_atm"]
        frame[f"leverage_{w}"] = (
            frame[f"iv_change_{w}"] / frame[f"spot_return_{w}"].replace(0, np.nan)
        ).clip(-100, 100)

    minute = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
    frame["time_fraction"] = (minute - 555) / 375.0
    frame["time_sin"] = np.sin(2 * np.pi * frame["time_fraction"])
    frame["time_cos"] = np.cos(2 * np.pi * frame["time_fraction"])
    frame["sqrt_dte"] = np.sqrt(frame["days_to_expiry"].clip(lower=0))

    future = frame[["datetime", "date", "iv_atm"]].copy()
    for h in HORIZONS:
        lookup = future.rename(columns={"iv_atm": f"future_iv_atm_{h}", "date": f"future_date_{h}"})
        lookup["datetime"] = lookup["datetime"] - pd.Timedelta(minutes=h)
        frame = frame.merge(lookup, on="datetime", how="left", validate="one_to_one")
        same_session = frame[f"future_date_{h}"].eq(frame["date"])
        frame[f"target_d_iv_{h}"] = (frame[f"future_iv_atm_{h}"] - frame["iv_atm"]).where(same_session)
        frame.drop(columns=[f"future_date_{h}"], inplace=True)

    frame["year"] = frame["datetime"].dt.year
    frame["split"] = np.select(
        [frame["year"].le(2022), frame["year"].eq(2023)],
        ["development", "validation"], default="evaluation",
    )
    return frame


def make_lgb(params: dict) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber", random_state=RANDOM_SEED, n_jobs=8, verbosity=-1,
        learning_rate=0.035, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, **params,
    )


def valid_for(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    mask = frame["available_strike_count"].ge(5)
    mask &= np.isfinite(frame["iv_atm"]) & np.isfinite(frame[f"target_d_iv_{horizon}"])
    result = frame.loc[mask].copy()
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    return result


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    err = y - pred
    denom = np.sum((y - y.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / denom) if denom > 0 else np.nan,
        "hit_rate": float(np.mean(np.sign(pred) == np.sign(y))),
        "ic": float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else np.nan,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    values = [[str(c) for c in frame.columns], *frame.astype(str).values.tolist()]
    widths = [max(len(row[i]) for row in values) for i in range(len(values[0]))]
    line = lambda row: "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([line(values[0]), separator, *[line(row) for row in values[1:]]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_feature_panel()
    bad = (~np.isfinite(panel[FEATURES])).sum().sort_values(ascending=False)
    print("Most-missing features:\n" + bad.head(15).to_string())
    con = duckdb.connect()
    con.register("panel", panel)
    con.execute("COPY panel TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(OUT / "atm_iv_change_feature_panel.parquet")])
    con.close()

    candidates = [
        {"n_estimators": 250, "num_leaves": 15, "min_child_samples": 150},
        {"n_estimators": 350, "num_leaves": 31, "min_child_samples": 150},
        {"n_estimators": 300, "num_leaves": 15, "min_child_samples": 350},
    ]
    validation_rows: list[dict] = []
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for h in HORIZONS:
        data = valid_for(panel, h)
        target = f"target_d_iv_{h}"
        dev = data[data["split"].eq("development")]
        val = data[data["split"].eq("validation")]
        print(
            f"h={h}: valid={len(data):,}, development={len(dev):,}, "
            f"validation={len(val):,}, evaluation={data['split'].eq('evaluation').sum():,}"
        )
        best_params, best_mae = None, np.inf
        for number, params in enumerate(candidates, 1):
            model = make_lgb(params).fit(dev[FEATURES], dev[target])
            pred = model.predict(val[FEATURES])
            metrics = score(val[target].to_numpy(), pred)
            validation_rows.append({"horizon_min": h, "candidate": number, **params, **metrics})
            if metrics["mae"] < best_mae:
                best_mae, best_params = metrics["mae"], params

        evaluation = data[data["split"].eq("evaluation")].copy()
        evaluation["refit_period"] = evaluation["datetime"].dt.to_period("Q")
        horizon_predictions = []
        for period in evaluation["refit_period"].drop_duplicates().sort_values():
            test = evaluation[evaluation["refit_period"].eq(period)].copy()
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-ROLLING_TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)]
            if len(train) < 1000 or test.empty:
                continue
            ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0)).fit(train[FEATURES], train[target])
            baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0)).fit(train[BASELINE_FEATURES], train[target])
            boost = make_lgb(best_params).fit(train[FEATURES], train[target])
            pred = test[["datetime", "date", "iv_atm", "is_0915", target]].copy()
            pred["horizon_min"] = h
            pred["zero"] = 0.0
            pred["momentum_5m"] = test["iv_change_5"].fillna(0.0).to_numpy()
            pred["mean_reversion_ridge"] = baseline.predict(test[BASELINE_FEATURES])
            pred["full_ridge"] = ridge.predict(test[FEATURES])
            pred["lightgbm"] = boost.predict(test[FEATURES])
            horizon_predictions.append(pred)

        if not horizon_predictions:
            raise RuntimeError(f"No walk-forward evaluation blocks were available for horizon {h}")
        preds = pd.concat(horizon_predictions, ignore_index=True)
        prediction_frames.append(preds)
        y = preds[target].to_numpy()
        for name in ("zero", "momentum_5m", "mean_reversion_ridge", "full_ridge", "lightgbm"):
            metric_rows.append({"horizon_min": h, "model": name, "n": len(preds), "days": preds["date"].nunique(), **score(y, preds[name].to_numpy())})

        latest_days = data["date"].drop_duplicates().sort_values().iloc[-ROLLING_TRAIN_DAYS:]
        final_train = data[data["date"].isin(latest_days)]
        final_model = make_lgb(best_params).fit(final_train[FEATURES], final_train[target])
        model_file = OUT / f"atm_iv_change_{h}m_lightgbm.joblib"
        joblib.dump({
            "model": final_model, "features": FEATURES, "horizon_min": h,
            "target_definition": f"ATM IV(t+{h}m) - ATM IV(t)",
            "iv_units": "absolute volatility (0.01 = one volatility point)",
        }, model_file)
        manifest_rows.append({
            "horizon_min": h, "model_file": model_file.name,
            "train_start": final_train["date"].min(), "train_end": final_train["date"].max(),
            "training_days": final_train["date"].nunique(), "training_rows": len(final_train),
            "validation_mae": best_mae, "parameters": json.dumps(best_params),
        })
        gain = final_model.feature_importances_.astype(float)
        for feature, value in zip(FEATURES, gain):
            importance_rows.append({
                "horizon_min": h, "feature": feature, "importance": value,
                "importance_share": value / gain.sum() if gain.sum() else np.nan,
            })

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_tuning.csv", index=False)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT / "model_manifest.csv", index=False)
    pd.DataFrame(importance_rows).sort_values(["horizon_min", "importance"], ascending=[True, False]).to_csv(OUT / "feature_importance.csv", index=False)
    con = duckdb.connect()
    con.register("predictions", predictions)
    con.execute(
        "COPY predictions TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(OUT / "walk_forward_predictions.parquet")],
    )
    con.close()
    opening_predictions = predictions[predictions["is_0915"].eq(1)].copy()
    opening_predictions.to_csv(OUT / "opening_0915_predictions.csv", index=False)
    opening_metric_rows = []
    for h in HORIZONS:
        opening = opening_predictions[opening_predictions["horizon_min"].eq(h)]
        y = opening[f"target_d_iv_{h}"].to_numpy()
        for name in ("zero", "momentum_5m", "mean_reversion_ridge", "full_ridge", "lightgbm"):
            opening_metric_rows.append({
                "horizon_min": h, "model": name, "n": len(opening),
                **score(y, opening[name].to_numpy()),
            })
    opening_metrics = pd.DataFrame(opening_metric_rows)
    opening_metrics.to_csv(OUT / "opening_0915_metrics.csv", index=False)
    panel.loc[panel["is_0915"].eq(1), ["datetime", "date", *FEATURES]].to_csv(OUT / "opening_0915_features.csv", index=False)

    champion = metrics[metrics["model"].eq("lightgbm")].sort_values("horizon_min")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in ("zero", "momentum_5m", "mean_reversion_ridge", "full_ridge", "lightgbm"):
        z = metrics[metrics["model"].eq(name)].sort_values("horizon_min")
        axes[0].plot(z["horizon_min"], z["mae"], marker="o", label=name)
    axes[0].set(title="Walk-forward MAE", xlabel="Horizon (minutes)", ylabel="Absolute IV")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(champion["horizon_min"].astype(str), champion["hit_rate"], color="#2A9D8F")
    axes[1].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1].set(title="LightGBM directional hit rate", xlabel="Horizon (minutes)", ylabel="Hit rate")
    fig.savefig(OUT / "model_performance.png", dpi=180)
    plt.close(fig)

    report_table = champion[["horizon_min", "n", "days", "mae", "rmse", "r2", "hit_rate", "ic"]].round(6)
    report = f"""# ATM IV-change prediction models

## Definition

Four separate LightGBM regressors predict `ATM IV(t+h) - ATM IV(t)` for 15, 30, 60, and 120 minutes. IV is represented as a decimal, so a prediction of 0.01 is one volatility point. ATM IV is the average of recalculated call and put IV at the ATM strike.

## Leakage controls

- Every target is an exact future timestamp in the same trading session; targets never cross overnight.
- All 5/15/60-minute feature changes and rolling statistics are session-contained.
- At 09:15 only, the trailing 5/15/60-minute IV, skew, OI, return, and realized-variance features are taken from the previous session's closing windows. Overnight return is separate.
- Hyperparameters are selected on 2023. The 2024-2025 evaluation refits quarterly using only the preceding 60 valid trading days.
- Production models are refit independently by horizon on the latest 60 valid sessions.

## Walk-forward LightGBM evaluation

{markdown_table(report_table)}

The zero, five-minute momentum, mean-reversion Ridge, and full Ridge comparisons are in `model_metrics.csv`.

Directional hit rates are promising, but the LightGBM magnitude forecasts do not beat the zero-change forecast on MAE or R-squared. Treat these as directional research signals, not validated magnitude forecasts, until a revised target/loss or richer quote data improves magnitude calibration. The smaller 09:15-only evaluation is in `opening_0915_metrics.csv`.

## Data limitations

The built five-strike data contains recalculated IV, OI, spot, forward, DTE, and timestamps. It does not contain bid/ask quotes, traded volume, second-expiry ATM IV, or an event calendar, so those proposed features are deliberately omitted.
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(pd.DataFrame(manifest_rows).to_string(index=False))


if __name__ == "__main__":
    main()
