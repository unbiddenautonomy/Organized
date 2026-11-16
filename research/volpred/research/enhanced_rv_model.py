"""Train and evaluate leakage-safe, spot-only enhanced RV models."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SPOT_FILE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
OUT = ROOT / "outputs" / "enhanced_rv_models"
HORIZONS = (15, 30, 60, 120)
EPS = 1e-12
TRAIN_DAYS = 60
SEED = 20261012

HAR_FEATURES = ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60"]
ENHANCED_FEATURES = HAR_FEATURES + [
    "log_current_r2", "log_ewma_r2", "log_rq_15",
    "downside_share_15", "jump_share_15",
    "time_sin", "time_cos", "minutes_to_close_fraction",
    "dow_sin", "dow_cos",
]


def qlike(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ratio = np.clip(y, EPS, None) / np.clip(prediction, EPS, None)
    return ratio - np.log(ratio) - 1.0


def prepare_spot() -> pd.DataFrame:
    raw = pd.read_csv(SPOT_FILE, usecols=["datetime", "spot_price"], parse_dates=["datetime"])
    raw = raw.dropna().loc[lambda x: x["spot_price"].gt(0)].drop_duplicates("datetime", keep="last").sort_values("datetime")
    raw = raw.set_index("datetime").between_time("09:15", "15:30").reset_index()
    frames = []
    for date, group in raw.groupby(raw["datetime"].dt.date, sort=True):
        group = group.set_index("datetime")[["spot_price"]].resample("1min").last()
        group["date"] = pd.Timestamp(date)
        frames.append(group.reset_index())
    spot = pd.concat(frames, ignore_index=True)
    spot["return"] = spot.groupby("date")["spot_price"].transform(lambda x: np.log(x / x.shift(1)))
    return spot


def build_panel() -> pd.DataFrame:
    spot = prepare_spot()
    parts = []
    for _, group in spot.groupby("date", sort=True):
        group = group.copy()
        r = group["return"]
        r2 = r.pow(2)
        for window in (5, 15, 60):
            rv = r2.rolling(window, min_periods=window).sum()
            group[f"log_rv_past_{window}"] = np.log(rv.clip(lower=EPS))

        ewma = r2.ewm(alpha=0.06, adjust=False).mean()
        rq15 = (15.0 / 3.0) * r.pow(4).rolling(15, min_periods=15).sum()
        downside = r2.where(r < 0, 0.0).rolling(15, min_periods=15).sum()
        rv15 = r2.rolling(15, min_periods=15).sum()
        bv15 = (np.pi / 2.0) * (r.abs() * r.abs().shift(1)).rolling(14, min_periods=14).sum()
        jump15 = (rv15 - bv15).clip(lower=0.0)
        group["log_current_r2"] = np.log(r2.clip(lower=EPS))
        group["log_ewma_r2"] = np.log(ewma.clip(lower=EPS))
        group["log_rq_15"] = np.log(rq15.clip(lower=EPS))
        group["downside_share_15"] = (downside / rv15.clip(lower=EPS)).clip(0, 1)
        group["jump_share_15"] = (jump15 / rv15.clip(lower=EPS)).clip(0, 1)

        minute = group["datetime"].dt.hour * 60 + group["datetime"].dt.minute
        fraction = (minute - 555) / 375.0
        group["time_sin"] = np.sin(2 * np.pi * fraction)
        group["time_cos"] = np.cos(2 * np.pi * fraction)
        group["minutes_to_close_fraction"] = (930 - minute).clip(lower=0) / 375.0
        dow = group["datetime"].dt.dayofweek
        group["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
        group["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
        group["time_bucket"] = (minute // 30).astype(int)

        price = group["spot_price"]
        for horizon in HORIZONS:
            future_r2 = r2.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
            future_price = price.shift(-horizon)
            group[f"future_var_{horizon}"] = future_r2
            group[f"future_abs_log_return_{horizon}"] = np.log(future_price / price).abs()
        parts.append(group)
    panel = pd.concat(parts, ignore_index=True)
    panel["year"] = panel["datetime"].dt.year
    return panel


def fit_model(frame: pd.DataFrame, features: list[str], target: str, alpha: float) -> tuple[object, np.ndarray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    log_y = np.log(frame[target].to_numpy(float).clip(min=EPS))
    model.fit(frame[features], log_y)
    return model, log_y - model.predict(frame[features])


def bounds(frame: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    return frame[features].quantile(0.001), frame[features].quantile(0.999)


def clipped(frame: pd.DataFrame, features: list[str], low: pd.Series, high: pd.Series) -> pd.DataFrame:
    return frame[features].clip(lower=low, upper=high, axis=1)


def smear_tables(frame: pd.DataFrame, residuals: np.ndarray) -> tuple[float, dict[int, float], dict[int, float]]:
    temp = pd.DataFrame({"bucket": frame["time_bucket"].to_numpy(), "residual": residuals})
    global_smear = float(np.mean(np.exp(residuals)))
    global_q90 = float(np.quantile(residuals, 0.90))
    smear, q90 = {}, {}
    for bucket, group in temp.groupby("bucket"):
        if len(group) >= 200:
            smear[int(bucket)] = float(np.mean(np.exp(group["residual"])))
            q90[int(bucket)] = float(group["residual"].quantile(0.90))
    smear[-1], q90[-1] = global_smear, global_q90
    return global_smear, smear, q90


def predict(model, x: pd.DataFrame, buckets: pd.Series, smear: dict[int, float]) -> np.ndarray:
    base = np.exp(model.predict(x))
    factors = buckets.map(lambda b: smear.get(int(b), smear[-1])).to_numpy(float)
    return base * factors


def daily_ci(frame: pd.DataFrame, candidate: str, baseline: str = "har", reps: int = 3000) -> tuple[float, float, float]:
    loss_diff = pd.DataFrame({
        "date": frame["date"],
        "difference": qlike(frame["actual"].to_numpy(), frame[candidate].to_numpy())
        - qlike(frame["actual"].to_numpy(), frame[baseline].to_numpy()),
    }).groupby("date")["difference"].mean().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(loss_diff, len(loss_diff), replace=True).mean() for _ in range(reps)])
    return float(loss_diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "enhanced_spot_panel.parquet"
    if cache.exists():
        panel = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(cache)]).fetchdf()
    else:
        panel = build_panel()
        connection = duckdb.connect()
        connection.register("panel", panel)
        connection.execute("COPY panel TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(cache)])
        connection.close()

    manifest, metrics = [], []
    for horizon in HORIZONS:
        target = f"future_var_{horizon}"
        required = ENHANCED_FEATURES + [target, f"future_abs_log_return_{horizon}"]
        data = panel[np.isfinite(panel[required]).all(axis=1) & panel[target].gt(0)].copy()
        dev, validation = data[data["year"].le(2022)], data[data["year"].eq(2023)]

        tuned = {}
        for name, features in {"har": HAR_FEATURES, "enhanced": ENHANCED_FEATURES}.items():
            best = (np.inf, None)
            low, high = bounds(dev, features)
            train = dev.copy(); score = validation.copy()
            train.loc[:, features] = clipped(train, features, low, high)
            score.loc[:, features] = clipped(score, features, low, high)
            for alpha in (0.1, 1.0, 10.0, 100.0, 300.0):
                model, residuals = fit_model(train, features, target, alpha)
                _, smear, _ = smear_tables(train, residuals)
                forecast = predict(model, score[features], score["time_bucket"], smear)
                loss = float(qlike(score[target].to_numpy(), forecast).mean())
                if loss < best[0]: best = (loss, alpha)
            tuned[name] = best[1]

        evaluation = data[data["year"].ge(2024)].copy()
        frames = []
        month_key = evaluation["datetime"].dt.to_period("M")
        for month in month_key.drop_duplicates().sort_values():
            test = evaluation[month_key.eq(month)].copy()
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)].copy()
            if len(train) < 2000: continue
            result = test[["datetime", "date", target]].rename(columns={target: "actual"}).copy()
            for name, features in {"har": HAR_FEATURES, "enhanced": ENHANCED_FEATURES}.items():
                low, high = bounds(train, features)
                train_x = train.copy(); train_x.loc[:, features] = clipped(train_x, features, low, high)
                test_x = clipped(test, features, low, high)
                model, residuals = fit_model(train_x, features, target, tuned[name])
                _, smear, _ = smear_tables(train_x, residuals)
                result[name] = predict(model, test_x, test["time_bucket"], smear)
            frames.append(result)
        predictions = pd.concat(frames, ignore_index=True)
        har_loss = float(qlike(predictions["actual"].to_numpy(), predictions["har"].to_numpy()).mean())
        enhanced_loss = float(qlike(predictions["actual"].to_numpy(), predictions["enhanced"].to_numpy()).mean())
        diff, ci_low, ci_high = daily_ci(predictions, "enhanced")
        candidate_passes_initial = enhanced_loss < har_loss and ci_high < 0
        promotion_approved = (OUT / "validation_120" / "PROMOTE").exists()
        promote = candidate_passes_initial and promotion_approved
        metrics.append({
            "horizon_min": horizon, "har_qlike": har_loss, "enhanced_qlike": enhanced_loss,
            "improvement_pct": 100 * (har_loss - enhanced_loss) / har_loss,
            "daily_loss_difference": diff, "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high, "candidate_passes_initial": candidate_passes_initial,
            "promotion_approved": promotion_approved, "promote": promote,
        })

        final_days = data["date"].drop_duplicates().sort_values().iloc[-TRAIN_DAYS:]
        final = data[data["date"].isin(final_days)].copy()
        selected_name = "enhanced" if promote else "har"
        selected_features = ENHANCED_FEATURES if promote else HAR_FEATURES
        low, high = bounds(final, selected_features)
        final.loc[:, selected_features] = clipped(final, selected_features, low, high)
        model, residuals = fit_model(final, selected_features, target, tuned[selected_name])
        global_smear, bucket_smear, bucket_q90 = smear_tables(final, residuals)
        standardized_abs = final[f"future_abs_log_return_{horizon}"] / np.sqrt(final[target].clip(lower=EPS))
        abs_distribution = {
            "mean": float(standardized_abs.mean()),
            "q80": float(standardized_abs.quantile(0.80)),
            "q90": float(standardized_abs.quantile(0.90)),
        }
        path = OUT / f"enhanced_rv_{horizon}m_model.joblib"
        joblib.dump({
            "model": model, "features": selected_features, "horizon_min": horizon,
            "algorithm": selected_name, "clip_lower": low.to_dict(), "clip_upper": high.to_dict(),
            "global_smear": global_smear, "bucket_smear": bucket_smear,
            "bucket_residual_q90": bucket_q90, "absolute_return_multiplier": abs_distribution,
        }, path)
        manifest.append({
            "horizon_min": horizon, "model_file": path.name, "algorithm": selected_name,
            "alpha": tuned[selected_name], "train_start": final["date"].min(),
            "train_end": final["date"].max(), "training_days": final["date"].nunique(),
            "training_rows": len(final), "feature_count": len(selected_features),
        })

    pd.DataFrame(metrics).to_csv(OUT / "evaluation.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUT / "model_manifest.csv", index=False)
    report = ["# Enhanced spot-only RV model", "", "```text", pd.DataFrame(metrics).to_string(index=False), "```", "",
              "Promotion requires lower rolling QLIKE and an upper 95% daily-block bootstrap bound below zero."]
    (OUT / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
