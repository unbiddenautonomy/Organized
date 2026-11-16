from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .atm_iv_change_prediction import HORIZONS, ROLLING_TRAIN_DAYS, score
except ImportError:  # Allow direct execution from the research directory.
    from atm_iv_change_prediction import HORIZONS, ROLLING_TRAIN_DAYS, score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "atm_iv_change_models" / "atm_iv_change_feature_panel.parquet"
OUT = ROOT / "outputs" / "atm_iv_change_models" / "simple_five_feature"
FEATURES = ["days_to_expiry", "iv_atm", "iv_atm_zscore", "overnight_return", "vol_of_vol_30"]
RANDOM_SEED = 20261008


class FastQuantileRegressor(BaseEstimator, RegressorMixin):
    """Small-feature linear quantile regression via convex pinball loss."""

    def __init__(self, quantile: float = 0.5, alpha: float = 0.0, max_iter: int = 500):
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(x)), x])

        def objective(beta):
            residual = y - design @ beta
            loss = np.where(residual >= 0, self.quantile * residual, (self.quantile - 1) * residual)
            return float(loss.mean() + 0.5 * self.alpha * np.dot(beta[1:], beta[1:]))

        def gradient(beta):
            residual = y - design @ beta
            d_prediction = np.where(residual > 0, -self.quantile, 1 - self.quantile)
            grad = design.T @ d_prediction / len(y)
            grad[1:] += self.alpha * beta[1:]
            return grad

        initial = np.linalg.lstsq(design, y, rcond=None)[0]
        result = minimize(
            objective, initial, jac=gradient, method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-12},
        )
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        self.converged_ = bool(result.success)
        return self

    def predict(self, x):
        return self.intercept_ + np.asarray(x, dtype=float) @ self.coef_


def pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def candidates() -> dict[str, list[tuple[str, object, dict]]]:
    return {
        "linear_5f": [("ols", LinearRegression(), {})],
        "ridge_5f": [
            (f"alpha={a:g}", Ridge(alpha=a), {"alpha": a})
            for a in (0.1, 1.0, 10.0, 100.0)
        ],
        "lasso_5f": [
            (f"alpha={a:g}", Lasso(alpha=a, max_iter=20000, random_state=RANDOM_SEED), {"alpha": a})
            for a in (1e-6, 1e-5, 1e-4, 1e-3)
        ],
        "elastic_net_5f": [
            (
                f"alpha={a:g},l1={ratio:g}",
                ElasticNet(alpha=a, l1_ratio=ratio, max_iter=20000, random_state=RANDOM_SEED),
                {"alpha": a, "l1_ratio": ratio},
            )
            for a in (1e-6, 1e-5, 1e-4, 1e-3)
            for ratio in (0.2, 0.5, 0.8)
        ],
        "quantile_median_5f": [
            (f"alpha={a:g}", FastQuantileRegressor(quantile=0.5, alpha=a), {"alpha": a, "quantile": 0.5})
            for a in (0.0, 1e-5, 1e-4, 1e-3)
        ],
    }


def load_panel() -> pd.DataFrame:
    cols = [
        "datetime", "date", "split", "available_strike_count", "is_0915",
        *FEATURES, *[f"target_d_iv_{h}" for h in HORIZONS],
    ]
    con = duckdb.connect()
    selected = ", ".join(f'"{c}"' for c in cols)
    frame = con.execute(f"SELECT {selected} FROM read_parquet(?)", [str(SOURCE)]).fetchdf()
    con.close()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame


def valid_for(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    target = f"target_d_iv_{h}"
    mask = frame["available_strike_count"].ge(5) & np.isfinite(frame[target])
    return frame.loc[mask].copy()


def select_models(dev: pd.DataFrame, val: pd.DataFrame, target: str):
    selected, rows = {}, []
    for model_name, variants in candidates().items():
        best = None
        for label, estimator, params in variants:
            fitted = pipeline(estimator).fit(dev[FEATURES], dev[target])
            metrics = score(val[target].to_numpy(), fitted.predict(val[FEATURES]))
            rows.append({"model": model_name, "candidate": label, **params, **metrics})
            if best is None or metrics["mae"] < best[0]:
                best = (metrics["mae"], label, estimator, params)
        selected[model_name] = best[2:]
    return selected, rows


def evaluate_subset(predictions: pd.DataFrame, subset: str) -> pd.DataFrame:
    rows = []
    models = list(candidates())
    for h in HORIZONS:
        part = predictions[predictions["horizon_min"].eq(h)]
        if subset == "0915":
            part = part[part["is_0915"].eq(1)]
        y = part[f"target_d_iv_{h}"].to_numpy()
        for model in models:
            rows.append({"horizon_min": h, "model": model, "n": len(part), **score(y, part[model].to_numpy())})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_panel()
    tuning_rows, manifest_rows, prediction_frames = [], [], []

    for h in HORIZONS:
        target = f"target_d_iv_{h}"
        data = valid_for(frame, h)
        dev = data[data["split"].eq("development")]
        val = data[data["split"].eq("validation")]
        selected, tuning = select_models(dev, val, target)
        tuning_rows.extend({"horizon_min": h, **row} for row in tuning)

        evaluation = data[data["split"].eq("evaluation")].copy()
        evaluation["refit_period"] = evaluation["datetime"].dt.to_period("Q")
        blocks = []
        for period in evaluation["refit_period"].drop_duplicates().sort_values():
            test = evaluation[evaluation["refit_period"].eq(period)].copy()
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-ROLLING_TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)]
            if len(train) < 1000 or test.empty:
                continue
            pred = test[["datetime", "date", "is_0915", target]].copy()
            for model_name, (estimator, _) in selected.items():
                fitted = pipeline(estimator).fit(train[FEATURES], train[target])
                pred[model_name] = fitted.predict(test[FEATURES])
            pred["horizon_min"] = h
            blocks.append(pred)
        prediction_frames.append(pd.concat(blocks, ignore_index=True))

        latest_days = data["date"].drop_duplicates().sort_values().iloc[-ROLLING_TRAIN_DAYS:]
        final_train = data[data["date"].isin(latest_days)]
        for model_name, (estimator, params) in selected.items():
            fitted = pipeline(estimator).fit(final_train[FEATURES], final_train[target])
            path = OUT / f"{model_name}_{h}m.joblib"
            joblib.dump({
                "model": fitted, "features": FEATURES, "horizon_min": h,
                "target_definition": f"ATM IV(t+{h}m) - ATM IV(t)",
            }, path)
            manifest_rows.append({
                "horizon_min": h, "model": model_name, "model_file": path.name,
                "train_start": final_train["date"].min(), "train_end": final_train["date"].max(),
                "training_days": final_train["date"].nunique(), "training_rows": len(final_train),
                "parameters": json.dumps(params),
            })

    predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics = evaluate_subset(predictions, "all")
    opening_metrics = evaluate_subset(predictions, "0915")
    pd.DataFrame(tuning_rows).to_csv(OUT / "validation_tuning.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT / "model_manifest.csv", index=False)
    all_metrics.to_csv(OUT / "intraday_metrics.csv", index=False)
    opening_metrics.to_csv(OUT / "opening_0915_metrics.csv", index=False)
    con = duckdb.connect()
    con.register("predictions", predictions)
    con.execute("COPY predictions TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(OUT / "walk_forward_predictions.parquet")])
    con.close()
    print("ALL INTRADAY")
    print(all_metrics.to_string(index=False))
    print("\n09:15 ONLY")
    print(opening_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
