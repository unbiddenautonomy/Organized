from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

try:
    from .atm_iv_change_prediction import HORIZONS, ROLLING_TRAIN_DAYS, score
    from .log_iv_change_prediction import TrainingWinsorizer, bootstrap_mae_difference, pinball
    from .simple_iv_change_models import FastQuantileRegressor
except ImportError:  # Allow direct execution from the research directory.
    from atm_iv_change_prediction import HORIZONS, ROLLING_TRAIN_DAYS, score
    from log_iv_change_prediction import TrainingWinsorizer, bootstrap_mae_difference, pinball
    from simple_iv_change_models import FastQuantileRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "log_iv_change_models" / "log_iv_feature_panel.parquet"
OUT = ROOT / "outputs" / "log_iv_change_models" / "compact_five_feature"
FEATURES = ["days_to_expiry", "log_iv_atm", "iv_atm_zscore", "overnight_return", "vol_of_vol_log_30"]
ALPHAS = (1e-5, 1e-4, 1e-3)


def model(alpha: float):
    return make_pipeline(
        SimpleImputer(strategy="median"), TrainingWinsorizer(), StandardScaler(),
        FastQuantileRegressor(quantile=0.5, alpha=alpha),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["datetime", "date", "split", "available_strike_count", "is_0915", *FEATURES, *[f"target_dlog_iv_{h}" for h in HORIZONS]]
    con = duckdb.connect()
    frame = con.execute(f"SELECT {', '.join(cols)} FROM read_parquet(?)", [str(SOURCE)]).fetchdf()
    con.close()
    frame["datetime"] = pd.to_datetime(frame["datetime"]); frame["date"] = pd.to_datetime(frame["date"])
    predictions, tuning_rows, manifest_rows = [], [], []
    for h in HORIZONS:
        target = f"target_dlog_iv_{h}"
        data = frame[frame["available_strike_count"].ge(5) & np.isfinite(frame[target])].copy()
        dev, val = data[data["split"].eq("development")], data[data["split"].eq("validation")]
        best_alpha, best_loss = None, np.inf
        for alpha in ALPHAS:
            fitted = model(alpha).fit(dev[FEATURES], dev[target])
            pred = fitted.predict(val[FEATURES])
            loss = pinball(val[target].to_numpy(), pred, 0.5)
            tuning_rows.append({"horizon_min": h, "alpha": alpha, "validation_pinball": loss})
            if loss < best_loss:
                best_alpha, best_loss = alpha, loss
        evaluation = data[data["split"].eq("evaluation")].copy()
        evaluation["period"] = evaluation["datetime"].dt.to_period("Q")
        for period in evaluation["period"].drop_duplicates().sort_values():
            test = evaluation[evaluation["period"].eq(period)]
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-ROLLING_TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)]
            if len(train) < 1000 or test.empty:
                continue
            fitted = model(best_alpha).fit(train[FEATURES], train[target])
            out = test[["datetime", "date", "is_0915", target]].copy()
            out["horizon_min"] = h; out["q50_compact"] = fitted.predict(test[FEATURES])
            predictions.append(out)
        latest = data["date"].drop_duplicates().sort_values().iloc[-ROLLING_TRAIN_DAYS:]
        train = data[data["date"].isin(latest)]
        fitted = model(best_alpha).fit(train[FEATURES], train[target])
        path = OUT / f"compact_log_iv_{h}m_q50.joblib"
        joblib.dump({"model": fitted, "features": FEATURES, "horizon_min": h, "quantile": 0.5}, path)
        manifest_rows.append({"horizon_min": h, "alpha": best_alpha, "validation_pinball": best_loss, "model_file": path.name, "training_rows": len(train)})
    pred = pd.concat(predictions, ignore_index=True)
    con = duckdb.connect()
    con.register("compact_predictions", pred)
    con.execute(
        "COPY compact_predictions TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(OUT / "walk_forward_predictions.parquet")],
    )
    con.close()
    rows, bootstrap_rows = [], []
    for subset, selected in (("intraday", pred), ("0915", pred[pred["is_0915"].eq(1)])):
        for h in HORIZONS:
            part = selected[selected["horizon_min"].eq(h)]
            y = part[f"target_dlog_iv_{h}"].to_numpy(); p = part["q50_compact"].to_numpy()
            rows.append({"subset": subset, "horizon_min": h, "n": len(part), **score(y, p), "pinball_loss": pinball(y, p, 0.5)})
            bootstrap_rows.append({
                "subset": subset, "horizon_min": h,
                **bootstrap_mae_difference(part, f"target_dlog_iv_{h}", "q50_compact"),
            })
    pd.DataFrame(rows).to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "validation_tuning.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT / "model_manifest.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "day_block_bootstrap.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
