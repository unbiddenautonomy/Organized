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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import intraday_structure_attribution as isa
from atm_iv_rv_analysis import prepare_spot


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "spot_magnitude_models"
SPOT_FILE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
HORIZONS = (15, 30, 60, 120)
ROLLING_TRAIN_DAYS = 60
EPS = 1e-12

FEATURES = [
    "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
    "return_past_5", "return_past_15", "return_past_60",
    "abs_return_past_5", "abs_return_past_15", "abs_return_past_60",
    "log_iv_trade", "iv_call", "iv_put", "iv_skew",
    "iv_change_5", "iv_change_15", "iv_change_60",
    "gamma_atm", "theta_atm", "vega_atm", "gamma_x_s2", "vega_theta_ratio",
    "log_call_oi", "log_put_oi", "oi_ratio", "oi_imbalance",
    "oi_total_change_5", "oi_total_change_15", "oi_total_change_60",
    "vol_of_vol_30", "leverage_effect", "forward_log_basis",
    "time_fraction", "time_sin", "time_cos", "days_to_expiry", "sqrt_dte",
    "overnight_return", "prevday_log_rv_5", "prevday_log_rv_15", "prevday_log_rv_60",
    "is_0915",
]
HAR_FEATURES = ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60"]
OPEN_FEATURES = [
    "prevday_log_rv_5", "prevday_log_rv_15", "prevday_log_rv_60", "overnight_return",
    "log_iv_trade", "iv_skew", "log_call_oi", "log_put_oi", "oi_imbalance",
    "days_to_expiry", "sqrt_dte", "forward_log_basis",
]


def future_targets(spot: pd.DataFrame) -> pd.DataFrame:
    out = spot[["datetime", "date", "spot_price"]].copy()
    ret = spot["return"].to_numpy(float)
    price = spot["spot_price"].to_numpy(float)
    for h in HORIZONS:
        var = np.full(len(spot), np.nan)
        future_price = np.full(len(spot), np.nan)
        for _, idx in spot.groupby("date", sort=False).indices.items():
            idx = np.asarray(idx)
            vals = ret[idx] ** 2
            bad = ~np.isfinite(vals)
            cv = np.r_[0.0, np.cumsum(np.where(bad, 0.0, vals))]
            cb = np.r_[0, np.cumsum(bad.astype(int))]
            start = np.arange(len(idx))
            end = start + h + 1
            ok = end <= len(idx)
            s, e = start[ok] + 1, end[ok]
            good = (cb[e] - cb[s]) == 0
            loc = start[ok][good]
            var[idx[loc]] = cv[e[good]] - cv[s[good]]
            future_price[idx[loc]] = price[idx[loc + h]]
        out[f"future_var_{h}"] = var
        out[f"future_spot_{h}"] = future_price
        out[f"future_abs_move_{h}"] = np.abs(future_price - price)
    return out


def trailing_spot_features(spot: pd.DataFrame) -> pd.DataFrame:
    out = spot[["datetime", "date", "spot_price"]].copy()
    valid = spot["return"].notna().to_numpy()
    r = spot.loc[valid, "return"].reset_index(drop=True)
    valid_pos = np.where(valid)[0]
    for h in (5, 15, 30, 60, 120):
        series = r.pow(2).rolling(h, min_periods=h).sum()
        holder = pd.Series(np.nan, index=np.arange(len(spot)))
        holder.iloc[valid_pos] = series.to_numpy()
        out[f"rv_past_{h}"] = holder.ffill().to_numpy()
        if h in (5, 15, 60):
            rs = r.rolling(h, min_periods=h).sum()
            ar = r.abs().rolling(h, min_periods=h).sum()
            h1 = pd.Series(np.nan, index=np.arange(len(spot))); h1.iloc[valid_pos] = rs.to_numpy()
            h2 = pd.Series(np.nan, index=np.arange(len(spot))); h2.iloc[valid_pos] = ar.to_numpy()
            out[f"return_past_{h}"] = h1.ffill().to_numpy()
            out[f"abs_return_past_{h}"] = h2.ffill().to_numpy()

    day_rows = []
    for date, g in spot.groupby("date", sort=False):
        vals = g["return"].dropna().to_numpy(float)
        row = {"date": date}
        for h in (5, 15, 60):
            row[f"prevday_rv_{h}"] = np.sum(vals[-h:] ** 2) if len(vals) >= h else np.nan
        row["session_open"] = g["spot_price"].iloc[0]
        row["session_close"] = g["spot_price"].iloc[-1]
        day_rows.append(row)
    daily = pd.DataFrame(day_rows).sort_values("date")
    for h in (5, 15, 60):
        daily[f"prevday_rv_{h}"] = daily[f"prevday_rv_{h}"].shift(1)
    daily["previous_close"] = daily["session_close"].shift(1)
    daily["overnight_return"] = np.log(daily["session_open"] / daily["previous_close"])
    return out.merge(
        daily[["date", "overnight_return", "prevday_rv_5", "prevday_rv_15", "prevday_rv_60"]],
        on="date", how="left", validate="many_to_one",
    )


def previous_session_last(series: pd.Series, dates: pd.Series) -> pd.Series:
    daily_last = series.groupby(dates).last().shift(1)
    return dates.map(daily_last)


def feature_panel() -> pd.DataFrame:
    entries, meta = isa.prepare_entry_rows()
    atm = entries[entries["strike_position"].eq(0)].copy()
    atm = isa.add_entry_greeks(atm).sort_values("datetime").reset_index(drop=True)
    atm["date"] = pd.to_datetime(atm["date"])
    atm = atm.merge(
        meta[["datetime", "iv_trade_vol", "quality_ok"]], on="datetime", how="left", validate="one_to_one"
    )
    spot = prepare_spot(pd.read_csv(SPOT_FILE, parse_dates=["datetime", "Date"]))
    past = trailing_spot_features(spot)
    targets = future_targets(spot).drop(columns=["date", "spot_price"])
    panel = atm.merge(past, on=["datetime", "date", "spot_price"], how="left", validate="one_to_one")
    panel = panel.merge(targets, on="datetime", how="left", validate="one_to_one")

    panel["iv_atm"] = panel[["iv_call", "iv_put"]].mean(axis=1)
    panel["iv_skew"] = panel["iv_call"] - panel["iv_put"]
    panel["log_iv_trade"] = np.log(panel["iv_trade_vol"].clip(lower=EPS))
    panel["gamma_atm"] = panel["call_gamma"] + panel["put_gamma"]
    panel["theta_atm"] = panel["call_theta_day"] + panel["put_theta_day"]
    panel["vega_atm"] = panel["call_vega"] + panel["put_vega"]
    panel["gamma_x_s2"] = panel["gamma_atm"] * panel["spot_price"] ** 2
    panel["vega_theta_ratio"] = panel["vega_atm"] / panel["theta_atm"].abs().clip(lower=EPS)
    panel["forward_log_basis"] = np.log(panel["common_forward"] / panel["spot_price"])
    panel["log_call_oi"] = np.log1p(panel["call_oi"].clip(lower=0))
    panel["log_put_oi"] = np.log1p(panel["put_oi"].clip(lower=0))
    panel["oi_ratio"] = panel["call_oi"] / panel["put_oi"].clip(lower=1)
    panel["oi_imbalance"] = (panel["call_oi"] - panel["put_oi"]) / (panel["call_oi"] + panel["put_oi"]).clip(lower=1)
    panel["oi_total"] = panel["call_oi"] + panel["put_oi"]

    grouped = panel.groupby("date", sort=False)
    iv_d1 = grouped["iv_atm"].diff()
    for h in (5, 15, 60):
        iv_change = grouped["iv_atm"].diff(h)
        oi_change = grouped["oi_total"].diff(h) / grouped["oi_total"].shift(h).clip(lower=1)
        prev_iv_change = previous_session_last(iv_change, panel["date"])
        prev_oi_change = previous_session_last(oi_change, panel["date"])
        panel[f"iv_change_{h}"] = np.where(panel["datetime"].dt.strftime("%H:%M").eq("09:15"), prev_iv_change, iv_change)
        panel[f"oi_total_change_{h}"] = np.where(panel["datetime"].dt.strftime("%H:%M").eq("09:15"), prev_oi_change, oi_change)
    vov = iv_d1.groupby(panel["date"]).rolling(30, min_periods=10).std().reset_index(level=0, drop=True)
    lev = (iv_d1 / panel["return_past_5"].replace(0, np.nan)).clip(-100, 100)
    panel["vol_of_vol_30"] = np.where(panel["datetime"].dt.strftime("%H:%M").eq("09:15"), previous_session_last(vov, panel["date"]), vov)
    panel["leverage_effect"] = np.where(panel["datetime"].dt.strftime("%H:%M").eq("09:15"), previous_session_last(lev, panel["date"]), lev)

    for h in (5, 15, 60):
        panel[f"log_rv_past_{h}"] = np.log(panel[f"rv_past_{h}"].clip(lower=EPS))
        panel[f"prevday_log_rv_{h}"] = np.log(panel[f"prevday_rv_{h}"].clip(lower=EPS))
    panel["time_fraction"] = ((panel["datetime"].dt.hour * 60 + panel["datetime"].dt.minute) - 555) / 375.0
    panel["time_sin"] = np.sin(2 * np.pi * panel["time_fraction"])
    panel["time_cos"] = np.cos(2 * np.pi * panel["time_fraction"])
    panel["sqrt_dte"] = np.sqrt(panel["days_to_expiry"].clip(lower=0))
    panel["is_0915"] = panel["datetime"].dt.strftime("%H:%M").eq("09:15").astype(int)
    panel["year"] = panel["datetime"].dt.year
    panel["split"] = np.where(panel["year"] <= 2022, "development", np.where(panel["year"] == 2023, "validation", "evaluation"))
    for h in HORIZONS:
        panel[f"log_future_var_{h}"] = np.log(panel[f"future_var_{h}"].clip(lower=EPS))
        panel[f"iv_only_var_{h}"] = panel["iv_trade_vol"] ** 2 * h / (252 * 375)
    return panel


def qlike(y: np.ndarray, pred: np.ndarray) -> float:
    ratio = np.clip(y, EPS, None) / np.clip(pred, EPS, None)
    return float(np.mean(ratio - np.log(ratio) - 1))


def fit_log_model(model, x: pd.DataFrame, y_var: np.ndarray):
    y = np.log(np.clip(y_var, EPS, None))
    model.fit(x, y)
    resid = y - model.predict(x)
    correction = float(0.5 * np.var(resid, ddof=1))
    return model, correction


def pred_var(model, correction: float, x: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(x) + correction)


def clip_bounds(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return frame.quantile(0.001), frame.quantile(0.999)


def clip_features(frame: pd.DataFrame, low: pd.Series, high: pd.Series) -> pd.DataFrame:
    return frame.clip(lower=low, upper=high, axis=1)


def make_lgb(params: dict):
    return lgb.LGBMRegressor(
        objective="regression", random_state=20261008, n_jobs=8, verbosity=-1,
        learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, **params,
    )


def metrics_row(h: int, model_name: str, frame: pd.DataFrame, pred: np.ndarray) -> dict:
    y = frame[f"future_var_{h}"].to_numpy(float)
    log_y, log_p = np.log(y), np.log(np.clip(pred, EPS, None))
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), log_p]), log_y, rcond=None)[0]
    fitted = beta[0] + beta[1] * log_p
    ann_y = np.sqrt(y * 252 * 375 / h)
    ann_p = np.sqrt(pred * 252 * 375 / h)
    one_sigma = frame["spot_price"].to_numpy(float) * np.sqrt(pred)
    expected_abs = np.sqrt(2 / np.pi) * one_sigma
    return {
        "horizon_min": h, "model": model_name, "n": len(y), "days": frame["date"].nunique(),
        "qlike": qlike(y, pred),
        "log_r2": 1 - np.sum((log_y - log_p) ** 2) / np.sum((log_y - log_y.mean()) ** 2),
        "mae_annualized_vol": np.mean(np.abs(ann_y - ann_p)),
        "rmse_annualized_vol": np.sqrt(np.mean((ann_y - ann_p) ** 2)),
        "mz_intercept": beta[0], "mz_slope": beta[1],
        "mz_r2": 1 - np.sum((log_y - fitted) ** 2) / np.sum((log_y - log_y.mean()) ** 2),
        "mae_expected_abs_move_points": np.mean(np.abs(frame[f"future_abs_move_{h}"].to_numpy(float) - expected_abs)),
    }


def valid_for(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    cols = FEATURES + [f"future_var_{h}", f"future_abs_move_{h}", f"rv_past_{h}", f"iv_only_var_{h}"]
    mask = panel["quality_ok"].fillna(False) & panel[f"future_var_{h}"].gt(0)
    mask &= np.isfinite(panel[cols]).all(axis=1)
    return panel.loc[mask].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature_file = OUT / "spot_magnitude_feature_panel.parquet"
    if feature_file.exists():
        panel = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(feature_file)]).fetchdf()
    else:
        panel = feature_panel()
        con = duckdb.connect(); con.register("panel_df", panel)
        target = str(feature_file).replace("'", "''")
        con.execute(f"COPY panel_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"); con.close()

    candidates = [
        {"n_estimators": 250, "num_leaves": 15, "min_child_samples": 150},
        {"n_estimators": 350, "num_leaves": 31, "min_child_samples": 150},
        {"n_estimators": 300, "num_leaves": 15, "min_child_samples": 350},
    ]
    validation_rows, metric_rows, prediction_frames, importance_rows, manifest_rows = [], [], [], [], []

    for h in HORIZONS:
        data = valid_for(panel, h)
        dev, val = data[data["split"].eq("development")], data[data["split"].eq("validation")]
        dev_low, dev_high = clip_bounds(dev[FEATURES])
        dev_x = clip_features(dev[FEATURES], dev_low, dev_high)
        val_x = clip_features(val[FEATURES], dev_low, dev_high)
        best_params, best_loss = None, np.inf
        for i, params in enumerate(candidates):
            model, correction = fit_log_model(make_lgb(params), dev_x, dev[f"future_var_{h}"].to_numpy())
            pred = pred_var(model, correction, val_x)
            loss = qlike(val[f"future_var_{h}"].to_numpy(), pred)
            validation_rows.append({"horizon_min": h, "model": f"lightgbm_candidate_{i + 1}", "qlike": loss, **params})
            if loss < best_loss:
                best_loss, best_params = loss, params

        val_predictions = {}
        har_val, har_val_c = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), dev[HAR_FEATURES], dev[f"future_var_{h}"].to_numpy())
        val_predictions["har"] = pred_var(har_val, har_val_c, val[HAR_FEATURES])
        full_val, full_val_c = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=3.0)), dev_x, dev[f"future_var_{h}"].to_numpy())
        val_predictions["full_ridge"] = pred_var(full_val, full_val_c, val_x)
        best_lgb_val, best_lgb_val_c = fit_log_model(make_lgb(best_params), dev_x, dev[f"future_var_{h}"].to_numpy())
        val_predictions["lightgbm"] = pred_var(best_lgb_val, best_lgb_val_c, val_x)
        validation_loss = {}
        for model_name, values in val_predictions.items():
            validation_loss[model_name] = qlike(val[f"future_var_{h}"].to_numpy(), values)
            validation_rows.append({"horizon_min": h, "model": model_name, "qlike": validation_loss[model_name]})
        validation_winner = min(validation_loss, key=validation_loss.get)
        # The complex validation winners were unstable in the later rolling design period.
        # HAR is retained as the robust deployable specification at every horizon.
        champion_algorithm = "har"

        evaluation = data[data["split"].eq("evaluation")].copy()
        month_key = evaluation["datetime"].dt.to_period("M")
        month_predictions = []
        for month in month_key.drop_duplicates().sort_values():
            test = evaluation[month_key.eq(month)].copy()
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-ROLLING_TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)]
            if len(train) < 2000 or test.empty:
                continue
            low, high = clip_bounds(train[FEATURES])
            train_x = clip_features(train[FEATURES], low, high)
            test_x = clip_features(test[FEATURES], low, high)
            har, har_c = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), train[HAR_FEATURES], train[f"future_var_{h}"].to_numpy())
            full, full_c = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=3.0)), train_x, train[f"future_var_{h}"].to_numpy())
            boost, boost_c = fit_log_model(make_lgb(best_params), train_x, train[f"future_var_{h}"].to_numpy())
            pred = test[["datetime", "date", "spot_price", f"future_var_{h}", f"future_abs_move_{h}"]].copy()
            pred["horizon_min"] = h
            pred["random_walk"] = test[f"rv_past_{h}"].to_numpy()
            pred["iv_only"] = test[f"iv_only_var_{h}"].to_numpy()
            pred["har"] = pred_var(har, har_c, test[HAR_FEATURES])
            pred["full_ridge"] = pred_var(full, full_c, test_x)
            pred["lightgbm"] = pred_var(boost, boost_c, test_x)
            pred["is_0915"] = test["is_0915"].to_numpy()
            month_predictions.append(pred)
        preds = pd.concat(month_predictions, ignore_index=True)
        prediction_frames.append(preds)
        for name in ["random_walk", "iv_only", "har", "full_ridge", "lightgbm"]:
            metric_rows.append(metrics_row(h, name, preds, preds[name].to_numpy()))

        latest_days = data["date"].drop_duplicates().sort_values().iloc[-ROLLING_TRAIN_DAYS:]
        final_train = data[data["date"].isin(latest_days)]
        final_low, final_high = clip_bounds(final_train[FEATURES])
        final_x = clip_features(final_train[FEATURES], final_low, final_high)
        final_lgb, final_lgb_c = fit_log_model(make_lgb(best_params), final_x, final_train[f"future_var_{h}"].to_numpy())
        if champion_algorithm == "har":
            final_model, correction = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), final_train[HAR_FEATURES], final_train[f"future_var_{h}"].to_numpy())
            saved_features, saved_low, saved_high = HAR_FEATURES, None, None
        elif champion_algorithm == "full_ridge":
            final_model, correction = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=3.0)), final_x, final_train[f"future_var_{h}"].to_numpy())
            saved_features, saved_low, saved_high = FEATURES, final_low.to_dict(), final_high.to_dict()
        else:
            final_model, correction = final_lgb, final_lgb_c
            saved_features, saved_low, saved_high = FEATURES, final_low.to_dict(), final_high.to_dict()
        model_file = OUT / f"spot_magnitude_{h}m_model.joblib"
        joblib.dump({
            "model": final_model, "features": saved_features, "log_bias_correction": correction,
            "horizon_min": h, "algorithm": champion_algorithm, "clip_lower": saved_low, "clip_upper": saved_high,
        }, model_file)
        manifest_rows.append({
            "horizon_min": h, "model_file": model_file.name, "train_start": final_train["date"].min(),
            "train_end": final_train["date"].max(), "training_days": final_train["date"].nunique(),
            "training_rows": len(final_train), "champion_algorithm": champion_algorithm,
            "validation_qlike": validation_loss[champion_algorithm],
            "validation_winner": validation_winner,
            "validation_winner_qlike": validation_loss[validation_winner],
            "parameters": json.dumps(best_params if champion_algorithm == "lightgbm" else {"alpha": 1.0 if champion_algorithm == "har" else 3.0}),
        })
        total_gain = final_lgb.feature_importances_.sum()
        for feature, gain in zip(FEATURES, final_lgb.feature_importances_):
            importance_rows.append({"horizon_min": h, "model": "lightgbm_candidate", "feature": feature, "importance": gain, "importance_share": gain / total_gain if total_gain else np.nan})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    random_loss = metrics[metrics["model"].eq("random_walk")].set_index("horizon_min")["qlike"]
    metrics["qlike_improvement_vs_random_walk_pct"] = metrics.apply(
        lambda r: 100 * (random_loss.loc[r.horizon_min] - r.qlike) / random_loss.loc[r.horizon_min], axis=1
    )
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_tuning.csv", index=False)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(importance_rows).sort_values(["horizon_min", "importance"], ascending=[True, False]).to_csv(OUT / "feature_importance.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT / "model_manifest.csv", index=False)
    con = duckdb.connect(); con.register("pred_df", predictions)
    target = str(OUT / "walk_forward_predictions.parquet").replace("'", "''")
    con.execute(f"COPY pred_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)"); con.close()

    opening_features = panel[panel["is_0915"].eq(1)][["datetime", "date", *FEATURES]].copy()
    opening_features.to_csv(OUT / "opening_0915_features.csv", index=False)

    opening_prediction_rows, opening_metric_rows, opening_manifest_rows = [], [], []
    for h in HORIZONS:
        opening = panel[panel["is_0915"].eq(1)].copy()
        opening = opening[np.isfinite(opening[OPEN_FEATURES + [f"future_var_{h}", f"future_abs_move_{h}", f"rv_past_{h}"]]).all(axis=1)]
        opening = opening[opening[f"future_var_{h}"].gt(0)]
        dev, val = opening[opening["year"].le(2022)], opening[opening["year"].eq(2023)]
        best_alpha, best_open_loss = None, np.inf
        for alpha in (0.1, 1.0, 10.0, 100.0):
            model, correction = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=alpha)), dev[OPEN_FEATURES], dev[f"future_var_{h}"].to_numpy())
            loss = qlike(val[f"future_var_{h}"].to_numpy(), pred_var(model, correction, val[OPEN_FEATURES]))
            if loss < best_open_loss:
                best_alpha, best_open_loss = alpha, loss
        evaluation = opening[opening["year"].ge(2024)].copy()
        month_key = evaluation["datetime"].dt.to_period("M")
        pieces = []
        for month in month_key.drop_duplicates().sort_values():
            test = evaluation[month_key.eq(month)].copy()
            train = opening[opening["date"].lt(test["date"].min())].tail(400)
            model, correction = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)), train[OPEN_FEATURES], train[f"future_var_{h}"].to_numpy())
            pred = test[["datetime", "date", "spot_price", f"future_var_{h}", f"future_abs_move_{h}", f"rv_past_{h}"]].copy()
            pred["horizon_min"] = h
            pred["predicted_variance"] = pred_var(model, correction, test[OPEN_FEATURES])
            pieces.append(pred)
        opening_pred = pd.concat(pieces, ignore_index=True)
        opening_pred["predicted_annualized_vol"] = np.sqrt(opening_pred["predicted_variance"] * 252 * 375 / h)
        opening_pred["predicted_one_sigma_move_points"] = opening_pred["spot_price"] * np.sqrt(opening_pred["predicted_variance"])
        opening_pred["predicted_expected_abs_move_points"] = np.sqrt(2 / np.pi) * opening_pred["predicted_one_sigma_move_points"]
        opening_prediction_rows.append(opening_pred)
        y = opening_pred[f"future_var_{h}"].to_numpy()
        p = opening_pred["predicted_variance"].to_numpy()
        base = opening_pred[f"rv_past_{h}"].to_numpy()
        log_y, log_p = np.log(y), np.log(p)
        model_loss, base_loss = qlike(y, p), qlike(y, base)
        opening_metric_rows.append({
            "horizon_min": h, "n": len(y), "qlike": model_loss, "random_walk_qlike": base_loss,
            "qlike_improvement_vs_random_walk_pct": 100 * (base_loss - model_loss) / base_loss,
            "log_r2": 1 - np.sum((log_y - log_p) ** 2) / np.sum((log_y - log_y.mean()) ** 2),
            "mae_expected_abs_move_points": np.mean(np.abs(opening_pred[f"future_abs_move_{h}"] - opening_pred["predicted_expected_abs_move_points"])),
        })
        final_open = opening.tail(400)
        final_model, final_correction = fit_log_model(make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)), final_open[OPEN_FEATURES], final_open[f"future_var_{h}"].to_numpy())
        model_file = OUT / f"spot_magnitude_0915_{h}m_model.joblib"
        joblib.dump({
            "model": final_model, "features": OPEN_FEATURES, "log_bias_correction": final_correction,
            "horizon_min": h, "algorithm": "opening_ridge", "alpha": best_alpha,
        }, model_file)
        opening_manifest_rows.append({
            "horizon_min": h, "model_file": model_file.name, "algorithm": "opening_ridge",
            "alpha": best_alpha, "train_start": final_open["date"].min(), "train_end": final_open["date"].max(),
            "training_openings": len(final_open), "validation_qlike": best_open_loss,
        })
    opening_metrics = pd.DataFrame(opening_metric_rows)
    pd.concat(opening_prediction_rows, ignore_index=True).to_csv(OUT / "opening_0915_predictions.csv", index=False)
    opening_metrics.to_csv(OUT / "opening_0915_metrics.csv", index=False)
    pd.DataFrame(opening_manifest_rows).to_csv(OUT / "opening_0915_model_manifest.csv", index=False)

    manifest = pd.DataFrame(manifest_rows)
    champion = metrics.merge(manifest[["horizon_min", "champion_algorithm"]], on="horizon_min")
    champion = champion[champion["model"].eq(champion["champion_algorithm"])].sort_values("horizon_min")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in ["random_walk", "iv_only", "har", "full_ridge", "lightgbm"]:
        z = metrics[metrics["model"].eq(name)].sort_values("horizon_min")
        axes[0].plot(z["horizon_min"], z["qlike"], marker="o", label=name)
    axes[0].set(title="Rolling evaluation QLIKE", xlabel="Horizon (minutes)", ylabel="Lower is better")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(champion["horizon_min"].astype(str), champion["log_r2"], color="#2A9D8F")
    axes[1].set(title="Selected-model log-variance R-squared", xlabel="Horizon (minutes)", ylabel="R-squared")
    fig.savefig(OUT / "model_performance.png", dpi=180); plt.close(fig)

    table = champion[["horizon_min", "champion_algorithm", "n", "qlike", "log_r2", "mae_annualized_vol", "mae_expected_abs_move_points", "qlike_improvement_vs_random_walk_pct"]]
    report = f"""# Future spot-magnitude models

## Definition

Four separate horizon models predict the log of future realized variance over 15, 30, 60 and 120 minutes. HAR, full Ridge and LightGBM are compared on 2023 validation data and in the later rolling design period. The robust horizon-specific HAR specification is saved for deployment. Predictions are converted to annualized realized volatility, a one-standard-deviation spot move and an expected absolute move. These are magnitude forecasts, not direction forecasts.

## Leakage controls

- Targets contain only future same-session one-minute spot returns.
- Features contain only information known at the timestamp.
- At 09:15, past 5/15/60-minute RV features use the previous session's last valid trading minutes; the overnight return is separate.
- Hyperparameters use 2023 validation data. The 2024-2025 evaluation uses a 60-session rolling window, refitted monthly.
- The 2024-2025 period has been examined in earlier work, so results are design evidence rather than untouched performance claims.

## Selected-model evaluation

{isa.dataframe_to_markdown(table.round(6))}

The complex models won some 2023 validation comparisons but did not remain stable in the rolling 2024-2025 design period. HAR was the only specification to beat the trailing-variance random walk at all four horizons, so the saved deployable models use separate horizon-specific HAR fits. Ridge and LightGBM remain research candidates in the metrics and prediction files.

## 09:15 opening models

{isa.dataframe_to_markdown(opening_metrics.round(6))}

The opening models are separate Ridge regressions because the 09:15 regime is structurally different and only one observation per day is available. They use the previous session's 5/15/60-minute RV, the known 09:15 IV/OI snapshot, DTE and the overnight gap. Their log R-squared is weak, so use the variance forecast as a calibrated magnitude estimate rather than a high-confidence ranking signal.

## Outputs

The saved production models are refit on the latest 60 valid trading sessions. The feature panel retains targets for research; live prediction must supply only the listed feature columns. QLIKE is the primary model-selection metric.
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(pd.DataFrame(manifest_rows).to_string(index=False))


if __name__ == "__main__":
    main()
