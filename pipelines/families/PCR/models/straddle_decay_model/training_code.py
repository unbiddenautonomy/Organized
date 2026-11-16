"""Train and backtest the shared expiry-week calibrated four-model stack.

The implementation follows the user-supplied architecture:

* one shared 60-minute net-decay regression;
* one shared 60-minute probability-of-profit classifier;
* one shared two-output (60/120 minute) holding-period regression;
* one shared large-loss classifier;
* entry thresholds calibrated separately for expiry weeks 0, 1, 2, and 3/4.

Leakage controls
----------------
Model families are selected on calendar 2021 after fitting only earlier data.
Final regressors are fitted through 2022.  Final classifiers are fitted through
2021 and probability-calibrated on 2022.  Calendar 2023 is used only to freeze
expiry-bucket entry thresholds.  January 2024 onward is used only once, for the
reported backtest.  Features are lagged one minute and exits retain the exact
entry expiry and strike via the imported causal dataset builder.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


HERE = Path(__file__).resolve().parent
ROOT = HERE / "output" / "expiry_week_shared_stack"
MODEL_ROOT = ROOT / "models"
WIN_CACHE = HERE / "output" / "win_classifier" / "all_m1_rows_features_v2.pkl"
TEST_START = pd.Timestamp("2024-01-01")
THRESHOLD_START = pd.Timestamp("2023-01-01")
PROBABILITY_CALIBRATION_START = pd.Timestamp("2022-01-01")
MODEL_SELECTION_START = pd.Timestamp("2021-01-01")
HORIZONS = (60, 120)
EXTENSION_BUFFER_POINTS = 5.0
LARGE_LOSS_POINTS = 40.0
SEED = 42


def expiry_bucket(dte: pd.Series) -> pd.Series:
    raw = np.floor(pd.to_numeric(dte, errors="coerce") / 7.0).clip(0, 4)
    return pd.Series(raw, index=dte.index).astype("Int64")


def calibration_group(bucket: pd.Series) -> pd.Series:
    return bucket.astype(str).replace({"3": "3_4", "4": "3_4"})


def regressors() -> dict[str, object]:
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=12.0)
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=240, min_samples_leaf=20, max_features=0.75,
                n_jobs=-1, random_state=SEED,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=220, max_leaf_nodes=31,
                min_samples_leaf=30, l2_regularization=2.0, random_state=SEED,
            ),
        ),
        "lightgbm": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                n_estimators=300, learning_rate=0.035, num_leaves=31,
                min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1,
            ),
        ),
        "xgboost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=300, learning_rate=0.035, max_depth=5,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, objective="reg:squarederror", tree_method="hist",
                random_state=SEED, n_jobs=-1,
            ),
        ),
    }


def classifiers() -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.25, max_iter=3000, random_state=SEED),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=260, min_samples_leaf=20, max_features=0.75,
                class_weight="balanced", n_jobs=-1, random_state=SEED,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=220, max_leaf_nodes=31,
                min_samples_leaf=30, l2_regularization=2.0, random_state=SEED,
            ),
        ),
        "lightgbm": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=300, learning_rate=0.035, num_leaves=31,
                min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=2.0, class_weight="balanced", random_state=SEED,
                n_jobs=-1, verbosity=-1,
            ),
        ),
        "xgboost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=300, learning_rate=0.035, max_depth=5,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=SEED, n_jobs=-1,
            ),
        ),
    }


def sigmoid_calibrator(raw_probability: np.ndarray, y: np.ndarray) -> LogisticRegression:
    eps = 1e-6
    score = np.log(np.clip(raw_probability, eps, 1 - eps) / np.clip(1 - raw_probability, eps, 1 - eps))
    return LogisticRegression(C=1e6, max_iter=2000, random_state=SEED).fit(score.reshape(-1, 1), y)


def calibrated_probability(model: object, calibrator: LogisticRegression, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)[:, 1]
    eps = 1e-6
    score = np.log(np.clip(raw, eps, 1 - eps) / np.clip(1 - raw, eps, 1 - eps))
    return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]


def build_learning_table() -> tuple[pd.DataFrame, list[str]]:
    sys.path.insert(0, str(HERE))
    import train_win_classifier as source

    features = list(source.FEATURES) + [f"expiry_week_{value}" for value in range(5)]
    learning_cache = ROOT / "learning_table_60_120.pkl"
    if learning_cache.exists():
        print(f"Using cached 60/120 learning table: {learning_cache}", flush=True)
        return pd.read_pickle(learning_cache), features

    raw = pd.read_pickle(WIN_CACHE)
    panel = source.build_atm_panel(raw)
    frames: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        print(f"Building causal {horizon}-minute outcomes", flush=True)
        part = source.horizon_dataset(panel, raw, horizon).copy()
        part["datetime"] = part.index
        part["net_points"] = part["net_rupees"] / source.lot_size(pd.DatetimeIndex(part.index.normalize()))
        frames[horizon] = part

    base_columns = ["datetime", "expiry", "strike", "days_to_expiry", *source.FEATURES]
    left = frames[60][base_columns + [
        "exit_datetime", "exit_straddle_premium", "synthetic_exit", "gross_points",
        "cost_rupees", "slippage_points", "net_rupees", "net_points", "win",
        "call_price", "put_price",
    ]].copy()
    left = left.reset_index(drop=True)
    left = left.rename(columns={
        "exit_datetime": "exit_datetime_60", "exit_straddle_premium": "exit_premium_60",
        "synthetic_exit": "synthetic_exit_60", "gross_points": "gross_points_60",
        "cost_rupees": "cost_rupees_60", "slippage_points": "slippage_points_60",
        "net_rupees": "net_rupees_60", "net_points": "net_points_60", "win": "win_60",
    })
    right = frames[120][[
        "datetime", "exit_datetime", "exit_straddle_premium", "synthetic_exit",
        "gross_points", "cost_rupees", "slippage_points", "net_rupees", "net_points", "win",
    ]].copy().reset_index(drop=True).rename(columns={
        "exit_datetime": "exit_datetime_120", "exit_straddle_premium": "exit_premium_120",
        "synthetic_exit": "synthetic_exit_120", "gross_points": "gross_points_120",
        "cost_rupees": "cost_rupees_120", "slippage_points": "slippage_points_120",
        "net_rupees": "net_rupees_120", "net_points": "net_points_120", "win": "win_120",
    })
    table = left.merge(right, on="datetime", how="inner", validate="one_to_one").set_index("datetime").sort_index()
    table["expiry_bucket"] = expiry_bucket(table["days_to_expiry"])
    table["calibration_group"] = calibration_group(table["expiry_bucket"])
    for value in range(5):
        table[f"expiry_week_{value}"] = (table["expiry_bucket"] == value).astype(float)
    table["tail_event"] = (
        table[["net_points_60", "net_points_120"]].min(axis=1) <= -LARGE_LOSS_POINTS
    ).astype(int)
    table.loc[:, features] = table[features].replace([np.inf, -np.inf], np.nan)
    table.to_pickle(learning_cache)
    return table, features


def select_regressor(table: pd.DataFrame, features: list[str], targets: list[str], label: str) -> tuple[str, pd.DataFrame]:
    fit = table.loc[table.index < MODEL_SELECTION_START]
    validation = table.loc[(table.index >= MODEL_SELECTION_START) & (table.index < PROBABILITY_CALIBRATION_START)]
    rows = []
    for name, template in regressors().items():
        print(f"Selecting {label} family: {name}", flush=True)
        model = (
            MultiOutputRegressor(clone(template), n_jobs=-1)
            if len(targets) > 1 else clone(template)
        )
        model.fit(fit[features], fit[targets] if len(targets) > 1 else fit[targets[0]])
        prediction = model.predict(validation[features])
        if len(targets) == 1:
            prediction = np.asarray(prediction).reshape(-1, 1)
        row = {"goal": label, "model": name}
        maes = []
        for index, target in enumerate(targets):
            mae = mean_absolute_error(validation[target], prediction[:, index])
            row[f"mae_{target}"] = mae
            maes.append(mae)
        row["selection_score"] = float(np.mean(maes))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["selection_score", "model"])
    return str(result.iloc[0]["model"]), result


def select_classifier(table: pd.DataFrame, features: list[str], target: str, label: str) -> tuple[str, pd.DataFrame]:
    fit = table.loc[table.index < MODEL_SELECTION_START]
    validation = table.loc[(table.index >= MODEL_SELECTION_START) & (table.index < PROBABILITY_CALIBRATION_START)]
    rows = []
    for name, template in classifiers().items():
        print(f"Selecting {label} family: {name}", flush=True)
        model = clone(template).fit(fit[features], fit[target])
        probability = model.predict_proba(validation[features])[:, 1]
        rows.append({
            "goal": label, "model": name,
            "brier_score": brier_score_loss(validation[target], probability),
            "roc_auc": roc_auc_score(validation[target], probability),
            "average_precision": average_precision_score(validation[target], probability),
        })
    result = pd.DataFrame(rows).sort_values(["brier_score", "model"])
    return str(result.iloc[0]["model"]), result


def fit_final_models(table: pd.DataFrame, features: list[str], selected: dict[str, str]) -> dict[str, object]:
    regression_fit = table.loc[table.index < THRESHOLD_START]
    classifier_fit = table.loc[table.index < PROBABILITY_CALIBRATION_START]
    probability_calibration = table.loc[
        (table.index >= PROBABILITY_CALIBRATION_START) & (table.index < THRESHOLD_START)
    ]

    decay = clone(regressors()[selected["decay"]]).fit(regression_fit[features], regression_fit["net_points_60"])
    holding = MultiOutputRegressor(
        clone(regressors()[selected["holding"]]), n_jobs=-1
    ).fit(
        regression_fit[features], regression_fit[["net_points_60", "net_points_120"]]
    )
    win = clone(classifiers()[selected["win"]]).fit(classifier_fit[features], classifier_fit["win_60"])
    win_calibrator = sigmoid_calibrator(
        win.predict_proba(probability_calibration[features])[:, 1], probability_calibration["win_60"].to_numpy()
    )
    tail = clone(classifiers()[selected["tail"]]).fit(classifier_fit[features], classifier_fit["tail_event"])
    tail_calibrator = sigmoid_calibrator(
        tail.predict_proba(probability_calibration[features])[:, 1], probability_calibration["tail_event"].to_numpy()
    )
    return {
        "decay": decay, "holding": holding, "win": win, "win_calibrator": win_calibrator,
        "tail": tail, "tail_calibrator": tail_calibrator,
    }


def add_predictions(frame: pd.DataFrame, features: list[str], models: dict[str, object]) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_decay_60_points"] = models["decay"].predict(result[features])
    holding = models["holding"].predict(result[features])
    result["predicted_pnl_60_points"] = holding[:, 0]
    result["predicted_pnl_120_points"] = holding[:, 1]
    result["win_probability"] = calibrated_probability(models["win"], models["win_calibrator"], result[features])
    result["large_loss_probability"] = calibrated_probability(models["tail"], models["tail_calibrator"], result[features])
    result["horizon_minutes"] = np.where(
        result["predicted_pnl_120_points"] > result["predicted_pnl_60_points"] + EXTENSION_BUFFER_POINTS,
        120, 60,
    )
    return result


def calibrate_thresholds(calibration: pd.DataFrame) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for group in ("0", "1", "2", "3_4"):
        rows = calibration.loc[calibration["calibration_group"].eq(group)]
        if rows.empty:
            raise RuntimeError(f"No 2023 calibration rows for expiry group {group}")
        quantile = 0.80 if group in {"0", "1"} else 0.85
        risk_limit = 0.15 if group in {"0", "1"} else (0.10 if group == "2" else 0.125)
        thresholds[group] = {
            "entry_quantile": quantile,
            "minimum_expected_points": max(
                0.0, float(rows["predicted_decay_60_points"].quantile(quantile))
            ),
            "minimum_win_probability": float(rows["win_probability"].quantile(quantile)),
            "maximum_large_loss_probability": risk_limit,
        }
    return thresholds


def apply_gates(frame: pd.DataFrame, thresholds: dict[str, dict[str, float]]) -> pd.DataFrame:
    result = frame.copy()
    for key in ("minimum_expected_points", "minimum_win_probability", "maximum_large_loss_probability"):
        result[key] = result["calibration_group"].map({group: values[key] for group, values in thresholds.items()})
    result["decay_pass"] = result["predicted_decay_60_points"] > result["minimum_expected_points"]
    result["win_pass"] = result["win_probability"] >= result["minimum_win_probability"]
    result["tail_pass"] = result["large_loss_probability"] <= result["maximum_large_loss_probability"]
    result["all_gates_pass"] = result["decay_pass"] & result["win_pass"] & result["tail_pass"]
    return result


def materialize_selected_outcome(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for base in (
        "exit_datetime", "exit_premium", "synthetic_exit", "gross_points",
        "cost_rupees", "slippage_points", "net_rupees", "net_points", "win",
    ):
        result[base] = np.where(
            result["horizon_minutes"].eq(120), result[f"{base}_120"], result[f"{base}_60"]
        )
    result["exit_datetime"] = pd.to_datetime(result["exit_datetime"])
    result["lot_size"] = result["net_rupees"] / result["net_points"].replace(0, np.nan)
    result["lot_size"] = result["lot_size"].fillna(15).round().astype(int)
    result["slippage_rupees"] = result["slippage_points"] * result["lot_size"]
    result["total_cost_rupees"] = result["cost_rupees"] + result["slippage_rupees"]
    result["gross_rupees"] = result["gross_points"] * result["lot_size"]
    return result


def enforce_one_open(signals: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    selected = []
    active_until = pd.Timestamp.min
    for timestamp, row in signals.sort_index().iterrows():
        if timestamp >= active_until:
            selected.append(timestamp)
            active_until = pd.Timestamp(row["exit_datetime"])
    return signals.loc[selected].copy(), len(signals) - len(selected)


def max_drawdown(values: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None, int]:
    equity = values.cumsum()
    peak = equity.cummax().clip(lower=0.0)
    drawdown = equity - peak
    trough = drawdown.idxmin() if len(drawdown) else None
    peak_date = equity.loc[:trough].idxmax() if trough is not None and len(equity.loc[:trough]) else None
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return float(drawdown.min()) if len(drawdown) else 0.0, peak_date, trough, longest


def aggregate_metrics(trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, object]:
    wins = trades.loc[trades["net_rupees"] > 0, "net_rupees"]
    losses = trades.loc[trades["net_rupees"] < 0, "net_rupees"]
    daily_series = pd.Series(
        daily["net_rupees"].to_numpy(float), index=pd.to_datetime(daily["date"])
    )
    dd, peak, trough, duration = max_drawdown(daily_series)
    daily_values = daily["net_rupees"].to_numpy(float)
    std = np.std(daily_values, ddof=1) if len(daily_values) > 1 else 0.0
    downside = np.sqrt(np.mean(np.minimum(daily_values, 0.0) ** 2)) if len(daily_values) else 0.0
    rng = np.random.default_rng(SEED)
    if len(trades):
        boot = rng.choice(trades["net_rupees"].to_numpy(), size=(10000, len(trades)), replace=True).mean(axis=1)
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        t_p = stats.ttest_1samp(trades["net_rupees"], 0.0, alternative="greater").pvalue if len(trades) > 1 else math.nan
    else:
        ci_low = ci_high = t_p = math.nan
    return {
        "trades": len(trades), "green_trades": int((trades["net_rupees"] > 0).sum()),
        "red_trades": int((trades["net_rupees"] < 0).sum()),
        "win_rate_pct": 100.0 * float((trades["net_rupees"] > 0).mean()) if len(trades) else math.nan,
        "gross_rupees": float(trades["gross_rupees"].sum()),
        "statutory_cost_rupees": float(trades["cost_rupees"].sum()),
        "slippage_rupees": float(trades["slippage_rupees"].sum()),
        "total_cost_rupees": float(trades["total_cost_rupees"].sum()),
        "net_rupees": float(trades["net_rupees"].sum()), "net_points": float(trades["net_points"].sum()),
        "mean_net_rupees": float(trades["net_rupees"].mean()) if len(trades) else math.nan,
        "median_net_rupees": float(trades["net_rupees"].median()) if len(trades) else math.nan,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else math.inf,
        "largest_win_rupees": float(trades["net_rupees"].max()) if len(trades) else math.nan,
        "largest_loss_rupees": float(trades["net_rupees"].min()) if len(trades) else math.nan,
        "market_days": len(daily), "active_days": int((daily["net_rupees"] != 0).sum()),
        "green_days": int((daily["net_rupees"] > 0).sum()), "red_days": int((daily["net_rupees"] < 0).sum()),
        "flat_days": int((daily["net_rupees"] == 0).sum()), "max_drawdown_rupees": dd,
        "drawdown_peak": str(peak.date()) if peak is not None else None,
        "drawdown_trough": str(trough.date()) if trough is not None else None,
        "max_drawdown_duration_market_days": duration,
        "daily_sharpe": math.sqrt(252) * float(np.mean(daily_values) / std) if std else math.nan,
        "daily_sortino": math.sqrt(252) * float(np.mean(daily_values) / downside) if downside else math.nan,
        "mean_trade_ttest_pvalue": float(t_p), "mean_trade_ci95_low": float(ci_low), "mean_trade_ci95_high": float(ci_high),
        "synthetic_exit_trades": int(trades["synthetic_exit"].sum()) if len(trades) else 0,
        "horizon_60_trades": int(trades["horizon_minutes"].eq(60).sum()),
        "horizon_120_trades": int(trades["horizon_minutes"].eq(120).sum()),
    }


def grouped_metrics(trades: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, group in trades.groupby(key, observed=True):
        wins = group.loc[group["net_rupees"] > 0, "net_rupees"].sum()
        losses = group.loc[group["net_rupees"] < 0, "net_rupees"].sum()
        rows.append({
            key: value, "trades": len(group), "green_trades": int((group["net_rupees"] > 0).sum()),
            "red_trades": int((group["net_rupees"] < 0).sum()),
            "win_rate_pct": 100 * float((group["net_rupees"] > 0).mean()),
            "net_points": float(group["net_points"].sum()), "costs_rupees": float(group["total_cost_rupees"].sum()),
            "net_rupees": float(group["net_rupees"].sum()),
            "profit_factor": float(wins / abs(losses)) if losses else math.inf,
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(summary: dict[str, object], expiry: pd.DataFrame, yearly: pd.DataFrame, funnel: pd.DataFrame, selected: dict[str, str]) -> None:
    lines = [
        "# Shared-model, expiry-week-calibrated BANKNIFTY backtest", "",
        "## Frozen architecture", "",
        "One shared decay model, one shared win classifier, one two-output 60/120-minute holding model, and one shared tail-risk classifier are used across all expiry weeks. Only the entry thresholds vary by expiry bucket; weeks 3 and 4 share a calibration group.", "",
        "No model prediction is used as a training input to another model. At inference, the four independently fitted outputs are combined only by the strategy decision rules.", "",
        "## Leakage controls", "",
        "- Family selection: calendar 2021, using models fitted only before 2021.",
        "- Final regressors: fitted only before 2023.",
        "- Final classifiers: fitted only before 2022 and probability-calibrated only on 2022.",
        "- Expiry-week thresholds: frozen from 2023 only.",
        "- Backtest: January 2024 onward; never used for fitting, calibration, family selection, or threshold choice.",
        "- Entry features are lagged one minute; the entry ATM expiry and strike remain fixed through exit.", "",
        "## Result", "",
        f"The frozen stack executed **{summary['trades']} trades** and earned **Rs {summary['net_rupees']:,.2f} net** after **Rs {summary['total_cost_rupees']:,.2f}** of statutory costs and slippage. Win rate was **{summary['win_rate_pct']:.2f}%**.", "",
        "## P&L and risk", "",
        f"- Gross P&L: Rs {summary['gross_rupees']:,.2f}",
        f"- Statutory costs: Rs {summary['statutory_cost_rupees']:,.2f}",
        f"- Slippage: Rs {summary['slippage_rupees']:,.2f}",
        f"- Net P&L: Rs {summary['net_rupees']:,.2f}",
        f"- Net points: {summary['net_points']:,.2f}",
        f"- Profit factor: {summary['profit_factor']:.3f}",
        f"- Maximum drawdown: Rs {summary['max_drawdown_rupees']:,.2f}",
        f"- Drawdown peak/trough: {summary['drawdown_peak']} / {summary['drawdown_trough']}",
        f"- Longest drawdown: {summary['max_drawdown_duration_market_days']} market days",
        f"- Daily Sharpe / Sortino: {summary['daily_sharpe']:.3f} / {summary['daily_sortino']:.3f}",
        f"- Average / median trade: Rs {summary['mean_net_rupees']:,.2f} / Rs {summary['median_net_rupees']:,.2f}",
        f"- Largest win / loss: Rs {summary['largest_win_rupees']:,.2f} / Rs {summary['largest_loss_rupees']:,.2f}",
        f"- Mean-trade 95% bootstrap CI: Rs {summary['mean_trade_ci95_low']:,.2f} to Rs {summary['mean_trade_ci95_high']:,.2f}",
        f"- One-sided mean-trade t-test p-value: {summary['mean_trade_ttest_pvalue']:.4f}", "",
        "## Trading days and exits", "",
        f"- Market days: {summary['market_days']}", f"- Active / green / red / flat days: {summary['active_days']} / {summary['green_days']} / {summary['red_days']} / {summary['flat_days']}",
        f"- Green / red trades: {summary['green_trades']} / {summary['red_trades']}",
        f"- 60-minute / 120-minute trades: {summary['horizon_60_trades']} / {summary['horizon_120_trades']}",
        f"- Synthetic same-strike exit valuations: {summary['synthetic_exit_trades']}",
        f"- Overlapping signals rejected: {summary['overlap_rejections']}", "",
        "## Selected families", "", *[f"- {goal}: {name}" for goal, name in selected.items()], "",
        "## Gate funnel", "", markdown_table(funnel), "", "## Expiry-week results", "", markdown_table(expiry), "",
        "## Calendar-year results", "", markdown_table(yearly), "", "## Interpretation", "",
        "This is an historical research result, not evidence of live profitability. The most important checks are whether performance is distributed across expiry buckets and years, whether the confidence interval excludes zero, and whether drawdown remains acceptable. A frozen forward paper-trading period is still required before considering deployment.", "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def package_models(models: dict[str, object], selected: dict[str, str], features: list[str], thresholds: dict[str, dict[str, float]], selection: pd.DataFrame, summary: dict[str, object]) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    specs = {
        "straddle_decay_model": {"model": models["decay"], "family": selected["decay"], "target": "net_points_60"},
        "win_probability_model": {"model": models["win"], "calibrator": models["win_calibrator"], "family": selected["win"], "target": "win_60"},
        "holding_period_model": {"model": models["holding"], "family": selected["holding"], "target": ["net_points_60", "net_points_120"]},
        "tail_risk_model": {"model": models["tail"], "calibrator": models["tail_calibrator"], "family": selected["tail"], "target": f"min(60m,120m) net points <= -{LARGE_LOSS_POINTS:g}"},
    }
    for name, payload in specs.items():
        folder = MODEL_ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({**payload, "features": features, "thresholds": thresholds}, folder / "model_bundle.joblib", compress=3)
        joblib.dump(payload["model"], folder / "model_pipeline.joblib", compress=3)
        if "calibrator" in payload:
            joblib.dump(payload["calibrator"], folder / "probability_calibrator.joblib", compress=3)
        steps = getattr(payload["model"], "steps", [])
        if isinstance(payload["model"], MultiOutputRegressor):
            for horizon, estimator in zip(HORIZONS, payload["model"].estimators_, strict=True):
                for step_name, step in getattr(estimator, "steps", []):
                    if isinstance(step, SimpleImputer):
                        joblib.dump(step, folder / f"imputer_{horizon}m.joblib", compress=3)
                    if isinstance(step, StandardScaler):
                        joblib.dump(step, folder / f"scaler_{horizon}m.joblib", compress=3)
        for step_name, step in steps:
            if isinstance(step, SimpleImputer):
                joblib.dump(step, folder / "imputer.joblib", compress=3)
            if isinstance(step, StandardScaler):
                joblib.dump(step, folder / "scaler.joblib", compress=3)
        has_direct_scaler = any(isinstance(step, StandardScaler) for _, step in steps)
        has_multi_scaler = any(folder.glob("scaler_*m.joblib"))
        stale_scaler_note = folder / "scaler_not_required.json"
        if (has_direct_scaler or has_multi_scaler) and stale_scaler_note.exists():
            stale_scaler_note.unlink()
        if not has_direct_scaler and not has_multi_scaler:
            (folder / "scaler_not_required.json").write_text(json.dumps({"scaler_used": False, "reason": "Selected tree model is scale invariant."}, indent=2), encoding="utf-8")
        (folder / "feature_schema.json").write_text(json.dumps({"features": features, "expiry_buckets": {"0": "0-6.99 DTE", "1": "7-13.99 DTE", "2": "14-20.99 DTE", "3": "21-27.99 DTE", "4": "28-34.99 DTE"}}, indent=2), encoding="utf-8")
        (folder / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        goal_rows = selection.loc[selection["goal"].eq(name.replace("_model", ""))]
        (folder / "metrics.json").write_text(json.dumps({"selected_family": payload["family"], "selection_metrics": goal_rows.replace({np.nan: None}).to_dict("records"), "locked_test_summary": summary}, indent=2, default=str), encoding="utf-8")
        card = [
            f"# Model Card: {name.replace('_', ' ').title()}", "", f"Selected family: `{payload['family']}`.", "",
            f"Target: `{payload['target']}`.", "", "## Data separation", "",
            "Family selection uses 2021 only after fitting earlier observations. Final regressors end before 2023. Final classifiers end before 2022, use 2022 only for probability calibration, and use 2023 only for expiry-week thresholds. January 2024 onward is the locked historical test.", "",
            "## Contract integrity", "", "Entry features are lagged by one minute. The ATM expiry and strike selected at entry remain fixed through the 60- or 120-minute exit. Costs include historical lot sizes, statutory charges, and two straddle points of slippage.", "",
            "## Limitations", "", "The source contains five strikes, so absent exact exit strikes use a transparent linear strike-surface valuation with an intrinsic-value floor. Historical backtest performance is not a guarantee of live performance.", "",
        ]
        (folder / "MODEL_CARD.md").write_text("\n".join(card), encoding="utf-8")
        inference_source = '''"""Minimal loader for this packaged model."""\nfrom pathlib import Path\nimport joblib\nimport pandas as pd\n\nHERE = Path(__file__).resolve().parent\nBUNDLE = joblib.load(HERE / "model_bundle.joblib")\n\ndef predict(frame: pd.DataFrame):\n    x = frame[BUNDLE["features"]]\n    model = BUNDLE["model"]\n    if "calibrator" not in BUNDLE:\n        return model.predict(x)\n    import numpy as np\n    raw = model.predict_proba(x)[:, 1]\n    eps = 1e-6\n    score = np.log(np.clip(raw, eps, 1 - eps) / np.clip(1 - raw, eps, 1 - eps))\n    return BUNDLE["calibrator"].predict_proba(score.reshape(-1, 1))[:, 1]\n'''
        (folder / "inference.py").write_text(inference_source, encoding="utf-8")
        (folder / "PACKAGE_README.md").write_text(
            "# Package contents\n\nLoad `model_bundle.joblib` for the complete fitted object, feature order, and expiry-week thresholds. Use `inference.py` for a minimal prediction example. See `MODEL_CARD.md` for training splits and limitations.\n",
            encoding="utf-8",
        )
        shutil.copy2(Path(__file__), folder / "training_code.py")
        (folder / "requirements.txt").write_text("pandas\nnumpy\nscipy\nscikit-learn\njoblib\nlightgbm\nxgboost\n", encoding="utf-8")
        manifest = {}
        for artifact in sorted(folder.iterdir()):
            if artifact.is_file() and artifact.name != "manifest.json":
                manifest[artifact.name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        (folder / "manifest.json").write_text(json.dumps({"algorithm": "sha256", "files": manifest}, indent=2), encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    table, features = build_learning_table()
    selected: dict[str, str] = {}
    selection_parts = []
    selected["decay"], part = select_regressor(table, features, ["net_points_60"], "straddle_decay")
    selection_parts.append(part)
    selected["holding"], part = select_regressor(table, features, ["net_points_60", "net_points_120"], "holding_period")
    selection_parts.append(part)
    selected["win"], part = select_classifier(table, features, "win_60", "win_probability")
    selection_parts.append(part)
    selected["tail"], part = select_classifier(table, features, "tail_event", "tail_risk")
    selection_parts.append(part)
    selection = pd.concat(selection_parts, ignore_index=True)
    models = fit_final_models(table, features, selected)

    calibration = add_predictions(table.loc[(table.index >= THRESHOLD_START) & (table.index < TEST_START)], features, models)
    thresholds = calibrate_thresholds(calibration)
    test = add_predictions(table.loc[table.index >= TEST_START], features, models)
    test = apply_gates(test, thresholds)
    test = materialize_selected_outcome(test)
    gated = test.loc[test["all_gates_pass"]].copy()
    trades, overlap_rejections = enforce_one_open(gated)

    market_dates = pd.date_range(test.index.min().normalize(), test.index.max().normalize(), freq="B")
    observed_dates = pd.DatetimeIndex(sorted(test.index.normalize().unique()))
    market_dates = observed_dates
    daily = trades.groupby(trades.index.normalize())["net_rupees"].sum().reindex(market_dates, fill_value=0.0).rename_axis("date").reset_index()
    summary = aggregate_metrics(trades, daily)
    summary["overlap_rejections"] = overlap_rejections
    summary["test_start"] = str(test.index.min())
    summary["test_end"] = str(test.index.max())
    funnel = pd.DataFrame([
        {"stage": "test candidates", "rows": len(test)},
        {"stage": "decay pass", "rows": int(test["decay_pass"].sum())},
        {"stage": "win pass", "rows": int(test["win_pass"].sum())},
        {"stage": "tail pass", "rows": int(test["tail_pass"].sum())},
        {"stage": "all gates", "rows": len(gated)},
        {"stage": "executed after one-open rule", "rows": len(trades)},
    ])
    trades = trades.copy()
    trades["year"] = trades.index.year
    expiry = grouped_metrics(trades, "calibration_group")
    yearly = grouped_metrics(trades, "year")

    selection.to_csv(ROOT / "model_family_selection_2021.csv", index=False)
    pd.DataFrame([summary]).to_csv(ROOT / "summary.csv", index=False)
    trades.reset_index().to_csv(ROOT / "trades.csv", index=False)
    daily.to_csv(ROOT / "daily_pnl.csv", index=False)
    expiry.to_csv(ROOT / "expiry_week_metrics.csv", index=False)
    yearly.to_csv(ROOT / "yearly_metrics.csv", index=False)
    funnel.to_csv(ROOT / "gate_funnel.csv", index=False)
    test.reset_index().to_csv(ROOT / "all_test_candidates.csv.gz", index=False, compression="gzip")
    (ROOT / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (ROOT / "run_metadata.json").write_text(json.dumps({
        "architecture": "shared models with expiry-week thresholds", "selected_models": selected,
        "features": features, "horizons": HORIZONS, "extension_buffer_points": EXTENSION_BUFFER_POINTS,
        "large_loss_points": LARGE_LOSS_POINTS, "family_selection_period": "2021",
        "classifier_probability_calibration_period": "2022", "threshold_calibration_period": "2023",
        "test_start": "2024-01-01", "thresholds": thresholds,
    }, indent=2), encoding="utf-8")
    write_report(summary, expiry, yearly, funnel, selected)
    package_models(models, selected, features, thresholds, selection, summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
