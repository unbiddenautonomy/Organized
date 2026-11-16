"""Near-daily long/short premium-direction backtest with strict time splits.

The model predicts the percentage change in a same-expiry straddle/strangle
premium. At each candidate timestamp it compares predicted long and short net
edge after an entry-time cost estimate, then enters on the first observable
threshold crossing. Coverage thresholds are selected on 2023 only.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import expanded_signal_hybrid as expanded
import hybrid_regime_strategy as hybrid
import retrained_short_vol_hybrid as retrained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "near_daily_premium_direction"
SLIPPAGE_BPS = 25.0
TARGET_MIN_DAYS = 210
TARGET_MAX_DAYS = 231
TARGET_QUANTILES = (0.03, 0.05, 0.075, 0.10)


def structure_pairs() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    short = {
        "atm_straddle": expanded.historical_structure_panel(
            "short_atm_straddle", expanded.SECONDARY_STRUCTURES["short_atm_straddle"]
        ),
        "one_strangle": expanded.historical_structure_panel(
            "short_1_strangle", expanded.SECONDARY_STRUCTURES["short_1_strangle"]
        ),
        "two_strangle": retrained.historical_structure_panel(),
    }
    long = {
        "atm_straddle": smb.read_parquet(expanded.OUT / "historical_long_atm_straddle_120m.parquet"),
        "one_strangle": smb.read_parquet(expanded.OUT / "historical_long_1_strangle_120m.parquet"),
        "two_strangle": smb.read_parquet(expanded.OUT / "historical_long_2_strangle_120m.parquet"),
    }
    return {name: (long[name], short[name]) for name in long}


def prepare_pair(name: str, long: pd.DataFrame, short: pd.DataFrame) -> pd.DataFrame:
    actual = short[[
        "entry_datetime", "net_pnl_rupees", "gross_pnl_rupees", "total_cost",
    ]].rename(columns={
        "net_pnl_rupees": "actual_short_net",
        "gross_pnl_rupees": "actual_short_gross",
        "total_cost": "actual_short_cost",
    })
    frame = long.merge(actual, on="entry_datetime", how="inner", validate="one_to_one")
    frame = frame.rename(columns={
        "net_pnl_rupees": "actual_long_net",
        "gross_pnl_rupees": "actual_long_gross",
        "total_cost": "actual_long_cost",
    })
    estimate = frame.copy()
    estimate["exit_turnover_points"] = estimate["entry_turnover_points"]
    estimate["buy_turnover_points"] = estimate["entry_turnover_points"]
    estimate["sell_turnover_points"] = estimate["entry_turnover_points"]
    frame["estimated_round_trip_cost"] = smb.charge_components(
        estimate, SLIPPAGE_BPS, current_2026=True
    )["total_cost"].to_numpy()
    frame["premium_return"] = frame["gross_pnl_points"] / frame["entry_value"].abs().clip(lower=1e-9)
    frame["structure_family"] = name
    frame["year"] = frame["date"].dt.year
    return frame


def models() -> dict[str, object]:
    return {
        "ridge_100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "lightgbm_huber": lgb.LGBMRegressor(
            objective="huber", n_estimators=300, learning_rate=0.025,
            num_leaves=7, min_child_samples=400, reg_lambda=10.0,
            reg_alpha=2.0, subsample=0.8, colsample_bytree=0.8,
            random_state=20261013, n_jobs=8, verbosity=-1,
        ),
    }


def score_model(model_name: str, model_factory: object, pairs: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    fitted = {}
    scored = []
    for structure, panel in pairs.items():
        train = panel[panel["year"].le(2022)].copy()
        lo, hi = train["premium_return"].quantile([0.005, 0.995])
        target = train["premium_return"].clip(lo, hi)
        if model_name.startswith("ridge"):
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        else:
            model = lgb.LGBMRegressor(**model_factory.get_params())
        model.fit(train[retrained.MODEL_FEATURES], target)
        fitted[structure] = model
        evaluation = panel[panel["year"].ge(2023)].copy()
        evaluation["predicted_premium_return"] = model.predict(evaluation[retrained.MODEL_FEATURES])
        evaluation["predicted_long_gross"] = (
            evaluation["predicted_premium_return"] * evaluation["entry_value"].abs() * evaluation["lot_size"]
        )
        evaluation["predicted_long_net"] = evaluation["predicted_long_gross"] - evaluation["estimated_round_trip_cost"]
        evaluation["predicted_short_net"] = -evaluation["predicted_long_gross"] - evaluation["estimated_round_trip_cost"]
        evaluation["trade_side"] = np.where(
            evaluation["predicted_long_net"].ge(evaluation["predicted_short_net"]), "long_vol", "short_vol"
        )
        evaluation["predicted_net_edge"] = evaluation[["predicted_long_net", "predicted_short_net"]].max(axis=1)
        evaluation["net_pnl_rupees"] = np.where(
            evaluation["trade_side"].eq("long_vol"), evaluation["actual_long_net"], evaluation["actual_short_net"]
        )
        evaluation["gross_pnl_rupees"] = np.where(
            evaluation["trade_side"].eq("long_vol"), evaluation["actual_long_gross"], evaluation["actual_short_gross"]
        )
        evaluation["total_cost"] = np.where(
            evaluation["trade_side"].eq("long_vol"), evaluation["actual_long_cost"], evaluation["actual_short_cost"]
        )
        evaluation["model_name"] = model_name
        scored.append(evaluation)
    all_scored = pd.concat(scored, ignore_index=True, sort=False)
    return fitted, all_scored


def first_edge_crossing(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Select the earliest executable crossing, breaking same-time ties by edge."""
    eligible = scored[scored["predicted_net_edge"].ge(threshold)].copy()
    return (
        eligible.sort_values(
            ["date", "entry_datetime", "predicted_net_edge"],
            ascending=[True, True, False],
        )
        .groupby("date", as_index=False, group_keys=False)
        .head(1)
        .copy()
    )


def pnl_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"days": 0, "net_pnl": 0.0, "hit_rate": np.nan, "profit_factor": np.nan, "max_drawdown": np.nan}
    pnl = frame["net_pnl_rupees"]
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    equity = frame.sort_values("date")["net_pnl_rupees"].cumsum()
    return {
        "days": int(frame["date"].nunique()), "net_pnl": float(pnl.sum()),
        "mean_net": float(pnl.mean()), "hit_rate": float(pnl.gt(0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else np.inf,
        "max_drawdown": float((equity.cummax() - equity).max()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = {name: prepare_pair(name, *frames) for name, frames in structure_pairs().items()}
    candidate_models = models()
    fitted_all, scored_all, validation_rows = {}, {}, []
    for model_name, factory in candidate_models.items():
        fitted, scored = score_model(model_name, factory, pairs)
        fitted_all[model_name] = fitted
        scored_all[model_name] = scored
        validation_scored = scored[scored["year"].eq(2023)].copy()
        calibration_scored = validation_scored[validation_scored["date"].lt("2023-07-01")]
        for quantile in TARGET_QUANTILES:
            threshold = float(calibration_scored["predicted_net_edge"].quantile(quantile))
            selected = first_edge_crossing(validation_scored, threshold)
            h1 = selected[selected["date"].lt("2023-07-01")]
            h2 = selected[selected["date"].ge("2023-07-01")]
            validation_rows.append({
                "model": model_name, "edge_quantile": quantile, "edge_threshold": threshold,
                **pnl_metrics(selected), "H1_net": h1["net_pnl_rupees"].sum(),
                "H2_net": h2["net_pnl_rupees"].sum(),
            })
    validation = pd.DataFrame(validation_rows)
    validation["target_coverage"] = validation["days"].between(TARGET_MIN_DAYS, TARGET_MAX_DAYS)
    validation["stable_positive"] = validation["target_coverage"] & validation["H1_net"].gt(0) & validation["H2_net"].gt(0)
    eligible = validation[validation["stable_positive"]]
    if eligible.empty:
        eligible = validation[validation["target_coverage"] & validation["net_pnl"].gt(0)]
    if eligible.empty:
        selected_rule = validation[validation["target_coverage"]].sort_values("net_pnl", ascending=False).iloc[0]
        decision = "REJECT_NO_POSITIVE_TARGET_COVERAGE_RULE"
    else:
        selected_rule = eligible.sort_values(["net_pnl", "profit_factor"], ascending=False).iloc[0]
        decision = "PASS_TARGET_COVERAGE_BACKTEST_FORWARD_PAPER_ONLY"
    model_name = selected_rule["model"]
    threshold = float(selected_rule["edge_threshold"])
    evaluation_scored = scored_all[model_name]
    evaluation_scored = evaluation_scored[evaluation_scored["year"].ge(2024)]
    trades = first_edge_crossing(evaluation_scored, threshold)
    smb.write_parquet(trades, OUT / "trade_ledger.parquet")

    rows = []
    for year, period, available in ((2024, "test_2024", 235), (2025, "holdout_2025_partial", 76)):
        sample = trades[trades["year"].eq(year)]
        rows.append({"period": period, "available_days": available, **pnl_metrics(sample)})
    rows.append({"period": "combined", "available_days": 311, **pnl_metrics(trades)})
    performance = pd.DataFrame(rows)
    performance["coverage"] = performance["days"] / performance["available_days"]
    performance["equivalent_trading_months"] = performance["days"] / 21.0
    validation.to_csv(OUT / "validation_selection.csv", index=False)
    performance.to_csv(OUT / "performance.csv", index=False)

    side_mix = trades.groupby(["year", "trade_side", "structure_family"], as_index=False).agg(
        days=("date", "nunique"), net_pnl=("net_pnl_rupees", "sum"),
        hit_rate=("net_pnl_rupees", lambda x: x.gt(0).mean()),
    )
    side_mix.to_csv(OUT / "side_structure_mix.csv", index=False)
    audit = pd.DataFrame([
        {"check": "one_trade_per_day", "violations": int(trades["date"].duplicated().sum())},
        {"check": "cross_day_window", "violations": int((trades["entry_datetime"].dt.normalize() != trades["exit_datetime"].dt.normalize()).sum())},
        {"check": "outside_0915_1530", "violations": int(((trades["entry_datetime"].dt.hour * 60 + trades["entry_datetime"].dt.minute).lt(555) | (trades["exit_datetime"].dt.hour * 60 + trades["exit_datetime"].dt.minute).gt(930)).sum())},
    ])
    audit["passed"] = audit["violations"].eq(0)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)
    policy = {
        "status": decision,
        "research_only": True,
        "model": model_name,
        "edge_threshold": threshold,
        "coverage_target_days": [TARGET_MIN_DAYS, TARGET_MAX_DAYS],
        "train": "2019-2022", "validation": "2023", "test": "2024",
        "holdout": "2025 through 2025-05-09", "same_monthly_expiry": True,
        "horizon_minutes": 120, "live_deployment": False,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    report = f"""# Near-daily premium-direction backtest

The model forecasts the 120-minute percentage change in the combined option
premium. It chooses long volatility when the predicted premium increase has
the higher after-cost edge and short volatility otherwise. It then chooses the
best ATM, one-strike, or two-strike straddle/strangle candidate for that day.

Train is 2019-2022. Model and coverage threshold selection use 2023 only. The
2024 and partial-2025 results do not select the rule.

Decision: `{decision}`.

## Selected validation rule

{dataframe_to_markdown(pd.DataFrame([selected_rule]).round(4))}

## Frozen evaluation

{dataframe_to_markdown(performance.round(4))}

## Long/short and structure mix

{dataframe_to_markdown(side_mix.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

The target is rejected unless validation net P&L is positive. A high hit rate
alone is insufficient because option losses are asymmetric and transaction
costs are material. Quotes do not prove fillability; this is research only.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(policy)
    print(validation.to_string(index=False))
    print(performance.to_string(index=False))
    print(side_mix.to_string(index=False))


if __name__ == "__main__":
    main()
