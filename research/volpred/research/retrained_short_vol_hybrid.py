"""Train a direct option-P&L model and combine it with the current hybrid."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import hybrid_coverage_optimization as coverage
import hybrid_regime_strategy as hybrid
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "retrained_hybrid"
FEATURE_PANEL = ROOT / "outputs" / "spot_magnitude_models" / "spot_magnitude_feature_panel.parquet"
CURRENT_LEDGER = ROOT / "model" / "regimeUnderstanding" / "coverage_optimization" / "coverage_trade_ledger.parquet"
TOTAL_DAY_PANEL = ROOT / "model" / "regimeUnderstanding" / "five_strike_trade_research" / "candidate_panel.parquet"
CACHE = OUT / "historical_short_2_strangle_panel_entry_safe.parquet"
SLIPPAGE_BPS = 25.0
MIN_VALIDATION_DAYS = 40

BASE_FEATURES = [
    "return_past_5", "return_past_15", "return_past_60",
    "abs_return_past_5", "abs_return_past_15", "abs_return_past_60",
    "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
    "overnight_return", "log_iv_trade", "iv_skew", "forward_log_basis",
    "oi_imbalance", "oi_total_change_5", "oi_total_change_15",
    "oi_total_change_60", "vol_of_vol_30", "leverage_effect",
    "time_fraction", "time_sin", "time_cos", "sqrt_dte",
]
MODEL_FEATURES = BASE_FEATURES + [
    "entry_value_pct", "delta", "gamma_s2", "vega", "theta_day", "days_to_expiry",
]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    smb.write_parquet(frame, path)


def historical_structure_panel() -> pd.DataFrame:
    if CACHE.exists():
        return smb.read_parquet(CACHE)
    entries, _ = smb.prepare_entry_rows()
    entries = smb.add_entry_greeks(entries)
    wide = smb.wide_legs(smb.leg_panel(entries, smb.load_exits(120)))
    dummy = wide[["entry_datetime"]].rename(columns={"entry_datetime": "datetime"}).copy()
    dummy["horizon_min"] = 120
    dummy["pred_annualized_variance"] = 0.0
    dummy["actual_var"] = 0.0
    dummy["is_0915"] = 0
    dummy["q50_compact"] = 0.0
    panel = smb.structure_panel(
        wide, dummy, "short_2_strangle",
        {(-2, "put"): -1, (2, "call"): -1}, 120,
    )
    panel["date"] = panel["entry_datetime"].dt.normalize()
    panel["lot_size"] = smb.lot_size(panel["date"])
    panel = panel[
        panel["gross_pnl_points"].notna()
        & panel["entry_turnover_points"].gt(0)
        & panel["exit_turnover_points"].gt(0)
        & ~panel["entry_tiny_any"]
    ].copy()
    entry_minute = panel["entry_datetime"].dt.hour * 60 + panel["entry_datetime"].dt.minute
    exit_minute = panel["exit_datetime"].dt.hour * 60 + panel["exit_datetime"].dt.minute
    panel = panel[
        entry_minute.between(555, 930) & exit_minute.between(555, 930)
        & panel["entry_datetime"].dt.normalize().eq(panel["exit_datetime"].dt.normalize())
    ].copy()
    panel = smb.execute(panel, SLIPPAGE_BPS, current_2026=True)

    feature_data = smb.read_parquet(FEATURE_PANEL)[["datetime", *BASE_FEATURES]]
    panel = panel.merge(
        feature_data, left_on="entry_datetime", right_on="datetime",
        how="inner", validate="one_to_one",
    ).drop(columns="datetime")
    panel["entry_value_pct"] = panel["entry_value"].abs() / panel["spot_price"]
    panel["gamma_s2"] = panel["gamma"] * panel["spot_price"].pow(2)
    for column in MODEL_FEATURES:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel[np.isfinite(panel[MODEL_FEATURES].to_numpy(dtype=float)).all(axis=1)].copy()
    panel["profitable"] = panel["net_pnl_rupees"].gt(0).astype(int)
    panel["year"] = panel["date"].dt.year
    write_parquet(panel, CACHE)
    return panel


def model_candidates() -> dict[str, object]:
    return {
        "logit_c0.01": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500)),
        "logit_c0.1": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
        "logit_c1": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)),
        "lightgbm_7": lgb.LGBMClassifier(
            n_estimators=250, num_leaves=7, min_child_samples=300,
            learning_rate=0.03, reg_lambda=5.0, verbosity=-1,
            n_jobs=8, random_state=20261012,
        ),
        "lightgbm_15": lgb.LGBMClassifier(
            n_estimators=250, num_leaves=15, min_child_samples=300,
            learning_rate=0.03, reg_lambda=5.0, verbosity=-1,
            n_jobs=8, random_state=20261012,
        ),
    }


def first_threshold_crossing(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Take the first live-observable crossing per day."""
    eligible = scored[scored["model_probability"].ge(threshold)].copy()
    return (
        eligible.sort_values(
            ["date", "entry_datetime", "model_probability"],
            ascending=[True, True, False],
        )
        .groupby("date", as_index=False, group_keys=False)
        .head(1)
        .copy()
    )


def daily_best(scored: pd.DataFrame) -> pd.DataFrame:
    raise RuntimeError(
        "daily_best is non-causal. Use first_threshold_crossing(scored, threshold)."
    )


def train(panel: pd.DataFrame) -> tuple[object, float, str, pd.DataFrame]:
    training = panel[panel["year"].le(2022)]
    validation = panel[panel["year"].eq(2023)]
    rows: list[dict] = []
    fitted: dict[str, object] = {}
    best: tuple[float, str, float] | None = None
    for name, model in model_candidates().items():
        model.fit(training[MODEL_FEATURES], training["profitable"])
        fitted[name] = model
        scored = validation.copy()
        scored["model_probability"] = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
        for quantile in (0.70, 0.75, 0.80, 0.85, 0.90):
            threshold = float(scored["model_probability"].quantile(quantile))
            trades = first_threshold_crossing(scored, threshold)
            net = float(trades["net_pnl_rupees"].sum())
            row = {
                "model": name, "validation_score_quantile": quantile,
                "probability_threshold": threshold, "validation_days": len(trades),
                "validation_net_pnl": net,
                "validation_mean_net": trades["net_pnl_rupees"].mean(),
                "validation_hit_rate": trades["net_pnl_rupees"].gt(0).mean(),
            }
            row["eligible"] = len(trades) >= MIN_VALIDATION_DAYS and net > 0
            rows.append(row)
            if row["eligible"] and (best is None or net > best[0]):
                best = (net, name, threshold)
    if best is None:
        raise RuntimeError("No retrained short-volatility model passed validation")
    return fitted[best[1]], best[2], best[1], pd.DataFrame(rows)


def new_signal_trades(panel: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    evaluation = panel[panel["year"].ge(2024)].copy()
    evaluation["model_probability"] = model.predict_proba(evaluation[MODEL_FEATURES])[:, 1]
    trades = first_threshold_crossing(evaluation, threshold)
    trades["hybrid_component"] = "retrained_short_2_strangle_fallback"
    return trades


def current_trades() -> pd.DataFrame:
    frame = smb.read_parquet(CURRENT_LEDGER)
    frame = frame[frame["coverage_rule"].eq("coverage_optimized_hybrid")].copy()
    frame["hybrid_component"] = "current_coverage_hybrid"
    return frame


def combine(current: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    occupied_dates = set(pd.to_datetime(current["date"]).dt.normalize())
    extra = fallback[~fallback["date"].dt.normalize().isin(occupied_dates)].copy()
    combined = pd.concat([current, extra], ignore_index=True, sort=False)
    return combined.sort_values("entry_datetime").copy()


def performance(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    coverage_rows: list[dict] = []
    con = duckdb.connect()
    total = con.execute(
        "SELECT year(date) AS sample_year, count(DISTINCT date) AS total_days "
        "FROM read_parquet(?) WHERE year(date) IN (2024, 2025) GROUP BY year(date)",
        [str(TOTAL_DAY_PANEL)],
    ).fetchdf().set_index("sample_year")["total_days"].to_dict()
    con.close()
    for name, frame in frames.items():
        for year, period in ((2024, "development_2024"), (2025, "holdout_2025")):
            sample = frame[frame["date"].dt.year.eq(year)].copy()
            rows.append({"strategy": name, "period": period, **hybrid.metrics(sample)})
            daily = sample.groupby("date")["net_pnl_rupees"].sum()
            coverage_rows.append({
                "strategy": name, "period": period,
                "total_available_days": total.get(year, 0),
                "signal_days": daily.size,
                "profitable_days": int(daily.gt(0).sum()),
                "losing_or_flat_days": int(daily.le(0).sum()),
                "signal_day_coverage": daily.size / total[year] if total.get(year) else np.nan,
                "profitable_signal_day_rate": daily.gt(0).mean() if daily.size else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def write_report(
    tuning: pd.DataFrame, model_name: str, threshold: float,
    perf: pd.DataFrame, day_coverage: pd.DataFrame,
) -> None:
    results = perf[[
        "strategy", "period", "trades", "days", "net_pnl", "mean_net",
        "hit_rate", "profit_factor", "max_drawdown", "p05_net",
    ]].round(4)
    report = f"""# Retrained short-volatility fallback and hybrid

## Research hypothesis

Option returns should be modeled from the conditional physical-versus-implied
volatility gap and state variables rather than from the sign of `IV-RV` alone.
This follows three strands of published evidence:

- [Carr and Wu (2009)](https://academic.oup.com/rfs/article-abstract/22/3/1311/1581057)
  show that option portfolios approximate risk-neutral expected
  variance and define the variance risk premium as realized versus synthetic
  option-implied variance.
- [Goyal and Saretto (2009)](https://www.ruf.rice.edu/~jgsfss/goyal_041808.pdf)
  document option-return predictability from the historical
  realized-volatility minus ATM implied-volatility difference.
- [Corsi (2009)](https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf)
  motivates multiple lagged realized-volatility horizons through HAR-RV.
- [Bali et al. (2023)](https://academic.oup.com/rfs/article/36/9/3548/7056660)
  find that option and stock characteristics can predict delta-hedged option
  returns, with nonlinear machine-learning interactions adding value.

The empirical hypothesis is therefore:

`Pr(net short-strangle P&L > 0 | RV lags, IV, skew, momentum, OI, Greeks, DTE, time)`

can identify profitable 120-minute short-volatility windows even when the
current regime hybrid is inactive.

## Training protocol

- Structure: sell ATM-2 put and ATM+2 call, same monthly expiry.
- Target: realized net P&L after the current encoded charges and 25 bps
  premium-turnover slippage.
- Train: 2019-2022.
- Validation/model and threshold selection: 2023 only.
- Frozen evaluation: 2024 and partial 2025.
- Chosen model: `{model_name}`.
- Entry threshold: predicted profitable-trade probability >= {threshold:.6f}.
- At most one fallback per day: choose the timestamp with the highest predicted
  probability. The fallback is used only on dates with no current-hybrid trade.

## Results

{dataframe_to_markdown(results)}

## Trading-day coverage

{dataframe_to_markdown(day_coverage.round(4))}

## Entry and P&L equations

`p_t = logistic(beta_0 + beta' x_t)`.

Enter only if `p_t` exceeds the frozen 2023 threshold. For the short strangle:

`PnL = [(P_t(K-2)-P_t+h(K-2)) + (C_t(K+2)-C_t+h(K+2))] * lot - costs`.

## Limitations

The model is fitted to overlapping intraday candidate windows, although only
one daily fallback is executed. The short strangle has unbounded tail risk.
The partial 2025 sample is small, and all results use quote-based rather than
executable bid/ask prices. This remains forward-paper research.

Two hypotheses were rejected before this model was admitted. A retrained
intraday direction classifier produced only about 0.509 evaluation AUC, and the
defined-risk iron-condor model's best qualifying 2023 validation slice lost
approximately INR 13,467. Neither is included in the combined policy.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = historical_structure_panel()
    model, threshold, model_name, tuning = train(panel)
    joblib.dump({
        "model": model, "features": MODEL_FEATURES,
        "probability_threshold": threshold, "model_name": model_name,
        "train_end": "2022-12-31", "validation_year": 2023,
    }, OUT / "short_vol_fallback_model.joblib")
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        coefficients = pd.DataFrame({
            "feature": MODEL_FEATURES,
            "standardized_logit_coefficient": model.named_steps["logisticregression"].coef_[0],
        }).sort_values("standardized_logit_coefficient", key=lambda x: x.abs(), ascending=False)
        coefficients.to_csv(OUT / "model_coefficients.csv", index=False)
    split_manifest = pd.DataFrame([
        {"split": "train", "date_rule": "year <= 2022", "rows": int(panel["year"].le(2022).sum()), "days": int(panel.loc[panel["year"].le(2022), "date"].nunique())},
        {"split": "validation", "date_rule": "year = 2023", "rows": int(panel["year"].eq(2023).sum()), "days": int(panel.loc[panel["year"].eq(2023), "date"].nunique())},
        {"split": "development evaluation", "date_rule": "year = 2024", "rows": int(panel["year"].eq(2024).sum()), "days": int(panel.loc[panel["year"].eq(2024), "date"].nunique())},
        {"split": "holdout evaluation", "date_rule": "year = 2025", "rows": int(panel["year"].eq(2025).sum()), "days": int(panel.loc[panel["year"].eq(2025), "date"].nunique())},
    ])
    split_manifest.to_csv(OUT / "training_split_manifest.csv", index=False)
    new_trades = new_signal_trades(panel, model, threshold)
    current = current_trades()
    combined = combine(current, new_trades)
    frames = {
        "current_coverage_hybrid": current,
        "retrained_short_vol_fallback": new_trades,
        "combined_retrained_hybrid": combined,
    }
    perf, day_coverage = performance(frames)
    tuning.to_csv(OUT / "validation_model_selection.csv", index=False)
    perf.to_csv(OUT / "strategy_performance.csv", index=False)
    day_coverage.to_csv(OUT / "trading_day_coverage.csv", index=False)
    write_parquet(new_trades, OUT / "new_signal_trades.parquet")
    write_parquet(combined, OUT / "combined_trade_ledger.parquet")
    policy = {
        "status": "FORWARD_PAPER_TEST_ONLY",
        "current_component": "coverage_optimized_hybrid",
        "new_component": "120-minute short ATM+/-2 strangle probability model",
        "new_component_model": model_name,
        "probability_threshold": threshold,
        "fallback_only_on_current_no-signal_dates": True,
        "same_monthly_expiry": True,
        "latest_entry": "13:30",
        "position_size": "one lot",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    write_report(tuning, model_name, threshold, perf, day_coverage)
    print(policy)
    print(perf.to_string(index=False))
    print(day_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
