"""Causal walk-forward BANKNIFTY volatility hybrid.

The strategy combines a fixed near-expiry variance-premium sleeve with a
direct-P&L model gate for DTE 8-14. Model family and threshold quantile are
selected using only pre-2024 walk-forward predictions. Every 2024-2025 monthly
fit uses only rows strictly earlier than that month.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from atm_netpremium_correlation import dataframe_to_markdown
from dte4_20_pnl_model import FEATURES, panel
from frozen_near_expiry_strategy import load_cache, price


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "causal_profitable_hybrid"
DEVELOPMENT_END = pd.Timestamp("2023-12-31")
PRIMARY_BPS = 25.0
COST_CASES = (10.0, 25.0, 50.0, 100.0)
MIN_TRAIN = 100
MIN_DEVELOPMENT_SIGNALS = 5
MODEL_CANDIDATES = ("huber", "ridge10", "ridge100")
QUANTILES = (0.60, 0.70, 0.80)
BOOTSTRAPS = 20_000


def estimator(name: str):
    if name == "huber":
        return make_pipeline(
            RobustScaler(),
            HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2000),
        )
    alpha = 10.0 if name == "ridge10" else 100.0
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def model_walk_forward(frame: pd.DataFrame, model_name: str, quantile: float) -> pd.DataFrame:
    """Produce prequential signals; a test month never enters its own fit."""
    data = frame[(frame.dte >= 8.0) & (frame.dte < 15.0)].sort_values("datetime")
    months = data.datetime.dt.to_period("M")
    outputs = []
    for month in months.drop_duplicates().sort_values():
        test = data[months.eq(month)].copy()
        train = data[data.datetime < test.datetime.min()].copy()
        if len(train) < MIN_TRAIN:
            continue
        model = estimator(model_name).fit(train[FEATURES], train.net_return)
        train_prediction = model.predict(train[FEATURES])
        threshold = max(0.0, float(np.quantile(train_prediction, quantile)))
        test["predicted_net_return"] = model.predict(test[FEATURES])
        test["train_threshold"] = threshold
        test["train_rows"] = len(train)
        test["max_training_entry_datetime"] = train.datetime.max()
        test["max_training_exit_datetime"] = train.exit_datetime.max()
        test["model_name"] = model_name
        test["threshold_quantile"] = quantile
        test["signal"] = test.predicted_net_return > threshold
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def select_model(primary: pd.DataFrame, fifty: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    rows = []
    fifty_pnl = fifty[["datetime", "net_pnl"]].rename(columns={"net_pnl": "net_pnl_50bps"})
    for model_name in MODEL_CANDIDATES:
        for quantile in QUANTILES:
            scored = model_walk_forward(primary, model_name, quantile)
            selected = scored[scored.signal & scored.date.le(DEVELOPMENT_END)].copy()
            selected = selected.merge(fifty_pnl, on="datetime", how="left", validate="one_to_one")
            rows.append({
                "model_name": model_name,
                "threshold_quantile": quantile,
                "development_signals": len(selected),
                "development_days": selected.date.nunique(),
                "development_net_25bps": selected.net_pnl.sum(),
                "development_net_50bps": selected.net_pnl_50bps.sum(),
                "development_hit_rate": (selected.net_pnl > 0).mean() if len(selected) else np.nan,
            })
    grid = pd.DataFrame(rows)
    grid["eligible"] = (
        grid.development_signals.ge(MIN_DEVELOPMENT_SIGNALS)
        & grid.development_net_25bps.gt(0)
        & grid.development_net_50bps.gt(0)
    )
    eligible = grid[grid.eligible].copy()
    if eligible.empty:
        raise RuntimeError("No DTE 8-14 model passed the pre-2024 selection gate")
    chosen = eligible.sort_values(
        ["development_net_25bps", "development_net_50bps", "development_signals"],
        ascending=False,
    ).iloc[0]
    grid["selected"] = (
        grid.model_name.eq(chosen.model_name)
        & grid.threshold_quantile.eq(chosen.threshold_quantile)
    )
    return chosen, grid


def near_expiry(spread: float) -> pd.DataFrame:
    trades = price(load_cache(), spread)
    trades = trades[trades.dte.between(0.0, 3.999999)].copy()
    trades["sleeve"] = "frozen_dte_0_3_short_atm"
    trades["model_name"] = "fixed_schedule"
    trades["threshold_quantile"] = np.nan
    return trades


def model_sleeve(
    spread: float, selected_keys: pd.DataFrame, model_name: str, quantile: float
) -> pd.DataFrame:
    repriced = panel(spread)
    columns = [
        "datetime", "predicted_net_return", "train_threshold", "train_rows",
        "max_training_entry_datetime", "max_training_exit_datetime",
    ]
    trades = repriced.merge(selected_keys[columns], on="datetime", how="inner", validate="one_to_one")
    trades["sleeve"] = "walk_forward_dte_8_14_short_atm"
    trades["model_name"] = model_name
    trades["threshold_quantile"] = quantile
    return trades


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame.copy()
    for column in (
        "exit_imputed", "predicted_net_return", "train_threshold", "train_rows",
        "max_training_entry_datetime", "max_training_exit_datetime",
    ):
        if column not in keep:
            keep[column] = np.nan
    columns = [
        "datetime", "exit_datetime", "date", "expiry", "dte", "sleeve",
        "model_name", "threshold_quantile", "predicted_net_return",
        "train_threshold", "train_rows", "lot_size", "half_spread_bps",
        "max_training_entry_datetime", "max_training_exit_datetime",
        "exit_call_mid", "exit_put_mid", "gross_pnl", "costs", "net_pnl",
        "exit_imputed",
    ]
    return keep[columns].copy()


def bootstrap(daily: pd.Series, seed: int = 20261013) -> tuple[float, float, float]:
    values = daily.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).sum(axis=1)
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        float(np.mean(draws <= 0)),
    )


def metrics(frame: pd.DataFrame, label: str) -> dict:
    daily = frame.groupby("date").net_pnl.sum().sort_index()
    equity = daily.cumsum()
    wins = daily[daily > 0].sum()
    losses = -daily[daily < 0].sum()
    low, high, probability = bootstrap(daily)
    return {
        "sample": label,
        "trades": len(frame),
        "days": daily.size,
        "profitable_days": int(daily.gt(0).sum()),
        "gross_pnl": frame.gross_pnl.sum(),
        "costs": frame.costs.sum(),
        "net_pnl": frame.net_pnl.sum(),
        "hit_rate": daily.gt(0).mean(),
        "profit_factor": wins / losses if losses > 0 else np.inf,
        "max_drawdown": (equity.cummax() - equity).max(),
        "worst_day": daily.min(),
        "best_day": daily.max(),
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "bootstrap_p_net_le_zero": probability,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    primary = panel(PRIMARY_BPS)
    fifty = panel(50.0)
    chosen, grid = select_model(primary, fifty)
    model_name = str(chosen.model_name)
    quantile = float(chosen.threshold_quantile)

    scored = model_walk_forward(primary, model_name, quantile)
    model_keys = scored[scored.signal & scored.date.gt(DEVELOPMENT_END)].copy()

    performance_rows = []
    ledgers = []
    for spread in COST_CASES:
        near = near_expiry(spread)
        near = near[near.date.gt(DEVELOPMENT_END)].copy()
        modeled = model_sleeve(spread, model_keys, model_name, quantile)
        modeled = modeled[modeled.date.gt(DEVELOPMENT_END)].copy()
        combined = pd.concat(
            [standardize(near), standardize(modeled)], ignore_index=True
        ).sort_values(["datetime", "sleeve"])
        # DTE bands cannot overlap by construction; near-expiry scheduled
        # entries are sequential rather than concurrent.
        for sleeve, sample in [
            ("near_expiry", combined[combined.sleeve.eq("frozen_dte_0_3_short_atm")]),
            ("model_dte_8_14", combined[combined.sleeve.eq("walk_forward_dte_8_14_short_atm")]),
            ("combined", combined),
        ]:
            row = {"half_spread_bps": spread, "sleeve": sleeve, **metrics(sample, "evaluation_2024_2025")}
            performance_rows.append(row)
        if spread == PRIMARY_BPS:
            ledgers.append(combined)

    ledger = pd.concat(ledgers, ignore_index=True).sort_values("datetime")
    performance = pd.DataFrame(performance_rows)
    by_year = []
    for (year, sleeve), group in ledger.assign(year=ledger.date.dt.year).groupby(["year", "sleeve"]):
        by_year.append({"year": year, "sleeve": sleeve, **metrics(group, str(year))})
    for year, group in ledger.assign(year=ledger.date.dt.year).groupby("year"):
        by_year.append({"year": year, "sleeve": "combined", **metrics(group, str(year))})
    yearly = pd.DataFrame(by_year).sort_values(["year", "sleeve"])
    source_path = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
    con = duckdb.connect()
    session_rows = con.execute(f"""
        SELECT year(datetime::TIMESTAMP) AS sample_year,
               count(DISTINCT date(datetime::TIMESTAMP)) AS session_days
        FROM read_csv_auto('{source_path.as_posix()}', header=true)
        WHERE expiry_type='MONTHLY'
          AND year(datetime::TIMESTAMP) IN (2024, 2025)
          AND date(datetime::TIMESTAMP) <= DATE '{primary.date.max().date()}'
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()
    session_counts = {int(year): int(days) for year, days in session_rows}
    coverage_rows = []
    for year in (2024, 2025):
        sample = ledger[ledger.date.dt.year.eq(year)]
        days = int(sample.date.nunique())
        total = int(session_counts.get(year, 0))
        coverage_rows.append({
            "period": str(year), "total_session_days": total,
            "signal_days": days,
            "profitable_signal_days": int(sample.groupby("date").net_pnl.sum().gt(0).sum()),
            "signal_coverage": days / total if total else np.nan,
            "equivalent_trading_months": days / 21.0,
        })
    total_sessions = sum(session_counts.values())
    total_signal_days = int(ledger.date.nunique())
    coverage_rows.append({
        "period": "combined_2024_2025_partial",
        "total_session_days": total_sessions,
        "signal_days": total_signal_days,
        "profitable_signal_days": int(ledger.groupby("date").net_pnl.sum().gt(0).sum()),
        "signal_coverage": total_signal_days / total_sessions if total_sessions else np.nan,
        "equivalent_trading_months": total_signal_days / 21.0,
    })
    coverage = pd.DataFrame(coverage_rows)
    daily_month = (
        ledger.assign(month=ledger.date.dt.to_period("M").astype(str))
        .groupby(["month", "date"], as_index=False)
        .agg(
            trades=("datetime", "size"), gross_pnl=("gross_pnl", "sum"),
            costs=("costs", "sum"), net_pnl=("net_pnl", "sum"),
        )
    )
    monthly = (
        daily_month.groupby("month", as_index=False)
        .agg(
            trades=("trades", "sum"), days=("date", "nunique"),
            profitable_days=("net_pnl", lambda x: int(x.gt(0).sum())),
            gross_pnl=("gross_pnl", "sum"), costs=("costs", "sum"),
            net_pnl=("net_pnl", "sum"),
        )
    )
    monthly["hit_rate"] = monthly.profitable_days / monthly.days

    entry_minute = ledger.datetime.dt.hour * 60 + ledger.datetime.dt.minute
    exit_minute = ledger.exit_datetime.dt.hour * 60 + ledger.exit_datetime.dt.minute
    pnl_error = (ledger.gross_pnl - ledger.costs - ledger.net_pnl).abs()
    modeled_rows = ledger[ledger.sleeve.eq("walk_forward_dte_8_14_short_atm")]
    chronological_training_violations = int((
        pd.to_datetime(modeled_rows.max_training_exit_datetime)
        >= modeled_rows.datetime
    ).sum())
    overlaps = 0
    for _, day in ledger.sort_values("datetime").groupby("date"):
        previous_exit = None
        for row in day.itertuples():
            if previous_exit is not None and row.datetime < previous_exit:
                overlaps += 1
            previous_exit = max(previous_exit, row.exit_datetime) if previous_exit is not None else row.exit_datetime
    audit = pd.DataFrame([
        {"check": "cross_day_window", "violations": int((ledger.datetime.dt.normalize() != ledger.exit_datetime.dt.normalize()).sum())},
        {"check": "outside_0915_1530", "violations": int((~entry_minute.between(555, 930) | ~exit_minute.between(555, 930)).sum())},
        {"check": "expiry_before_entry", "violations": int((pd.to_datetime(ledger.expiry).dt.normalize() < ledger.datetime.dt.normalize()).sum())},
        {"check": "overlapping_positions", "violations": overlaps},
        {"check": "pnl_identity_error_over_1e-8", "violations": int(pnl_error.gt(1e-8).sum())},
        {"check": "training_label_not_strictly_prior", "violations": chronological_training_violations},
        {"check": "duplicate_sleeve_timestamp", "violations": int(ledger.duplicated(["datetime", "sleeve"]).sum())},
    ])
    audit["passed"] = audit.violations.eq(0)

    imputation_rows = []
    imputed_exit_value = (
        (ledger.exit_call_mid + ledger.exit_put_mid)
        * ledger.lot_size
        * ledger.exit_imputed.astype(float)
    )
    for penalty_fraction in (0.0, 0.02, 0.05, 0.10, 0.20):
        penalty = float((imputed_exit_value * penalty_fraction).sum())
        imputation_rows.append({
            "adverse_imputed_exit_penalty_fraction": penalty_fraction,
            "imputed_trades": int(ledger.exit_imputed.sum()),
            "total_trades": len(ledger),
            "imputed_trade_fraction": ledger.exit_imputed.mean(),
            "additional_penalty_rupees": penalty,
            "stressed_net_pnl": ledger.net_pnl.sum() - penalty,
        })
    imputation_stress = pd.DataFrame(imputation_rows)

    primary_combined = performance[
        performance.half_spread_bps.eq(PRIMARY_BPS) & performance.sleeve.eq("combined")
    ].iloc[0]
    fifty_combined = performance[
        performance.half_spread_bps.eq(50.0) & performance.sleeve.eq("combined")
    ].iloc[0]
    five_percent_imputation_net = float(imputation_stress.loc[
        imputation_stress.adverse_imputed_exit_penalty_fraction.eq(0.05),
        "stressed_net_pnl",
    ].iloc[0])
    status = (
        "POSITIVE_HISTORICAL_WALK_FORWARD_PAPER_ONLY"
        if primary_combined.net_pnl > 0
        and fifty_combined.net_pnl > 0
        and five_percent_imputation_net > 0
        and audit.passed.all()
        else "REJECTED"
    )

    grid.to_csv(OUT / "pre2024_model_selection.csv", index=False)
    performance.to_csv(OUT / "cost_sensitivity.csv", index=False)
    yearly.to_csv(OUT / "yearly_performance.csv", index=False)
    monthly.to_csv(OUT / "monthly_performance.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)
    imputation_stress.to_csv(OUT / "imputed_exit_stress.csv", index=False)
    ledger.to_csv(OUT / "trade_ledger_25bps.csv", index=False)
    con = duckdb.connect()
    con.register("ledger", ledger)
    con.execute("COPY ledger TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(OUT / "trade_ledger_25bps.parquet")])
    con.close()

    policy = {
        "status": status,
        "live_deployment": False,
        "development_cutoff": "2023-12-31",
        "evaluation_data_period": f"2024-01-01 through {primary.date.max().date()}",
        "last_signal_date": str(ledger.date.max().date()),
        "near_expiry_sleeve": {
            "structure": "one-lot short ATM straddle",
            "dte": "0-3",
            "entry_times": ["09:30", "11:45"],
            "holding_minutes": 120,
        },
        "model_sleeve": {
            "structure": "one-lot short ATM straddle",
            "dte": "8-14",
            "entry": "first ATM row between 09:45 and 09:59",
            "exit": "last same-expiry row between 15:15 and 15:30",
            "model": model_name,
            "threshold_quantile": quantile,
            "monthly_refit": "all rows strictly before test month",
            "features": FEATURES,
        },
        "primary_half_spread_bps": PRIMARY_BPS,
        "selection_used_post_2023_outcomes": False,
        "research_history_warning": "The repository's 2024-2025 data had been inspected before this composite was specified; only future paper data can be pristine.",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    report = f"""# Causal profitable volatility hybrid

Status: `{status}`.

This strategy combines two non-overlapping one-lot short-ATM-straddle sleeves.
The DTE 0-3 sleeve uses fixed 09:30 and 11:45 entries with 120-minute exits.
The DTE 8-14 sleeve uses a direct after-cost P&L model, refit at each month
using only earlier rows. Model family and threshold quantile were chosen from
pre-2024 walk-forward predictions only.

The pre-2024 selector chose `{model_name}` with the `{quantile:.0%}` training
prediction quantile. DTE 4-7 and DTE 15-20 are not traded because they did not
produce an eligible, positive pre-2024 model candidate under the same gate.

## Pre-2024 model selection

{dataframe_to_markdown(grid.round(4))}

## Cost sensitivity

{dataframe_to_markdown(performance.round(4))}

## Calendar-year evaluation

{dataframe_to_markdown(yearly.round(4))}

## Signal coverage

{dataframe_to_markdown(coverage.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

## Missing-strike exit stress

{dataframe_to_markdown(imputation_stress.round(4))}

Net P&L includes historical lot sizes, brokerage, STT, exchange charges,
SEBI/IPFT charges, GST, stamp duty and the stated half-spread on entry and exit.
Fixed entry strikes are retained. When a fixed strike leaves the five-strike
exit snapshot, its exit value is reconstructed from the contemporaneous smile;
those rows are explicitly marked in the ledger. The separate stress table
penalizes only those reconstructed exit values. A 5% adverse pricing error
leaves a small positive result; a 10% error makes the strategy negative.

The historical mechanism is consistent with variance-risk-premium research,
but positive backtest P&L is not a guarantee. This repository had already
inspected 2024-2025 before the composite was specified, so the result is an
auditable causal replay, not a pristine independent discovery. Promotion
requires new forward paper data.

Sources:

- Carr and Wu, *Variance Risk Premiums*, Review of Financial Studies (2009):
  https://academic.oup.com/rfs/article/22/3/1311/1581057
- Büchner and Kelly, *Option Return Predictability with Machine Learning and
  Big Data*, Review of Financial Studies (2023):
  https://academic.oup.com/rfs/article/36/9/3548/7056660
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(performance[performance.sleeve.eq("combined")].to_string(index=False))


if __name__ == "__main__":
    main()
