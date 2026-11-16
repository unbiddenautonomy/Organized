"""Superseded non-causal near-daily short-volatility research.

This file is retained only to reproduce the historical analysis. Its
``daily_best`` entry selection sees every timestamp in a completed session and
then chooses that day's maximum model score. Do not use it for performance
claims or live inference; use ``near_daily_short_vol_causal.py`` instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import expanded_signal_hybrid as expanded
import hybrid_regime_strategy as hybrid
import retrained_short_vol_hybrid as retrained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "near_daily_short_vol_risk_managed"
RAW = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
PATH_CACHE = OUT / "atm_straddle_paths.parquet"
SLIPPAGE_BPS = 25.0
HORIZONS = (120, 180)
TARGET_MIN_DAYS = 210
TARGET_MAX_DAYS = 231
SCORE_QUANTILES = (0.00, 0.03, 0.05, 0.075, 0.10)
DTE_FLOORS = (0, 1, 2, 3, 4)
STOP_LEVELS = (0.10, 0.15, 0.20, 0.30, 0.50)
TAKE_LEVELS = (0.05, 0.10, 0.15, 0.20, 1.00)
SIZE_QUANTILES = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
HIGH_CONFIDENCE_LOTS = (2, 3)


def historical_atm_panel(horizon: int) -> pd.DataFrame:
    if horizon == 120:
        return expanded.historical_structure_panel(
            "short_atm_straddle", expanded.SECONDARY_STRUCTURES["short_atm_straddle"]
        )
    cache = OUT / f"historical_short_atm_straddle_{horizon}m_entry_safe.parquet"
    if cache.exists():
        return smb.read_parquet(cache)
    entries, _ = smb.prepare_entry_rows()
    entries = smb.add_entry_greeks(entries)
    exit_cache = OUT / f"fixed_strike_exits_{horizon}m.parquet"
    if not exit_cache.exists():
        keys = entries[["source_index", "datetime", "expiry", "strike", "strike_position"]].copy()
        con = duckdb.connect()
        con.execute("SET threads=4")
        con.execute("SET memory_limit='8GB'")
        con.register("entry_keys", keys)
        latest_hour = 12
        latest_minute = 30
        con.execute(f"""
            COPY (
              WITH k AS (
                SELECT source_index, datetime AS entry_datetime, expiry, strike, strike_position,
                       datetime + INTERVAL {horizon} MINUTE AS exit_datetime
                FROM entry_keys
                WHERE CAST(datetime AS TIME) <= TIME '{latest_hour:02d}:{latest_minute:02d}:00'
              ), x AS (
                SELECT datetime::TIMESTAMP AS exit_datetime, expiry::DATE AS expiry,
                       strike::DOUBLE AS strike, call_price::DOUBLE AS exit_call_price,
                       put_price::DOUBLE AS exit_put_price
                FROM read_csv_auto('{smb.RAW_CHAIN.as_posix()}', header=true)
                WHERE expiry_type='MONTHLY'
              )
              SELECT k.*, '{horizon}m' AS horizon, x.exit_call_price, x.exit_put_price
              FROM k LEFT JOIN x USING(exit_datetime, expiry, strike)
              WHERE CAST(k.entry_datetime AS DATE)=CAST(k.exit_datetime AS DATE)
            ) TO '{exit_cache.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.close()
    wide = smb.wide_legs(smb.leg_panel(entries, smb.read_parquet(exit_cache)))
    dummy = wide[["entry_datetime"]].rename(columns={"entry_datetime": "datetime"}).copy()
    dummy["horizon_min"] = horizon
    dummy["pred_annualized_variance"] = 0.0
    dummy["actual_var"] = 0.0
    dummy["is_0915"] = 0
    dummy["q50_compact"] = 0.0
    panel = smb.structure_panel(
        wide, dummy, "short_atm_straddle",
        expanded.SECONDARY_STRUCTURES["short_atm_straddle"], horizon,
    )
    panel["date"] = panel["entry_datetime"].dt.normalize()
    panel["lot_size"] = smb.lot_size(panel["date"])
    panel = panel[
        panel["gross_pnl_points"].notna()
        & panel["entry_turnover_points"].gt(0)
        & panel["exit_turnover_points"].gt(0)
        # Exit-derived ``tiny_any`` and ``stale_any`` are audit fields only.
        & ~panel["entry_tiny_any"]
    ].copy()
    entry_minute = panel["entry_datetime"].dt.hour * 60 + panel["entry_datetime"].dt.minute
    exit_minute = panel["exit_datetime"].dt.hour * 60 + panel["exit_datetime"].dt.minute
    panel = panel[
        entry_minute.between(555, 930) & exit_minute.between(555, 930)
        & panel["entry_datetime"].dt.normalize().eq(panel["exit_datetime"].dt.normalize())
    ].copy()
    panel = smb.execute(panel, SLIPPAGE_BPS, current_2026=True)
    features = smb.read_parquet(retrained.FEATURE_PANEL)[["datetime", *retrained.BASE_FEATURES]]
    panel = panel.merge(
        features, left_on="entry_datetime", right_on="datetime",
        how="inner", validate="one_to_one",
    ).drop(columns="datetime")
    panel["entry_value_pct"] = panel["entry_value"].abs() / panel["spot_price"]
    panel["gamma_s2"] = panel["gamma"] * panel["spot_price"].pow(2)
    for column in retrained.MODEL_FEATURES:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel[np.isfinite(panel[retrained.MODEL_FEATURES].to_numpy(dtype=float)).all(axis=1)].copy()
    panel["profitable"] = panel["net_pnl_rupees"].gt(0).astype(int)
    panel["year"] = panel["date"].dt.year
    smb.write_parquet(panel, cache)
    return panel


def daily_candidates(horizon: int) -> tuple[object, pd.DataFrame]:
    panel = historical_atm_panel(horizon)
    train = panel[panel["year"].le(2022)]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    model.fit(train[retrained.MODEL_FEATURES], train["profitable"])
    scored = panel[panel["year"].ge(2023)].copy()
    scored["model_probability"] = model.predict_proba(scored[retrained.MODEL_FEATURES])[:, 1]
    raise RuntimeError(
        "This superseded routine ranked a completed session and is disabled. "
        "Use causal_validation_short_vol.py for first-crossing validation."
    )


def load_paths(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    cache = OUT / f"atm_straddle_paths_{horizon}m.parquet"
    if cache.exists():
        return smb.read_parquet(cache)
    keys = daily[["trade_id", "entry_datetime", "exit_datetime", "expiry"]].copy()
    con = duckdb.connect()
    con.register("trade_keys", keys)
    con.execute(f"""
        CREATE TEMP TABLE raw_small AS
        SELECT datetime::TIMESTAMP AS quote_datetime, expiry::DATE AS expiry,
               strike_position::INTEGER AS strike_position, strike::DOUBLE AS strike,
               call_price::DOUBLE AS call_price, put_price::DOUBLE AS put_price
        FROM read_csv_auto('{RAW.as_posix()}', header=true)
        WHERE expiry_type='MONTHLY' AND datetime >= TIMESTAMP '2023-01-01'
    """)
    paths = con.execute("""
        WITH entries AS (
          SELECT k.*, r.strike,
                 r.call_price + r.put_price AS entry_premium
          FROM trade_keys k
          JOIN raw_small r ON r.quote_datetime=k.entry_datetime
                          AND r.expiry=k.expiry AND r.strike_position=0
        )
        SELECT e.trade_id, e.entry_datetime, e.exit_datetime AS scheduled_exit,
               e.expiry, e.strike, e.entry_premium, r.quote_datetime,
               r.call_price + r.put_price AS premium
        FROM entries e
        JOIN raw_small r ON r.expiry=e.expiry AND r.strike=e.strike
                        AND r.quote_datetime > e.entry_datetime
                        AND r.quote_datetime <= e.exit_datetime
        ORDER BY e.trade_id, r.quote_datetime
    """).fetchdf()
    con.close()
    smb.write_parquet(paths, cache)
    return paths


def apply_exits(daily: pd.DataFrame, paths: pd.DataFrame, stop: float, take: float) -> pd.DataFrame:
    p = paths.copy()
    p["loss_fraction"] = (p["premium"] - p["entry_premium"]) / p["entry_premium"]
    p["profit_fraction"] = (p["entry_premium"] - p["premium"]) / p["entry_premium"]
    p["trigger"] = p["loss_fraction"].ge(stop) | p["profit_fraction"].ge(take)
    triggered = p[p["trigger"]].sort_values("quote_datetime").groupby("trade_id").head(1)
    scheduled = p.sort_values("quote_datetime").groupby("trade_id").tail(1)
    chosen = scheduled.set_index("trade_id")
    chosen.update(triggered.set_index("trade_id"))
    chosen = chosen.reset_index()[["trade_id", "quote_datetime", "premium", "trigger"]]
    out = daily.merge(chosen, on="trade_id", how="inner", validate="one_to_one")
    out["scheduled_exit_datetime"] = out["exit_datetime"]
    out["exit_datetime"] = out["quote_datetime"]
    out["exit_value"] = -out["premium"]
    out["gross_pnl_points"] = out["entry_value"].abs() - out["premium"]
    out["exit_turnover_points"] = out["premium"]
    out["sell_turnover_points"] = out["entry_value"].abs()
    out["buy_turnover_points"] = out["premium"]
    out["exit_reason"] = np.where(
        out["trigger"],
        np.where((out["premium"] - out["entry_value"].abs()) / out["entry_value"].abs() >= stop,
                 "stop_loss", "take_profit"),
        "scheduled_exit",
    )
    out = smb.execute(out, SLIPPAGE_BPS, current_2026=True)
    out["year"] = out["date"].dt.year
    return out


def metrics(frame: pd.DataFrame) -> dict:
    m = hybrid.metrics(frame)
    return {**m, "profitable_days": int(frame.groupby("date")["net_pnl_rupees"].sum().gt(0).sum())}


def apply_sizing(frame: pd.DataFrame, threshold: float, high_lots: int, slippage_bps: float = SLIPPAGE_BPS) -> pd.DataFrame:
    out = frame.copy()
    out["base_lot_size"] = out["lot_size"]
    out["lot_multiplier"] = np.where(out["model_probability"].ge(threshold), high_lots, 1)
    out["lot_size"] = out["base_lot_size"] * out["lot_multiplier"]
    return smb.execute(out, slippage_bps, current_2026=True)


def main() -> None:
    raise RuntimeError(
        "Superseded: daily maximum-score selection has intraday look-ahead. "
        "Run research/near_daily_short_vol_causal.py instead."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ledgers = [], {}
    fitted_models = {}
    for horizon in HORIZONS:
        model, daily = daily_candidates(horizon)
        fitted_models[horizon] = model
        paths = load_paths(daily, horizon)
        validation_daily = daily[daily["year"].eq(2023)]
        for stop in STOP_LEVELS:
            for take in TAKE_LEVELS:
                exited = apply_exits(daily, paths, stop, take)
                for score_q in SCORE_QUANTILES:
                    threshold = float(validation_daily["model_probability"].quantile(score_q))
                    for dte_floor in DTE_FLOORS:
                        selected = exited[
                            exited["model_probability"].ge(threshold)
                            & exited["days_to_expiry"].ge(dte_floor)
                        ].copy()
                        key = (horizon, stop, take, score_q, dte_floor)
                        ledgers[key] = selected
                        val = selected[selected["year"].eq(2023)]
                        h1 = val[val["date"].lt("2023-07-01")]
                        h2 = val[val["date"].ge("2023-07-01")]
                        rows.append({
                            "horizon_min": horizon,
                            "stop_fraction": stop, "take_fraction": take,
                            "score_quantile": score_q, "score_threshold": threshold,
                            "dte_floor": dte_floor, **metrics(val),
                            "H1_net": h1["net_pnl_rupees"].sum(),
                            "H2_net": h2["net_pnl_rupees"].sum(),
                        })
    search = pd.DataFrame(rows)
    search["target_coverage"] = search["days"].between(TARGET_MIN_DAYS, TARGET_MAX_DAYS)
    search["stable_positive"] = search["target_coverage"] & search["H1_net"].gt(0) & search["H2_net"].gt(0)
    eligible = search[search["stable_positive"]]
    if eligible.empty:
        decision = "REJECT_NO_STABLE_POSITIVE_10_TO_11_MONTH_RULE"
        selected = search[search["target_coverage"]].sort_values(
            ["net_pnl", "profit_factor"], ascending=False
        ).iloc[0]
    else:
        decision = "PASS_BACKTEST_FORWARD_PAPER_ONLY"
        selected = eligible.sort_values(
            ["net_pnl", "profit_factor", "max_drawdown"], ascending=[False, False, True]
        ).iloc[0]
    key = (int(selected.horizon_min), float(selected.stop_fraction), float(selected.take_fraction),
           float(selected.score_quantile), int(selected.dte_floor))
    trades = ledgers[key]

    sizing_rows, sized_ledgers = [], {}
    validation_base = trades[trades["year"].eq(2023)]
    for size_q in SIZE_QUANTILES:
        size_threshold = float(validation_base["model_probability"].quantile(size_q))
        for high_lots in HIGH_CONFIDENCE_LOTS:
            sized = apply_sizing(trades, size_threshold, high_lots)
            val = sized[sized["year"].eq(2023)]
            h1 = val[val["date"].lt("2023-07-01")]
            h2 = val[val["date"].ge("2023-07-01")]
            row = {
                "size_quantile": size_q, "size_threshold": size_threshold,
                "high_confidence_lots": high_lots, **metrics(val),
                "H1_net": h1["net_pnl_rupees"].sum(), "H2_net": h2["net_pnl_rupees"].sum(),
            }
            row["stable_positive"] = row["H1_net"] > 0 and row["H2_net"] > 0
            sizing_rows.append(row)
            sized_ledgers[(size_q, high_lots)] = sized
    sizing_search = pd.DataFrame(sizing_rows)
    sizing_eligible = sizing_search[sizing_search["stable_positive"]]
    selected_sizing = sizing_eligible.sort_values(
        ["net_pnl", "profit_factor", "max_drawdown"], ascending=[False, False, True]
    ).iloc[0]
    sized = sized_ledgers[(float(selected_sizing.size_quantile), int(selected_sizing.high_confidence_lots))]
    base_evaluation = trades[trades["year"].ge(2024)].copy()
    evaluation = sized[sized["year"].ge(2024)].copy()
    smb.write_parquet(evaluation, OUT / "trade_ledger.parquet")
    perf_rows, base_perf_rows = [], []
    for year, period, available in ((2024, "test_2024", 235), (2025, "holdout_2025_partial", 76)):
        sample = evaluation[evaluation["year"].eq(year)]
        perf_rows.append({"period": period, "available_days": available, **metrics(sample)})
        base_sample = base_evaluation[base_evaluation["year"].eq(year)]
        base_perf_rows.append({"period": period, "available_days": available, **metrics(base_sample)})
    perf_rows.append({"period": "combined", "available_days": 311, **metrics(evaluation)})
    base_perf_rows.append({"period": "combined", "available_days": 311, **metrics(base_evaluation)})
    performance = pd.DataFrame(perf_rows)
    base_performance = pd.DataFrame(base_perf_rows)
    performance["coverage"] = performance["days"] / performance["available_days"]
    performance["equivalent_trading_months"] = performance["days"] / 21.0
    base_performance["coverage"] = base_performance["days"] / base_performance["available_days"]
    base_performance["equivalent_trading_months"] = base_performance["days"] / 21.0
    search.to_csv(OUT / "validation_search.csv", index=False)
    pd.DataFrame([selected]).to_csv(OUT / "selected_rule.csv", index=False)
    sizing_search.to_csv(OUT / "validation_sizing_search.csv", index=False)
    pd.DataFrame([selected_sizing]).to_csv(OUT / "selected_sizing.csv", index=False)
    base_performance.to_csv(OUT / "one_lot_performance.csv", index=False)
    performance.to_csv(OUT / "performance.csv", index=False)
    exposure_rows = []
    for year, period in ((2024, "test_2024"), (2025, "holdout_2025_partial")):
        sample = evaluation[evaluation["year"].eq(year)]
        exposure_rows.append({
            "period": period,
            "one_lot_days": int(sample["lot_multiplier"].eq(1).sum()),
            "three_lot_days": int(sample["lot_multiplier"].eq(3).sum()),
            "lot_equivalents": int(sample["lot_multiplier"].sum()),
            "net_pnl": sample["net_pnl_rupees"].sum(),
            "net_pnl_per_lot_equivalent": (
                sample["net_pnl_rupees"].sum() / sample["lot_multiplier"].sum()
            ),
        })
    exposure = pd.DataFrame(exposure_rows)
    exposure.to_csv(OUT / "sizing_exposure.csv", index=False)
    exit_mix = evaluation.groupby(["year", "exit_reason"], as_index=False).agg(
        trades=("date", "size"), net_pnl=("net_pnl_rupees", "sum"),
        hit_rate=("net_pnl_rupees", lambda x: x.gt(0).mean()),
    )
    exit_mix.to_csv(OUT / "exit_mix.csv", index=False)

    cost_rows = []
    for bps in (0.0, 10.0, 25.0, 50.0, 100.0):
        stressed = apply_sizing(
            trades, float(selected_sizing.size_threshold),
            int(selected_sizing.high_confidence_lots), bps,
        )
        for year, period in ((2024, "test_2024"), (2025, "holdout_2025_partial")):
            cost_rows.append({
                "slippage_bps": bps, "period": period,
                **metrics(stressed[stressed["year"].eq(year)]),
            })
    cost_sensitivity = pd.DataFrame(cost_rows)
    cost_sensitivity.to_csv(OUT / "cost_sensitivity.csv", index=False)

    rng = np.random.default_rng(20261013)
    bootstrap_rows = []
    for year, period in ((2024, "test_2024"), (2025, "holdout_2025_partial")):
        values = evaluation[evaluation["year"].eq(year)]["net_pnl_rupees"].to_numpy(float)
        draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        bootstrap_rows.append({
            "period": period, "days": len(values), "observed_mean": values.mean(),
            "mean_ci_low": np.quantile(draws, 0.025), "mean_ci_high": np.quantile(draws, 0.975),
            "bootstrap_probability_mean_positive": np.mean(draws > 0),
        })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "bootstrap_uncertainty.csv", index=False)
    audit = pd.DataFrame([
        {"check": "one_trade_per_day", "violations": int(evaluation["date"].duplicated().sum())},
        {"check": "cross_day_window", "violations": int((evaluation["entry_datetime"].dt.normalize() != evaluation["exit_datetime"].dt.normalize()).sum())},
        {"check": "outside_0915_1530", "violations": int((
            ~(evaluation["entry_datetime"].dt.hour * 60 + evaluation["entry_datetime"].dt.minute).between(555, 930)
            | ~(evaluation["exit_datetime"].dt.hour * 60 + evaluation["exit_datetime"].dt.minute).between(555, 930)
        ).sum())},
        {"check": "holding_exceeds_180_minutes", "violations": int((
            (evaluation["exit_datetime"] - evaluation["entry_datetime"]).dt.total_seconds() / 60 > 180
        ).sum())},
        {"check": "expiry_before_entry", "violations": int((
            pd.to_datetime(evaluation["expiry"]).dt.normalize() < evaluation["entry_datetime"].dt.normalize()
        ).sum())},
    ])
    audit["passed"] = audit["violations"].eq(0)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)
    policy = {
        "status": decision, "research_only": True,
        "structure": "short_atm_straddle", "same_monthly_expiry": True,
        "horizon_max_minutes": key[0], "stop_fraction": key[1], "take_fraction": key[2],
        "take_profit_interpretation": "1.0 means effectively disabled; otherwise exit at 180 minutes",
        "score_quantile": key[3], "dte_floor": key[4],
        "ordinary_lots": 1,
        "high_confidence_lots": int(selected_sizing.high_confidence_lots),
        "high_confidence_probability_threshold": float(selected_sizing.size_threshold),
        "train": "2019-2022", "validation_selection": "2023 only",
        "test": "2024", "holdout": "2025 through 2025-05-09",
        "slippage_bps": SLIPPAGE_BPS, "live_deployment": False,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    report = f"""# Near-daily risk-managed short-volatility backtest

This test sells one same-expiry ATM straddle on each eligible date. The entry
timestamp is the day's highest score from a model trained on 2019-2022. Stop,
take-profit, score, and DTE settings are selected using 2023 only. Stops use
the first subsequently observed fixed-strike premium satisfying the rule;
there is no end-of-window look-ahead.

Decision: `{decision}`.

## Trading equations

The entry model estimates

`p_t = Pr(net short-straddle P&L > 0 | IV, RV lags, skew, returns, OI, Greeks, DTE, time)`.

For each day, use the candidate timestamp with maximum `p_t` and trade only
when `p_t >= {float(selected.score_threshold):.6f}`. Position size is

`lots_t = 3 if p_t >= {float(selected_sizing.size_threshold):.6f}, otherwise 1`.

For a fixed ATM strike and the same monthly expiry,

`net P&L = lots_t * contract_lot * [(C_entry + P_entry) - (C_exit + P_exit)] - all costs`.

Exit at the first fixed-strike quote where combined premium is at least 150%
of entry premium; otherwise exit after 180 minutes. The selected take-profit
parameter is 100%, which is effectively disabled because option premium cannot
fall below zero.

## Selected 2023 rule

{dataframe_to_markdown(pd.DataFrame([selected]).round(4))}

## Frozen evaluation

{dataframe_to_markdown(performance.round(4))}

## One-lot benchmark

{dataframe_to_markdown(base_performance.round(4))}

## Selected confidence sizing

{dataframe_to_markdown(pd.DataFrame([selected_sizing]).round(4))}

## Exit attribution

{dataframe_to_markdown(exit_mix.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

## Cost sensitivity

{dataframe_to_markdown(cost_sensitivity[["slippage_bps", "period", "days", "net_pnl", "profit_factor", "max_drawdown"]].round(4))}

## Bootstrap uncertainty

{dataframe_to_markdown(bootstrap.round(4))}

The rule is rejected if it cannot produce positive net P&L independently in
both halves of 2023 at 210-231 signal days. Mid-price paths and modeled
slippage do not guarantee executable fills. Short straddles retain gap and
tail risk even with an intraday stop.

The higher net P&L uses up to three lots and therefore requires materially
more margin and tail-risk capacity. Historical margin data is unavailable, so
the report does not claim a return-on-margin improvement from raw rupee P&L.

## Sizing exposure

{dataframe_to_markdown(exposure.round(4))}

## Research basis

- Carr and Wu (2009), *Variance Risk Premiums*, motivate measuring the wedge
  between risk-neutral option-implied variance and expected physical variance.
  https://academic.oup.com/rfs/article-abstract/22/3/1311/1581057
- Goyal and Saretto (2009), *Cross-Section of Option Returns and Volatility*,
  motivate conditioning long/short option returns on realized versus implied
  volatility. https://doi.org/10.1016/j.jfineco.2009.01.001
- Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*,
  motivates using realized-volatility signals at several horizons rather than
  one lag. https://doi.org/10.1093/jjfinec/nbp001
- Bali et al. (2023), *Option Return Predictability with Machine Learning and
  Big Data*, motivate combining option and underlying characteristics while
  testing nonlinearities out of sample.
  https://academic.oup.com/rfs/article/36/9/3548/7056660
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(policy)
    print(pd.DataFrame([selected]).to_string(index=False))
    print(performance.to_string(index=False))
    print(exit_mix.to_string(index=False))


if __name__ == "__main__":
    main()
