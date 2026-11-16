from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

import train_fixed_atm_horizons as base
import train_high_win_rate_models as high


HISTORICAL = Path("artifacts/fixed_atm_horizons_leakage_free/prepared_fixed_atm_labels.parquet")
ENTRY_2026 = Path(os.environ.get("VOLTR_43W_2026_ENTRY_SOURCE", Path(__file__).resolve().parents[3] / "data" / "banknifty_options_model_input_2026.csv"))
CHAIN_2026 = Path(os.environ.get("VOLTR_43W_2026_CHAIN_SOURCE", Path(__file__).resolve().parents[3] / "data" / "BANKNIFTY_2026.parquet"))
FROZEN_CLASSIFIER_METRICS = Path("artifacts/high_win_rate_models/metrics.json")
FROZEN_HYBRID_METRICS = Path("artifacts/high_win_rate_hybrid/metrics.json")
OUTPUT = Path("artifacts/frozen_models_2026_holdout")


def build_2026_prepared() -> pd.DataFrame:
    connection = duckdb.connect()
    entry_sql = """
    WITH raw AS (
        SELECT
            CAST(datetime AS TIMESTAMP) AS signal_datetime,
            CAST(Date AS DATE) AS trade_date,
            CAST(tick_time AS VARCHAR) AS signal_tick_time,
            CAST(expiry AS DATE) AS expiry,
            CAST(strike AS DOUBLE) AS strike,
            CAST(atm_strike AS DOUBLE) AS atm_strike,
            CAST(days_to_expiry AS DOUBLE) AS days_to_expiry,
            CAST(spot_price AS DOUBLE) AS spot_price,
            COALESCE(TRY_CAST(future_open AS DOUBLE), CAST(synthetic_future AS DOUBLE)) AS future_open,
            TRY_CAST(Future_oi AS DOUBLE) AS future_oi,
            CAST(call_price AS DOUBLE) AS call_price,
            CAST(put_price AS DOUBLE) AS put_price,
            CAST(call_oi AS DOUBLE) AS call_oi,
            CAST(put_oi AS DOUBLE) AS put_oi,
            CAST(iv_call AS DOUBLE) AS iv_call,
            CAST(iv_put AS DOUBLE) AS iv_put,
            CAST(delta_call AS DOUBLE) AS delta_call,
            CAST(delta_put AS DOUBLE) AS delta_put,
            CAST(vega_call AS DOUBLE) AS vega_call,
            CAST(vega_put AS DOUBLE) AS vega_put,
            CAST(gamma_call AS DOUBLE) AS gamma_call,
            CAST(gamma_put AS DOUBLE) AS gamma_put,
            CAST(theta_call AS DOUBLE) AS theta_call,
            CAST(theta_put AS DOUBLE) AS theta_put,
            dte_flag,
            TRUE AS is_monthly
        FROM read_csv_auto(?, header=true)
        WHERE dte_flag = 'M1'
          AND days_to_expiry >= 21.0 AND days_to_expiry < 35.0
    ), chosen AS (
        SELECT signal_datetime, MIN(expiry) AS expiry
        FROM raw GROUP BY signal_datetime
    )
    SELECT r.*, r.call_price + r.put_price AS signal_straddle
    FROM raw r INNER JOIN chosen c USING (signal_datetime, expiry)
    WHERE r.strike = r.atm_strike
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY r.signal_datetime, r.expiry ORDER BY r.strike
    ) = 1
    ORDER BY r.signal_datetime
    """
    frame = connection.execute(entry_sql, [str(ENTRY_2026)]).fetchdf()
    frame["signal_datetime"] = pd.to_datetime(frame["signal_datetime"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    # The source timestamp is treated as the time the completed one-minute
    # quote becomes known. Execution is delayed to the next minute.
    frame["signal_quote_available_at"] = frame["signal_datetime"]
    frame["datetime"] = frame["signal_datetime"] + pd.Timedelta(1, unit="m")
    frame["trade_date"] = frame["datetime"].dt.normalize()
    frame = frame.reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))

    wanted_parts = []
    for horizon in (0, *base.HORIZONS):
        wanted_parts.append(
            pd.DataFrame(
                {
                    "row_id": frame["row_id"],
                    "horizon": horizon,
                    "quote_datetime": frame["datetime"] + pd.Timedelta(horizon, unit="m"),
                    "expiry": frame["expiry"],
                    "strike": frame["strike"],
                }
            )
        )
    wanted = pd.concat(wanted_parts, ignore_index=True)
    connection.register("wanted_quotes", wanted)
    connection.execute("SET TimeZone='Asia/Calcutta'")
    quote_sql = """
    WITH chain AS (
        SELECT
            CAST(timestamp AS TIMESTAMP) AS quote_datetime,
            CAST(expiry AS DATE) AS expiry,
            CAST(strike AS DOUBLE) AS strike,
            option_type,
            CAST(close AS DOUBLE) AS price
        FROM read_parquet(?)
        WHERE underlying = 'BANKNIFTY'
    ), matched AS (
        SELECT
            w.row_id, w.horizon, w.quote_datetime, w.expiry, w.strike,
            MAX(CASE WHEN c.option_type = 'CE' THEN c.price END) AS call_price,
            MAX(CASE WHEN c.option_type = 'PE' THEN c.price END) AS put_price
        FROM wanted_quotes w
        INNER JOIN chain c
          ON c.quote_datetime = w.quote_datetime
         AND c.expiry = w.expiry
         AND c.strike = w.strike
        GROUP BY w.row_id, w.horizon, w.quote_datetime, w.expiry, w.strike
    )
    SELECT * FROM matched WHERE call_price IS NOT NULL AND put_price IS NOT NULL
    """
    quotes = connection.execute(quote_sql, [str(CHAIN_2026)]).fetchdf()
    connection.close()

    entry = quotes[quotes["horizon"] == 0].copy().rename(
        columns={
            "quote_datetime": "entry_quote_datetime",
            "strike": "entry_strike",
            "call_price": "entry_call_price",
            "put_price": "entry_put_price",
        }
    )
    entry["entry_quote_available_at"] = pd.to_datetime(entry["entry_quote_datetime"])
    entry["entry_straddle"] = entry["entry_call_price"] + entry["entry_put_price"]
    frame = frame.merge(
        entry[["row_id", "entry_quote_datetime", "entry_quote_available_at", "entry_strike", "entry_straddle"]],
        on="row_id", how="left", validate="one_to_one",
    )
    for horizon in base.HORIZONS:
        part = quotes[quotes["horizon"] == horizon].copy().rename(
            columns={
                "quote_datetime": f"exit_datetime_{horizon}",
                "strike": f"exit_strike_{horizon}",
                "call_price": f"exit_call_price_{horizon}",
                "put_price": f"exit_put_price_{horizon}",
            }
        )
        part[f"exit_quote_available_at_{horizon}"] = pd.to_datetime(part[f"exit_datetime_{horizon}"])
        part[f"exit_straddle_{horizon}"] = part[f"exit_call_price_{horizon}"] + part[f"exit_put_price_{horizon}"]
        keep = [
            "row_id", f"exit_datetime_{horizon}", f"exit_quote_available_at_{horizon}",
            f"exit_strike_{horizon}", f"exit_straddle_{horizon}",
        ]
        frame = frame.merge(part[keep], on="row_id", how="left", validate="one_to_one")
    return frame


def evaluate_classifier_models(hist: pd.DataFrame, holdout: pd.DataFrame, features: list[str]) -> list[dict]:
    frozen = json.loads(FROZEN_CLASSIFIER_METRICS.read_text(encoding="utf-8"))
    frozen_by_horizon = {row["horizon_minutes"]: row for row in frozen["results"]}
    results = []
    for horizon in base.HORIZONS:
        historical, target = high.horizon_data(hist, horizon)
        test, test_target = high.horizon_data(holdout, horizon)
        spec = frozen_by_horizon[horizon]
        model = high.classifiers()[spec["selected_model"]]
        model.fit(historical[features], historical[target].gt(0).astype(int))
        probability = model.predict_proba(test[features])[:, 1]
        metrics, trades = base.non_overlapping_strategy(
            test, test[test_target].to_numpy(), probability, horizon, spec["probability_threshold"]
        )
        wins = int((trades["actual_net_points_after_2pt_cost"] > 0).sum())
        metrics["wilson_win_rate_lower_95"] = high.wilson_lower_bound(wins, len(trades))
        artifact_name = f"frozen_classifier_{horizon}m.joblib"
        joblib.dump(
            {
                "model": model, "model_name": spec["selected_model"], "features": features,
                "horizon_minutes": horizon, "probability_threshold": spec["probability_threshold"],
                "trained_through": str(hist["datetime"].max()), "holdout_year": 2026,
            },
            OUTPUT / artifact_name,
        )
        trades.to_csv(OUTPUT / f"classifier_2026_trades_{horizon}m.csv", index=False)
        results.append(
            {
                "horizon_minutes": horizon,
                "model": spec["selected_model"],
                "probability_threshold": spec["probability_threshold"],
                "historical_rows": len(historical),
                "holdout_rows": len(test),
                "artifact": artifact_name,
                "metrics_2026": metrics,
            }
        )
    return results


def evaluate_hybrid(hist: pd.DataFrame, holdout: pd.DataFrame, features: list[str]) -> dict:
    frozen = json.loads(FROZEN_HYBRID_METRICS.read_text(encoding="utf-8"))
    policy = frozen["selected_validation_policy"]
    historical, target = high.horizon_data(hist, 90)
    test, test_target = high.horizon_data(holdout, 90)
    classifier = high.classifiers()["extra_trees_classifier"]
    regressor = base.candidate_models()["lightgbm_huber"]
    classifier.fit(historical[features], historical[target].gt(0).astype(int))
    regressor.fit(historical[features], historical[target])
    probability = classifier.predict_proba(test[features])[:, 1]
    edge = regressor.predict(test[features])
    score = np.where(
        (probability >= policy["probability_threshold"])
        & (edge >= policy["edge_threshold"]),
        probability,
        -np.inf,
    )
    metrics, trades = base.non_overlapping_strategy(
        test, test[test_target].to_numpy(), score, 90, policy["probability_threshold"]
    )
    wins = int((trades["actual_net_points_after_2pt_cost"] > 0).sum())
    metrics["wilson_win_rate_lower_95"] = high.wilson_lower_bound(wins, len(trades))
    joblib.dump(
        {
            "classifier": classifier, "regressor": regressor, "features": features,
            "horizon_minutes": 90,
            "probability_threshold": policy["probability_threshold"],
            "predicted_net_edge_threshold": policy["edge_threshold"],
            "trained_through": str(hist["datetime"].max()), "holdout_year": 2026,
        },
        OUTPUT / "frozen_hybrid_90m.joblib",
    )
    trades.to_csv(OUTPUT / "hybrid_2026_trades_90m.csv", index=False)
    return {
        "probability_threshold": policy["probability_threshold"],
        "edge_threshold": policy["edge_threshold"],
        "historical_rows": len(historical),
        "holdout_rows": len(test),
        "metrics_2026": metrics,
    }


def write_report(classifiers: list[dict], hybrid: dict, feature_count: int) -> None:
    lines = [
        "# Frozen Models 2026 Unseen Holdout Results",
        "",
        "All model families, horizons, and thresholds were frozen before loading 2026. Models were refit on all available development rows through May 2025 and evaluated once on January-June 2026.",
        "",
        f"Entry-time features: {feature_count}.",
        "",
        "## Classifier models",
        "",
        "| Horizon | Model | Threshold | Trades | Win rate | Net points | Profit factor | Max drawdown | Wilson lower 95% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in classifiers:
        m = row["metrics_2026"]
        lines.append(
            f"| {row['horizon_minutes']}m | {row['model']} | {row['probability_threshold']:.2f} | "
            f"{m['trades']} | {m['win_rate_after_slippage']:.2%} | {m['net_points_after_2pt_slippage']:.1f} | "
            f"{m['profit_factor']:.2f} | {m['maximum_drawdown_points']:.1f} | {m['wilson_win_rate_lower_95']:.2%} |"
        )
    hm = hybrid["metrics_2026"]
    hybrid_summary = (
        f"It produced **{hm['trades']} trades**, **{hm['win_rate_after_slippage']:.2%} wins**, **{hm['net_points_after_2pt_slippage']:.1f} net points**, profit factor **{hm['profit_factor']:.2f}**, and maximum drawdown **{hm['maximum_drawdown_points']:.1f} points**."
        if hm["trades"]
        else "It produced **no qualifying trades** in the 2026 holdout. Win rate, profit factor, and drawdown are therefore not defined."
    )
    lines.extend(
        [
            "",
            "## Frozen hybrid 90 minute model",
            "",
            f"The frozen hybrid used probability threshold {hybrid['probability_threshold']:.2f} and predicted net-edge threshold {hybrid['edge_threshold']:.1f} points.",
            "",
            hybrid_summary,
            "",
            "The 2026 data was not used for model, feature, horizon, or threshold selection.",
            "",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    historical = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(HISTORICAL)]).fetchdf()
    holdout_cache = OUTPUT / "prepared_2026_holdout.parquet"
    if holdout_cache.exists():
        holdout = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(holdout_cache)]).fetchdf()
    else:
        holdout = build_2026_prepared()
        connection = duckdb.connect()
        connection.register("holdout", holdout)
        connection.execute("COPY holdout TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(holdout_cache)])
        connection.close()
    historical, features = base.add_entry_features(historical)
    holdout, holdout_features = base.add_entry_features(holdout)
    if features != holdout_features:
        raise AssertionError("Historical and 2026 feature lists differ")
    forbidden = [name for name in features if name.startswith(("exit_", "target_", "label_"))]
    forbidden.extend(name for name in features if name.startswith("entry_"))
    if forbidden:
        raise AssertionError(f"Future/execution fields entered features: {forbidden}")
    classifier_results = evaluate_classifier_models(historical, holdout, features)
    hybrid_result = evaluate_hybrid(historical, holdout, features)
    manifest = {
        "development_end": str(historical["datetime"].max()),
        "holdout_start": str(holdout["datetime"].min()),
        "holdout_end": str(holdout["datetime"].max()),
        "features": features,
        "classifier_results": classifier_results,
        "hybrid_result": hybrid_result,
        "leakage_audit": {
            "2026_used_for_selection": False,
            "frozen_classifier_source": str(FROZEN_CLASSIFIER_METRICS),
            "frozen_hybrid_source": str(FROZEN_HYBRID_METRICS),
            "fixed_signal_atm_strike": True,
            "one_minute_execution_delay": True,
            "future_or_execution_fields_in_features": forbidden,
        },
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(classifier_results, hybrid_result, len(features))
    print(json.dumps({"classifiers": classifier_results, "hybrid": hybrid_result}, indent=2))


if __name__ == "__main__":
    main()
