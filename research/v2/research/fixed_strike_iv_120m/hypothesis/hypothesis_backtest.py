from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


FIXED_COLUMNS = [
    "row_id", "datetime", "exit_datetime", "Date", "dte_flag", "expiry", "entry_strike",
    "entry_spot_price", "entry_call_price", "entry_put_price", "entry_iv_call", "entry_iv_put",
    "entry_vega_call", "entry_vega_put", "exit_call_price", "exit_put_price", "exit_iv_call",
    "exit_iv_put", "exit_vega_call", "exit_vega_put", "exit_matched_strike",
    "exit_source_datetime", "quality_trainable",
]

PATH_COLUMNS = [
    "datetime", "Date", "dte_flag", "expiry", "strike", "call_price", "put_price",
]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def vega_weighted_iv(call_iv: pd.Series, put_iv: pd.Series,
                     call_vega: pd.Series, put_vega: pd.Series) -> pd.Series:
    call_weight = call_vega.abs()
    put_weight = put_vega.abs()
    total = call_weight + put_weight
    weighted = (call_iv * call_weight + put_iv * put_weight) / total
    fallback = (call_iv + put_iv) / 2.0
    return weighted.where(total.gt(0) & total.notna(), fallback)


def load_spot_history(path: str, chunksize: int) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(path, usecols=["datetime", "spot_price"], chunksize=chunksize, low_memory=False):
        chunk = chunk.dropna().drop_duplicates("datetime")
        pieces.append(chunk)
    spot = pd.concat(pieces, ignore_index=True).drop_duplicates("datetime", keep="last")
    spot["datetime"] = pd.to_datetime(spot["datetime"])
    return spot.rename(columns={"spot_price": "entry_spot_price"}).sort_values("datetime")


def realized_volatility_map(frame: pd.DataFrame, window_minutes: int, min_covered_minutes: int,
                            max_gap_minutes: int, annual_trading_days: int,
                            minutes_per_day: int) -> pd.Series:
    spot = frame[["datetime", "entry_spot_price"]].dropna().copy()
    spot = spot.groupby("datetime", as_index=False)["entry_spot_price"].median().sort_values("datetime")
    spot["date"] = spot["datetime"].dt.date
    pieces = []
    annual_minutes = annual_trading_days * minutes_per_day
    for _, day in spot.groupby("date", sort=False):
        day = day.sort_values("datetime").copy()
        elapsed_minutes = day["datetime"].diff().dt.total_seconds().div(60.0)
        valid = elapsed_minutes.gt(0) & elapsed_minutes.le(max_gap_minutes)
        returns = np.log(day["entry_spot_price"]).diff().where(valid)
        index = day["datetime"]
        squared = pd.Series(returns.to_numpy() ** 2, index=index)
        exposure = pd.Series(elapsed_minutes.where(valid).to_numpy(), index=index)
        variance_sum = squared.rolling(f"{window_minutes}min", min_periods=1).sum()
        covered_minutes = exposure.rolling(f"{window_minutes}min", min_periods=1).sum()
        per_minute_variance = (variance_sum / covered_minutes).where(covered_minutes >= min_covered_minutes)
        pieces.append(pd.Series(np.sqrt(per_minute_variance * annual_minutes).to_numpy(), index=index))
    return pd.concat(pieces).sort_index() if pieces else pd.Series(dtype=float)


def lot_sizes(expiry: pd.Series, schedule: list[dict]) -> pd.Series:
    output = pd.Series(np.nan, index=expiry.index, dtype=float)
    parsed = pd.to_datetime(expiry)
    for rule in schedule:
        mask = output.isna() & parsed.lt(pd.Timestamp(rule["before"]))
        output.loc[mask] = int(rule["lot_size"])
    if output.isna().any():
        raise ValueError("Lot-size schedule does not cover every expiry")
    return output.astype(int)


def build_hypothesis_table(config: dict) -> pd.DataFrame:
    source = Path(config["data"]["fixed_strike_dataset"])
    frame = pd.read_csv(source, usecols=FIXED_COLUMNS, low_memory=False)
    for column in ("datetime", "exit_datetime", "exit_source_datetime", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    settings = config["hypothesis"]
    spot_history = load_spot_history(config["data"]["entry_surface_csv"], int(config["data"]["chunksize"]))
    rv = realized_volatility_map(
        spot_history, int(settings["rv_window_minutes"]), int(settings["rv_min_covered_minutes"]),
        int(settings["rv_max_gap_minutes"]),
        int(settings["annual_trading_days"]), int(settings["minutes_per_trading_day"]),
    )
    frame = frame.loc[as_bool(frame["quality_trainable"])].copy()
    frame = frame.loc[
        frame["entry_strike"].eq(frame["exit_matched_strike"])
        & frame["exit_source_datetime"].eq(frame["exit_datetime"])
    ].copy()
    frame["rv_current"] = frame["datetime"].map(rv)
    frame["rv_t120"] = frame["exit_datetime"].map(rv)
    frame["iv_current"] = vega_weighted_iv(
        frame["entry_iv_call"], frame["entry_iv_put"], frame["entry_vega_call"], frame["entry_vega_put"]
    )
    frame["iv_t120"] = vega_weighted_iv(
        frame["exit_iv_call"], frame["exit_iv_put"], frame["exit_vega_call"], frame["exit_vega_put"]
    )
    frame["entry_straddle_price"] = frame["entry_call_price"] + frame["entry_put_price"]
    frame["exit_straddle_price"] = frame["exit_call_price"] + frame["exit_put_price"]
    frame["lot_size"] = lot_sizes(frame["expiry"], config["risk"]["lot_size_by_expiry"])
    frame["horizon_pnl_points"] = frame["entry_straddle_price"] - frame["exit_straddle_price"]
    frame["horizon_pnl_per_lot"] = frame["horizon_pnl_points"] * frame["lot_size"]

    frame["condition_1_iv_current_gt_rv_current"] = frame["iv_current"] > frame["rv_current"]
    frame["condition_3_converged_down"] = (
        frame["iv_t120"].lt(frame["iv_current"])
        & (frame["iv_t120"] - frame["rv_t120"]).abs().lt(
            (frame["iv_current"] - frame["rv_current"]).abs()
        )
    )
    for side in ("call", "put"):
        frame[f"condition_1_{side}"] = frame[f"entry_iv_{side}"] > frame["rv_current"]
        frame[f"condition_3_{side}"] = (
            frame[f"exit_iv_{side}"].lt(frame[f"entry_iv_{side}"])
            & (frame[f"exit_iv_{side}"] - frame["rv_t120"]).abs().lt(
                (frame[f"entry_iv_{side}"] - frame["rv_current"]).abs()
            )
        )

    primary = float(settings["primary_near_tolerance_iv"])
    for tolerance in [float(value) for value in settings["tolerance_grid_iv"]]:
        label = int(round(tolerance * 10000))
        frame[f"condition_2_near_{label}bp"] = (frame["iv_t120"] - frame["rv_t120"]).abs() <= tolerance
        frame[f"hypothesis_composite_{label}bp"] = (
            frame["condition_1_iv_current_gt_rv_current"]
            & frame[f"condition_2_near_{label}bp"]
            & frame["condition_3_converged_down"]
        )
        both_legs = (
            frame["condition_1_call"] & frame["condition_1_put"]
            & ((frame["exit_iv_call"] - frame["rv_t120"]).abs() <= tolerance)
            & ((frame["exit_iv_put"] - frame["rv_t120"]).abs() <= tolerance)
            & frame["condition_3_call"] & frame["condition_3_put"]
        )
        frame[f"hypothesis_both_legs_{label}bp"] = both_legs
    primary_label = int(round(primary * 10000))
    primary_column = (
        f"hypothesis_both_legs_{primary_label}bp" if bool(settings.get("require_both_legs", False))
        else f"hypothesis_composite_{primary_label}bp"
    )
    frame["hypothesis_primary"] = frame[primary_column]
    frame["primary_definition"] = primary_column
    return frame


def scan_same_strike_paths(signals: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, int]:
    if signals.empty:
        return pd.DataFrame(columns=PATH_COLUMNS), 0
    keys = signals[["Date", "dte_flag", "expiry", "entry_strike"]].drop_duplicates().copy()
    keys["Date"] = keys["Date"].astype(str)
    keys["expiry"] = pd.to_datetime(keys["expiry"]).dt.strftime("%Y-%m-%d")
    keys = keys.rename(columns={"entry_strike": "strike"})
    earliest, latest = keys["Date"].min(), keys["Date"].max()
    matches = []
    rows_scanned = 0
    for chunk in pd.read_csv(
        config["data"]["full_chain_csv"], usecols=PATH_COLUMNS,
        chunksize=int(config["data"]["chunksize"]), low_memory=False,
    ):
        rows_scanned += len(chunk)
        chunk["Date"] = chunk["Date"].astype(str)
        if chunk["Date"].max() < earliest:
            continue
        if chunk["Date"].min() > latest:
            break
        candidate = chunk.loc[chunk["Date"].between(earliest, latest)]
        if candidate.empty:
            continue
        found = candidate.merge(keys, on=["Date", "dte_flag", "expiry", "strike"], how="inner")
        if not found.empty:
            matches.append(found)
    if not matches:
        return pd.DataFrame(columns=PATH_COLUMNS), rows_scanned
    paths = pd.concat(matches, ignore_index=True)
    paths["datetime"] = pd.to_datetime(paths["datetime"])
    paths["expiry"] = pd.to_datetime(paths["expiry"])
    paths = paths.drop_duplicates(["datetime", "dte_flag", "expiry", "strike"], keep="last")
    return paths.sort_values(["Date", "dte_flag", "expiry", "strike", "datetime"]), rows_scanned


def first_barrier(path: pd.DataFrame, entry_premium: float, lot_size: int,
                  stop: float, target: float | None, cost: float) -> tuple[pd.Timestamp | None, str | None, float | None]:
    if path.empty:
        return None, None, None
    pnl = (entry_premium - (path["call_price"] + path["put_price"])) * lot_size - cost
    events = pd.Series(False, index=path.index)
    stop_hit = pnl <= -stop
    events |= stop_hit
    if target is not None:
        target_hit = pnl >= target
        events |= target_hit
    if not events.any():
        return None, None, None
    first = events[events].index[0]
    reason = "stop_loss" if bool(stop_hit.loc[first]) else "profit_target"
    return pd.Timestamp(path.loc[first, "datetime"]), reason, float(pnl.loc[first])


def backtest_signals(signals: pd.DataFrame, paths: pd.DataFrame, config: dict) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    risk = config["risk"]
    stop = float(risk["stop_loss_per_lot"])
    target = float(risk["profit_target_per_lot"])
    cost = float(risk.get("round_trip_cost_per_lot", 0.0))
    grouped = {
        key: group.sort_values("datetime").set_index("datetime", drop=False)
        for key, group in paths.groupby(["Date", "dte_flag", "expiry", "strike"], sort=False)
    }
    records = []
    for row in signals.itertuples(index=False):
        key = (str(row.Date), row.dte_flag, pd.Timestamp(row.expiry), row.entry_strike)
        path = grouped.get(key, pd.DataFrame())
        if not path.empty:
            path = path.loc[(path["datetime"] > row.datetime) & (path["datetime"] <= row.exit_datetime)].copy()
        record = row._asdict()
        record["path_observations"] = len(path)
        for policy, policy_target in (("stop_only", None), ("bracket", target)):
            event_time, reason, event_pnl = first_barrier(
                path, row.entry_straddle_price, int(row.lot_size), stop, policy_target, cost
            )
            if event_time is None:
                record[f"{policy}_exit_datetime"] = row.exit_datetime
                record[f"{policy}_exit_reason"] = "time_exit"
                record[f"{policy}_pnl_per_lot"] = row.horizon_pnl_per_lot - cost
                record[f"{policy}_holding_minutes"] = 120.0
            else:
                record[f"{policy}_exit_datetime"] = event_time
                record[f"{policy}_exit_reason"] = reason
                record[f"{policy}_pnl_per_lot"] = event_pnl
                record[f"{policy}_holding_minutes"] = (event_time - row.datetime).total_seconds() / 60.0
        records.append(record)
    return pd.DataFrame(records)


def non_overlapping(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    chosen = []
    available = pd.Timestamp.min
    exit_column = f"{policy}_exit_datetime"
    for row in frame.sort_values(["datetime", "row_id"]).itertuples():
        if row.datetime >= available:
            chosen.append(row.Index)
            available = pd.Timestamp(getattr(row, exit_column))
    return frame.loc[chosen].copy()


def summary_row(frame: pd.DataFrame, pnl_column: str, label: str) -> dict:
    pnl = frame[pnl_column].dropna()
    return {
        "sample": label, "trades": int(len(pnl)), "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "mean_pnl": float(pnl.mean()) if len(pnl) else np.nan,
        "median_pnl": float(pnl.median()) if len(pnl) else np.nan,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "loss_rate": float((pnl < 0).mean()) if len(pnl) else np.nan,
        "profit_factor": float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()) if (pnl < 0).any() else np.nan,
        "minimum_pnl": float(pnl.min()) if len(pnl) else np.nan,
        "maximum_pnl": float(pnl.max()) if len(pnl) else np.nan,
    }


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    for key in ("fixed_strike_dataset", "entry_surface_csv", "full_chain_csv"):
        config["data"][key] = str((base / config["data"][key]).resolve())
    output = (base / config["output"]["directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    hypothesis = build_hypothesis_table(config)
    hypothesis.to_csv(output / "all_actual_value_hypotheses.csv", index=False)
    signals = hypothesis.loc[hypothesis["hypothesis_primary"]].copy()
    paths, rows_scanned = scan_same_strike_paths(signals, config)
    paths.to_csv(output / "same_strike_intraperiod_prices.csv", index=False)
    trades = backtest_signals(signals, paths, config)
    trades.to_csv(output / "primary_hypothesis_trades.csv", index=False)

    summaries = []
    for policy in ("stop_only", "bracket"):
        summaries.append(summary_row(trades, f"{policy}_pnl_per_lot", f"all_overlapping_{policy}"))
        sequential = non_overlapping(trades, policy)
        summaries.append(summary_row(sequential, f"{policy}_pnl_per_lot", f"non_overlapping_{policy}"))
        sequential.to_csv(output / f"non_overlapping_{policy}_trades.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "pnl_summary.csv", index=False)

    sensitivity = []
    for tolerance in [float(value) for value in config["hypothesis"]["tolerance_grid_iv"]]:
        label = int(round(tolerance * 10000))
        for definition in ("composite", "both_legs"):
            mask = hypothesis[f"hypothesis_{definition}_{label}bp"]
            selected = hypothesis.loc[mask]
            sensitivity.append({
                "tolerance_iv": tolerance, "tolerance_vol_points": tolerance * 100,
                "definition": definition, "observations": int(mask.sum()),
                "mean_horizon_pnl_per_lot": float(selected["horizon_pnl_per_lot"].mean()),
                "median_horizon_pnl_per_lot": float(selected["horizon_pnl_per_lot"].median()),
                "horizon_win_rate": float((selected["horizon_pnl_per_lot"] > 0).mean()),
            })
    pd.DataFrame(sensitivity).to_csv(output / "tolerance_sensitivity.csv", index=False)

    audit = {
        "input_rows": int(len(hypothesis)), "primary_signals": int(len(signals)),
        "same_strike_rate": float(hypothesis["entry_strike"].eq(hypothesis["exit_matched_strike"]).mean()),
        "exact_120m_rate": float(hypothesis["exit_source_datetime"].eq(hypothesis["exit_datetime"]).mean()),
        "rv_available_rows": int(hypothesis[["rv_current", "rv_t120"]].notna().all(axis=1).sum()),
        "full_chain_rows_scanned": int(rows_scanned), "intraperiod_rows": int(len(paths)),
        "minimum_path_observations": int(trades["path_observations"].min()) if len(trades) else 0,
        "median_path_observations": float(trades["path_observations"].median()) if len(trades) else 0,
        "uses_model_predictions": False,
        "lookahead_warning": "Conditions 2 and 3 use t+120 actual values; results test an ex-post hypothesis and are not a deployable entry rule.",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    run(parser.parse_args().config.resolve())
