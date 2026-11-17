from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "analysis_outputs" / "stress_test_2026-10-27"
PREDICTIONS = ROOT / "analysis_outputs" / "three_iv_rv_conditions" / "three_rule_predictions.csv.gz"
PRICES = OUTPUT / "replay_prices_exact.csv.gz"
SPOT = Path(os.environ.get("VOLTR_SPOT_1MIN_SOURCE", Path(__file__).resolve().parents[3] / "research" / "v2" / "data" / "icharts_banknifty_spot_1min.csv"))
LIVE_DIR = ROOT.parents[2] / "pipelines" / "ivrv"  # integrity check covers the live IVRV code

LOT_SIZE = 30
BASE_TARGET = 1500.0
BASE_STOP = -750.0
BASE_HOLD = 120
BASE_PROB = 0.75
BASE_EVENT = 0.0
BASE_CAP = 3
NEAR_GAP = 0.005
SQUARE_OFF_MINUTE = 15 * 60 + 14
ANNUALIZATION = 252
SEED = 20261027


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def live_hashes() -> dict[str, str]:
    return {
        str(path): sha256(path)
        for path in sorted(LIVE_DIR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def explicit_cost(entry_price: float, exit_price: float) -> float:
    entry_turnover = entry_price * LOT_SIZE
    exit_turnover = exit_price * LOT_SIZE
    turnover = entry_turnover + exit_turnover
    stt = entry_turnover * 0.0015
    nse = turnover * 0.0003553
    sebi = turnover * 0.000001
    stamp = exit_turnover * 0.00003
    gst = 0.18 * (nse + sebi)
    return stt + nse + sebi + stamp + gst


def sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1)
    return float(values.mean() / sd * math.sqrt(ANNUALIZATION)) if sd > 0 else math.nan


def max_drawdown(values: np.ndarray) -> float:
    equity = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=float))))
    return float(np.max(np.maximum.accumulate(equity) - equity))


def prepare_data() -> tuple[pd.DataFrame, dict, pd.DatetimeIndex]:
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["datetime", "expiry"], low_memory=False)
    predictions = predictions.loc[predictions["base_eligible"].astype(bool)].copy()
    predictions = predictions.sort_values("datetime", kind="stable").reset_index(drop=True)
    predictions["candidate_id"] = np.arange(len(predictions))
    predictions["trade_date"] = predictions["datetime"].dt.normalize()

    prices = pd.read_csv(PRICES, parse_dates=["datetime", "expiry"], low_memory=False)
    prices = prices.loc[
        prices["datetime"].between(predictions["datetime"].min().normalize(), predictions["datetime"].max().normalize() + pd.Timedelta(days=1))
    ].copy()
    prices["trade_date"] = prices["datetime"].dt.normalize()
    prices["straddle_price"] = prices["call_price"].astype(float) + prices["put_price"].astype(float)
    price_groups = {}
    for key, group in prices.groupby(["trade_date", "dte_flag", "expiry", "strike"], sort=False):
        group = group.sort_values("datetime", kind="stable")
        price_groups[key] = (
            group["datetime"].to_numpy(dtype="datetime64[ns]"),
            group["straddle_price"].to_numpy(dtype=float),
        )

    path_cache = {}
    for delay in [0, 1]:
        paths = {}
        for row in predictions.itertuples():
            key = (row.trade_date, row.dte_flag, row.expiry, float(row.entry_strike))
            group = price_groups.get(key)
            if group is None:
                continue
            timestamps, straddles = group
            intended_entry = np.datetime64(row.datetime + pd.Timedelta(minutes=delay), "ns")
            position = int(np.searchsorted(timestamps, intended_entry, side="left"))
            if position >= len(timestamps) or timestamps[position] != intended_entry:
                continue
            exit_limit = np.datetime64(
                row.trade_date + pd.Timedelta(hours=15, minutes=14, seconds=59), "ns"
            )
            end = int(np.searchsorted(timestamps, exit_limit, side="right"))
            observed_times = timestamps[position:end]
            observed_prices = straddles[position:end]
            elapsed = (observed_times - timestamps[position]).astype("timedelta64[s]").astype(float) / 60.0
            pnl = (straddles[position] - observed_prices) * LOT_SIZE
            paths[int(row.candidate_id)] = {
                "entry_datetime": pd.Timestamp(timestamps[position]),
                "entry_price": float(straddles[position]),
                "times": observed_times,
                "elapsed": elapsed,
                "pnl": pnl,
            }
        path_cache[delay] = paths
    eligible_dates = pd.DatetimeIndex(sorted(predictions["trade_date"].unique()))
    return predictions, path_cache, eligible_dates


def route_mask(frame: pd.DataFrame, probability_cutoff: float, event_cutoff: float) -> pd.Series:
    gap = frame["pred_no120_gate_iv_minus_rv"].astype(float)
    score = frame["pred_no120_event_score"].astype(float)
    probability = frame["pred_convergence_probability_classifier"].astype(float)
    rule = frame["pred_strategy_rule_iv_above_rv_iv_declines_to_rv"].astype(bool)
    above = (gap > NEAR_GAP) & rule & (probability > probability_cutoff)
    near = (gap.abs() <= NEAR_GAP) & (score >= event_cutoff)
    below = (gap < -NEAR_GAP) & (score >= event_cutoff)
    return above | near | below


def add_route(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    gap = result["pred_no120_gate_iv_minus_rv"].astype(float)
    result["route"] = np.select(
        [gap > NEAR_GAP, gap.abs() <= NEAR_GAP, gap < -NEAR_GAP],
        ["iv_above_rv", "iv_near_rv", "iv_below_rv"],
        default="unrouted",
    )
    return result


def outcomes_for(
    candidates: pd.DataFrame,
    paths: dict,
    target: float,
    stop: float,
    hold: int,
) -> pd.DataFrame:
    records = []
    for row in candidates.itertuples():
        path = paths.get(int(row.candidate_id))
        if path is None:
            continue
        cutoff = pd.Timestamp(row.trade_date) + pd.Timedelta(hours=15, minutes=14, seconds=59)
        deadline = min(cutoff, pd.Timestamp(path["entry_datetime"]) + pd.Timedelta(minutes=hold))
        deadline64 = np.datetime64(deadline.to_datetime64(), "ns")
        end_index = int(np.searchsorted(path["times"], deadline64, side="right"))
        chosen = None
        chosen_reason = None
        used_deadline_fallback = False
        for index in range(0, end_index):
            stamp = pd.Timestamp(path["times"][index])
            pnl = float(path["pnl"][index])
            if stamp.hour * 60 + stamp.minute >= SQUARE_OFF_MINUTE:
                chosen, chosen_reason = index, "session_square_off"
            elif math.isfinite(target) and pnl >= target:
                chosen, chosen_reason = index, "profit_target"
            elif math.isfinite(stop) and pnl <= stop:
                chosen, chosen_reason = index, "stop_loss"
            elif float(path["elapsed"][index]) >= hold:
                chosen, chosen_reason = index, "time_exit"
            if chosen is not None:
                break
        if chosen is None:
            chosen = max(end_index - 1, 0)
            chosen_reason = "session_square_off" if cutoff <= pd.Timestamp(path["entry_datetime"]) + pd.Timedelta(minutes=hold) else "time_exit"
            used_deadline_fallback = True
        entry_price = float(path["entry_price"])
        pnl = float(path["pnl"][chosen])
        exit_price = entry_price - pnl / LOT_SIZE
        cost = explicit_cost(entry_price, exit_price)
        records.append({
            "candidate_id": int(row.candidate_id),
            "signal_datetime": row.datetime,
            "entry_datetime": path["entry_datetime"],
            "exit_datetime": deadline if used_deadline_fallback else pd.Timestamp(path["times"][chosen]),
            "trade_date": row.trade_date,
            "expiry": row.expiry,
            "entry_strike": float(row.entry_strike),
            "probability": float(row.pred_convergence_probability_classifier),
            "event_score": float(row.pred_no120_event_score),
            "route": row.route if hasattr(row, "route") else "unassigned",
            "exit_reason": chosen_reason,
            "gross_pnl_inr": pnl,
            "explicit_costs_inr": cost,
            "net_pnl_inr": pnl - cost,
        })
    return pd.DataFrame(records)


def apply_cap(outcomes: pd.DataFrame, cap: int) -> pd.DataFrame:
    heap: list[int] = []
    accepted = []
    for index, row in outcomes.sort_values("entry_datetime", kind="stable").iterrows():
        entry_ns = int(pd.Timestamp(row["entry_datetime"]).value)
        while heap and heap[0] <= entry_ns:
            heapq.heappop(heap)
        if len(heap) >= cap:
            continue
        heapq.heappush(heap, int(pd.Timestamp(row["exit_datetime"]).value))
        accepted.append(index)
    return outcomes.loc[accepted].sort_values("entry_datetime", kind="stable").reset_index(drop=True)


def metrics(ledger: pd.DataFrame, eligible_dates: pd.DatetimeIndex) -> dict:
    if ledger.empty:
        daily = np.zeros(len(eligible_dates))
        pnl = np.array([], dtype=float)
    else:
        pnl = ledger["net_pnl_inr"].to_numpy(dtype=float)
        daily_series = ledger.groupby("trade_date")["net_pnl_inr"].sum()
        daily = daily_series.reindex(eligible_dates, fill_value=0.0).to_numpy(dtype=float)
    return {
        "trades": int(len(ledger)),
        "net_pnl_inr": float(pnl.sum()),
        "sharpe": sharpe(daily),
        "max_drawdown_inr": max_drawdown(pnl),
        "win_rate_pct": float(np.mean(pnl > 0) * 100) if len(pnl) else None,
    }


def ex_ante_interval_selection(outcomes: pd.DataFrame, cap: int, priorities: np.ndarray) -> pd.DataFrame:
    work = outcomes.copy()
    work["priority"] = priorities
    selected: list[int] = []
    selected_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for index, row in work.sort_values("priority", kind="stable").iterrows():
        start, end = pd.Timestamp(row.entry_datetime), pd.Timestamp(row.exit_datetime)
        checkpoints = [start] + [s for s, _ in selected_intervals if start < s < end]
        feasible = True
        for checkpoint in checkpoints:
            active = sum(s <= checkpoint < e for s, e in selected_intervals)
            if active >= cap:
                feasible = False
                break
        if feasible:
            selected.append(index)
            selected_intervals.append((start, end))
    return work.loc[selected].sort_values("entry_datetime", kind="stable").reset_index(drop=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes_before = live_hashes()
    predictions, path_cache, eligible_dates = prepare_data()
    predictions = add_route(predictions)
    baseline_mask = route_mask(predictions, BASE_PROB, BASE_EVENT)
    baseline_candidates = predictions.loc[baseline_mask].copy()
    baseline_all_outcomes = outcomes_for(
        predictions, path_cache[0], BASE_TARGET, BASE_STOP, BASE_HOLD
    )
    baseline_outcomes = baseline_all_outcomes.loc[
        baseline_all_outcomes["candidate_id"].isin(baseline_candidates["candidate_id"])
    ].copy()
    baseline_ledger = apply_cap(baseline_outcomes, BASE_CAP)
    baseline_metrics = metrics(baseline_ledger, eligible_dates)
    baseline_ledger.to_csv(OUTPUT / "tier2_replayed_baseline_trades.csv", index=False)

    validation = {
        "expected_trades": 304,
        "replayed_trades": int(len(baseline_ledger)),
        "expected_gross_pnl_inr": 129517.5,
        "replayed_gross_pnl_inr": float(baseline_ledger["gross_pnl_inr"].sum()),
    }
    validation["matched"] = (
        validation["expected_trades"] == validation["replayed_trades"]
        and abs(validation["expected_gross_pnl_inr"] - validation["replayed_gross_pnl_inr"]) < 1e-6
    )
    if not validation["matched"]:
        raise RuntimeError(f"Replay baseline mismatch: {validation}")

    outcome_cache: dict[tuple[float, float, int], pd.DataFrame] = {
        (BASE_TARGET, BASE_STOP, BASE_HOLD): baseline_all_outcomes
    }

    def get_outcomes(target: float, stop: float, hold: int) -> pd.DataFrame:
        key = (float(target), float(stop), int(hold))
        if key not in outcome_cache:
            outcome_cache[key] = outcomes_for(predictions, path_cache[0], *key)
        return outcome_cache[key]

    parameter_values = {
        "target": [750.0, 1125.0, 1500.0, 1875.0, 2250.0],
        "stop": [-375.0, -562.5, -750.0, -937.5, -1125.0],
        "hold": [90, 120, 150],
        "probability": [0.70, 0.75, 0.80, 0.85],
        "event_score": [-0.02, -0.01, 0.0, 0.01, 0.02],
        "cap": [1, 2, 3, 4, 5],
    }
    defaults = {
        "target": BASE_TARGET,
        "stop": BASE_STOP,
        "hold": BASE_HOLD,
        "probability": BASE_PROB,
        "event_score": BASE_EVENT,
        "cap": BASE_CAP,
    }
    sensitivity_rows = []
    scenario_keys = set()
    for left, right in itertools.combinations(parameter_values, 2):
        for left_value in parameter_values[left]:
            for right_value in parameter_values[right]:
                config = defaults.copy()
                config[left] = left_value
                config[right] = right_value
                key = tuple(config[name] for name in defaults)
                scenario_keys.add(key)
                mask = route_mask(predictions, config["probability"], config["event_score"])
                outcomes = get_outcomes(config["target"], config["stop"], config["hold"])
                selected = outcomes.loc[outcomes["candidate_id"].isin(predictions.loc[mask, "candidate_id"])]
                ledger = apply_cap(selected, int(config["cap"]))
                result = metrics(ledger, eligible_dates)
                sensitivity_rows.append({
                    "parameter_x": left,
                    "value_x": left_value,
                    "parameter_y": right,
                    "value_y": right_value,
                    **config,
                    **result,
                })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUTPUT / "tier2_pairwise_sensitivity.csv", index=False)

    no_stop = get_outcomes(BASE_TARGET, -math.inf, BASE_HOLD)
    no_stop = no_stop.loc[no_stop["candidate_id"].isin(baseline_candidates["candidate_id"])]
    no_stop_metrics = metrics(apply_cap(no_stop, BASE_CAP), eligible_dates)
    no_target = get_outcomes(math.inf, BASE_STOP, BASE_HOLD)
    no_target = no_target.loc[no_target["candidate_id"].isin(baseline_candidates["candidate_id"])]
    no_target_metrics = metrics(apply_cap(no_target, BASE_CAP), eligible_dates)

    delayed_all = outcomes_for(predictions, path_cache[1], BASE_TARGET, BASE_STOP, BASE_HOLD)
    delayed = delayed_all.loc[delayed_all["candidate_id"].isin(baseline_candidates["candidate_id"])]
    delay_metrics = metrics(apply_cap(delayed, BASE_CAP), eligible_dates)

    lag_rows = []
    baseline_signal_ids = set(baseline_candidates["candidate_id"].astype(int))
    by_datetime = predictions.set_index("datetime")["candidate_id"].to_dict()
    for lag in [1, 5, 15]:
        shifted_ids = []
        for candidate in baseline_candidates.itertuples():
            shifted_time = candidate.datetime + pd.Timedelta(minutes=lag)
            shifted_id = by_datetime.get(shifted_time)
            if shifted_id is not None and shifted_time.normalize() == candidate.datetime.normalize():
                shifted_ids.append(int(shifted_id))
        shifted = baseline_all_outcomes.loc[baseline_all_outcomes["candidate_id"].isin(shifted_ids)]
        result = metrics(apply_cap(shifted, BASE_CAP), eligible_dates)
        lag_rows.append({"lag_minutes": lag, "qualified_entries": len(shifted_ids), **result})
    lag_frame = pd.DataFrame(lag_rows)
    lag_frame.to_csv(OUTPUT / "tier2_lag_test.csv", index=False)

    rng = np.random.default_rng(SEED + 300)
    candidate_minute = predictions["datetime"].dt.hour * 60 + predictions["datetime"].dt.minute
    qualified_minute_counts = (
        candidate_minute.loc[baseline_mask].value_counts().sort_index().to_dict()
    )
    minute_pools = {
        int(minute): group["candidate_id"].to_numpy(dtype=int)
        for minute, group in predictions.assign(minute=candidate_minute).groupby("minute")
    }
    placebo_totals = np.empty(1000)
    placebo_sharpes = np.empty(1000)
    for simulation in range(1000):
        chosen = []
        for minute, count in qualified_minute_counts.items():
            pool = minute_pools[int(minute)]
            chosen.extend(rng.choice(pool, size=int(count), replace=False).tolist())
        sample = baseline_all_outcomes.loc[baseline_all_outcomes["candidate_id"].isin(chosen)]
        result = metrics(apply_cap(sample, BASE_CAP), eligible_dates)
        placebo_totals[simulation] = result["net_pnl_inr"]
        placebo_sharpes[simulation] = result["sharpe"]

    day_counts = baseline_candidates.groupby("trade_date").size().to_dict()
    day_pools = {
        day: group["candidate_id"].to_numpy(dtype=int)
        for day, group in predictions.groupby("trade_date")
    }
    permuted_totals = np.empty(1000)
    for simulation in range(1000):
        chosen = []
        for day, count in day_counts.items():
            chosen.extend(rng.choice(day_pools[day], size=int(count), replace=False).tolist())
        sample = baseline_all_outcomes.loc[baseline_all_outcomes["candidate_id"].isin(chosen)]
        permuted_totals[simulation] = metrics(apply_cap(sample, BASE_CAP), eligible_dates)["net_pnl_inr"]

    placebo = {
        "random_entry_same_time_of_day": {
            "net_pnl_p05_p50_p95_inr": np.percentile(placebo_totals, [5, 50, 95]).tolist(),
            "sharpe_p05_p50_p95": np.percentile(placebo_sharpes, [5, 50, 95]).tolist(),
            "strategy_net_percentile": float(np.mean(placebo_totals <= baseline_metrics["net_pnl_inr"]) * 100),
            "strategy_beats_95th_percentile": bool(baseline_metrics["net_pnl_inr"] > np.percentile(placebo_totals, 95)),
        },
        "within_day_label_permutation": {
            "net_pnl_p05_p50_p95_inr": np.percentile(permuted_totals, [5, 50, 95]).tolist(),
            "strategy_net_percentile": float(np.mean(permuted_totals <= baseline_metrics["net_pnl_inr"]) * 100),
        },
    }

    highest = ex_ante_interval_selection(
        baseline_outcomes, BASE_CAP, -baseline_outcomes["probability"].to_numpy(dtype=float)
    )
    lowest = ex_ante_interval_selection(
        baseline_outcomes, BASE_CAP, baseline_outcomes["probability"].to_numpy(dtype=float)
    )
    random_cap_totals = np.empty(200)
    for simulation in range(200):
        sample = ex_ante_interval_selection(baseline_outcomes, BASE_CAP, rng.random(len(baseline_outcomes)))
        random_cap_totals[simulation] = sample["net_pnl_inr"].sum()
    cap_selection = {
        "method_note": "Ex-ante daily opportunity selection under interval-overlap cap; random/high/low rankings use future opportunity availability and are diagnostic, not directly executable live rules.",
        "first_come": baseline_metrics,
        "highest_probability": metrics(highest, eligible_dates),
        "lowest_probability": metrics(lowest, eligible_dates),
        "random_200_net_pnl_p10_p50_p90_inr": np.percentile(random_cap_totals, [10, 50, 90]).tolist(),
        "first_come_percentile_within_random": float(np.mean(random_cap_totals <= baseline_metrics["net_pnl_inr"]) * 100),
        "first_come_within_central_80pct": bool(
            np.percentile(random_cap_totals, 10)
            <= baseline_metrics["net_pnl_inr"]
            <= np.percentile(random_cap_totals, 90)
        ),
    }

    baseline_ledger["month"] = pd.to_datetime(baseline_ledger["trade_date"]).dt.to_period("M").astype(str)
    monthly = baseline_ledger.groupby("month").agg(
        trades=("net_pnl_inr", "size"),
        net_pnl_inr=("net_pnl_inr", "sum"),
        win_rate_pct=("net_pnl_inr", lambda values: float((values > 0).mean() * 100)),
    ).reset_index()
    monthly.to_csv(OUTPUT / "tier2_monthly_results.csv", index=False)

    lomo_rows = []
    for month in sorted(baseline_ledger["month"].unique()):
        ledger = baseline_ledger.loc[baseline_ledger["month"].ne(month)]
        dates = eligible_dates[eligible_dates.to_period("M").astype(str) != month]
        lomo_rows.append({"month_removed": month, **metrics(ledger, dates)})
    pd.DataFrame(lomo_rows).to_csv(OUTPUT / "tier2_leave_one_month_out.csv", index=False)

    route_results = {
        route: metrics(group, eligible_dates)
        for route, group in baseline_ledger.groupby("route", sort=True)
    }
    expiry_day = pd.to_datetime(baseline_ledger["trade_date"]).dt.date == pd.to_datetime(baseline_ledger["expiry"]).dt.date
    expiry_split = {
        "expiry_day": metrics(baseline_ledger.loc[expiry_day], eligible_dates),
        "non_expiry_day": metrics(baseline_ledger.loc[~expiry_day], eligible_dates),
    }
    regime_dates = pd.to_datetime(baseline_ledger["trade_date"])
    regulatory_splits = {
        "pre_oct_2024_stt_change": metrics(baseline_ledger.loc[regime_dates < "2024-10-01"], eligible_dates),
        "post_oct_2024_stt_change": metrics(baseline_ledger.loc[regime_dates >= "2024-10-01"], eligible_dates),
        "pre_nov_20_2024_structure_change": metrics(baseline_ledger.loc[regime_dates < "2024-11-20"], eligible_dates),
        "post_nov_20_2024_structure_change": metrics(baseline_ledger.loc[regime_dates >= "2024-11-20"], eligible_dates),
    }

    spot = pd.read_csv(SPOT, usecols=["datetime", "spot_price"], parse_dates=["datetime"])
    spot = spot.loc[spot["datetime"].dt.normalize().isin(eligible_dates)].copy()
    spot["trade_date"] = spot["datetime"].dt.normalize()
    spot_daily = spot.groupby("trade_date")["spot_price"].agg(open="first", close="last")
    spot_daily["previous_close"] = spot_daily["close"].shift(1)
    spot_daily["overnight_gap_pct"] = (spot_daily["open"] / spot_daily["previous_close"] - 1) * 100
    spot_daily["intraday_move_pct"] = (spot_daily["close"] / spot_daily["open"] - 1) * 100
    daily_strategy = baseline_ledger.groupby("trade_date")["net_pnl_inr"].sum().reindex(eligible_dates, fill_value=0.0)
    shock_table = spot_daily.join(daily_strategy.rename("strategy_net_pnl_inr"), how="left")
    shock_table.to_csv(OUTPUT / "tier2_daily_shock_table.csv")
    gap_days = shock_table.loc[shock_table["overnight_gap_pct"].abs() > 1.5].reset_index()
    worst_day = daily_strategy.idxmin()
    rolling_three = daily_strategy.rolling(3).sum()
    worst_three_end = rolling_three.idxmin()
    stop_ledger = baseline_ledger.loc[baseline_ledger["exit_reason"].eq("stop_loss")]
    simultaneous_stops = stop_ledger.groupby("exit_datetime").size()
    historical_shocks = {
        "june_2024_election_gap": "outside evaluation window",
        "october_2024": metrics(baseline_ledger.loc[baseline_ledger["month"].eq("2024-10")], eligible_dates),
        "april_2025": metrics(baseline_ledger.loc[baseline_ledger["month"].eq("2025-04")], eligible_dates),
        "gap_days_over_1_5pct": gap_days[["trade_date", "overnight_gap_pct", "strategy_net_pnl_inr"]].astype({"trade_date": str}).to_dict(orient="records"),
        "worst_day": {"date": str(worst_day.date()), "net_pnl_inr": float(daily_strategy.loc[worst_day])},
        "worst_three_trading_day_window": {
            "end_date": str(worst_three_end.date()),
            "net_pnl_inr": float(rolling_three.loc[worst_three_end]),
        },
        "maximum_positions_stopped_same_minute": int(simultaneous_stops.max()) if len(simultaneous_stops) else 0,
    }

    average_cost = float(baseline_ledger["explicit_costs_inr"].mean())
    synthetic_rows = []
    for gap_pct in [2, 3, 5, -2, -3, -5]:
        gross_loss = -3 * 2 * abs(BASE_STOP)
        net_loss = gross_loss - 3 * average_cost
        synthetic_rows.append({
            "gap_pct": gap_pct,
            "assumed_open_positions": 3,
            "gross_loss_inr": gross_loss,
            "estimated_net_loss_inr": net_loss,
            "multiple_of_historical_max_drawdown": abs(net_loss) / baseline_metrics["max_drawdown_inr"],
            "pass_under_2x_drawdown_rule": abs(net_loss) < 2 * baseline_metrics["max_drawdown_inr"],
        })
    pd.DataFrame(synthetic_rows).to_csv(OUTPUT / "tier2_synthetic_gap_shocks.csv", index=False)

    target_stop_25 = sensitivity.loc[
        sensitivity["parameter_x"].eq("target")
        & sensitivity["parameter_y"].eq("stop")
        & sensitivity["value_x"].isin([1125.0, 1500.0, 1875.0])
        & sensitivity["value_y"].isin([-562.5, -750.0, -937.5])
    ]
    minimum_25_sharpe = float(target_stop_25["sharpe"].min())

    tested_sharpes = sensitivity.drop_duplicates(
        ["target", "stop", "hold", "probability", "event_score", "cap"]
    )["sharpe"].dropna().to_numpy(dtype=float) / math.sqrt(ANNUALIZATION)
    n_trials = int(len(tested_sharpes) + 20 + 4 + 2 + 3)
    daily_baseline = baseline_ledger.groupby("trade_date")["net_pnl_inr"].sum().reindex(eligible_dates, fill_value=0.0).to_numpy(dtype=float)
    observed_sr = sharpe(daily_baseline) / math.sqrt(ANNUALIZATION)
    trial_sr_std = float(np.std(tested_sharpes, ddof=1))
    euler_gamma = 0.5772156649015329
    expected_max_sr = trial_sr_std * (
        (1 - euler_gamma) * norm.ppf(1 - 1 / n_trials)
        + euler_gamma * norm.ppf(1 - 1 / (n_trials * math.e))
    )
    daily_skew = float(skew(daily_baseline, bias=False))
    daily_kurtosis = float(kurtosis(daily_baseline, fisher=False, bias=False))
    denominator = math.sqrt(
        max(1e-12, 1 - daily_skew * observed_sr + ((daily_kurtosis - 1) / 4) * observed_sr**2)
    )
    dsr_probability = float(norm.cdf((observed_sr - expected_max_sr) * math.sqrt(len(daily_baseline) - 1) / denominator))
    dsr = {
        "configuration_count_lower_bound": n_trials,
        "observed_annualized_sharpe": baseline_metrics["sharpe"],
        "expected_max_annualized_sharpe_under_multiple_testing": expected_max_sr * math.sqrt(ANNUALIZATION),
        "deflated_sharpe_probability": dsr_probability,
        "pass_probability_above_0_95": dsr_probability > 0.95,
        "warning": "N counts configurations in this stress run only; prior informal experiments are not fully enumerable, so this probability is optimistic.",
    }

    review_path = OUTPUT / "single_page_review.csv"
    review = pd.read_csv(review_path)
    tier2_rows = [
        {
            "test": "5-second entry delay",
            "baseline": f"Net ₹{baseline_metrics['net_pnl_inr']:.2f}",
            "stressed_value": f"UNAVAILABLE; nearest 60-second proxy net ₹{delay_metrics['net_pnl_inr']:.2f}",
            "criterion": "Net > 60% of baseline",
            "result": "NOT TESTABLE",
        },
        {
            "test": "Random-entry placebo",
            "baseline": f"Net ₹{baseline_metrics['net_pnl_inr']:.2f}",
            "stressed_value": f"95th percentile ₹{np.percentile(placebo_totals, 95):.2f}",
            "criterion": "Strategy beats placebo 95th percentile",
            "result": "PASS" if placebo["random_entry_same_time_of_day"]["strategy_beats_95th_percentile"] else "FAIL",
        },
        {
            "test": "Parameter ±25% target/stop",
            "baseline": f"Sharpe {baseline_metrics['sharpe']:.3f}",
            "stressed_value": f"Minimum 3x3-grid Sharpe {minimum_25_sharpe:.3f}",
            "criterion": "No cell Sharpe < 2",
            "result": "PASS" if minimum_25_sharpe >= 2 else "FAIL",
        },
        {
            "test": "Cap randomization",
            "baseline": f"First-come net ₹{baseline_metrics['net_pnl_inr']:.2f}",
            "stressed_value": f"Random P10/P90 ₹{np.percentile(random_cap_totals, 10):.2f}/₹{np.percentile(random_cap_totals, 90):.2f}",
            "criterion": "First-come within central 80%",
            "result": "PASS" if cap_selection["first_come_within_central_80pct"] else "FAIL",
        },
        {
            "test": "3% gap shock, all three stopped",
            "baseline": f"Historical drawdown ₹{baseline_metrics['max_drawdown_inr']:.2f}",
            "stressed_value": f"Estimated net loss ₹{abs(synthetic_rows[1]['estimated_net_loss_inr']):.2f}",
            "criterion": "Loss < 2x historical drawdown",
            "result": "PASS" if synthetic_rows[1]["pass_under_2x_drawdown_rule"] else "FAIL",
        },
        {
            "test": "Deflated Sharpe",
            "baseline": f"Sharpe {baseline_metrics['sharpe']:.3f}",
            "stressed_value": f"Probability {dsr_probability:.4f}; N lower bound {n_trials}",
            "criterion": "Probability > 0.95",
            "result": "PASS" if dsr_probability > 0.95 else "FAIL",
        },
        {
            "test": "Tick-level exits / bid-ask fills",
            "baseline": "Minute LTP/close data",
            "stressed_value": "Tick/1-second bid-ask data unavailable",
            "criterion": "Required for valid test",
            "result": "NOT TESTABLE",
        },
    ]
    review = pd.concat([review, pd.DataFrame(tier2_rows)], ignore_index=True)
    review.to_csv(review_path, index=False)

    unavailable = {
        "entry_delay_1_2_5_10_30_seconds": "Source option paths are one-minute observations with no bid/ask quotes. A 60-second delayed-entry proxy was run, but it cannot grade the 5-second criterion.",
        "tick_level_exit_and_intrabar_touch": "No tick or one-second option path data is present.",
        "stop_slippage_k_times_spread": "No historical bid/ask spread field is present.",
        "vix_terciles": "No India VIX series was found in the supplied local datasets.",
        "event_calendar_split": "No dated event calendar was supplied; gap days are reported separately and were not relabeled as event days.",
        "vol_shift_iv_plus_5_10_and_crush": "A full option repricing surface/Greeks path for synthetic IV shocks was not supplied; no substitute metric was invented.",
        "liquidity_spreads_x3_x5": "Historical spreads are unavailable; Tier 1 contains flat-slippage and 70%-fill proxies instead.",
    }

    hashes_after = live_hashes()
    integrity = {
        "unchanged": hashes_before == hashes_after,
        "files_checked": len(hashes_before),
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }
    (OUTPUT / "live_code_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    payload = {
        "tier": 2,
        "replay_validation": validation,
        "baseline": baseline_metrics,
        "one_minute_entry_delay_proxy": delay_metrics,
        "lag_tests": lag_rows,
        "placebo_tests": placebo,
        "cap_selection": cap_selection,
        "stop_removed": no_stop_metrics,
        "target_removed": no_target_metrics,
        "minimum_sharpe_target_stop_plus_minus_25pct_grid": minimum_25_sharpe,
        "route_results": route_results,
        "expiry_split": expiry_split,
        "regulatory_splits": regulatory_splits,
        "historical_shocks": historical_shocks,
        "synthetic_gap_shocks": synthetic_rows,
        "deflated_sharpe": dsr,
        "unavailable_tests": unavailable,
        "live_code_integrity": {"unchanged": integrity["unchanged"], "files_checked": integrity["files_checked"]},
    }
    (OUTPUT / "tier2_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
