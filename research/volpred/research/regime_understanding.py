"""Describe ATM Greek regimes and their P&L effects on the training sample only."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import ndtr

from atm_iv_rv_analysis import (
    prepare_spot,
    remaining_trading_minutes,
    rolling_variances,
    seasonality_profile,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm.csv"
SPOT_SOURCE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
OUT = ROOT / "outputs" / "regime_understanding"
MODEL_DIR = ROOT / "model" / "regimeUnderstanding"
TIME_ANALYSIS_DIR = MODEL_DIR / "greek_time_evolution"
SCENARIO_ANALYSIS_DIR = MODEL_DIR / "iv_rv_scenarios"
TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2022-12-31 23:59:59")
HORIZONS = (15, 30, 60, 120, 180)
GREEKS = ("gamma", "vega", "theta", "vanna", "volga", "charm")
SESSION_OPEN_MINUTE = 9 * 60 + 15
SESSION_CLOSE_MINUTE = 15 * 60 + 30
IV_RV_EQUAL_TOLERANCE = 0.005  # one-half volatility point in decimal-vol units
EPS = 1e-12


def _pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def black_greeks(
    forward: np.ndarray,
    strike: np.ndarray,
    maturity_years: np.ndarray,
    volatility: np.ndarray,
    spot: np.ndarray,
    is_call: bool,
) -> dict[str, np.ndarray]:
    """Return consistently scaled Black Greeks for one option leg.

    Vega is points per 1.00 absolute volatility, theta and charm are per
    calendar day, and vanna/volga use 1.00 absolute-volatility units.
    """
    f, k, t, v, s = np.broadcast_arrays(
        np.asarray(forward, float), np.asarray(strike, float),
        np.asarray(maturity_years, float), np.asarray(volatility, float),
        np.asarray(spot, float),
    )
    valid = (
        np.isfinite(f) & np.isfinite(k) & np.isfinite(t)
        & np.isfinite(v) & np.isfinite(s)
        & (f > 0) & (k > 0) & (t > 0) & (v > 0) & (s > 0)
    )
    t = np.maximum(t, EPS)
    v = np.maximum(v, EPS)
    root_t = np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (np.log(f / k) + 0.5 * v * v * t) / (v * root_t)
        d2 = d1 - v * root_t
        density = _pdf(d1)
        forward_to_spot = f / s
        delta_forward = ndtr(d1) if is_call else ndtr(d1) - 1.0
        delta = delta_forward * forward_to_spot
        gamma = density / (f * v * root_t) * forward_to_spot**2
        vega = f * density * root_t
        theta_day = (-f * density * v / (2.0 * root_t)) / 365.0
        vanna = -density * d2 / v * forward_to_spot
        volga = vega * d1 * d2 / v

        one_day = 1.0 / 365.0
        t_after_day = np.maximum(t - one_day, EPS)
        root_after_day = np.sqrt(t_after_day)
        d1_after_day = (
            np.log(f / k) + 0.5 * v * v * t_after_day
        ) / (v * root_after_day)
        delta_forward_after_day = (
            ndtr(d1_after_day) if is_call else ndtr(d1_after_day) - 1.0
        )
        charm_day = delta_forward_after_day * forward_to_spot - delta

    result = {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_day,
        "vanna": vanna,
        "volga": volga,
        "charm": charm_day,
    }
    return {name: np.where(valid, values, np.nan) for name, values in result.items()}


def load_training_atm() -> pd.DataFrame:
    source = str(SOURCE).replace("'", "''")
    query = f"""
        SELECT
            source_index::BIGINT AS source_index,
            datetime::TIMESTAMP AS datetime,
            expiry::DATE AS expiry,
            strike::DOUBLE AS strike,
            days_to_expiry::DOUBLE AS days_to_expiry,
            spot_price::DOUBLE AS spot_price,
            synthetic_future::DOUBLE AS synthetic_future,
            call_price::DOUBLE AS call_price,
            put_price::DOUBLE AS put_price,
            iv_call::DOUBLE AS iv_call,
            iv_put::DOUBLE AS iv_put
        FROM read_csv_auto('{source}', header=true)
        WHERE datetime::TIMESTAMP >= TIMESTAMP '2019-01-01'
          AND datetime::TIMESTAMP < TIMESTAMP '2023-01-01'
          AND dte_flag = 'M1'
        ORDER BY datetime::TIMESTAMP
    """
    frame = duckdb.connect().execute(query).fetchdf()
    frame = frame.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    frame["date"] = frame["datetime"].dt.normalize()
    minute_of_day = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
    frame = frame[
        minute_of_day.between(SESSION_OPEN_MINUTE, SESSION_CLOSE_MINUTE)
    ].copy().reset_index(drop=True)
    frame["premium"] = frame["call_price"] + frame["put_price"]
    frame["atm_iv"] = frame[["iv_call", "iv_put"]].mean(axis=1)
    valid = (
        frame["datetime"].between(TRAIN_START, TRAIN_END)
        & frame["spot_price"].gt(0) & frame["synthetic_future"].gt(0)
        & frame["strike"].gt(0) & frame["days_to_expiry"].gt(0)
        & frame["call_price"].gt(0) & frame["put_price"].gt(0)
        & frame["iv_call"].between(0.05, 1.50)
        & frame["iv_put"].between(0.05, 1.50)
    )
    frame["quality_ok"] = valid
    return frame


def add_consistent_greeks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    maturity = out["days_to_expiry"].to_numpy(float) / 365.0
    for side, is_call in (("call", True), ("put", False)):
        values = black_greeks(
            out["synthetic_future"].to_numpy(float),
            out["strike"].to_numpy(float),
            maturity,
            out[f"iv_{side}"].to_numpy(float),
            out["spot_price"].to_numpy(float),
            is_call,
        )
        for greek, data in values.items():
            out[f"{side}_{greek}"] = data
    for greek in ("delta", *GREEKS):
        out[greek] = out[f"call_{greek}"] + out[f"put_{greek}"]
    return out


def add_forward_realized_volatility(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach trading-clock IV and forward RV without using option-chain rows."""
    spot = pd.read_csv(SPOT_SOURCE, usecols=["datetime", "spot_price"], parse_dates=["datetime"])
    spot = prepare_spot(spot)
    seasonal = seasonality_profile(spot)
    variances = rolling_variances(spot, seasonal, HORIZONS)
    for name, values in variances.items():
        spot[name] = values

    out = frame.copy()
    position = pd.Index(spot["datetime"]).get_indexer(out["datetime"])
    matched = position >= 0
    for horizon in HORIZONS:
        values = np.full(len(out), np.nan)
        values[matched] = spot[f"future_var_{horizon}"].to_numpy()[position[matched]]
        out[f"future_rv_{horizon}"] = np.sqrt(values)

    remaining = remaining_trading_minutes(
        pd.DatetimeIndex(out["datetime"]),
        pd.DatetimeIndex(out["expiry"]),
        spot["date"],
    )
    out["remaining_trading_minutes"] = remaining
    out["iv_trade_vol"] = out["atm_iv"] * np.sqrt(
        (out["days_to_expiry"] / 365.0)
        * (252.0 * 375.0 / np.maximum(remaining, 1))
    )
    return out


def learn_thresholds(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    good = frame[frame["quality_ok"]]
    thresholds: dict[str, dict[str, float]] = {}
    for greek in GREEKS:
        low, high = good[greek].quantile([1.0 / 3.0, 2.0 / 3.0])
        thresholds[greek] = {"low_max": float(low), "high_min": float(high)}
    return thresholds


def assign_regimes(
    frame: pd.DataFrame, thresholds: dict[str, dict[str, float]]
) -> pd.DataFrame:
    out = frame.copy()
    for greek, limits in thresholds.items():
        values = out[greek]
        out[f"{greek}_regime"] = np.select(
            [values.le(limits["low_max"]), values.ge(limits["high_min"])],
            ["low", "high"],
            default="normal",
        )
        out.loc[~out["quality_ok"] | values.isna(), f"{greek}_regime"] = pd.NA
    return out


def regime_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    """Find contiguous intraday runs; a gap over two minutes starts a new run."""
    episode_frames: list[pd.DataFrame] = []
    for greek in GREEKS:
        state_col = f"{greek}_regime"
        x = frame.loc[frame[state_col].notna(), [
            "datetime", "date", "expiry", greek, state_col,
        ]].copy()
        gap = x["datetime"].diff().dt.total_seconds().div(60.0)
        new_episode = (
            x[state_col].ne(x[state_col].shift())
            | x["date"].ne(x["date"].shift())
            | x["expiry"].ne(x["expiry"].shift())
            | gap.gt(2.0)
        )
        x["episode_id"] = new_episode.cumsum()
        episodes = x.groupby("episode_id", as_index=False).agg(
            regime=(state_col, "first"),
            start=("datetime", "min"),
            end=("datetime", "max"),
            samples=(greek, "size"),
            start_value=(greek, "first"),
            end_value=(greek, "last"),
            mean_value=(greek, "mean"),
            min_value=(greek, "min"),
            max_value=(greek, "max"),
        )
        episodes["duration_minutes"] = (
            (episodes["end"] - episodes["start"]).dt.total_seconds() / 60.0 + 1.0
        )
        episodes.insert(0, "greek", greek)
        episode_frames.append(episodes.drop(columns="episode_id"))
    return pd.concat(episode_frames, ignore_index=True)


def duration_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    return episodes.groupby(["greek", "regime"], as_index=False).agg(
        episodes=("duration_minutes", "size"),
        total_minutes=("duration_minutes", "sum"),
        mean_duration_minutes=("duration_minutes", "mean"),
        median_duration_minutes=("duration_minutes", "median"),
        p90_duration_minutes=("duration_minutes", lambda s: s.quantile(0.90)),
        max_duration_minutes=("duration_minutes", "max"),
    )


def build_windows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    future_cols = [
        "datetime", "date", "expiry", "strike", "spot_price", "premium",
        "call_price", "put_price", "iv_call", "iv_put", "quality_ok",
        *GREEKS,
    ]
    future = frame[future_cols].copy().rename(
        columns={column: f"exit_{column}" for column in future_cols if column != "datetime"}
    )
    windows: list[pd.DataFrame] = []
    audit: list[dict] = []
    for horizon in HORIZONS:
        entry = frame.copy()
        entry["target_datetime"] = entry["datetime"] + pd.Timedelta(minutes=horizon)
        target_minute = entry["target_datetime"].dt.hour * 60 + entry["target_datetime"].dt.minute
        completes_intraday = (
            entry["target_datetime"].dt.normalize().eq(entry["date"])
            & target_minute.between(SESSION_OPEN_MINUTE, SESSION_CLOSE_MINUTE)
        )
        entry = entry[completes_intraday].copy()
        joined = entry.merge(
            future,
            left_on="target_datetime",
            right_on="datetime",
            how="left",
            suffixes=("", "_matched"),
            validate="many_to_one",
        ).drop(columns=["datetime_matched"], errors="ignore")
        joined["horizon_min"] = horizon
        joined["future_rv"] = joined[f"future_rv_{horizon}"]
        joined["iv_minus_rv"] = joined["iv_trade_vol"] - joined["future_rv"]
        joined["iv_rv_scene"] = np.select(
            [
                joined["iv_minus_rv"].gt(IV_RV_EQUAL_TOLERANCE),
                joined["iv_minus_rv"].lt(-IV_RV_EQUAL_TOLERANCE),
            ],
            ["IV > RV", "IV < RV"],
            default="IV = RV",
        )
        joined.loc[
            joined[["iv_trade_vol", "future_rv"]].isna().any(axis=1), "iv_rv_scene"
        ] = pd.NA
        joined["same_session"] = joined["date"].eq(joined["exit_date"])
        joined["same_expiry"] = joined["expiry"].eq(joined["exit_expiry"])
        joined["same_strike"] = joined["strike"].eq(joined["exit_strike"])
        joined["future_atm_available"] = joined["exit_spot_price"].notna() & joined["same_session"] & joined["same_expiry"]
        joined["fixed_contract_available"] = joined["future_atm_available"] & joined["same_strike"]
        joined["spot_change"] = joined["exit_spot_price"] - joined["spot_price"]
        joined["actual_long_straddle_pnl"] = np.where(
            joined["fixed_contract_available"], joined["exit_premium"] - joined["premium"], np.nan
        )
        elapsed_days = horizon / 1440.0
        joined["gamma_pnl"] = 0.5 * joined["gamma"] * joined["spot_change"] ** 2
        joined["vega_pnl"] = (
            joined["call_vega"] * (joined["exit_iv_call"] - joined["iv_call"])
            + joined["put_vega"] * (joined["exit_iv_put"] - joined["iv_put"])
        )
        joined["theta_pnl"] = joined["theta"] * elapsed_days
        joined["vanna_pnl"] = joined["spot_change"] * (
            joined["call_vanna"] * (joined["exit_iv_call"] - joined["iv_call"])
            + joined["put_vanna"] * (joined["exit_iv_put"] - joined["iv_put"])
        )
        joined["volga_pnl"] = 0.5 * (
            joined["call_volga"] * (joined["exit_iv_call"] - joined["iv_call"]) ** 2
            + joined["put_volga"] * (joined["exit_iv_put"] - joined["iv_put"]) ** 2
        )
        joined["charm_pnl"] = joined["charm"] * joined["spot_change"] * elapsed_days
        joined["requested_greeks_pnl"] = joined[[f"{g}_pnl" for g in GREEKS]].sum(
            axis=1, min_count=len(GREEKS)
        )
        joined["unexplained_pnl"] = (
            joined["actual_long_straddle_pnl"] - joined["requested_greeks_pnl"]
        )
        for greek in GREEKS:
            joined[f"{greek}_change"] = joined[f"exit_{greek}"] - joined[greek]
        usable = (
            joined["quality_ok"] & joined["exit_quality_ok"].fillna(False)
            & joined["future_atm_available"]
        )
        joined["analysis_ok"] = usable
        audit.append({
            "horizon_min": horizon,
            "latest_allowed_entry": str(
                (pd.Timestamp("2000-01-01 15:30") - pd.Timedelta(minutes=horizon)).time()
            ),
            "entry_rows": len(joined),
            "quality_entry_rows": int(joined["quality_ok"].sum()),
            "future_atm_rows": int((joined["quality_ok"] & joined["future_atm_available"]).sum()),
            "fixed_contract_pnl_rows": int((joined["quality_ok"] & joined["exit_quality_ok"].fillna(False) & joined["fixed_contract_available"]).sum()),
        })
        keep = [
            "datetime", "target_datetime", "horizon_min", "expiry", "strike",
            "spot_price", "exit_spot_price", "spot_change", "premium", "exit_premium",
            "iv_trade_vol", "future_rv", "iv_minus_rv", "iv_rv_scene",
            "same_strike", "future_atm_available", "fixed_contract_available", "analysis_ok",
            "actual_long_straddle_pnl", *[f"{g}_pnl" for g in GREEKS],
            "requested_greeks_pnl", "unexplained_pnl",
        ]
        for greek in GREEKS:
            keep.extend([greek, f"exit_{greek}", f"{greek}_change", f"{greek}_regime"])
        windows.append(joined[keep])
    return pd.concat(windows, ignore_index=True), pd.DataFrame(audit)


def summarize_changes(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in HORIZONS:
        sample = windows[(windows["horizon_min"] == horizon) & windows["analysis_ok"]]
        for greek in GREEKS:
            change = sample[f"{greek}_change"].dropna()
            rows.append({
                "horizon_min": horizon,
                "greek": greek,
                "n": len(change),
                "mean_entry_value": sample.loc[change.index, greek].mean(),
                "mean_exit_value": sample.loc[change.index, f"exit_{greek}"].mean(),
                "mean_change": change.mean(),
                "median_change": change.median(),
                "median_absolute_change": change.abs().median(),
                "p10_change": change.quantile(0.10),
                "p90_change": change.quantile(0.90),
            })
    return pd.DataFrame(rows)


def summarize_pnl(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in HORIZONS:
        sample = windows[
            (windows["horizon_min"] == horizon)
            & windows["analysis_ok"]
            & windows["fixed_contract_available"]
            & windows["actual_long_straddle_pnl"].notna()
        ]
        actual = sample["actual_long_straddle_pnl"]
        for greek in GREEKS:
            term = sample[f"{greek}_pnl"]
            rows.append({
                "horizon_min": horizon,
                "greek": greek,
                "n": int((actual.notna() & term.notna()).sum()),
                "mean_attributed_pnl_points": term.mean(),
                "median_attributed_pnl_points": term.median(),
                "mean_absolute_attributed_pnl_points": term.abs().mean(),
                "correlation_with_actual_pnl": term.corr(actual),
                "mean_actual_long_straddle_pnl_points": actual.mean(),
            })
    return pd.DataFrame(rows)


def summarize_scenario_changes(windows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon in HORIZONS:
        base = windows[
            (windows["horizon_min"] == horizon)
            & windows["analysis_ok"]
            & windows["iv_rv_scene"].notna()
        ]
        for scene in ("IV < RV", "IV = RV", "IV > RV"):
            sample = base[base["iv_rv_scene"].eq(scene)]
            for greek in GREEKS:
                change = sample[f"{greek}_change"].dropna()
                rows.append({
                    "scene": scene,
                    "horizon_min": horizon,
                    "greek": greek,
                    "n": len(change),
                    "mean_iv_minus_rv": sample.loc[change.index, "iv_minus_rv"].mean(),
                    "mean_entry_value": sample.loc[change.index, greek].mean(),
                    "mean_exit_value": sample.loc[change.index, f"exit_{greek}"].mean(),
                    "mean_change": change.mean(),
                    "median_change": change.median(),
                    "median_absolute_change": change.abs().median(),
                    "p10_change": change.quantile(0.10),
                    "p90_change": change.quantile(0.90),
                })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{x:.6f}" if isinstance(x, float) else str(x) for x in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    frame: pd.DataFrame,
    durations: pd.DataFrame,
    changes: pd.DataFrame,
    pnl: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    duration_view = durations[[
        "greek", "regime", "episodes", "total_minutes",
        "median_duration_minutes", "p90_duration_minutes", "max_duration_minutes",
    ]].round(2)
    change_view = changes[[
        "horizon_min", "greek", "n", "mean_change", "median_absolute_change",
        "p10_change", "p90_change",
    ]].round(6)
    pnl_view = pnl[[
        "horizon_min", "greek", "n", "mean_attributed_pnl_points",
        "mean_absolute_attributed_pnl_points", "correlation_with_actual_pnl",
        "mean_actual_long_straddle_pnl_points",
    ]].round(4)
    report = f"""# ATM Greek regime analysis — training period

## Scope

- Source: `data/raw/icharts_option_banknifty_monthly_atm.csv` only.
- Training observations: {frame['datetime'].min()} through {frame['datetime'].max()}.
- Horizons: 15, 30, 60, 120, and 180 minutes.
- Every entry and exit is restricted to the same 09:15–15:30 trading session;
  a window is omitted if its horizon would finish after 15:30.
- Reference position: one long ATM call plus one long ATM put (an ATM straddle).
- Regimes: training-sample terciles for each combined straddle Greek. `low` is at
  or below the 33.33% threshold, `high` is at or above the 66.67% threshold,
  and `normal` is between them.
- A regime episode ends when the state changes, the session or expiry changes,
  or observations are separated by more than two minutes.

## Regime duration

{markdown_table(duration_view)}

## How Greeks change by horizon

The exit Greek is the Greek of the ATM straddle at the future timestamp. This
describes the ATM regime even when the ATM strike changes.

{markdown_table(change_view)}

## Effect on option P&L

P&L is evaluated only when the original entry strike is also the ATM row at the
future timestamp. That keeps the contract fixed, but creates a selection bias
toward windows in which spot did not move far enough to change the ATM strike.
The table is therefore explanatory training evidence, not a trading backtest.

{markdown_table(pnl_view)}

## Coverage audit

{markdown_table(audit)}

## Units and limitations

- Gamma contribution: `0.5 * gamma * spot_change^2`.
- Vega contribution uses change in decimal IV; vega is per 1.00 absolute IV.
- Theta and charm use calendar-day units and elapsed minutes divided by 1,440.
- Vanna and volga use decimal-IV changes.
- The requested six terms are a local Taylor attribution. Delta, higher-order
  effects, quote noise, discrete strike changes, and model error can remain in
  unexplained P&L.
- Thresholds were learned only from 2019–2022. The 2023 validation and
  2024–2025 test periods are not read by this analysis.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def write_model_analysis(changes: pd.DataFrame, durations: pd.DataFrame) -> None:
    TIME_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    changes.to_csv(TIME_ANALYSIS_DIR / "greek_change_by_horizon.csv", index=False)
    durations.to_csv(TIME_ANALYSIS_DIR / "regime_duration_summary.csv", index=False)
    pivot = changes.pivot(index="greek", columns="horizon_min", values="mean_change").reindex(GREEKS)
    pivot = pivot.reset_index().round(6)
    text = f"""# Greek change over time

Training-only ATM long-straddle analysis for the 09:15–15:30 session. Values
are mean changes from entry to the stated horizon.

{markdown_table(pivot)}

Regime-duration details are in `regime_duration_summary.csv`. The underlying
definition and unit conventions are documented in the parent README.
"""
    (TIME_ANALYSIS_DIR / "ANALYSIS.md").write_text(text, encoding="utf-8")


def write_scenario_analysis(summary: pd.DataFrame) -> None:
    SCENARIO_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SCENARIO_ANALYSIS_DIR / "greek_change_by_scenario.csv", index=False)
    sections = []
    for scene in ("IV < RV", "IV = RV", "IV > RV"):
        view = summary[summary["scene"].eq(scene)].pivot(
            index="greek", columns="horizon_min", values="mean_change"
        ).reindex(GREEKS).reset_index().round(6)
        counts = summary[
            summary["scene"].eq(scene) & summary["greek"].eq("gamma")
        ][["horizon_min", "n"]]
        sections.append(
            f"## {scene}\n\nMean Greek change:\n\n{markdown_table(view)}"
            f"\n\nAvailable observations:\n\n{markdown_table(counts)}"
        )
    report = f"""# Greek changes conditional on IV versus RV

This uses the 2019–2022 training sample and intraday windows that finish by
15:30. Greek values come from the ATM-only option CSV. Forward RV is computed
from the underlying one-minute spot path over the matching 15, 30, 60, 120, or
180-minute horizon.

Because exact equality is not meaningful for continuous estimates, `IV = RV`
means absolute IV minus RV is at most {100 * IV_RV_EQUAL_TOLERANCE:.1f}
volatility points. Values above that band are `IV > RV`; values below it are
`IV < RV`. IV is converted to the same trading-time annualized clock as RV.

{"\n\n".join(sections)}
"""
    (SCENARIO_ANALYSIS_DIR / "ANALYSIS.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    frame = add_forward_realized_volatility(add_consistent_greeks(load_training_atm()))
    thresholds = learn_thresholds(frame)
    frame = assign_regimes(frame, thresholds)
    episodes = regime_episodes(frame)
    durations = duration_summary(episodes)
    windows, audit = build_windows(frame)
    changes = summarize_changes(windows)
    pnl = summarize_pnl(windows)
    scenarios = summarize_scenario_changes(windows)

    artifact = {
        "model_family": "regimeUnderstanding",
        "model_type": "training_quantile_regime_classifier",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "train_start": str(TRAIN_START.date()),
        "train_end": str(TRAIN_END.date()),
        "observed_train_start": str(frame["datetime"].min()),
        "observed_train_end": str(frame["datetime"].max()),
        "session_open": "09:15",
        "session_close": "15:30",
        "horizons_must_finish_by_session_close": True,
        "iv_rv_equal_tolerance_vol_points": 100 * IV_RV_EQUAL_TOLERANCE,
        "horizons_minutes": list(HORIZONS),
        "greeks": list(GREEKS),
        "regime_definition": "low <= q33; normal q33 < x < q67; high >= q67",
        "thresholds": thresholds,
    }
    (MODEL_DIR / "artifacts" / "regime_thresholds.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    episodes.to_csv(OUT / "greek_regime_episodes.csv", index=False)
    durations.to_csv(OUT / "regime_duration_summary.csv", index=False)
    changes.to_csv(OUT / "greek_change_by_horizon.csv", index=False)
    pnl.to_csv(OUT / "pnl_attribution_by_horizon.csv", index=False)
    scenarios.to_csv(OUT / "greek_change_by_iv_rv_scenario.csv", index=False)
    audit.to_csv(OUT / "data_audit.csv", index=False)
    connection = duckdb.connect()
    connection.register("windows_df", windows)
    target = str(OUT / "greek_windows.parquet").replace("'", "''")
    connection.execute(f"COPY windows_df TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    connection.close()
    write_report(frame, durations, changes, pnl, audit)
    write_model_analysis(changes, durations)
    write_scenario_analysis(scenarios)
    print(durations.to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
