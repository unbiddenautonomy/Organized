"""Clock-consistent, joint-scenario backtest for the 120-minute short straddle.

The rule and risk limits in this file are fixed before inspecting its results.
It evaluates the same 10:15/12:15 non-overlapping grid used by the earlier HAR
q90 backtest.  Historical bid/ask is unavailable, so 25 bps modeled slippage is
used in addition to statutory charges and brokerage.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from atm_iv_rv_analysis import prepare_spot, seasonality_profile
from economic_evaluation import (
    PRIMARY_SLIPPAGE_BPS, attach_pnl, estimated_roundtrip_cost, performance,
)
from five_strike_spot_hypothesis import black_price


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "corrected_short_straddle"
ECONOMIC_PANEL = ROOT / "outputs" / "economic_evaluation" / "economic_trade_panel.parquet"
IV_PREDICTIONS = ROOT / "outputs" / "log_iv_change_models" / "compact_five_feature" / "walk_forward_predictions.parquet"
HAR_PREDICTIONS = ROOT / "outputs" / "enhanced_rv_models" / "validation_120" / "oos_120m.parquet"
SPOT_FILE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"

HORIZON = 120
TRAIN_DAYS = 60
MIN_SCENARIOS = 1000
MAX_SCENARIOS = 5000
MARGIN_FRACTION = 0.15
ANNUAL_MARGIN_RATE = 0.08
COST_MULTIPLE = 2.0
MAX_LOSS_PROBABILITY = 0.45
MAX_ES_LOSS_CREDIT_MULTIPLE = 1.0


def read_parquet(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
    con.close()
    return frame


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute("COPY frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    con.close()


def source_spot() -> pd.DataFrame:
    raw = pd.read_csv(SPOT_FILE, usecols=["datetime", "spot_price"], parse_dates=["datetime"])
    raw = raw.dropna().loc[lambda x: x["spot_price"].gt(0)].drop_duplicates("datetime", keep="last").sort_values("datetime")
    return prepare_spot(raw)


def causal_overnight_variance(spot: pd.DataFrame) -> tuple[pd.Series, float, float]:
    daily = spot.groupby("date", as_index=False).agg(
        first_time=("datetime", "first"), open=("spot_price", "first"), close=("spot_price", "last")
    ).sort_values("date")
    daily["previous_close"] = daily["close"].shift(1)
    daily["previous_date"] = daily["date"].shift(1)
    daily["gap_days"] = (daily["date"] - daily["previous_date"]).dt.days
    daily["overnight_r2"] = np.log(daily["open"] / daily["previous_close"]).pow(2)
    train_mask = (
        daily["date"].le(pd.Timestamp("2022-12-31"))
        & daily["open"].gt(0) & daily["previous_close"].gt(0)
        & daily["gap_days"].gt(0) & np.isfinite(daily["overnight_r2"])
    )
    daily["overnight_per_calendar_day"] = daily["overnight_r2"] / daily["gap_days"].clip(lower=1)
    cap = float(daily.loc[train_mask, "overnight_per_calendar_day"].quantile(0.995))
    daily["overnight_per_calendar_day"] = daily["overnight_per_calendar_day"].clip(upper=cap)
    fallback = float(daily.loc[train_mask, "overnight_per_calendar_day"].mean())
    daily["causal_overnight_daily_var"] = (
        daily["overnight_per_calendar_day"].shift(1).rolling(60, min_periods=30).mean().fillna(fallback)
    )
    return daily.set_index("date")["causal_overnight_daily_var"], fallback, cap


def corrected_intraday_iv_variance(
    row: pd.Series, seasonal: pd.Series, sessions: set[pd.Timestamp],
) -> float:
    """Allocate expiry IV variance after removing an expected overnight budget."""
    current = pd.Timestamp(row["datetime"])
    expiry = pd.Timestamp(row["expiry"]).normalize()
    current_date = current.normalize()
    dte = max(float(row["days_to_expiry"]), 1.0 / 1440.0)
    total_option_variance = float(row["iv_atm"]) ** 2 * dte / 365.0
    calendar_nights = max((expiry - current_date).days, 0)
    overnight_budget = float(row["overnight_daily_var"]) * calendar_nights
    intraday_pool = max(total_option_variance - overnight_budget, 0.0)

    minute = current.hour * 60 + current.minute - 555
    start = int(np.clip(minute + 1, 1, 375))
    end = int(np.clip(minute + HORIZON, 1, 375))
    window_mass = float(seasonal.loc[start:end].sum()) if end >= start else 0.0
    current_remaining = float(seasonal.loc[start:375].sum())
    future_sessions = sum(current_date < date <= expiry for date in sessions)
    total_mass = current_remaining + future_sessions * float(seasonal.loc[1:375].sum())
    return intraday_pool * window_mass / total_mass if total_mass > 0 else np.nan


def joined_panel() -> tuple[pd.DataFrame, pd.Series, float, float]:
    panel = read_parquet(ECONOMIC_PANEL)
    panel = panel[(panel["horizon_min"].eq(HORIZON)) & panel["scope"].eq("intraday")].copy()
    panel["join_minute"] = panel["datetime"].dt.floor("min")

    har = read_parquet(HAR_PREDICTIONS)[["datetime", "har", "har_q90"]]
    har = har.rename(columns={"datetime": "join_minute", "har": "har_mean"})
    panel = panel.merge(har, on="join_minute", how="inner", validate="many_to_one")

    iv = read_parquet(IV_PREDICTIONS)
    iv = iv[iv["horizon_min"].eq(HORIZON)][["datetime", "target_dlog_iv_120"]].copy()
    panel = panel.merge(iv, on="datetime", how="inner", validate="many_to_one")
    panel = panel.sort_values("datetime").drop_duplicates("datetime", keep="last")
    panel["rv_log_residual"] = np.log(panel["actual_var"] / panel["har_mean"])
    panel["iv_log_residual"] = panel["target_dlog_iv_120"] - panel["q50_compact"]
    panel["terminal_return"] = np.log(panel["exit_spot"] / panel["spot_price"])
    panel["standardized_terminal_return"] = panel["terminal_return"] / np.sqrt(panel["actual_var"])
    needed = [
        "har_mean", "har_q90", "actual_var", "iv_atm", "q50_compact",
        "rv_log_residual", "iv_log_residual", "standardized_terminal_return",
        "entry_premium", "exit_premium", "strike", "days_to_expiry", "lot_size",
    ]
    panel = panel[np.isfinite(panel[needed]).all(axis=1)].copy()
    spot = source_spot()
    seasonal = seasonality_profile(spot)
    overnight_by_date, overnight_fallback, overnight_cap = causal_overnight_variance(spot)
    sessions = set(pd.to_datetime(spot["date"].unique()))
    panel["overnight_daily_var"] = panel["date"].map(overnight_by_date).fillna(overnight_fallback)
    panel["corrected_intraday_iv_variance"] = panel.apply(
        corrected_intraday_iv_variance, axis=1,
        args=(seasonal, sessions),
    )
    panel["tail_cushion_corrected"] = panel["corrected_intraday_iv_variance"] / panel["har_q90"]
    return panel, seasonal, overnight_fallback, overnight_cap


def scenario_metrics(row: pd.Series, history: pd.DataFrame) -> dict | None:
    prior_dates = history.loc[history["date"].lt(row["date"]), "date"].drop_duplicates().sort_values()
    use_dates = prior_dates.iloc[-TRAIN_DAYS:]
    scenarios = history[history["date"].isin(use_dates)].tail(MAX_SCENARIOS)
    if len(scenarios) < MIN_SCENARIOS:
        return None

    simulated_variance = float(row["har_mean"]) * np.exp(scenarios["rv_log_residual"].to_numpy(float))
    simulated_log_return = scenarios["standardized_terminal_return"].to_numpy(float) * np.sqrt(simulated_variance)
    simulated_spot = float(row["spot_price"]) * np.exp(simulated_log_return)
    simulated_dlog_iv = float(row["q50_compact"]) + scenarios["iv_log_residual"].to_numpy(float)
    simulated_exit_iv = np.clip(float(row["iv_atm"]) * np.exp(simulated_dlog_iv), 0.01, 5.0)
    exit_dte = max(float(row["days_to_expiry"]) - HORIZON / 1440.0, 1.0 / 1440.0)
    t_years = np.full(len(scenarios), exit_dte / 365.0)
    strike = np.full(len(scenarios), float(row["strike"]))
    exit_call = black_price(simulated_spot, strike, t_years, simulated_exit_iv, True)
    exit_put = black_price(simulated_spot, strike, t_years, simulated_exit_iv, False)
    simulated_exit_premium = exit_call + exit_put
    gross_points = float(row["entry_premium"]) - simulated_exit_premium
    lot = float(row["lot_size"])

    one = pd.DataFrame([row])
    trading_cost = float(estimated_roundtrip_cost(one, -1, PRIMARY_SLIPPAGE_BPS)[0])
    margin_proxy = MARGIN_FRACTION * float(row["spot_price"]) * lot
    margin_carry = margin_proxy * ANNUAL_MARGIN_RATE * HORIZON / (252.0 * 375.0)
    total_friction = trading_cost + margin_carry
    net_rupees = gross_points * lot - total_friction
    cutoff = np.quantile(net_rupees, 0.05)
    tail = net_rupees[net_rupees <= cutoff]
    expected_delta_iv = float(row["iv_atm"]) * (float(np.mean(np.exp(simulated_dlog_iv))) - 1.0)
    expected_gross = float(np.mean(gross_points) * lot)
    expected_net = float(np.mean(net_rupees))
    expected_shortfall = float(np.mean(tail))
    credit_rupees = float(row["entry_premium"]) * lot
    return {
        "scenario_count": len(scenarios), "expected_delta_iv": expected_delta_iv,
        "expected_gross_rupees": expected_gross, "expected_net_rupees": expected_net,
        "estimated_trading_cost": trading_cost, "margin_carry": margin_carry,
        "total_friction": total_friction, "loss_probability": float(np.mean(net_rupees < 0)),
        "var_95_rupees": float(cutoff), "es_95_rupees": expected_shortfall,
        "cost_gate": bool(expected_gross > COST_MULTIPLE * total_friction),
        "loss_probability_gate": bool(np.mean(net_rupees < 0) <= MAX_LOSS_PROBABILITY),
        "tail_loss_gate": bool(expected_shortfall >= -MAX_ES_LOSS_CREDIT_MULTIPLE * credit_rupees),
    }


def backtest() -> tuple[pd.DataFrame, pd.DataFrame, dict, float, float]:
    panel, _, overnight, overnight_cap = joined_panel()
    minute = panel["join_minute"].dt.hour * 60 + panel["join_minute"].dt.minute
    candidates = panel[minute.isin([615, 735])].copy()  # 10:15 and 12:15
    rows = []
    for _, row in candidates.iterrows():
        metrics = scenario_metrics(row, panel)
        if metrics is not None:
            rows.append({**row.to_dict(), **metrics})
    evaluated = pd.DataFrame(rows)
    evaluated["iv_tail_gate"] = evaluated["corrected_intraday_iv_variance"] > evaluated["har_q90"]
    evaluated["signal"] = (
        evaluated["iv_tail_gate"] & evaluated["cost_gate"]
        & evaluated["loss_probability_gate"] & evaluated["tail_loss_gate"]
    )
    selected = evaluated[evaluated["signal"]].copy()
    selected["action"] = -1
    trades = attach_pnl(selected, PRIMARY_SLIPPAGE_BPS)
    if len(trades):
        trades["margin_carry"] = (
            MARGIN_FRACTION * trades["spot_price"] * trades["lot_size"]
            * ANNUAL_MARGIN_RATE * HORIZON / (252.0 * 375.0)
        )
        trades["net_pnl_before_carry"] = trades["net_pnl_rupees"]
        trades["net_pnl_rupees"] -= trades["margin_carry"]
    else:
        trades["margin_carry"] = []
        trades["net_pnl_before_carry"] = []
    result = performance(trades)
    if len(trades):
        result["total_margin_carry"] = float(trades["margin_carry"].sum())
        result["total_friction_including_carry"] = float(trades["total_cost"].sum() + trades["margin_carry"].sum())
        daily = trades.groupby("date")["net_pnl_rupees"].sum().to_numpy(float)
        rng = np.random.default_rng(20261012)
        boot = rng.choice(daily, size=(10000, len(daily)), replace=True).mean(axis=1)
        result["mean_daily_net_bootstrap_ci"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        wins = int((daily > 0).sum())
        result["winning_days"] = wins
        result["one_sided_sign_pvalue"] = float(stats.binomtest(wins, len(daily), 0.5, alternative="greater").pvalue)
    return evaluated, trades, result, overnight, overnight_cap


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evaluated, trades, result, overnight, overnight_cap = backtest()
    diagnostics = {
        "rule": {
            "horizon_minutes": HORIZON, "entry_times": ["10:15", "12:15"],
            "training_days_for_joint_scenarios": TRAIN_DAYS,
            "iv_tail_gate": "overnight-stripped seasonality-allocated IV variance > HAR q90",
            "cost_gate": f"expected gross P&L > {COST_MULTIPLE:.1f}x total friction",
            "max_loss_probability": MAX_LOSS_PROBABILITY,
            "max_es95_loss_as_entry_credit": MAX_ES_LOSS_CREDIT_MULTIPLE,
            "slippage_bps": PRIMARY_SLIPPAGE_BPS,
            "margin_proxy_fraction_of_spot_notional": MARGIN_FRACTION,
            "annual_margin_carry_rate": ANNUAL_MARGIN_RATE,
        },
        "clock_audit": {
            "greek_theta_unit": "calendar day (verified from Black implementation /365)",
            "new_pricing_clock": "calendar DTE in full nonlinear repricing",
            "overnight_variance_fallback_2019_2022": overnight,
            "overnight_variance_winsor_cap_99_5pct_2019_2022": overnight_cap,
            "overnight_estimator": "prior 60 sessions, per calendar day, pre-2023 cap",
        },
        "coverage": {
            "evaluated_candidates": len(evaluated), "signals": int(evaluated["signal"].sum()),
            "start": str(evaluated["date"].min().date()) if len(evaluated) else None,
            "end": str(evaluated["date"].max().date()) if len(evaluated) else None,
        },
        "performance": result,
        "gate_counts": {
            name: int(evaluated[name].sum()) for name in
            ["iv_tail_gate", "cost_gate", "loss_probability_gate", "tail_loss_gate", "signal"]
        },
        "execution_limit": "Source has no historical bid/ask; 25 bps slippage is modeled, not observed.",
        "stop_limit": "No intrawindow fixed-strike bid/ask path exists, so a stop-loss is not backtested.",
    }
    (OUT / "results.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    evaluated.to_csv(OUT / "evaluated_candidates.csv", index=False)
    write_parquet(trades, OUT / "executed_trades.parquet")
    quarterly = pd.DataFrame()
    if len(trades):
        quarterly = trades.assign(quarter=trades["date"].dt.to_period("Q").astype(str)).groupby("quarter", as_index=False).agg(
            trades=("net_pnl_rupees", "size"), total_net=("net_pnl_rupees", "sum"),
            mean_net=("net_pnl_rupees", "mean"), hit_rate=("net_pnl_rupees", lambda x: float((x > 0).mean())),
        )
    quarterly.to_csv(OUT / "quarterly_performance.csv", index=False)

    def num(name: str) -> str:
        value = result.get(name, np.nan)
        return "NA" if not np.isfinite(value) else f"{value:,.2f}"

    report = [
        "# Corrected 120-minute short-straddle backtest", "",
        "## Method", "",
        "- Black repricing uses calendar DTE throughout; the old mixed theta/trading-minute approximation is removed.",
        "- Expiry IV variance has a 2019-2022 expected overnight budget removed, then the remaining intraday pool is allocated with the frozen minute-of-day variance profile.",
        "- The IV forecast uses the mean of exp(predicted log-IV change + prior residual), rather than exp(median)-1.",
        "- Paired RV residual, IV residual and standardized terminal-return observations from the preceding 60 trading days form joint scenarios. Full option repricing captures their dependence plus delta, vanna and volga nonlinearity.",
        "- A trade requires the corrected IV tail gate, expected gross P&L above twice modeled friction, loss probability no more than 45%, and 5% expected-shortfall loss no worse than one entry credit.",
        "- Entries are fixed at 10:15 and 12:15, so 120-minute holdings do not overlap.", "",
        "## Results", "",
        f"- Evaluated candidates: {len(evaluated)}",
        f"- Trades: {int(result.get('trades', 0)) if np.isfinite(result.get('trades', np.nan)) else 0}",
        f"- Total gross P&L: Rs {num('total_gross')}",
        f"- Total modeled costs: Rs {num('total_cost')}",
        f"- Margin carry proxy: Rs {num('total_margin_carry')}",
        f"- Total net P&L after costs and margin carry: Rs {num('total_net')}",
        f"- Mean net per trade: Rs {num('mean_net')}",
        f"- Hit rate: {num('hit_rate')}",
        f"- Daily Sharpe: {num('daily_sharpe')}",
        f"- Maximum drawdown: Rs {num('max_drawdown')}", "",
        f"The 95% day-bootstrap interval for mean daily net P&L is {result.get('mean_daily_net_bootstrap_ci', ['NA', 'NA'])}; the one-sided exact sign-test p-value is {result.get('one_sided_sign_pvalue', 'NA')}.", "",
        "## Limits", "",
        "Historical bid/ask is unavailable. The result includes brokerage, taxes, exchange fees, 25 bps slippage and an explicit margin-carry proxy, but cannot prove executable fills. Intrawindow fixed-strike bid/ask paths are also unavailable, so a stop-loss cannot be tested honestly.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, default=str))


if __name__ == "__main__":
    main()
