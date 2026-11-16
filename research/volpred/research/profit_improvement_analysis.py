from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from economic_evaluation import (
    OUT as ECON_OUT,
    attach_pnl,
    bootstrap_daily,
    estimated_roundtrip_cost,
    non_overlapping,
    performance,
    read_parquet,
    write_parquet,
)
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "profit_improvement"
SELECTION_SLIPPAGE_BPS = 10.0


@dataclass(frozen=True)
class Rule:
    signal: str
    hurdle: float
    direction: str
    agreement: bool
    confidence_q: float = 0.0
    time_filter: str = "all"
    dte_filter: str = "all"
    max_trades_day: int = 99
    cooldown_min: int = 0


def signal_points(frame: pd.DataFrame, signal: str) -> np.ndarray:
    if signal == "rv_only":
        return (frame["theta_decay_points"] * frame["gamma_theta_score"]).to_numpy(float)
    if signal == "iv_only":
        return (frame["theta_decay_points"] * frame["iv_only_score"]).to_numpy(float)
    if signal == "div_only":
        return (frame["vega_s_decimal"] * frame["pred_d_iv"]).to_numpy(float)
    if signal.startswith("blend_"):
        weight = float(signal.split("_", 1)[1])
        return (
            frame["theta_decay_points"]
            * (frame["gamma_theta_score"] + weight * frame["vega_score"])
        ).to_numpy(float)
    raise ValueError(signal)


def raw_actions(frame: pd.DataFrame, rule: Rule, confidence_cut: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    expected_rupees = signal_points(frame, rule.signal) * frame["lot_size"].to_numpy(float)
    long_cost = estimated_roundtrip_cost(frame, 1, SELECTION_SLIPPAGE_BPS)
    short_cost = estimated_roundtrip_cost(frame, -1, SELECTION_SLIPPAGE_BPS)
    action = np.where(
        expected_rupees > rule.hurdle * long_cost,
        1,
        np.where(expected_rupees < -rule.hurdle * short_cost, -1, 0),
    ).astype(int)
    if rule.direction == "long":
        action[action < 0] = 0
    elif rule.direction == "short":
        action[action > 0] = 0
    if rule.agreement:
        agrees = np.sign(frame["gamma_theta_score"].to_numpy(float)) == np.sign(frame["vega_score"].to_numpy(float))
        action[~agrees] = 0
    conviction = np.divide(
        np.abs(expected_rupees),
        np.where(expected_rupees >= 0, long_cost, short_cost),
        out=np.zeros_like(expected_rupees),
        where=np.where(expected_rupees >= 0, long_cost, short_cost) > 0,
    )
    if confidence_cut is not None:
        action[conviction < confidence_cut] = 0
    return action, conviction


def filter_frame(frame: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    z = frame
    minute = z["datetime"].dt.hour * 60 + z["datetime"].dt.minute
    if rule.time_filter == "early":
        z = z[minute < 690]
    elif rule.time_filter == "late":
        z = z[minute >= 690]
    if rule.dte_filter == "0_3":
        z = z[z["days_to_expiry"] < 4]
    elif rule.dte_filter == "4_plus":
        z = z[z["days_to_expiry"] >= 4]
    return z.copy()


def execute(frame: pd.DataFrame, rule: Rule, confidence_cut: float | None = None) -> pd.DataFrame:
    z = filter_frame(frame, rule)
    if z.empty:
        return z.assign(action=pd.Series(dtype=int))
    action, conviction = raw_actions(z, rule, confidence_cut)
    z["action"] = action
    z["conviction"] = conviction
    z = z[z["action"].ne(0)].sort_values("datetime")
    chosen_pos: list[int] = []
    next_free_ns = np.iinfo(np.int64).min
    current_day = None
    count_day = 0
    dt_ns = z["datetime"].astype("int64").to_numpy()
    exit_ns = z["exit_datetime"].astype("int64").to_numpy()
    days = z["date"].astype("int64").to_numpy()
    cooldown_ns = int(rule.cooldown_min * 60 * 1_000_000_000)
    for pos in range(len(z)):
        day = days[pos]
        if current_day is None or day != current_day:
            current_day = day
            count_day = 0
            next_free_ns = np.iinfo(np.int64).min
        if count_day >= rule.max_trades_day or dt_ns[pos] < next_free_ns:
            continue
        chosen_pos.append(pos)
        count_day += 1
        next_free_ns = exit_ns[pos] + cooldown_ns
    return z.iloc[chosen_pos].copy()


def score_rule(frame: pd.DataFrame, rule: Rule, confidence_cut: float | None = None) -> tuple[dict, pd.DataFrame]:
    selected = execute(frame, rule, confidence_cut)
    trades = attach_pnl(selected, SELECTION_SLIPPAGE_BPS)
    return performance(trades), trades


def robust_dev_score(trades: pd.DataFrame) -> tuple[float, float, float]:
    if trades.empty:
        return -999.0, np.nan, np.nan
    h1 = trades[trades["datetime"] < pd.Timestamp("2024-07-01")]
    h2 = trades[trades["datetime"] >= pd.Timestamp("2024-07-01")]
    p1, p2, pa = performance(h1), performance(h2), performance(trades)
    if pa["trades"] < 20 or p1["trades"] < 5 or p2["trades"] < 5:
        return -999.0, p1["daily_sharpe"], p2["daily_sharpe"]
    vals = np.array([pa["daily_sharpe"], p1["daily_sharpe"], p2["daily_sharpe"]], float)
    if not np.all(np.isfinite(vals)):
        return -999.0, p1["daily_sharpe"], p2["daily_sharpe"]
    stability_penalty = 1.5 if (p1["total_net"] <= 0 or p2["total_net"] <= 0) else 0.0
    score = 0.50 * vals[0] + 0.25 * vals[1] + 0.25 * vals[2] - stability_penalty
    return score, p1["daily_sharpe"], p2["daily_sharpe"]


def optimize_cell(frame: pd.DataFrame, scope: str, horizon: int) -> tuple[Rule, float, pd.DataFrame]:
    dev = frame[(frame["datetime"] < pd.Timestamp("2025-01-01"))].copy()
    signals = ["rv_only", "iv_only", "div_only", "blend_0.10", "blend_0.25", "blend_0.50", "blend_1.00"]
    stage1: list[dict] = []
    for signal in signals:
        for hurdle in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            for direction in ["both", "long", "short"]:
                for agreement in [False, True]:
                    rule = Rule(signal, hurdle, direction, agreement)
                    perf, trades = score_rule(dev, rule)
                    robust, h1, h2 = robust_dev_score(trades)
                    stage1.append({**asdict(rule), **perf, "robust_score": robust, "h1_sharpe": h1, "h2_sharpe": h2})
    s1 = pd.DataFrame(stage1).sort_values(["robust_score", "total_net"], ascending=False)
    seeds = s1.head(3)

    rows: list[dict] = []
    for _, seed in seeds.iterrows():
        for q in [0.0, 0.75, 0.90]:
            for time_filter in ["all", "early", "late"]:
                for dte_filter in ["all", "0_3", "4_plus"]:
                    for max_day in [1, 2]:
                        for cooldown in [0]:
                            rule = Rule(
                                seed["signal"], float(seed["hurdle"]), seed["direction"], bool(seed["agreement"]),
                                q, time_filter, dte_filter, int(max_day), int(cooldown),
                            )
                            base_for_cut = filter_frame(dev, rule)
                            _, conviction = raw_actions(base_for_cut, rule)
                            active = conviction[conviction >= rule.hurdle]
                            cut = float(np.quantile(active, q)) if q > 0 and len(active) else None
                            perf, trades = score_rule(dev, rule, cut)
                            robust, h1, h2 = robust_dev_score(trades)
                            rows.append({
                                "scope": scope, "horizon_min": horizon, **asdict(rule), "confidence_cut": cut,
                                **perf, "robust_score": robust, "h1_sharpe": h1, "h2_sharpe": h2,
                            })
    candidates = pd.DataFrame(rows).drop_duplicates(
        ["signal", "hurdle", "direction", "agreement", "confidence_q", "time_filter", "dte_filter", "max_trades_day", "cooldown_min"]
    )
    eligible = candidates[(candidates["robust_score"] > -900) & (candidates["trades"] >= 20)].copy()
    if eligible.empty:
        best = candidates.sort_values(["robust_score", "total_net"], ascending=False).iloc[0]
    else:
        best = eligible.sort_values(["robust_score", "total_net"], ascending=False).iloc[0]
    rule = Rule(
        best.signal, float(best.hurdle), best.direction, bool(best.agreement), float(best.confidence_q),
        best.time_filter, best.dte_filter, int(best.max_trades_day), int(best.cooldown_min),
    )
    return rule, float(best.confidence_cut) if pd.notna(best.confidence_cut) else None, candidates


def evaluate_scenarios(raw: pd.DataFrame, scope: str, horizon: int, rule: Rule, cut: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, trade_frames = [], []
    periods = {
        "development_2024": raw[raw["datetime"] < pd.Timestamp("2025-01-01")],
        "holdout_2025": raw[raw["datetime"] >= pd.Timestamp("2025-01-01")],
        "full_sample": raw,
    }
    scenarios = [
        ("zero_cost", 0.0, False, True),
        ("historical_zerodha_only", 0.0, False, False),
        ("historical_plus_10bps", 10.0, False, False),
        ("historical_plus_25bps", 25.0, False, False),
        ("current_2026_zerodha_only", 0.0, True, False),
        ("current_2026_plus_10bps", 10.0, True, False),
        ("current_2026_plus_25bps", 25.0, True, False),
    ]
    for period, frame in periods.items():
        chosen = execute(frame, rule, cut)
        for scenario, bps, current, zero in scenarios:
            if zero:
                trades = chosen.copy()
                trades["gross_pnl_rupees"] = trades["action"] * trades["long_pnl_points"] * trades["lot_size"]
                trades["total_cost"] = 0.0
                trades["net_pnl_rupees"] = trades["gross_pnl_rupees"]
            else:
                trades = attach_pnl(chosen, bps, current_2026=current)
            p = performance(trades)
            rows.append({"scope": scope, "horizon_min": horizon, "period": period, "cost_scenario": scenario, **p})
            if period == "holdout_2025" and scenario in {"historical_plus_10bps", "current_2026_plus_10bps"}:
                z = trades.copy()
                z["scope"] = scope; z["horizon_min"] = horizon; z["period"] = period; z["cost_scenario"] = scenario
                trade_frames.append(z)
    return pd.DataFrame(rows), pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()


def baseline_comparison(raw: pd.DataFrame, scope: str, horizon: int) -> pd.DataFrame:
    rows = []
    holdout = raw[raw["datetime"] >= pd.Timestamp("2025-01-01")]
    for signal in ["rv_only", "iv_only", "div_only", "blend_1.00"]:
        rule = Rule(signal, 1.0, "both", False)
        chosen = execute(holdout, rule)
        for scenario, bps, current in [("historical_plus_25bps", 25.0, False), ("current_2026_plus_10bps", 10.0, True)]:
            trades = attach_pnl(chosen, bps, current_2026=current)
            rows.append({"scope": scope, "horizon_min": horizon, "method": signal, "cost_scenario": scenario, **performance(trades)})
    return pd.DataFrame(rows)


def cost_waterfall(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    cost_cols = ["brokerage", "stt", "exchange", "sebi", "ipft", "stamp", "gst", "slippage", "total_cost"]
    keys = ["scope", "horizon_min", "cost_scenario"]
    return trades.groupby(keys)[cost_cols].agg(["sum", "mean"]).rename_axis(columns=["component", "stat"]).stack(0, future_stack=True).reset_index()


def plot_results(results: pd.DataFrame, baselines: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    h = results[(results["period"] == "holdout_2025") & (results["cost_scenario"] == "current_2026_plus_10bps")]
    for scope, g in h.groupby("scope"):
        axes[0].plot(g["horizon_min"], g["total_net"], marker="o", label=scope)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Selected rules: 2025 holdout net P&L")
    axes[0].set_xlabel("Horizon (minutes)"); axes[0].set_ylabel("Net P&L per lot (₹)"); axes[0].legend()
    b = baselines[(baselines["scope"] == "intraday") & (baselines["cost_scenario"] == "current_2026_plus_10bps")]
    for method, g in b.groupby("method"):
        axes[1].plot(g["horizon_min"], g["daily_sharpe"], marker="o", label=method)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Unoptimized signals: 2025 holdout Sharpe")
    axes[1].set_xlabel("Horizon (minutes)"); axes[1].set_ylabel("Daily Sharpe"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "profit_improvement.png", dpi=180); plt.close(fig)


def write_report(selected: pd.DataFrame, results: pd.DataFrame, baselines: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    hold = results[(results["period"] == "holdout_2025") & (results["cost_scenario"] == "current_2026_plus_10bps")].copy()
    hold = hold.sort_values("daily_sharpe", ascending=False)
    best = hold.iloc[0] if len(hold) else None
    lines = [
        "# Cost reduction and profit-improvement study", "",
        "Rules were selected only on 2024 data. The reported validation period is 2025-01-01 onward. "
        "This limits rule-selection leakage, although the underlying forecasting models were previously developed using the broader sample.", "",
        "## Decision", "",
    ]
    if best is not None:
        lines.append(
            f"The best 2025 holdout cell under current 2026 Zerodha charges plus a modeled 10 bps execution allowance was "
            f"{best.scope} {int(best.horizon_min)} minutes: {int(best.trades)} trades, net ₹{best.total_net:,.0f}, "
            f"daily Sharpe {best.daily_sharpe:.2f}, hit rate {best.hit_rate:.1%}."
        )
    lines += [
        "", "The main cost reduction comes from rejecting weak entries and limiting same-day turnover. Brokerage and statutory charges "
        "are not negotiable within Zerodha's standard pricing; the 10 bps scenario represents disciplined limit-order execution and must "
        "be checked in paper/live fills because the source file has no bid/ask quotes.", "",
        "## Selected rules", "", dataframe_to_markdown(selected), "",
        "## 2025 holdout results", "", dataframe_to_markdown(hold), "",
        "## Unoptimized holdout benchmarks", "", dataframe_to_markdown(baselines), "",
        "## Bootstrap checks", "", dataframe_to_markdown(bootstrap), "",
        "## Guardrails", "",
        "- Use one lot until actual spread and slippage are recorded for at least 50 trades.",
        "- Do not treat a positive cell as proven unless its day-block confidence interval and later paper-trading period remain positive.",
        "- Do not replace the two-leg straddle with a one-leg option solely to save brokerage; that introduces directional risk and is a different strategy.",
        "- Re-optimize only on a scheduled expanding window, never after individual losses.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = read_parquet(ECON_OUT / "economic_trade_panel.parquet")
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["exit_datetime"] = pd.to_datetime(panel["exit_datetime"])
    panel["date"] = pd.to_datetime(panel["date"])

    candidate_frames, selected_rows, result_frames, baseline_frames, holdout_trades = [], [], [], [], []
    for scope in ["intraday", "09:15"]:
        # Prior analysis showed the shorter horizons cannot cover costs. Optimize the
        # economically plausible 120-minute horizon instead of data-mining all cells.
        for horizon in [120]:
            raw = panel[(panel["scope"] == scope) & (panel["horizon_min"] == horizon)].copy()
            raw = raw[raw["is_0915"] == (1 if scope == "09:15" else 0)]
            if raw.empty:
                continue
            print(f"Optimizing {scope} {horizon}m ({len(raw):,} opportunities)...", flush=True)
            rule, cut, candidates = optimize_cell(raw, scope, horizon)
            candidate_frames.append(candidates)
            selected_rows.append({"scope": scope, "horizon_min": horizon, **asdict(rule), "confidence_cut": cut})
            results, trades = evaluate_scenarios(raw, scope, horizon, rule, cut)
            result_frames.append(results); holdout_trades.append(trades)
            baseline_frames.append(baseline_comparison(raw, scope, horizon))

    candidates = pd.concat(candidate_frames, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    results = pd.concat(result_frames, ignore_index=True)
    baselines = pd.concat(baseline_frames, ignore_index=True)
    trades = pd.concat(holdout_trades, ignore_index=True)
    boot_rows = []
    for keys, z in trades.groupby(["scope", "horizon_min", "cost_scenario"]):
        boot_rows.append({"scope": keys[0], "horizon_min": keys[1], "cost_scenario": keys[2], **bootstrap_daily(z, n_boot=2000)})
    bootstrap = pd.DataFrame(boot_rows)
    waterfall = cost_waterfall(trades)

    candidates.to_csv(OUT / "optimization_candidates.csv", index=False)
    selected.to_csv(OUT / "selected_rules.csv", index=False)
    results.to_csv(OUT / "selected_rule_performance.csv", index=False)
    baselines.to_csv(OUT / "holdout_baselines.csv", index=False)
    bootstrap.to_csv(OUT / "holdout_bootstrap.csv", index=False)
    waterfall.to_csv(OUT / "cost_reduction_breakdown.csv", index=False)
    write_parquet(trades, OUT / "optimized_holdout_trades.parquet")
    plot_results(results, baselines)
    write_report(selected, results, baselines, bootstrap)
    print("SELECTED")
    print(selected.to_string(index=False))
    print("\nHOLDOUT CURRENT+10BPS")
    print(results[(results.period == "holdout_2025") & (results.cost_scenario == "current_2026_plus_10bps")].to_string(index=False))


if __name__ == "__main__":
    main()
