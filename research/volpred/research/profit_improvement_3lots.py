from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from economic_evaluation import charge_components, performance, read_parquet, write_parquet


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "economic_evaluation" / "economic_trade_panel.parquet"
OUT = ROOT / "outputs" / "profit_improvement_3lots"
N_LOTS = 3
SELECTION_BPS = 10.0


@dataclass(frozen=True)
class Rule:
    signal: str
    hurdle: float
    agreement: str = "none"
    confidence_q: float = 0.0
    time_filter: str = "all"
    dte_filter: str = "all"
    max_trades_day: int = 2


def signal_points(frame: pd.DataFrame, name: str) -> np.ndarray:
    theta = frame["theta_decay_points"].to_numpy(float)
    if name == "iv_only":
        return theta * frame["iv_only_score"].to_numpy(float)
    if name == "rv_only":
        return theta * frame["gamma_theta_score"].to_numpy(float)
    if name == "div_only":
        return frame["vega_s_decimal"].to_numpy(float) * frame["pred_d_iv"].to_numpy(float)
    if name.startswith("blend_"):
        w = float(name.split("_", 1)[1])
        return theta * (frame["gamma_theta_score"].to_numpy(float) + w * frame["vega_score"].to_numpy(float))
    raise ValueError(name)


def costs(frame: pd.DataFrame, action: np.ndarray, bps: float, current: bool, order_batches: int, entry_estimate: bool = False) -> pd.DataFrame:
    exit_premium = frame["entry_premium"].to_numpy(float) if entry_estimate else frame["exit_premium"].to_numpy(float)
    c = charge_components(
        frame["entry_premium"].to_numpy(float), exit_premium, action,
        frame["lot_size"].to_numpy(float) * N_LOTS, frame["date"],
        slippage_bps=bps, current_2026=current,
    )
    if order_batches != 1:
        extra_brokerage = 80.0 * (order_batches - 1)
        c["brokerage"] += extra_brokerage
        c["gst"] += 0.18 * extra_brokerage
        c["total_cost"] += 1.18 * extra_brokerage
    return c


def filter_frame(frame: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    z = frame
    minute = z["datetime"].dt.hour * 60 + z["datetime"].dt.minute
    if rule.time_filter == "before_1130":
        z = z[minute < 690]
    elif rule.time_filter == "after_1130":
        z = z[minute >= 690]
    if rule.dte_filter == "0_3":
        z = z[z["days_to_expiry"] < 4]
    elif rule.dte_filter == "4_plus":
        z = z[z["days_to_expiry"] >= 4]
    return z.copy()


def agreement_mask(frame: pd.DataFrame, agreement: str) -> np.ndarray:
    rv_short = frame["gamma_theta_score"].to_numpy(float) < 0
    iv_short = frame["iv_only_score"].to_numpy(float) < 0
    div_short = frame["pred_d_iv"].to_numpy(float) < 0
    if agreement == "none": return np.ones(len(frame), dtype=bool)
    if agreement == "rv_div": return rv_short & div_short
    if agreement == "iv_div": return iv_short & div_short
    if agreement == "all_short": return rv_short & iv_short & div_short
    raise ValueError(agreement)


def action_and_conviction(frame: pd.DataFrame, rule: Rule, cut: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    expected_rupees = signal_points(frame, rule.signal) * frame["lot_size"].to_numpy(float) * N_LOTS
    action = -np.ones(len(frame), dtype=int)
    est = costs(frame, action, SELECTION_BPS, current=True, order_batches=1, entry_estimate=True)["total_cost"].to_numpy(float)
    conviction = np.divide(-expected_rupees, est, out=np.full(len(frame), -np.inf), where=est > 0)
    active = (expected_rupees < -rule.hurdle * est) & agreement_mask(frame, rule.agreement)
    if cut is not None:
        active &= conviction >= cut
    return np.where(active, -1, 0), conviction


def execute(frame: pd.DataFrame, rule: Rule, cut: float | None = None) -> pd.DataFrame:
    z = filter_frame(frame, rule)
    action, conviction = action_and_conviction(z, rule, cut)
    z["action"] = action; z["conviction"] = conviction
    z = z[z["action"] != 0].sort_values("datetime")
    if z.empty: return z
    dt = z["datetime"].astype("int64").to_numpy(); ex = z["exit_datetime"].astype("int64").to_numpy(); days = z["date"].astype("int64").to_numpy()
    chosen, current_day, count, next_free = [], None, 0, np.iinfo(np.int64).min
    for pos in range(len(z)):
        if current_day is None or days[pos] != current_day:
            current_day, count, next_free = days[pos], 0, np.iinfo(np.int64).min
        if count >= rule.max_trades_day or dt[pos] < next_free: continue
        chosen.append(pos); count += 1; next_free = ex[pos]
    return z.iloc[chosen].copy()


def attach(frame: pd.DataFrame, bps: float, current: bool, order_batches: int) -> pd.DataFrame:
    if frame.empty: return frame.assign(gross_pnl_rupees=[], total_cost=[], net_pnl_rupees=[])
    out = frame.reset_index(drop=True).copy()
    c = costs(out, out["action"].to_numpy(int), bps, current, order_batches)
    for col in c.columns: out[col] = c[col].to_numpy()
    out["gross_pnl_rupees"] = out["action"] * out["long_pnl_points"] * out["lot_size"] * N_LOTS
    out["net_pnl_rupees"] = out["gross_pnl_rupees"] - out["total_cost"]
    return out


def dev_score(trades: pd.DataFrame) -> tuple[float, int, float, float]:
    if len(trades) < 20: return -999.0, 0, np.nan, np.nan
    p = performance(trades)
    q = trades.assign(q=trades["datetime"].dt.quarter).groupby("q")
    q_net = q["net_pnl_rupees"].sum()
    positive_q = int((q_net > 0).sum())
    h1 = performance(trades[trades.datetime < pd.Timestamp("2024-07-01")])
    h2 = performance(trades[trades.datetime >= pd.Timestamp("2024-07-01")])
    if h1["trades"] < 5 or h2["trades"] < 5 or not np.isfinite(p["daily_sharpe"]): return -999.0, positive_q, h1["daily_sharpe"], h2["daily_sharpe"]
    penalty = 1.0 * (4 - positive_q) + (1.0 if h1["total_net"] <= 0 else 0) + (1.0 if h2["total_net"] <= 0 else 0)
    robust = 0.5 * p["daily_sharpe"] + 0.25 * h1["daily_sharpe"] + 0.25 * h2["daily_sharpe"] - penalty
    return robust, positive_q, h1["daily_sharpe"], h2["daily_sharpe"]


def optimize(frame: pd.DataFrame, scope: str) -> tuple[Rule, float | None, pd.DataFrame]:
    dev = frame[frame.datetime < pd.Timestamp("2025-01-01")].copy()
    stage1 = []
    signals = ["iv_only", "rv_only", "div_only", "blend_0.10", "blend_0.25", "blend_0.50", "blend_1.00"]
    for signal in signals:
        for hurdle in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            for agreement in ["none", "rv_div", "iv_div", "all_short"]:
                rule = Rule(signal, hurdle, agreement, max_trades_day=1 if scope == "09:15" else 2)
                z = attach(execute(dev, rule), SELECTION_BPS, True, 1)
                p = performance(z); robust, posq, h1, h2 = dev_score(z)
                stage1.append({**asdict(rule), **p, "robust_score": robust, "positive_quarters": posq, "h1_sharpe": h1, "h2_sharpe": h2})
    seeds = pd.DataFrame(stage1).sort_values(["robust_score", "total_net"], ascending=False).head(6)
    rows = []
    for _, s in seeds.iterrows():
        for q in [0.0, 0.50, 0.75, 0.90]:
            for time_filter in ["all", "before_1130", "after_1130"]:
                for dte_filter in ["all", "0_3", "4_plus"]:
                    for max_day in ([1] if scope == "09:15" else [1, 2]):
                        rule = Rule(s.signal, float(s.hurdle), s.agreement, q, time_filter, dte_filter, max_day)
                        base = filter_frame(dev, rule)
                        action, conviction = action_and_conviction(base, rule)
                        active = conviction[action != 0]
                        cut = float(np.quantile(active, q)) if q > 0 and len(active) else None
                        z = attach(execute(dev, rule, cut), SELECTION_BPS, True, 1)
                        p = performance(z); robust, posq, h1, h2 = dev_score(z)
                        rows.append({"scope": scope, **asdict(rule), "confidence_cut": cut, **p, "robust_score": robust, "positive_quarters": posq, "h1_sharpe": h1, "h2_sharpe": h2})
    candidates = pd.DataFrame(rows).drop_duplicates(["signal", "hurdle", "agreement", "confidence_q", "time_filter", "dte_filter", "max_trades_day"])
    eligible = candidates[(candidates.robust_score > -900) & (candidates.trades >= 20) & (candidates.positive_quarters >= 3)]
    best = (eligible if len(eligible) else candidates).sort_values(["robust_score", "total_net"], ascending=False).iloc[0]
    rule = Rule(best.signal, float(best.hurdle), best.agreement, float(best.confidence_q), best.time_filter, best.dte_filter, int(best.max_trades_day))
    cut = float(best.confidence_cut) if pd.notna(best.confidence_cut) else None
    return rule, cut, candidates


def prior_rule(scope: str) -> Rule:
    return Rule("iv_only", 2.0, max_trades_day=2) if scope == "intraday" else Rule("div_only", 0.5, max_trades_day=1)


def evaluate(frame: pd.DataFrame, scope: str, name: str, rule: Rule, cut: float | None) -> tuple[list[dict], list[pd.DataFrame]]:
    rows, save = [], []
    periods = {"development_2024": frame[frame.datetime < pd.Timestamp("2025-01-01")], "holdout_2025": frame[frame.datetime >= pd.Timestamp("2025-01-01")], "full_sample": frame}
    scenarios = [
        ("zero_cost", 0.0, True, 1, True),
        ("current_plus_10bps_combined_orders", 10.0, True, 1, False),
        ("current_plus_25bps_combined_orders", 25.0, True, 1, False),
        ("current_plus_10bps_split_orders", 10.0, True, 3, False),
        ("historical_plus_25bps_combined_orders", 25.0, False, 1, False),
    ]
    for period, pframe in periods.items():
        chosen = execute(pframe, rule, cut)
        for scenario, bps, current, batches, zero in scenarios:
            if zero:
                z = chosen.copy(); z["gross_pnl_rupees"] = z.action * z.long_pnl_points * z.lot_size * N_LOTS; z["total_cost"] = 0.0; z["net_pnl_rupees"] = z.gross_pnl_rupees
            else: z = attach(chosen, bps, current, batches)
            rows.append({"scope": scope, "rule_name": name, "lots": N_LOTS, "period": period, "cost_scenario": scenario, **performance(z)})
            if period == "holdout_2025" and scenario == "current_plus_10bps_combined_orders":
                z = z.copy(); z["scope"] = scope; z["rule_name"] = name; z["cost_scenario"] = scenario; save.append(z)
    return rows, save


def simple_bootstrap(trades: pd.DataFrame, n=5000) -> dict:
    daily = trades.groupby("date").net_pnl_rupees.sum().to_numpy(float)
    if len(daily) < 10: return {"days": len(daily), "mean_daily": np.mean(daily) if len(daily) else np.nan, "mean_ci_low": np.nan, "mean_ci_high": np.nan, "prob_mean_positive": np.nan}
    rng = np.random.default_rng(7319); draws = rng.choice(daily, (n, len(daily)), replace=True).mean(axis=1)
    return {"days": len(daily), "mean_daily": daily.mean(), "mean_ci_low": np.quantile(draws, .025), "mean_ci_high": np.quantile(draws, .975), "prob_mean_positive": np.mean(draws > 0)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = read_parquet(SOURCE)
    for c in ["datetime", "exit_datetime", "date"]: panel[c] = pd.to_datetime(panel[c])
    selected_rows, candidate_frames, perf_rows, saved = [], [], [], []
    for scope in ["intraday", "09:15"]:
        frame = panel[(panel.scope == scope) & (panel.horizon_min == 120) & (panel.is_0915 == (1 if scope == "09:15" else 0))].copy()
        print(f"Optimizing {scope}: {len(frame):,} opportunities", flush=True)
        rule, cut, candidates = optimize(frame, scope)
        candidate_frames.append(candidates)
        selected_rows.append({"scope": scope, "lots": N_LOTS, **asdict(rule), "confidence_cut": cut})
        for name, use_rule, use_cut in [("prior_rule_scaled", prior_rule(scope), None), ("improved_2024_selected", rule, cut)]:
            rows, trades = evaluate(frame, scope, name, use_rule, use_cut); perf_rows += rows; saved += trades
    # Pure sizing comparison: preserve the exact one-lot holdout entries from the
    # prior study and change quantity only. This is distinct from recalculating the
    # entry hurdle after fixed brokerage is diluted over three lots.
    exact_source = read_parquet(ROOT / "outputs" / "profit_improvement" / "optimized_holdout_trades.parquet")
    exact_source["datetime"] = pd.to_datetime(exact_source["datetime"]); exact_source["exit_datetime"] = pd.to_datetime(exact_source["exit_datetime"]); exact_source["date"] = pd.to_datetime(exact_source["date"])
    exact_source = exact_source[exact_source.cost_scenario == "current_2026_plus_10bps"].copy()
    for scope, chosen in exact_source.groupby("scope"):
        for scenario, bps, batches in [("current_plus_10bps_combined_orders", 10.0, 1), ("current_plus_25bps_combined_orders", 25.0, 1), ("current_plus_10bps_split_orders", 10.0, 3)]:
            z = attach(chosen, bps, True, batches)
            perf_rows.append({"scope": scope, "rule_name": "exact_same_entries_3lots", "lots": N_LOTS, "period": "holdout_2025", "cost_scenario": scenario, **performance(z)})
            if scenario == "current_plus_10bps_combined_orders":
                z = z.copy(); z["scope"] = scope; z["rule_name"] = "exact_same_entries_3lots"; z["cost_scenario"] = scenario; saved.append(z)

    selected = pd.DataFrame(selected_rows); candidates = pd.concat(candidate_frames, ignore_index=True); perf = pd.DataFrame(perf_rows); trades = pd.concat(saved, ignore_index=True)
    compare = perf[(perf.period == "holdout_2025") & (perf.cost_scenario.isin(["current_plus_10bps_combined_orders", "current_plus_25bps_combined_orders", "current_plus_10bps_split_orders"]))].copy()
    boot = []
    for keys, z in trades.groupby(["scope", "rule_name"]): boot.append({"scope": keys[0], "rule_name": keys[1], **simple_bootstrap(z)})
    boot = pd.DataFrame(boot)
    selected.to_csv(OUT / "selected_rules_3lots.csv", index=False)
    perf.to_csv(OUT / "three_lot_performance.csv", index=False)
    compare.to_csv(OUT / "improvement_comparison.csv", index=False)
    boot.to_csv(OUT / "holdout_robustness.csv", index=False)
    candidates.sort_values(["scope", "robust_score", "total_net"], ascending=[True, False, False]).head(500).to_csv(OUT / "top_candidates.csv", index=False)
    write_parquet(trades, OUT / "optimized_3lot_holdout_trades.parquet")
    lines = ["# Three-lot profit improvement", "", "Rules were selected on 2024 only and evaluated on 2025. Costs assume three lots are placed as one quantity per leg, so the four-order straddle round trip retains ₹80 brokerage; split-order stress uses ₹240 brokerage.", "", "## Selected rules", "", selected.to_markdown(index=False) if False else selected.to_string(index=False), "", "## Holdout comparison", "", compare.to_string(index=False), "", "## Statistical caution", "", boot.to_string(index=False), "", "Return on capital is not reported because historical SPAN/exposure margin snapshots are unavailable. Three-lot P&L is not permission to scale live capital; margin and tail loss also triple approximately."]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSELECTED\n", selected.to_string(index=False)); print("\nHOLDOUT\n", compare.to_string(index=False)); print("\nROBUSTNESS\n", boot.to_string(index=False))


if __name__ == "__main__": main()
