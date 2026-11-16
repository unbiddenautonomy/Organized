from __future__ import annotations

"""Regime-switching, non-directional DTE 4-14 structure engine.

Every structure is scored at entry by the same second-order option P&L
approximation. At most one fixed-strike structure is held, and it is always
closed on the entry date at or after 15:15.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import monthly_dte4_research as research
import monthly_dte4_structure_hybrid as structures


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "monthly_dte4_research"
OUT = ROOT / "outputs" / "monthly_dte4_multi_strategy"
PRIMARY_BPS = 25.0
LOOKBACK_CYCLES = 6
MIN_VALIDATION_TRADES = 5


# leg tuple = (option type, strike position, position sign); +1 long, -1 short
STRATEGIES: dict[str, list[tuple[str, int, int]]] = {
    "short_atm_straddle": [("call", 0, -1), ("put", 0, -1)],
    "short_1_strangle": [("call", 1, -1), ("put", -1, -1)],
    "long_atm_straddle": [("call", 0, 1), ("put", 0, 1)],
    "long_1_strangle": [("call", 1, 1), ("put", -1, 1)],
    "short_iron_fly_2": [("call", 0, -1), ("put", 0, -1), ("call", 2, 1), ("put", -2, 1)],
    "short_iron_condor_1_2": [("call", 1, -1), ("put", -1, -1), ("call", 2, 1), ("put", -2, 1)],
}


def load_days() -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute(
        "SELECT * FROM read_parquet(?) WHERE days_to_expiry>=4 AND days_to_expiry<15",
        [str(BASE / "oos_predictions.parquet")],
    ).fetchdf()
    con.close()
    return frame


def leg_columns(strategy: str, prefix: str) -> list[str]:
    return [f"{prefix}_{kind}_{pos:+d}" for kind, pos, _ in STRATEGIES[strategy]]


def arrays(frame: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    legs = STRATEGIES[strategy]
    entry = frame[leg_columns(strategy, "price")].to_numpy(float)
    exit_ = frame[leg_columns(strategy, "exit_price")].to_numpy(float)
    gamma = frame[leg_columns(strategy, "gamma")].to_numpy(float)
    theta = frame[leg_columns(strategy, "theta")].to_numpy(float)
    vega = frame[leg_columns(strategy, "vega")].to_numpy(float)
    direction = np.array([sign for _, _, sign in legs], float)
    return entry, exit_, gamma, theta, vega, direction


def normalize_wide(raw_wide: pd.DataFrame) -> pd.DataFrame:
    wide = raw_wide.copy()
    # Explicit aliases make the strategy equations easy to audit.
    for pos in (-2, -1, 0, 1, 2):
        wide[f"price_call_{pos:+d}"] = wide[f"call_price_{pos:+d}"]
        wide[f"price_put_{pos:+d}"] = wide[f"put_price_{pos:+d}"]
        wide[f"exit_price_call_{pos:+d}"] = wide[f"exit_call_price_{pos:+d}"]
        wide[f"exit_price_put_{pos:+d}"] = wide[f"exit_put_price_{pos:+d}"]
        wide[f"delta_call_{pos:+d}"] = wide[f"delta_call_{pos:+d}"]
        wide[f"delta_put_{pos:+d}"] = wide[f"delta_put_{pos:+d}"]
        wide[f"gamma_call_{pos:+d}"] = wide[f"gamma_call_{pos:+d}"]
        wide[f"theta_call_{pos:+d}"] = wide[f"theta_call_{pos:+d}"]
        wide[f"vega_call_{pos:+d}"] = wide[f"vega_call_{pos:+d}"]
        wide[f"gamma_put_{pos:+d}"] = wide[f"gamma_put_{pos:+d}"]
        wide[f"theta_put_{pos:+d}"] = wide[f"theta_put_{pos:+d}"]
        wide[f"vega_put_{pos:+d}"] = wide[f"vega_put_{pos:+d}"]
    return wide


def score_all(days: pd.DataFrame, leg_rows: pd.DataFrame) -> pd.DataFrame:
    wide = normalize_wide(structures.wide_legs(leg_rows))
    frame = days.merge(wide, on=["datetime", "expiry"], how="left", suffixes=("", "_structure"), validate="one_to_one")
    valid_exit = [f"valid_close_exit_{p:+d}" for p in (-2, -1, 0, 1, 2)]
    frame = frame[frame[valid_exit].fillna(False).all(axis=1)].copy()
    outputs = []
    for strategy in STRATEGIES:
        entry, exit_, gamma, theta, vega, direction = arrays(frame, strategy)
        valid = (
            np.isfinite(entry).all(axis=1) & np.isfinite(exit_).all(axis=1)
            & np.isfinite(gamma).all(axis=1) & np.isfinite(theta).all(axis=1) & np.isfinite(vega).all(axis=1)
            & (entry >= 5).all(axis=1) & (exit_ >= 0).all(axis=1)
        )
        z = frame.loc[valid].copy().reset_index(drop=True)
        ent, ex, ga, th, ve = entry[valid], exit_[valid], gamma[valid], theta[valid], vega[valid]
        d = direction[None, :]
        # Stored vega is points per one IV percentage point; pred_d_iv is decimal.
        expected_leg_long = (
            0.5 * ga * z["spot_price"].to_numpy(float)[:, None] ** 2 * z["pred_future_variance"].to_numpy(float)[:, None]
            + th * z["elapsed_days"].to_numpy(float)[:, None]
            + ve * (100.0 * z["pred_d_iv"].to_numpy(float)[:, None])
        )
        expected_points = (d * expected_leg_long).sum(axis=1)
        selection_cost = structures.leg_level_cost(
            ent, ent, direction, z["lot_size"].to_numpy(), z["date"], PRIMARY_BPS
        )["total_cost"].to_numpy()
        z["strategy"] = strategy
        z["expected_gross_rupees"] = expected_points * z["lot_size"].to_numpy(float)
        z["selection_cost_rupees"] = selection_cost
        z["predicted_net_rupees"] = z["expected_gross_rupees"] - selection_cost
        active = z["predicted_net_rupees"].gt(0)
        z = z[active].copy().reset_index(drop=True)
        if z.empty:
            continue
        ent, ex = ent[active.to_numpy()], ex[active.to_numpy()]
        costs = structures.leg_level_cost(
            ent, ex, direction, z["lot_size"].to_numpy(), z["date"], PRIMARY_BPS
        )
        gross_points = ((ex - ent) * direction[None, :]).sum(axis=1)
        for col in costs:
            z[col] = costs[col].to_numpy()
        z["gross_pnl_rupees"] = gross_points * z["lot_size"].to_numpy(float)
        z["net_pnl_rupees"] = z["gross_pnl_rupees"] - z["total_cost"]
        z["profitable"] = z["net_pnl_rupees"].gt(0)
        outputs.append(z)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def daily_max_edge(scored: pd.DataFrame) -> pd.DataFrame:
    chosen = scored.sort_values(["date", "predicted_net_rupees"], ascending=[True, False]).drop_duplicates("date")
    assert not chosen["date"].duplicated().any()
    return chosen.copy()


def rolling_hybrid(scored: pd.DataFrame, cycles: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, decisions = [], []
    for pos, cycle in enumerate(cycles):
        current = scored[scored["outer_test_cycle"].eq(cycle)]
        if pos < LOOKBACK_CYCLES:
            decisions.append({"test_cycle": cycle, "eligible_strategies": "warmup", "trades": 0})
            continue
        history = scored[scored["outer_test_cycle"].isin(cycles[pos - LOOKBACK_CYCLES:pos])]
        stats = history.groupby("strategy").agg(
            trades=("net_pnl_rupees", "size"), net=("net_pnl_rupees", "sum")
        )
        eligible = stats[stats["trades"].ge(MIN_VALIDATION_TRADES) & stats["net"].gt(0)].index.tolist()
        if not eligible:
            decisions.append({"test_cycle": cycle, "eligible_strategies": "none", "trades": 0})
            continue
        day = current[current["strategy"].isin(eligible)].sort_values(
            ["date", "predicted_net_rupees"], ascending=[True, False]
        ).drop_duplicates("date")
        decisions.append({"test_cycle": cycle, "eligible_strategies": "|".join(sorted(eligible)), "trades": len(day)})
        selected.append(day)
    out = pd.concat(selected, ignore_index=True) if selected else scored.iloc[:0].copy()
    assert not out["date"].duplicated().any()
    assert out["datetime"].dt.normalize().eq(out["exit_datetime_structure"].dt.normalize()).all()
    return out, pd.DataFrame(decisions)


def cycle_best_short_structure(scored: pd.DataFrame, cycles: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose straddle or strangle for a full cycle from the prior six cycles."""
    allowed = ["short_atm_straddle", "short_1_strangle"]
    selected, decisions = [], []
    for pos, cycle in enumerate(cycles):
        if pos < LOOKBACK_CYCLES:
            decisions.append({"test_cycle": cycle, "selected_strategy": "warmup_no_trade"})
            continue
        history = scored[
            scored["outer_test_cycle"].isin(cycles[pos - LOOKBACK_CYCLES:pos])
            & scored["strategy"].isin(allowed)
        ]
        stats = history.groupby("strategy").agg(
            trades=("net_pnl_rupees", "size"), net=("net_pnl_rupees", "sum"),
            mean=("net_pnl_rupees", "mean"),
        )
        eligible = stats[stats["trades"].ge(MIN_VALIDATION_TRADES) & stats["net"].gt(0)].sort_values(
            ["net", "mean"], ascending=False
        )
        if eligible.empty:
            decisions.append({"test_cycle": cycle, "selected_strategy": "no_trade"})
            continue
        chosen = eligible.index[0]
        decisions.append({"test_cycle": cycle, "selected_strategy": chosen,
                          "validation_trades": int(eligible.iloc[0].trades),
                          "validation_net": float(eligible.iloc[0].net)})
        selected.append(scored[scored["outer_test_cycle"].eq(cycle) & scored["strategy"].eq(chosen)])
    out = pd.concat(selected, ignore_index=True) if selected else scored.iloc[:0].copy()
    assert not out["date"].duplicated().any()
    return out, pd.DataFrame(decisions)


def reprice_selected(selected: pd.DataFrame, leg_rows: pd.DataFrame, bps: float) -> pd.DataFrame:
    # Re-run the full scorer's realized leg pricing at a different friction,
    # while keeping the 25-bps-selected date/strategy decisions fixed.
    days = load_days()
    primary_scored = score_all(days, leg_rows)
    keys = selected[["datetime", "strategy"]].drop_duplicates()
    base = primary_scored.merge(keys, on=["datetime", "strategy"], how="inner")
    if bps == PRIMARY_BPS:
        return base
    # Adjust only the explicitly modeled slippage component; statutory charges,
    # brokerage and selected positions remain fixed.
    turnover = base["buy_turnover"] + base["sell_turnover"]
    base["total_cost"] = base["total_cost"] - base["slippage"] + turnover * bps / 10000.0
    base["slippage"] = turnover * bps / 10000.0
    base["net_pnl_rupees"] = base["gross_pnl_rupees"] - base["total_cost"]
    base["profitable"] = base["net_pnl_rupees"].gt(0)
    return base


def metrics(name: str, frame: pd.DataFrame) -> dict:
    return {"portfolio": name, **research.performance(frame), **research.cycle_inference(frame)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = []
    for name, legs in STRATEGIES.items():
        registry.append({"strategy": name, "legs": len(legs), "definition": str(legs)})
    pd.DataFrame(registry).to_csv(OUT / "strategy_registry.csv", index=False)
    days = load_days()
    leg_rows, audit = structures.load_legs(days)
    pd.DataFrame([audit]).to_csv(OUT / "data_audit.csv", index=False)
    scored = score_all(days, leg_rows)
    scored.to_csv(OUT / "all_cost_qualified_strategy_trades_25bps.csv", index=False)
    standalone_rows = []
    for strategy, group in scored.groupby("strategy"):
        standalone_rows.append({"strategy": strategy, **research.performance(group), **research.cycle_inference(group)})
    standalone = pd.DataFrame(standalone_rows)
    standalone.to_csv(OUT / "standalone_signals.csv", index=False)
    edge = daily_max_edge(scored)
    cycles = list(pd.Series(days["outer_test_cycle"].unique()).sort_values())
    rolling, decisions = rolling_hybrid(scored, cycles)
    decisions.to_csv(OUT / "cycle_eligibility.csv", index=False)
    rolling.to_csv(OUT / "hybrid_trade_ledger_25bps.csv", index=False)
    short_switch, short_decisions = cycle_best_short_structure(scored, cycles)
    short_decisions.to_csv(OUT / "straddle_strangle_cycle_decisions.csv", index=False)
    short_switch.to_csv(OUT / "straddle_strangle_trade_ledger_25bps.csv", index=False)
    rows = []
    for name, chosen in (
        ("daily_max_predicted_edge", edge),
        ("six_cycle_filtered_hybrid", rolling),
        ("cycle_switch_straddle_strangle", short_switch),
    ):
        for bps in (10.0, 25.0, 50.0):
            z = reprice_selected(chosen, leg_rows, bps)
            rows.append({"cost_bps": bps, **metrics(name, z)})
    portfolio = pd.DataFrame(rows)
    portfolio.to_csv(OUT / "portfolio_performance.csv", index=False)
    lines = [
        "# DTE 4-14 multi-strategy non-directional engine", "",
        "## Mathematical entry rule", "",
        "For every structure j, forecast its entry-time net edge as:", "",
        "`E[PnL_j] = lot * sum_i position_i * (0.5*Gamma_i*S^2*forecast_variance + Theta_i*elapsed_days + Vega_i*100*forecast_delta_IV) - complete_leg_level_cost_j`.", "",
        "A structure is eligible only if this value is positive. When several structures qualify, select the one with maximum predicted net edge. The filtered hybrid additionally requires positive realized aggregate net P&L and at least five signals in the preceding six expiry cycles. There is one position per day, with a fixed strike and a same-day exit at or after 15:15.", "",
        "## Standalone cost-qualified signals at 25 bps", "", standalone.to_string(index=False), "",
        "## Non-overlapping portfolios", "", portfolio.to_string(index=False), "",
        "## Audit", "", pd.DataFrame([audit]).to_string(index=False), "",
        "This is exploratory retrospective evidence. It does not modify the incumbent model and cannot establish executable profitability without historical bid/ask quotes and a frozen forward shadow period.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
