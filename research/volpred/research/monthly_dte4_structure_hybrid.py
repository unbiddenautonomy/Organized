from __future__ import annotations

"""Three-structure hybrid for the DTE 4-14 combined short-vol signal.

Structures are fixed at entry and liquidated on the same date at/after 15:15.
The hybrid selects exactly one structure per expiry cycle using only the prior
six cycles, preventing overlapping exposure and future-outcome selection.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import monthly_dte4_research as research


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "monthly_dte4_research"
OUT = ROOT / "outputs" / "monthly_dte4_structure_hybrid"
CHAIN = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
EXITS = ROOT / "outputs" / "intraday_structure_attribution" / "_fixed_strike_exit_prices.parquet"
PRIMARY_BPS = 25.0
LOOKBACK_CYCLES = 6
MIN_VALIDATION_TRADES = 5

STRUCTURES = pd.DataFrame([
    {"structure": "short_atm_straddle", "definition": "short call(0) + short put(0)", "legs": 2},
    {"structure": "short_1_strangle", "definition": "short call(+1) + short put(-1)", "legs": 2},
    {"structure": "iron_condor_1_2", "definition": "short call(+1), short put(-1), long call(+2), long put(-2)", "legs": 4},
])


def load_signal_days() -> pd.DataFrame:
    con = duckdb.connect()
    pred = con.execute(
        "SELECT * FROM read_parquet(?)", [str(BASE / "oos_predictions.parquet")]
    ).fetchdf()
    con.close()
    trades = research.build_trades(pred)
    return trades[
        trades["candidate"].eq("combined_greeks_short")
        & trades["days_to_expiry"].ge(4)
        & trades["days_to_expiry"].lt(15)
    ].copy()


def load_legs(signals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = signals[["datetime", "expiry"]].drop_duplicates().copy()
    con = duckdb.connect()
    con.register("signal_keys", keys)
    query = """
    SELECT c.datetime, c.expiry, c.strike_position, c.source_index, c.strike,
           c.call_price, c.put_price,
           c.delta_call, c.delta_put,
           c.gamma_call, c.theta_call, c.vega_call,
           c.gamma_put, c.theta_put, c.vega_put,
           x.exit_datetime, x.exit_call_price, x.exit_put_price
    FROM read_csv(?, header=true, auto_detect=true) c
    INNER JOIN signal_keys s ON c.datetime=s.datetime AND c.expiry=s.expiry
    LEFT JOIN read_parquet(?) x
      ON c.source_index=x.source_index AND x.horizon='session_end'
    WHERE c.strike_position BETWEEN -2 AND 2
    """
    legs = con.execute(query, [str(CHAIN), str(EXITS)]).fetchdf()
    con.close()
    legs["entry_date"] = legs["datetime"].dt.normalize()
    legs["exit_date"] = legs["exit_datetime"].dt.normalize()
    exit_minute = legs["exit_datetime"].dt.hour * 60 + legs["exit_datetime"].dt.minute
    legs["valid_close_exit"] = legs["exit_date"].eq(legs["entry_date"]) & exit_minute.ge(915)
    audit = {
        "signal_days": int(keys.shape[0]),
        "leg_rows": int(len(legs)),
        "signal_days_with_all_five_strikes": int(legs.groupby("datetime")["strike_position"].nunique().ge(5).sum()),
        "signal_days_with_all_valid_close_exits": int(legs.groupby("datetime")["valid_close_exit"].all().sum()),
    }
    return legs, audit


def wide_legs(legs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_index", "strike", "call_price", "put_price",
        "delta_call", "delta_put",
        "gamma_call", "theta_call", "vega_call", "gamma_put", "theta_put", "vega_put",
        "exit_call_price", "exit_put_price", "valid_close_exit",
    ]
    wide = legs.pivot_table(index=["datetime", "expiry", "exit_datetime"], columns="strike_position", values=cols, aggfunc="first")
    wide.columns = [f"{name}_{int(pos):+d}" for name, pos in wide.columns]
    return wide.reset_index()


def historical_rates(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    d = pd.to_datetime(dates)
    stt = np.where(d < pd.Timestamp("2024-10-01"), .000625, .001)
    exchange = np.where(d < pd.Timestamp("2024-04-01"), .0005,
                        np.where(d < pd.Timestamp("2024-10-01"), .000495, .0003503))
    return stt, exchange


def leg_level_cost(
    entry: np.ndarray, exit_: np.ndarray, directions: np.ndarray,
    lots: np.ndarray, dates: pd.Series, bps: float,
) -> pd.DataFrame:
    entry = np.asarray(entry, float)
    exit_ = np.asarray(exit_, float)
    directions = np.asarray(directions, float)
    lots = np.asarray(lots, float)[:, None]
    buy = np.where(directions[None, :] > 0, entry, exit_) * lots
    sell = np.where(directions[None, :] > 0, exit_, entry) * lots
    buy_turnover, sell_turnover = buy.sum(axis=1), sell.sum(axis=1)
    turnover = buy_turnover + sell_turnover
    stt_rate, exchange_rate = historical_rates(dates)
    brokerage = np.full(len(entry), 40.0 * entry.shape[1])  # entry+exit, Rs20 per leg/order
    stt = np.rint(sell_turnover * stt_rate)
    exchange = turnover * exchange_rate
    sebi, ipft = turnover * .000001, turnover * .000000001
    stamp = buy_turnover * .00003
    gst = .18 * (brokerage + exchange + sebi + ipft)
    slippage = turnover * bps / 10000.0
    return pd.DataFrame({
        "brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi,
        "ipft": ipft, "stamp": stamp, "gst": gst, "slippage": slippage,
        "total_cost": brokerage + stt + exchange + sebi + ipft + stamp + gst + slippage,
        "buy_turnover": buy_turnover, "sell_turnover": sell_turnover,
    })


def structure_arrays(frame: pd.DataFrame, structure: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if structure == "short_atm_straddle":
        entry = frame[["call_price_+0", "put_price_+0"]].to_numpy(float)
        exit_ = frame[["exit_call_price_+0", "exit_put_price_+0"]].to_numpy(float)
        direction = np.array([-1.0, -1.0])
    elif structure == "short_1_strangle":
        entry = frame[["call_price_+1", "put_price_-1"]].to_numpy(float)
        exit_ = frame[["exit_call_price_+1", "exit_put_price_-1"]].to_numpy(float)
        direction = np.array([-1.0, -1.0])
    elif structure == "iron_condor_1_2":
        entry = frame[["call_price_+1", "put_price_-1", "call_price_+2", "put_price_-2"]].to_numpy(float)
        exit_ = frame[["exit_call_price_+1", "exit_put_price_-1", "exit_call_price_+2", "exit_put_price_-2"]].to_numpy(float)
        direction = np.array([-1.0, -1.0, 1.0, 1.0])
    else:
        raise ValueError(structure)
    return entry, exit_, direction


def price_structures(signals: pd.DataFrame, legs: pd.DataFrame, bps: float) -> pd.DataFrame:
    wide = wide_legs(legs)
    frame = signals.merge(wide, on=["datetime", "expiry"], how="left", suffixes=("", "_structure"), validate="one_to_one")
    valid_cols = [f"valid_close_exit_{p:+d}" for p in (-2, -1, 0, 1, 2)]
    frame = frame[frame[valid_cols].fillna(False).all(axis=1)].copy()
    priced = []
    for structure in STRUCTURES["structure"]:
        entry, exit_, directions = structure_arrays(frame, structure)
        valid = np.isfinite(entry).all(axis=1) & np.isfinite(exit_).all(axis=1) & (entry >= 5).all(axis=1) & (exit_ >= 0).all(axis=1)
        z = frame.loc[valid].copy().reset_index(drop=True)
        ent, ex = entry[valid], exit_[valid]
        gross_points = ((ex - ent) * directions[None, :]).sum(axis=1)
        costs = leg_level_cost(ent, ex, directions, z["lot_size"].to_numpy(), z["date"], bps)
        for col in costs:
            z[col] = costs[col].to_numpy()
        z["structure"] = structure
        z["gross_pnl_rupees"] = gross_points * z["lot_size"].to_numpy(float)
        z["net_pnl_rupees"] = z["gross_pnl_rupees"] - z["total_cost"]
        z["profitable"] = z["net_pnl_rupees"].gt(0)
        z["cost_bps"] = bps
        priced.append(z)
    return pd.concat(priced, ignore_index=True)


def standalone_summary(all_costs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (structure, bps), group in all_costs.groupby(["structure", "cost_bps"]):
        rows.append({"structure": structure, "cost_bps": bps, **research.performance(group),
                     **research.cycle_inference(group)})
    return pd.DataFrame(rows)


def hybrid(primary: pd.DataFrame, all_cycles: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, decisions = [], []
    for pos, cycle in enumerate(all_cycles):
        if pos < LOOKBACK_CYCLES:
            decisions.append({"test_cycle": cycle, "structure": "warmup_no_trade"})
            continue
        history_cycles = all_cycles[pos - LOOKBACK_CYCLES:pos]
        history = primary[primary["outer_test_cycle"].isin(history_cycles)]
        score = history.groupby("structure").agg(
            trades=("net_pnl_rupees", "size"), net=("net_pnl_rupees", "sum"),
            mean=("net_pnl_rupees", "mean"),
        )
        eligible = score[score["trades"].ge(MIN_VALIDATION_TRADES) & score["net"].gt(0)].sort_values(
            ["net", "mean"], ascending=False
        )
        if eligible.empty:
            decisions.append({"test_cycle": cycle, "structure": "no_trade"})
            continue
        chosen = eligible.index[0]
        decisions.append({"test_cycle": cycle, "structure": chosen,
                          "validation_trades": int(eligible.iloc[0].trades),
                          "validation_net": float(eligible.iloc[0].net)})
        selected.append(primary[primary["outer_test_cycle"].eq(cycle) & primary["structure"].eq(chosen)])
    out = pd.concat(selected, ignore_index=True) if selected else primary.iloc[:0].copy()
    assert not out["date"].duplicated().any()
    assert out["datetime"].dt.normalize().eq(out["exit_datetime_structure"].dt.normalize()).all()
    return out, pd.DataFrame(decisions)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STRUCTURES.to_csv(OUT / "structure_registry.csv", index=False)
    signals = load_signal_days()
    legs, audit = load_legs(signals)
    pd.DataFrame([audit]).to_csv(OUT / "data_audit.csv", index=False)
    by_cost = [price_structures(signals, legs, bps) for bps in (10.0, 25.0, 50.0)]
    all_costs = pd.concat(by_cost, ignore_index=True)
    summary = standalone_summary(all_costs)
    summary.to_csv(OUT / "standalone_performance.csv", index=False)
    primary = all_costs[all_costs["cost_bps"].eq(PRIMARY_BPS)].copy()
    cycles = list(pd.Series(signals["outer_test_cycle"].unique()).sort_values())
    hybrid_primary, decisions = hybrid(primary, cycles)
    decisions.to_csv(OUT / "hybrid_cycle_decisions.csv", index=False)
    hybrid_primary.to_csv(OUT / "hybrid_trade_ledger_25bps.csv", index=False)
    hybrid_rows = []
    chosen_keys = hybrid_primary[["datetime", "structure"]].drop_duplicates()
    for bps in (10.0, 25.0, 50.0):
        z = all_costs[all_costs["cost_bps"].eq(bps)].merge(chosen_keys, on=["datetime", "structure"], how="inner")
        hybrid_rows.append({"portfolio": "past_only_structure_selector", "cost_bps": bps,
                            **research.performance(z), **research.cycle_inference(z)})
    hybrid_perf = pd.DataFrame(hybrid_rows)
    hybrid_perf.to_csv(OUT / "hybrid_performance.csv", index=False)
    lines = [
        "# DTE 4-14 three-structure hybrid",
        "",
        "All positions are fixed-strike, intraday-only and exit at or after 15:15. The incumbent forecasting models were not modified.",
        "",
        "## Structures", "", STRUCTURES.to_string(index=False), "",
        "## Standalone results", "", summary.to_string(index=False), "",
        "## Non-overlapping hybrid", "", hybrid_perf.to_string(index=False), "",
        "## Mathematical selector", "",
        f"For cycle k, use only the preceding {LOOKBACK_CYCLES} expiry cycles. Require at least {MIN_VALIDATION_TRADES} trades and positive 25-bps aggregate net P&L. Select the eligible structure with maximum validation net P&L, breaking ties by mean net P&L. If none qualifies, do not trade. Exactly one structure is permitted per cycle and one position per day.",
        "",
        "## Audit", "", pd.DataFrame([audit]).to_string(index=False), "",
        "This is exploratory historical evidence, not an untouched validation. Bid/ask quotes are unavailable, so execution is represented by historical statutory charges plus explicit 10/25/50 bps turnover friction.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
