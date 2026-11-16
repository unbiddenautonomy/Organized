from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SOURCE = ARTIFACTS / "leakage_safe_classifier_trades_with_costs.csv"
DETAIL = ARTIFACTS / "leakage_safe_classifier_trades_current_costs_no_brokerage.csv"
SUMMARY = ARTIFACTS / "current_cost_no_brokerage_pnl_summary.csv"

# Current NSE equity-options/statutory rates checked on 17 October 2026.
STT_SELL_PREMIUM_RATE = 0.0015
NSE_TRANSACTION_RATE = 0.0003553
SEBI_TURNOVER_RATE = 0.000001
STAMP_BUY_PREMIUM_RATE = 0.00003
GST_RATE = 0.18


def maximum_drawdown(values: pd.Series) -> float:
    equity = values.cumsum()
    return float((equity - equity.cummax().clip(lower=0.0)).min())


def metrics(values: pd.Series) -> dict[str, float | int]:
    values = values.astype(float)
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return {
        "trades": int(len(values)),
        "total_pnl": float(values.sum()),
        "mean_pnl": float(values.mean()),
        "median_pnl": float(values.median()),
        "win_rate_pct": float((values > 0).mean() * 100),
        "profit_factor": gains / losses if losses else np.inf,
        "worst_trade": float(values.min()),
        "best_trade": float(values.max()),
        "maximum_drawdown": maximum_drawdown(values),
    }


def main() -> None:
    trades = pd.read_csv(SOURCE, parse_dates=["datetime"])

    gross = trades["stop_only_pnl_per_lot"].astype(float)
    lots = trades["lot_size"].astype(float)
    entry_turnover = trades["entry_straddle_price"].astype(float) * lots
    exit_premium = trades["entry_straddle_price"].astype(float) - gross / lots
    exit_turnover = exit_premium * lots
    total_turnover = entry_turnover + exit_turnover

    trades["current_brokerage"] = 0.0
    trades["current_stt"] = entry_turnover * STT_SELL_PREMIUM_RATE
    trades["current_nse_transaction_charges"] = total_turnover * NSE_TRANSACTION_RATE
    trades["current_sebi_charges"] = total_turnover * SEBI_TURNOVER_RATE
    trades["current_stamp_duty"] = exit_turnover * STAMP_BUY_PREMIUM_RATE
    trades["current_gst"] = GST_RATE * (
        trades["current_nse_transaction_charges"] + trades["current_sebi_charges"]
    )
    charge_columns = [
        "current_brokerage",
        "current_stt",
        "current_nse_transaction_charges",
        "current_sebi_charges",
        "current_stamp_duty",
        "current_gst",
    ]
    trades["current_explicit_costs_no_brokerage"] = trades[charge_columns].sum(axis=1)
    trades["net_pnl_current_costs_no_brokerage"] = (
        gross - trades["current_explicit_costs_no_brokerage"]
    )

    # Four fills per round trip: two sold at entry and two bought at exit.
    trades["slippage_0.5pt_per_fill"] = 4.0 * 0.5 * lots
    trades["net_pnl_current_costs_slip_0.5pt"] = (
        trades["net_pnl_current_costs_no_brokerage"]
        - trades["slippage_0.5pt_per_fill"]
    )
    trades["slippage_1pt_per_fill"] = 4.0 * lots
    trades["net_pnl_current_costs_slip_1pt"] = (
        trades["net_pnl_current_costs_no_brokerage"]
        - trades["slippage_1pt_per_fill"]
    )

    scenarios = {
        "gross_no_costs": gross,
        "current_exchange_statutory_no_brokerage": trades[
            "net_pnl_current_costs_no_brokerage"
        ],
        "current_costs_plus_0.5pt_per_fill": trades[
            "net_pnl_current_costs_slip_0.5pt"
        ],
        "current_costs_plus_1pt_per_fill": trades[
            "net_pnl_current_costs_slip_1pt"
        ],
    }
    summary = pd.DataFrame(
        [{"scenario": name, **metrics(values)} for name, values in scenarios.items()]
    )

    # Retain the input ledger as the historical-cost audit trail, but do not
    # expose obsolete charge fields in the active current-cost ledger.
    obsolete_cost_columns = [
        "brokerage",
        "stt",
        "exchange_charges",
        "sebi_charges",
        "stamp_duty",
        "gst",
        "explicit_costs",
        "net_pnl_explicit_costs",
        "net_pnl_costs_slip_0.5pt",
        "net_pnl_costs_slip_1pt",
    ]
    trades = trades.drop(
        columns=[column for column in obsolete_cost_columns if column in trades.columns]
    )
    trades.to_csv(DETAIL, index=False)
    summary.to_csv(SUMMARY, index=False)

    daily = pd.DataFrame(
        {
            "date": trades["datetime"].dt.date,
            "gross": gross,
            "net": trades["net_pnl_current_costs_no_brokerage"],
        }
    ).groupby("date", as_index=False).sum()

    print(summary.to_string(index=False))
    print("\nCost totals:")
    print(trades[charge_columns + ["current_explicit_costs_no_brokerage"]].sum())
    print("\nCurrent-cost trade/day signs:")
    print(
        {
            "green_trades": int((trades["net_pnl_current_costs_no_brokerage"] > 0).sum()),
            "red_trades": int((trades["net_pnl_current_costs_no_brokerage"] < 0).sum()),
            "flat_trades": int((trades["net_pnl_current_costs_no_brokerage"] == 0).sum()),
            "green_days": int((daily["net"] > 0).sum()),
            "red_days": int((daily["net"] < 0).sum()),
            "flat_days": int((daily["net"] == 0).sum()),
        }
    )


if __name__ == "__main__":
    main()
