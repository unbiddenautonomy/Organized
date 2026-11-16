from __future__ import annotations

from pathlib import Path

import pandas as pd

from hypothesis_backtest import first_barrier


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    trades = pd.read_csv(
        ARTIFACTS / "primary_hypothesis_trades.csv",
        parse_dates=["datetime", "exit_datetime", "expiry"],
    ).sort_values(["datetime", "row_id"])
    paths = pd.read_csv(
        ARTIFACTS / "same_strike_intraperiod_prices.csv",
        parse_dates=["datetime", "expiry"],
    )
    grouped = {
        key: group.sort_values("datetime")
        for key, group in paths.groupby(["Date", "dte_flag", "expiry", "strike"], sort=False)
    }
    rows = []
    for stop in (400.0, 500.0, 600.0, 750.0, 900.0, 1000.0, 1250.0):
        candidates = []
        for trade in trades.itertuples(index=False):
            key = (str(trade.Date), trade.dte_flag, trade.expiry, trade.entry_strike)
            path = grouped[key]
            path = path.loc[(path["datetime"] > trade.datetime) & (path["datetime"] <= trade.exit_datetime)]
            timestamp, reason, pnl = first_barrier(
                path, trade.entry_straddle_price, int(trade.lot_size), stop, None, 0.0
            )
            candidates.append({
                "datetime": trade.datetime,
                "exit": timestamp if timestamp is not None else trade.exit_datetime,
                "pnl": pnl if pnl is not None else trade.horizon_pnl_per_lot,
                "reason": reason if reason is not None else "time_exit",
            })
        candidate = pd.DataFrame(candidates)
        selected = []
        available = pd.Timestamp.min
        for trade in candidate.itertuples():
            if trade.datetime >= available:
                selected.append(trade.Index)
                available = trade.exit
        result = candidate.loc[selected]
        pnl = result["pnl"]
        equity = pnl.cumsum()
        drawdown = equity - equity.cummax()
        rows.append({
            "stop_per_lot": stop, "trades": len(result), "net_pnl": pnl.sum(),
            "mean_pnl": pnl.mean(), "loss_rate": (pnl < 0).mean(),
            "stop_rate": result["reason"].eq("stop_loss").mean(),
            "average_stopped_loss": result.loc[result["reason"].eq("stop_loss"), "pnl"].mean(),
            "profit_factor": pnl[pnl > 0].sum() / -pnl[pnl < 0].sum(),
            "maximum_drawdown": drawdown.min(),
        })
    output = pd.DataFrame(rows)
    output.to_csv(ARTIFACTS / "stop_sensitivity.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
