# Rising-RV convergence model replay: 29 October 2026

The `iv_gt_rv_120m_no_120m_features` LightGBM artifact was loaded by the existing
IV/RV inference pipeline as a separate score. It did not replace the current
convergence classifier. The live runtime now has a separate long-straddle paper
book and dashboard panel for this signal. The new score uses
the original 19 training inputs and requires mean ATM call/put IV to exceed
trailing 120-minute spot RV. Volatility inputs are annualized percentage points.

## Replay inputs and signal

- Source: `analysis/Data/banknifty_atm5_minute.csv` and `banknifty_future.csv`.
- The replay uses future LTP as its spot proxy and bid/ask midpoint to calculate IV.
- 315 warmed prediction rows; 257 eligible rows; 65 signals at probability >= 0.5.
- Quality gate passed on all 65 signals. The model's reported validation Brier
  score is 0.281, so 0.5 is a provisional threshold rather than a calibrated
  trading cutoff.

## P&L

The current IV/RV paper method enters at the next minute, limits each book to
three positions, and exits at a ₹1,500 target, -₹750 stop, 120-minute horizon,
or square-off. The five-strike minute file loses the fixed entry strike for all
three entered positions before their path exit can be observed. It therefore
provides **no realized stop/target P&L** for this model. See
`rising_rv_pnl_summary.json` and `rising_rv_skipped.csv`.

The source also saves fixed-strike quotes at exactly 120 minutes for each entry
row. Using the next-minute entry row and its saved 120-minute exit quote, with
the same three-position cap and 30-unit lot, gives this **horizon-only** result:

| Position | Trades | Gross P&L | Configured charges | Net P&L |
|---|---:|---:|---:|---:|
| Short ATM straddle | 3 | -₹2,233.50 | ₹357.70 | **-₹2,591.20** |
| Long ATM straddle | 3 | ₹1,425.00 | ₹360.45 | **₹1,064.55** |

Entries after 13:29 are excluded from this horizon-only book because a full
120-minute exit would fall after the 15:29 quote feed. Charges use the existing
IV/RV backtest rates and zero brokerage. This result does not enforce the stop,
target, or square-off, and is based on a single day with proxy spot and only
three entries per side. It is a diagnostic, not a deployable P&L estimate.

To reproduce, run `run_previous_day_ivrv.py` with
`IVRV_OUTPUT_DIR=rising_rv_120m_2026-10-29`, then run
`replay_rising_rv_120m.py` with the same environment variable.
