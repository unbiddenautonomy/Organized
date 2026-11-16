# banknifty_fixed_atm_90m_high_win_rate_hybrid

## Purpose

Combine a 90-minute positive-outcome classifier with a net-decay regressor. A trade requires both the frozen win-probability and predicted-edge gates.

## Frozen configuration

- Classifier: Extra Trees classifier
- Regressor: LightGBM Huber regressor
- Horizon: 90 minutes
- Win-probability threshold: 0.70
- Predicted net-edge threshold: 5.0 points
- Entry strike: ATM at signal time and fixed through exit
- Execution: next one-minute full-chain quote after the signal snapshot
- Development data end: 2025-05-08 15:27:59
- Unseen holdout: 2026-01-01 09:16:00 through 2026-07-02 15:30:00
- Feature count: 66

## 2026 unseen holdout

- Trades: 0
- Net points: 0.00
- Result: No qualifying trades; win rate and profit factor are undefined.

## Preprocessing

The classifier and regressor pipelines are embedded in `model.joblib`. Their fitted preprocessors and estimators are also supplied separately. Neither tree model requires feature scaling; the scaler artifacts record this explicitly.

## Limitations

- The frozen hybrid did not trigger in the unseen 2026 holdout.
- Statutory charges and bid/ask spreads are not included.
- Exact fixed-strike quote availability can reduce sample coverage.
- The 2026 source uses synthetic futures when a futures quote is unavailable.
- This artifact is research-only and is not approved for live deployment.
