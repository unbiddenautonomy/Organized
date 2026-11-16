# banknifty_fixed_atm_120m_lightgbm_classifier

## Purpose

Predict whether a fixed-signal-ATM BANKNIFTY short strangle will produce positive net premium decay over 120 minutes after two points of modeled round-trip cost.

## Frozen configuration

- Model: lightgbm_classifier
- Horizon: 120 minutes
- Win-probability threshold: 0.80
- Entry strike: ATM at signal time and fixed through exit
- Execution: next one-minute full-chain quote after the signal snapshot
- Development data end: 2025-05-08 15:27:59
- Unseen holdout: 2026-01-01 09:16:00 through 2026-07-02 15:30:00
- Feature count: 66

## 2026 unseen holdout

- Trades: 37
- Win rate: 45.95%
- Net points: -563.29
- Profit factor: 0.49366735880771934
- Maximum drawdown points: -719.190000000001
- 95% Wilson win-rate lower bound: 31.04%

## Preprocessing

Use `pipeline.joblib` or `model.joblib` for inference. The fitted preprocessor is provided separately in `preprocessor.joblib`. `scaler.joblib` contains the fitted scaler when one is used; for tree models it records that scaling is not required.

## Limitations

- Statutory charges and bid/ask spreads are not included.
- Results use minute-close execution with two points of modeled round-trip cost.
- Exact fixed-strike quote availability can reduce sample coverage.
- The 2026 source uses synthetic futures when a futures quote is unavailable.
- This artifact is research-only and is not approved for live deployment.
