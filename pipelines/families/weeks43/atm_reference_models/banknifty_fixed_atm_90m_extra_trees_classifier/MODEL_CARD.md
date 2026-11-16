# banknifty_fixed_atm_90m_extra_trees_classifier

## Purpose

Predict whether a fixed-signal-ATM BANKNIFTY short strangle will produce positive net premium decay over 90 minutes after two points of modeled round-trip cost.

## Frozen configuration

- Model: extra_trees_classifier
- Horizon: 90 minutes
- Win-probability threshold: 0.60
- Entry strike: ATM at signal time and fixed through exit
- Execution: next one-minute full-chain quote after the signal snapshot
- Development data end: 2025-05-08 15:27:59
- Unseen holdout: 2026-01-01 09:16:00 through 2026-07-02 15:30:00
- Feature count: 66

## 2026 unseen holdout

- Trades: 7
- Win rate: 42.86%
- Net points: 76.60
- Profit factor: 2.999999999999988
- Maximum drawdown points: -20.799999999999955
- 95% Wilson win-rate lower bound: 15.82%

## Preprocessing

Use `pipeline.joblib` or `model.joblib` for inference. The fitted preprocessor is provided separately in `preprocessor.joblib`. `scaler.joblib` contains the fitted scaler when one is used; for tree models it records that scaling is not required.

## Limitations

- Statutory charges and bid/ask spreads are not included.
- Results use minute-close execution with two points of modeled round-trip cost.
- Exact fixed-strike quote availability can reduce sample coverage.
- The 2026 source uses synthetic futures when a futures quote is unavailable.
- This artifact is research-only and is not approved for live deployment.
