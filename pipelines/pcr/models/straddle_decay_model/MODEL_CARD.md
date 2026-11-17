# Model Card: Straddle Decay Model

Selected family: `ridge`.

Target: `net_points_60`.

## Data separation

Family selection uses 2021 only after fitting earlier observations. Final regressors end before 2023. Final classifiers end before 2022, use 2022 only for probability calibration, and use 2023 only for expiry-week thresholds. January 2024 onward is the locked historical test.

## Contract integrity

Entry features are lagged by one minute. The ATM expiry and strike selected at entry remain fixed through the 60- or 120-minute exit. Costs include historical lot sizes, statutory charges, and two straddle points of slippage.

## Limitations

The source contains five strikes, so absent exact exit strikes use a transparent linear strike-surface valuation with an intrinsic-value floor. Historical backtest performance is not a guarantee of live performance.
