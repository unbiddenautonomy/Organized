# Frozen Models 2026 Unseen Holdout Results

All model families, horizons, and thresholds were frozen before loading 2026. Models were refit on all available development rows through May 2025 and evaluated once on January-June 2026.

Entry-time features: 66.

## Classifier models

| Horizon | Model | Threshold | Trades | Win rate | Net points | Profit factor | Max drawdown | Wilson lower 95% |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30m | extra_trees_classifier | 0.55 | 163 | 52.76% | 216.9 | 1.16 | -196.1 | 45.12% |
| 60m | lightgbm_classifier | 0.75 | 48 | 54.17% | 10.5 | 1.02 | -328.3 | 40.29% |
| 90m | extra_trees_classifier | 0.60 | 7 | 42.86% | 76.6 | 3.00 | -20.8 | 15.82% |
| 120m | lightgbm_classifier | 0.80 | 37 | 45.95% | -563.3 | 0.49 | -719.2 | 31.04% |
| 180m | logistic_c01 | 0.75 | 1 | 100.00% | 14.8 | inf | 0.0 | 20.65% |

## Frozen hybrid 90 minute model

The frozen hybrid used probability threshold 0.70 and predicted net-edge threshold 5.0 points.

It produced **no qualifying trades** in the 2026 holdout. Win rate, profit factor, and drawdown are therefore not defined.

The 2026 data was not used for model, feature, horizon, or threshold selection.
