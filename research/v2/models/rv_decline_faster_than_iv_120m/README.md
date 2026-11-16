# RV decline faster than IV over 120 minutes

This project trains a leakage-safe classifier on the combined fixed-strike ATM IV/RV feature set.

The target is positive when, over the next 120 minutes:

1. annualized realized volatility falls by at least 2 volatility points; and
2. realized volatility falls at least 1 volatility point more than fixed-strike combined ATM IV.

The thresholds are configurable in `config.yaml`. A direct probability classifier is more appropriate than subtracting two noisy point forecasts because the requested output is an event.

Train from this directory:

```powershell
py -3.13 train.py --config config.yaml
```

Score a feature CSV:

```powershell
py -3.13 predict.py input_features.csv --model artifacts/winner.joblib --output predictions.csv
```

Only entry-time features with lookback windows of 60 minutes or less are used. The split is chronological, and training rows whose 120-minute labels cross a split boundary are purged.

