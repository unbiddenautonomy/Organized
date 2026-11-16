# BANKNIFTY fixed-ATM 30-minute tail-veto overlay (experimental)

## Goal

Increase the win rate and reduce tail losses of the 30-minute fixed-ATM short-strangle entry model without changing the strike between signal time and t+30 minutes.

## Construction

- Entry model: frozen Extra Trees classifier, probability threshold 0.55.
- Risk overlay: soft-voting tail-loss classifier (logistic, Extra Trees, histogram gradient boosting).
- Veto rule: reject a candidate when its tail-risk score is at or above the rolling 80th percentile of the prior 1,000 scores.
- Timing: the current score is appended only after the decision, so the threshold is backward-looking.
- Horizon: 30 minutes.
- Strike: signal-time ATM CE/PE strike, fixed through exit.

## Diagnostic comparison on 2026

- Base: 163 trades, 52.76% wins, +216.9 net points, 1.16 profit factor, -196.1 max drawdown.
- Overlay: 152 trades, 55.92% wins, +287.85 net points, 1.24 profit factor, -176.45 max drawdown.

## Leakage and validation status

The sequential veto calculation does not use future scores. However, 2026 had already been inspected before this overlay evaluation, so this comparison is **not a fresh holdout** and must not be treated as confirmed out-of-sample improvement. Validate once on a later untouched period before deployment.

## Files

- `model.joblib`: complete composite artifact and rolling-state seed.
- `entry_pipeline.joblib`, `tail_veto_pipeline.joblib`: fitted component pipelines.
- `preprocessor.joblib`, `scaler.joblib`, `estimators.joblib`: separated fitted components.
- `predict.py`: stateful sequential inference helper.
- `feature_manifest.json`, `scaler_metadata.json`: input and scaling documentation.
- `metrics.json`, trade CSVs, and `EVALUATION_REPORT.md`: diagnostic evidence.

Research-only; costs model two points of round-trip slippage and excludes some real-world charges/spread effects.
