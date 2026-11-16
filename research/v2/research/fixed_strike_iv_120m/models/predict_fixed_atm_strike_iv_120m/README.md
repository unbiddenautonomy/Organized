# Predict fixed entry-ATM strike IV at t+120 minutes

Goal: predict call and put IV at `t+120` for the exact strike that was ATM at entry `t`.

The suite trains and tunes:

- HAR-style Ridge
- Elastic Net
- Histogram Gradient Boosting
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- HAR + LightGBM residual hybrid
- LightGBM 10th/50th/90th conditional quantiles

Leakage controls:

- Features come only from the explicit `feature_manifest.json` whitelist.
- `exit_`, `label_`, and `quality_` columns are rejected.
- Hyperparameter folds are expanding and purged by the actual label availability time.
- Hyperparameters use training data only.
- Family selection uses validation only.
- Test data is evaluated once after selection.

Models predict

```text
log(IV(K_t,t+120) / IV(K_t,t))
```

and are scored after conversion back to future IV. This preserves positive IV and reduces scale differences between calm and high-volatility periods.

Run:

```powershell
py -3.13 train.py --config config.yaml
```

Outputs include every fitted family, target-specific winners, full test predictions, quantile intervals, tuning history, a leaderboard, and a Markdown report.

## Completed run

The full run completed successfully on 2026-10-16. Histogram Gradient Boosting was selected on validation data for both targets.

| Target | Test MAE | Vol points | Test R² | Improvement vs persistence | Improvement vs median-ratio baseline |
|---|---:|---:|---:|---:|---:|
| Call IV | 0.007238 | 0.724 | 0.7567 | 1.78% | 0.12% |
| Put IV | 0.006782 | 0.678 | 0.7937 | 3.78% | 0.25% |

The nominal 80% LightGBM quantile intervals cover 73.51% of call outcomes and 72.76% of put outcomes, so interval calibration is required before using them as risk limits. Balanced direction accuracy is close to 50%, which means the models are much better at forecasting the future IV level than the sign of the IV move.

