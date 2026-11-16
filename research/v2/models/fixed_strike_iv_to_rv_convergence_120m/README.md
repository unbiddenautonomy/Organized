# Tuned 120-minute fixed-strike IV → RV convergence models

Goal: estimate whether and by how much the strike that is ATM at entry moves toward an out-of-sample forecast of exact 120-minute realized volatility.

The suite trains and tunes:

1. Log-HARQ realized-volatility regression.
2. LightGBM realized-volatility regression.
3. Validation-weighted HARQ + LightGBM RV ensemble.
4. Error-correction ridge models for call and put fixed-strike IV changes.
5. Monotonic XGBoost call and put IV-change models.
6. Calibrated LightGBM call and put convergence classifiers.
7. LightGBM 10th/50th/90th percentile IV-change models.
8. NGBoost probabilistic call and put IV-change models.

All searches use expanding walk-forward folds with a 120-minute purge. Rows sharing a timestamp remain in the same partition. RV predictions used by convergence models are out-of-fold, walk-forward, or held-out predictions; in-sample RV predictions are never used.

Run after the full feature table has been built:

```powershell
cd E:\v2\models\fixed_strike_iv_to_rv_convergence_120m
py -3.13 train_all.py --config config.yaml
```

Outputs under `artifacts/` include fitted models, hyperparameter searches, test predictions, and a consolidated Markdown/JSON report. No model is called production-ready unless it beats its simple baseline on the untouched test period.

The checked-in configuration uses a bounded research budget: up to 12,000 chronological rows for standard models, 2,000 for NGBoost, and two worker threads. This keeps every multi-candidate, three-fold search reproducible on a shared workstation. Increase these caps only in a separate sensitivity run; do not tune them against the held-out test metrics.

The checked-in configuration intentionally uses a bounded research budget: up to 12,000 chronological rows for standard models, 2,000 for NGBoost, and two worker threads. This keeps all multi-candidate, three-fold searches reproducible on a shared workstation. Increase these caps only in a separate sensitivity run; do not tune them against the held-out test metrics.

