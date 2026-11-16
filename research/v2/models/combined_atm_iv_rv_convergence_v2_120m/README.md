# V2 combined ATM IV to RV models

This project trains one model stack for the combined call-put IV of the strike that was ATM at entry. The strike remains fixed at the exact 120-minute exit.

## Model families

1. Log-HARQ RV regression.
2. LightGBM RV regression.
3. Validation-weighted HARQ and LightGBM RV ensemble.
4. Error-correction Ridge combined-IV-change regression.
5. Monotonic XGBoost combined-IV-change regression.
6. Direct LightGBM combined-IV-change regression.
7. Validation-selected Ridge, XGBoost, and LightGBM point ensemble.
8. LightGBM convergence classifier.
9. LightGBM q10, q50, and q90 IV-change regressions.
10. Standardized-target NGBoost probabilistic regression.

All hyperparameters are selected with expanding walk-forward folds and a 120-minute purge. RV predictions used to construct the IV-to-RV gap are out-of-fold, walk-forward, or held-out. Test data is never used to select hyperparameters or ensemble weights.

```powershell
cd E:\v2\models\combined_atm_iv_rv_convergence_v2_120m
py -3.13 train_all.py --config config.yaml
```

The configured research budget caps ordinary model fits at 12,000 chronological observations and NGBoost fits at 2,000. These caps must not be adjusted after inspecting held-out test metrics.
