# 120-minute RV improvement study

## Outcome

The full leakage-safe search selected a regularized depth-3 XGBoost model. On the untouched test period it achieved:

| Model | QLIKE | MAE | RMSE |
|---|---:|---:|---:|
| Persistence | 0.250183 | 0.032130 | 0.042822 |
| Corrected log-HAR | 0.197628 | 0.030694 | 0.038577 |
| Retuned LightGBM | 0.169820 | 0.025176 | 0.033594 |
| Retuned XGBoost | **0.166675** | 0.024103 | **0.033127** |
| Validation-optimized blend | 0.187008 | 0.029065 | 0.036812 |

The previous checked-in LightGBM scored 0.167739 QLIKE, so the new XGBoost result is a modest 0.63% QLIKE improvement. The previous ensemble had lower MAE (0.022237), however, so this is an improvement for the primary QLIKE objective rather than a universal win across every loss.

The validation-optimized blend concentrated on log-HAR and did not transfer to the held-out regime. It is retained for auditability but should not replace the individual XGBoost forecast.

## What changed

- Increased the chronological training cap from 12,000 to 60,000 rows.
- Expanded walk-forward cross-validation from three to four purged folds.
- Forecast log variance and invert it to positive volatility.
- Added safe log-RV, log-RQ, absolute-return, and IV-minus-RV features.
- Tuned five Ridge, eight LightGBM, and six XGBoost configurations.
- Tested L1, L2, Huber, Fair, and pseudo-Huber objectives where supported.
- Estimated multiplicative bias calibration and blend weights using validation data only.
- Kept the final 25,381-row test block untouched until all choices were frozen.

## Research rationale

- HAR/HARQ remains a serious benchmark because realized quarticity can proxy time-varying RV measurement error; the HARQ paper reports forecast improvements from modeling this effect: <https://public.econ.duke.edu/~get/browse/courses/pbc/Readings/BollerslevPattonQuaedvliegJoE2016.pdf>
- A broad 2024 comparison finds no general evidence that nonlinear ML always beats simpler linear volatility models, supporting side-by-side evaluation rather than assuming a complex winner: <https://doi.org/10.1016/j.jempfin.2024.101524>
- LightGBM officially supports robust regression objectives including L1, Huber, Fair, Gamma, and Tweedie: <https://github.com/lightgbm-org/LightGBM/blob/main/docs/Parameters.rst>
- XGBoost officially supports squared-error and pseudo-Huber regression objectives: <https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst>

Pseudo-Huber was numerically unsuitable for this dataset and target representation, producing extremely poor CV QLIKE. That result is recorded in `metrics.json` rather than silently discarded.

## Reproduce

```powershell
cd E:\v2\models\fixed_strike_iv_to_rv_convergence_120m
py -3.13 improve_rv.py --config rv_improvement_full_config.yaml
```

Production candidate: `artifacts_rv_improved_full/xgboost.joblib`.

The saved payload contains `model`, `features`, `params`, and `calibration`. Raw model output is log variance, so inference must apply `sqrt(exp(raw_prediction)) * calibration`.
