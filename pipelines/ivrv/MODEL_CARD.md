# IV/RV live inference model card

The live `ivrv` worker combines three forecasting model families at a fixed BANKNIFTY entry strike and 120-minute horizon:

| Family | Saved artifacts | Prediction |
| --- | --- | --- |
| Combined IV | Seven `combined_*.joblib` artifacts under `../models/combined_atm_iv_rv_convergence_v2_120m/artifacts` | Direct combined IV change, corrected estimates, convergence classification, and uncertainty estimates |
| Separate call/put IV | Eight `call_*.joblib` and `put_*.joblib` artifacts under `../research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m` | Fixed-strike leg IV log ratios and quantiles |
| Forward RV | `../models/fixed_strike_iv_to_rv_convergence_120m/artifacts_rv_improved_full/xgboost.joblib` | Calibrated annualized forward 120-minute realized volatility |

The exact saved feature names and order are in `model_feature_lists.csv`; the live runtime loads the serialized feature lists from each artifact and enforces a strict feature contract. The combined IV, leg IV, and RV models use 151, 143, and 168 features respectively in their active primary forecasts. The supplementary sharp RV decline model is in `../../rv_iv_no120_best_model_package` and has its own model card.

The entry strike and expiry remain fixed for the forecast horizon. The RV artifact needs its saved inverse target transformation and calibration; the leg IV artifacts need their saved log-ratio inverse transformation. `model.py` applies these transforms and combines the predictions.
