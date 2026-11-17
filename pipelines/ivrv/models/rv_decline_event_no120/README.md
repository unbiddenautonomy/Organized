# RV versus IV 120-minute event model

This package contains the best no-120-minute-input model and everything needed for batch inference. The model forecasts whether realized volatility will decline sharply relative to fixed-strike combined ATM implied volatility during the next 120 minutes.

## Install

Python 3.13 was used for training and verification.

```powershell
py -3.13 -m pip install -r requirements.txt
```

## Predict

```powershell
py -3.13 predict.py path\to\features.csv --output predictions.csv
```

`winner.joblib` contains both fitted ElasticNet estimators, their median imputers, their fitted `StandardScaler` objects, the ordered feature list, the decision threshold, and the target settings. No separate scaler file is required.

The output includes the signed event score, the Boolean prediction, and the predicted 120-minute RV and IV changes. A score at or above zero produces a positive event prediction.

See `MODEL_CARD.md` for the target, evaluation, limitations, and leakage controls. `input_features.json` contains the exact ordered model input schema. Use `MANIFEST.sha256` to verify package files.
