# Model card: sharper RV decline versus IV

## Model details

- Model name: `elastic_a0.001_l10.5_two_target`
- Architecture: two independent ElasticNet regressions
- Regularization: alpha `0.001`, L1 ratio `0.5`
- Preprocessing per target: median imputation followed by a fitted `StandardScaler`
- Input count: 168
- Maximum input lookback: 60 minutes
- Input names containing `120m`: zero
- Training period: October 2019 through May 2025
- Random seed: 42

Both preprocessing pipelines and regression estimators are embedded in `winner.joblib`.

## Intended prediction

The model predicts a positive event when, over the next 120 minutes:

1. Realized volatility declines by at least `0.02`.
2. Realized volatility declines at least `0.01` more than fixed-strike combined ATM implied volatility.

The model first predicts the future changes in RV and IV. It then calculates the minimum margin by which those two conditions are satisfied. The saved decision threshold is zero.

The prediction horizon is 120 minutes, but the model does not consume any feature containing `120m`. The future 120-minute columns were used only to construct training labels and evaluation outcomes.

## Model selection

Thirteen specifications were compared using chronologically separated historical validation data: ordinary linear regression, Ridge, ElasticNet, logistic regression, Random Forest, Extra Trees, and LightGBM classifier and regression variants. ElasticNet had the best validation accuracy after historical threshold fitting at 78.39% and the best validation ROC AUC at 0.8499.

The released operating threshold is the signed regression margin's semantic boundary of zero. This avoids carrying a period-specific fitted offset into later market regimes.

## Evaluation

On 13,045 eligible rows from the 2026 out-of-time comparison period:

| Metric | Value |
|---|---:|
| Accuracy | 80.72% |
| Balanced accuracy | 81.81% |
| Precision | 61.97% |
| Recall | 84.34% |
| F1 | 71.44% |
| ROC AUC | 0.9038 |
| Average precision | 0.8027 |

Confusion matrix: 7,384 true negatives, 1,931 false positives, 584 false negatives, and 3,146 true positives.

The non-overlapping 120-minute sample contained 205 rows and reached 91.71% accuracy. Because this smaller sample has much wider uncertainty, the full-period 80.72% result is the primary reported accuracy.

## Leakage controls

- Inputs are restricted to contemporaneous, lagged, or trailing measurements with no lookback longer than 60 minutes.
- All columns beginning with future-derived target or exit prefixes are rejected from the input list.
- The saved bundle is checked for any feature name containing `120m`.
- The chronological training/validation split purges labels whose 120-minute outcome crosses the boundary.
- Imputers and scalers are fitted inside each model pipeline, using training data only during validation.

The 2026 period was examined in earlier project iterations. Its metrics are useful for comparative out-of-time analysis, but it is no longer a completely untouched final test. Reserve a later unseen period for a final deployment estimate.

## Appropriate use and limitations

Use this model as a research signal on data generated with the same feature definitions and units as the training pipeline. It is not a trading strategy by itself and does not account for transaction costs, liquidity, execution delay, position sizing, or market impact. Accuracy can change under volatility-regime or data-distribution shifts. Monitor feature distributions, signal rate, realized precision, and calibration after deployment.

## Package contents

- `winner.joblib`: model, imputers, scalers, feature order, threshold, and target settings
- `predict.py`: standalone CSV inference
- `model_components.py`: class definition required to deserialize the model
- `input_features.json`: ordered feature schema
- `feature_coefficients.csv`: standardized-model coefficients
- `metrics.json`: complete evaluation metrics
- `validation_leaderboard.csv`: all compared models
- `requirements.txt`: verified Python dependencies
- `MANIFEST.sha256`: file integrity hashes
