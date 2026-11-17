# Model card: LightGBM without 120-minute RV inputs

## Purpose

Predict whether realized volatility rises and the IV–RV gap shrinks during the next 120 minutes, for Bank Nifty monthly-option entries where mean call/put IV at `t` exceeds trailing 120-minute spot RV. The option strike is the one that was ATM at `t` and remains fixed when future IV is measured.

Positive label: `RV[t+120] > RV[t]` and `|IV[t+120] - RV[t+120]| < |IV[t] - RV[t]|`.

## Removed inputs

This model has **19 inputs**. It excludes both `current_rv` (trailing 120-minute RV) and `current_gap` (`current_iv - current_rv`). See `features.json` for the exact order and `features.md` for definitions. The entry filter still needs the trailing 120-minute RV to decide whether `IV[t] > RV[t]`; this is a model-input ablation, not a removal of the original study condition.

## Training

- Fitted model: LightGBM classifier in `best_model.joblib`.
- Hyperparameters: 300 estimators, 7 leaves, learning rate 0.1, minimum child samples 100, L2 regularization 2. Twelve configurations were sampled and scored with three expanding training-era time folds.
- Source: `data/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv`.
- Eligible examples: 1,426 entry minutes over 21 expiries. Each trailing and forward 120-minute window must contain at least 60 observed spot returns, with converged call/put IV at entry and exit.
- Chronological expiry split: 806 training, 273 validation, 347 test examples. Split groups and tuning folds do not share trading dates.

| Metric | With 120-minute features | Without 120-minute features |
|---|---:|---:|
| Validation ROC AUC | 0.759 | 0.853 |
| Test ROC AUC | 0.980 | 0.915 |
| Test average precision | 0.919 | 0.768 |
| Test Brier score | 0.048 | 0.137 |

The with-feature reference is the previous LightGBM package in `outputs/iv_gt_rv_120m_best_model`. This is an exploratory comparison: both versions were developed against the same test period, so the test result is no longer a pristine final estimate. The validation Brier score for this version was 0.281, which also cautions against treating its raw probabilities as calibrated. Full metrics and per-expiry results are in `report.json`.

## Limitations

Future IV convergence and future spot coverage determine which examples receive labels. That eligibility is unknown at entry and limits the scope of the reported metrics. Windows contain missing minutes, neighboring entry labels overlap, and the test covers only five expiries. This is a research classifier, not a trading backtest or production-ready signal. A fresh later-period walk-forward evaluation is needed before deployment.

## Reproduce

With the source CSV under a parent workspace `data/` folder, run `py -3.13 train.py` from this folder after installing `requirements.txt`. The script rebuilds the events, retunes LightGBM, and writes the model, report, and predictions here. Load the `joblib` artifact only from a trusted source.
