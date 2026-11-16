# 120-minute fixed-strike IV/RV inference analysis

> This is a model and research guide. Its `E:\v2` examples are historical;
> use the repository root README and `setup/voltr.py` to launch this checkout.

## Purpose

At entry time `t`, identify the BANKNIFTY strike `K_t` that is ATM, keep that exact strike and expiry fixed, and estimate whether its combined call-put implied volatility will be closer to the predicted forward 120-minute realized volatility at `t+120`.

This folder consolidates the saved-model contracts, data cards, feature lists, transforms, performance, and the correct way to combine the three model families. The exact feature order extracted from the serialized artifacts is in [`model_feature_lists.csv`](model_feature_lists.csv).

## Executive conclusion

There are two possible IV forecasting routes:

1. The combined-IV LightGBM directly predicts the change in arithmetic-mean call/put IV over 120 minutes.
2. The separate call and put HistGradientBoosting models predict each leg's log IV ratio; their future-IV predictions can then be averaged.

The separate XGBoost model predicts forward 120-minute annualized RV. It estimates volatility magnitude, not spot direction.

For the stated goal, the clean inference comparison is:

```text
entry_combined_iv = (entry_iv_call + entry_iv_put) / 2
predicted_exit_combined_iv = entry_combined_iv + predicted_combined_iv_change

entry_gap = abs(entry_combined_iv - predicted_rv_120m)
exit_gap  = abs(predicted_exit_combined_iv - predicted_rv_120m)

predicted_convergence = exit_gap < entry_gap
```

The optional two-leg cross-check is:

```text
predicted_call_iv_120m = entry_iv_call * exp(clip(predicted_call_log_ratio, -2, 2))
predicted_put_iv_120m  = entry_iv_put  * exp(clip(predicted_put_log_ratio,  -2, 2))
predicted_exit_combined_iv_from_legs =
    (predicted_call_iv_120m + predicted_put_iv_120m) / 2
```

Do not switch to the strike that is ATM at `t+120`. All three data pipelines were designed around the original entry strike.

## Model inventory

| Role | Saved model | Algorithm | Output | Saved features |
|---|---|---|---|---:|
| Direct combined-IV forecast | `E:\v2\models\combined_atm_iv_rv_convergence_v2_120m\artifacts\combined_direct_lightgbm.joblib` | LightGBM regressor | `exit_combined_iv - entry_combined_iv` | 151 |
| Fixed-strike call-IV forecast | `E:\v2\research\fixed_strike_iv_120m\models\predict_fixed_atm_strike_iv_120m\artifacts_inference_60m\call_hist_gradient_boosting.joblib` | Imputer + HistGradientBoostingRegressor | `log(exit_call_iv / entry_call_iv)` | 143 |
| Fixed-strike put-IV forecast | `E:\v2\research\fixed_strike_iv_120m\models\predict_fixed_atm_strike_iv_120m\artifacts_inference_60m\put_hist_gradient_boosting.joblib` | Imputer + HistGradientBoostingRegressor | `log(exit_put_iv / entry_put_iv)` | 143 |
| Forward realized-volatility forecast | `E:\v2\models\fixed_strike_iv_to_rv_convergence_120m\artifacts_rv_improved_full\xgboost.joblib` | XGBoost regressor | calibrated annualized forward 120-minute RV | 168 |

The call and put winners use the same 169-feature schema and ordering.

## Data cards

### 1. Combined fixed-strike IV-change LightGBM

**Target.** `target_fixed_combined_iv_change_120m`:

```text
combined_iv = (call_iv + put_iv) / 2
target = exit_fixed_atm_iv_combined - entry_atm_iv_combined
```

The model output is an absolute IV change in decimal volatility units. It is not a future IV level, log ratio, or convergence probability.

**Strike and time alignment.** The entry strike must equal the source ATM strike. Exit data is joined at exactly `t+120` on timestamp, DTE flag, expiry, and the unchanged entry strike.

**Training data.** The feature builder processed 233,126 rows over 1,239 days. There were 191,647 entry-quality rows, 53,237 label-available rows, and 45,834 training-eligible rows. The model report lists 22,485 train out-of-fold rows, 7,050 validation rows, and 8,501 held-out test rows. The test interval was 23 August 2024 09:16:59 through 9 May 2025 13:21:59.

**Validation protocol.** Three expanding walk-forward folds with a 120-minute purge. Ordinary fits were capped at 12,000 chronological observations.

**Held-out performance.** MAE `0.00384493` (0.3845 volatility points), RMSE `0.00925134`, R² `-0.1380`, and IV-change direction accuracy `72.66%`. MAE improved `13.42%` over the zero-change baseline, but RMSE was slightly worse and R² remained negative. This is research-grade, not production-approved.

**Feature contract.** The active retrained model has 151 features after removing historical 120-minute lag, change, return, RV, and RQ inputs. Training appends:

- `rv_pred_120m`
- `entry_gap_to_rv_pred_combined = entry_atm_iv_combined - rv_pred_120m`

Its features comprise 48 exact lag fields, 48 corresponding changes, 4 spot returns, 8 trailing RV/RQ measures, 7 quality flags, calendar/session fields, current option/Greek/OI/surface fields, and the two RV-forecast fields.

**Important semantic distinction.** A separate saved artifact named `combined_convergence_classifier.joblib` is the binary convergence model. The requested `combined_direct_lightgbm.joblib` predicts IV change. Its forecast must be converted to a future combined-IV level and compared with predicted RV using the decision rule above.

### 2. Fixed-strike call and put IV HistGradientBoosting models

**Targets.** Each model predicts a log ratio:

```text
y_call = log(call_iv(K_t,t+120) / call_iv(K_t,t))
y_put  = log( put_iv(K_t,t+120) /  put_iv(K_t,t))
```

The saved estimator's raw output must therefore be converted back to a future-IV level. The training code clips the raw log-ratio forecast to `[-2, 2]` before exponentiation.

**Saved pipeline.** Each artifact is a dictionary with `estimator`, `features`, `side`, and `target_transform`. The estimator is a scikit-learn pipeline containing `SimpleImputer` followed by `HistGradientBoostingRegressor`. Selected parameters were 150 iterations, learning rate 0.025, 31 maximum leaf nodes, and L2 regularization 1.0.

**Training data and split.** The target table has 133,989 rows: 92,999 train rows, 19,950 validation rows, 20,903 test rows, and 137 purged rows. The held-out test period ran from 19 September 2024 12:56:59 through 9 May 2025 13:26:59.

**Held-out performance.** Call-IV MAE was `0.007238` (0.724 volatility points), R² `0.7567`, and balanced change-direction accuracy `51.57%`. Put-IV MAE was `0.006782` (0.678 volatility points), R² `0.7937`, and balanced direction accuracy `51.86%`. The models forecast future levels reasonably well but barely classify the direction of the IV move.

**Feature contract.** Both active models use the same 143 ordered features: 35 current entry/surface/Greek fields, 52 exact lag fields, 52 changes at 5/15/30/60 minutes, plus four time/DTE fields. They do not directly consume predicted RV.

### 3. Forward 120-minute RV XGBoost

**Target.** `target_rv_forward_120m`, the annualized realized volatility over the next exact 120 one-minute returns. The label requires a complete 121-price window from `t` through `t+120`.

**Target transform and inverse.** Training uses log variance:

```text
z = log(max(RV, 1e-8)^2)
raw_rv = sqrt(exp(clip(model_prediction, -20, 5)))
RVpred = raw_rv * 1.046232793493275
```

The multiplicative value is the validation-derived calibration stored in the joblib. Calling `model.predict()` alone does not produce the final RV forecast; inverse transformation and calibration are mandatory.

**Training data.** 83,031 train rows, 20,145 validation rows, and 25,381 held-out test rows. Model fitting was capped at 60,000 chronologically sampled development rows.

**Validation protocol.** Four expanding walk-forward folds with a 120-minute purge. The test block was not used for hyperparameter or blend selection.

**Held-out performance.** XGBoost QLIKE `0.166675`, MAE `0.024103` (2.410 volatility points), and RMSE `0.033127`, versus persistence QLIKE `0.250183` and MAE `0.032130`. Although the validation-selected three-model blend excluded XGBoost, XGBoost had the best reported held-out result. That is a post-test observation and should not be treated as clean model selection evidence without another untouched period.

**Feature contract.** The active retrained model has 168 ordered features. Historical 120-minute inputs and their engineered forms were removed; its trailing return, RV and realized-quarticity inputs stop at 60 minutes.

## Feature-list summary

| Model contract | Current/other | Lags | Changes | Spot returns | RV/RQ | Extra RV engineering | Quality | Calendar/DTE | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Combined-IV LightGBM | 31 | 48 | 48 | 4 | 8 | 0 | 7 | 5 | 151 |
| Call/put HistGradientBoosting | 35 | 52 | 52 | 0 | 0 | 0 | 0 | 4 | 143 |
| RV XGBoost | 36 | 48 | 48 | 4 | 8 | 12 | 7 | 5 | 168 |

The exact row-by-row feature ordering is intentionally stored separately in `model_feature_lists.csv`; inference data must be reindexed to that order before prediction.

## Recommended inference sequence

1. Select the source ATM row at time `t`; store `entry_strike = K_t`, expiry, and DTE flag.
2. Build all features using information available at or before `t`. Exact lags must refer to the same feature pipeline semantics used in training.
3. Run the RV XGBoost using its exact 168-column order. Apply the inverse log-variance transform and saved calibration.
4. For the direct combined-IV model, add `rv_pred_120m` and calculate `entry_gap_to_rv_pred_combined`.
5. Reindex to the combined model's exact 178-column order and predict the IV change.
6. Calculate `predicted_exit_combined_iv = entry_atm_iv_combined + predicted_change`.
7. Classify predicted convergence by comparing the absolute entry and predicted-exit gaps to the same `RVpred`.
8. Optionally run the call and put models and compare their averaged future-IV forecast with the direct combined forecast. Large disagreement should be treated as model uncertainty, not automatically averaged away.
9. Convert RV into a probable terminal spot range only after RV inference:

```text
sigma_120 = RVpred * sqrt(120 / (252 * 375))
range(z)  = [S0 * exp(-z * sigma_120), S0 * exp(z * sigma_120)]
```

Use `z=1` for a one-sigma range and `z=1.645` for an approximate 90% lognormal range. This is a terminal-move range under the volatility scaling assumption, not a guaranteed intraperiod barrier range.

## Critical integration warning

The combined-IV LightGBM was trained with an out-of-sample `rv_pred_120m` generated by the combined-model project's own RV stack, not necessarily the separately saved improved-full XGBoost named in this analysis. Replacing the training-time RV forecast with the separate XGBoost changes both:

- `rv_pred_120m`
- `entry_gap_to_rv_pred_combined`

That is a feature-distribution change. It may still be economically sensible, but it is not automatically equivalent to the validated model card. Before relying on the joint pipeline, rebuild or rescore the combined-IV model with out-of-sample predictions from this exact RV XGBoost, or run an untouched walk-forward comparison demonstrating that the substitution is harmless.

## Other implementation cautions

- Preserve decimal volatility units: `0.20` means 20%, not `20`.
- Preserve case-sensitive feature names such as `Future_oi`.
- Do not recompute ATM at exit.
- Do not feed future, exit, label-availability, or training-eligibility columns into inference.
- Do not call `xgboost_payload["model"].predict()` and interpret the result as RV; apply the inverse transform and calibration.
- Do not interpret the call/put models' raw output as IV; it is a log ratio.
- The combined model's raw output is an additive IV change, whereas the leg models' raw outputs are multiplicative log changes.
- Tree gain/split importance is descriptive, not causal. In the combined model, `rv_pred_120m` is among the most frequently used features; in the RV model, current IV and short-window RV measures dominate the reported feature importance.
- Model metrics exclude trading frictions. A convergence forecast is not by itself a profitable short-straddle signal; delta/gamma exposure, premium decay, stops, bid/ask, slippage, taxes, and brokerage still matter.

## Source cards inspected

- `E:\v2\models\combined_atm_iv_rv_convergence_v2_120m\README.md`
- `E:\v2\models\combined_atm_iv_rv_convergence_v2_120m\artifacts\REPORT.md`
- `E:\v2\models\combined_atm_iv_rv_convergence_v2_120m\artifacts\metrics.json`
- `E:\v2\models\combined_atm_iv_rv_convergence_v2_120m\artifacts\run_manifest.json`
- `E:\v2\research\combined_atm_iv_rv_convergence_v2\README.md`
- `E:\v2\research\combined_atm_iv_rv_convergence_v2\artifacts\combined_atm_iv_convergence_v2_features_column_manifest.json`
- `E:\v2\research\fixed_strike_iv_120m\models\predict_fixed_atm_strike_iv_120m\README.md`
- `E:\v2\research\fixed_strike_iv_120m\models\predict_fixed_atm_strike_iv_120m\artifacts\REPORT.md`
- `E:\v2\research\fixed_strike_iv_120m\artifacts\feature_manifest.json`
- `E:\v2\models\fixed_strike_iv_to_rv_convergence_120m\artifacts_rv_improved_full\REPORT.md`
- `E:\v2\models\fixed_strike_iv_to_rv_convergence_120m\artifacts_rv_improved_full\metrics.json`
- `E:\v2\research\fixed_strike_iv_rv_convergence\README.md`
- `E:\v2\research\fixed_strike_iv_rv_convergence\artifacts\fixed_strike_convergence_features_column_manifest.json`

The serialized `.joblib` payloads were also loaded directly to verify container keys, estimator types, target transforms, calibrations, and exact feature order.

## Implemented stacked inference pipeline

[`inference_pipeline.py`](inference_pipeline.py) loads the saved artifacts without refitting and uses their outputs together. Its model paths and inference constants are declared in [`config.yaml`](config.yaml).

The pipeline emits:

- calibrated XGBoost forward RV;
- combined-IV deltas from Ridge, monotonic XGBoost, direct LightGBM, their saved validation-weighted ensemble, and NGBoost;
- combined-IV q10/q50/q90 forecasts;
- saved convergence-classifier probability and decision;
- call and put HistGradientBoosting point forecasts after log-ratio inversion;
- call and put q10/q50/q90 forecasts after log-ratio inversion;
- arithmetic-mean combined IV reconstructed from the leg forecasts;
- entry and predicted-exit IV-to-RV gaps for each principal forecast;
- deterministic convergence flags, a three-signal consensus vote, and a robust whole-interval convergence flag;
- one-sigma and approximate 90% terminal spot ranges when spot is supplied.

### Python use

```python
import pandas as pd

from inference_pipeline import FixedStrikeInferencePipeline

pipeline = FixedStrikeInferencePipeline(r"E:\v2\Inference\config.yaml")

combined = pd.read_csv("combined_features_for_inference.csv")
legs = pd.read_csv("call_put_features_for_inference.csv")
rv = pd.read_csv("rv_features_for_inference.csv")

predictions = pipeline.predict(combined, legs, rv)
predictions.to_csv("inference_predictions.csv", index=False)
```

### Command-line use

```powershell
cd E:\v2\Inference
py -3.13 inference_pipeline.py `
  --combined-features combined_features_for_inference.csv `
  --leg-features call_put_features_for_inference.csv `
  --rv-features rv_features_for_inference.csv `
  --output inference_predictions.csv
```

The three input frames must describe identical rows in identical order. When identity fields are available, the pipeline checks `row_id`, timestamp, DTE flag, expiry, and entry strike before scoring. A single wide CSV may be passed only as `--combined-features` when it contains the union of all three saved feature contracts.

The pipeline deliberately exposes model disagreement. It does not silently average every candidate into a supposedly validated forecast. Its three-signal consensus uses the direct combined-IV LightGBM comparison, the independently trained call/put HGB comparison, and the saved convergence classifier. Quantile and NGBoost outputs remain visible for risk and uncertainty analysis.

### Diagnostic three-rule signal

The separate no-120-minute-input ElasticNet bundle in
`..\..\rv_iv_no120_best_model_package\winner.joblib` is evaluated on every
completed inference minute. The following research-rule outputs remain visible
in every prediction, but they no longer authorize a paper entry:

1. Current combined IV is above trailing 60-minute annualized RV, and the
   direct combined-IV LightGBM predicts that IV will decline and finish closer
   to the calibrated forward 120-minute XGBoost RV prediction.
2. Current combined IV and trailing 60-minute RV are within 0.5 volatility
   points: `abs(entry_atm_iv_combined - rv_trailing_60m) <= 0.005`.
3. Current combined IV is below trailing 60-minute RV and the no-120m model
   predicts its sharp-RV-decline event. That event means at least a 0.02 RV
   decline and an RV decline at least 0.01 greater than the fixed-strike
   combined-IV decline over 120 minutes.

### Regime-routed one-lot paper position engine

Paper entries are routed by the current combined-IV minus trailing-60-minute-RV
gap. When the gap is greater than `0.005`, the IV-above-RV convergence rule must
pass and `pred_convergence_probability_classifier > 0.75` (strictly). When the
absolute gap is at most `0.005`, or the gap is below `-0.005`, the saved no-120m
sharp-RV-decline model is used instead and `pred_no120_event_score >= 0.0`.
Thus the same RV-decline model covers both the near-equal and IV-below-RV
regimes, while the convergence classifier covers clearly IV-above-RV entries.

A qualifying routed signal opens one lot at fresh live call/put prices for the
strike that was ATM at the signal minute only when fewer than three paper
positions are already open. That strike and expiry remain fixed. Each accepted
position has its own entry price, clock, P&L, target, and stop. Qualifying
signals received while three positions are open are retained in the prediction
log and dashboard as `QUALIFIED_BUT_SKIPPED`.

The position is marked from live ticks and closes on the first of:

- gross P&L at or above +₹1,500 per lot;
- gross P&L at or below -₹750 per lot;
- 120 minutes from the actual paper entry time; or
- the first market update at or after 15:14:30.

No new position is opened at or after 15:14:30. Entry and exit events are
written to `logs/paper_trade_events.jsonl`, while the recoverable current state
is stored in `logs/paper_position_state.json`. The latest prediction/status API
also includes a `paper_trading` object. This engine remains paper-only and has
no broker or order-placement path. Reported P&L is gross before transaction
costs and slippage. The dashboard restores up to 500 records from the active
prediction log, shows every timestamped signal, and provides an expandable
`All values` view containing the complete saved event for each timestamp.

### Fixed-strike paths for after-market backtesting

The live runtime writes `logs/fixed_strike_120m_paths.jsonl`. For every valid
entry minute, it retains the strike that was ATM at that entry and records that
same strike's call and put prices on every subsequent completed minute through
minute 120. It does not replace the entry strike with the observation-time ATM.

Each path row contains the stable `entry_id`, entry and observation timestamps,
holding minute, expiry, fixed strike, entry and observation spot/future/option
prices, IVs, OI, straddle values, gross short-straddle P&L per unit and per lot,
the entry-time convergence probability and threshold, legacy no-120m
gate/model/signal fields, target/stop flags, and an
`is_horizon_exit` marker. For a backtest, group by `entry_id`, retain entries
where `entry_short_straddle_signal` is true, sort by `holding_minutes`, and take
the first row where the profit-target flag, stop-loss flag, or horizon marker is
true. Target and stop flags are gross and do not include costs or slippage.

Historical UDP replay writes the same schema to its timestamped
`logs/historicalReplay/YYYY-MM-DD/fixed_strike_120m_paths_HHMMSS.jsonl` file.

### Validation performed

- Saved-artifact contract smoke tests: passed.
- Transform tests for leg log ratios and RV log variance: passed.
- Real-row integration test across the three historical feature tables: passed for the held-out row at 23 August 2024 09:16:59, fixed strike 50,900.

Run the repeatable tests with:

```powershell
cd E:\v2\Inference
py -3.13 -m pytest -q test_inference_pipeline.py
```

The leakage audit and corrected common-window P&L are documented in
[`artifacts/LEAKAGE_AUDIT.md`](artifacts/LEAKAGE_AUDIT.md) and
[`artifacts/LEAKAGE_SAFE_TEST_PNL_REPORT.md`](artifacts/LEAKAGE_SAFE_TEST_PNL_REPORT.md).
The active gross-to-net calculation, using current exchange/statutory rates and
zero brokerage, is in
[`artifacts/CURRENT_COST_NO_BROKERAGE_PNL_REPORT.md`](artifacts/CURRENT_COST_NO_BROKERAGE_PNL_REPORT.md).
The earlier historical-rate, brokerage-inclusive report is retained and marked
as superseded for auditability.

The consolidated machine-readable export is
[`artifacts/inference_test_predictions_and_costs.json`](artifacts/inference_test_predictions_and_costs.json).
It contains all 7,423 predictions from the common leakage-safe test window, all
324 executed trades with current zero-brokerage costs, the cost assumptions,
and the aggregate scenario summary. Regenerate it with
`py -3.13 export_predictions_and_costs_json.py`.

## Live UDP inference and dashboard

The live runtime is organized under [`live/`](live/):

- [`live/live_inference.py`](live/live_inference.py) aggregates the live feed,
  constructs the exact saved-model feature contracts, runs all 16 artifacts,
  and serves the dashboard.
- [`live/live_config.json`](live/live_config.json) contains only inference-side
  overrides and paths. The multicast addresses, ports, and interface addresses
  used by the receiver are declared in the `live_udp` section of
  [`config.yaml`](config.yaml).
- [`live/udp_receiver.py`](live/udp_receiver.py) owns the local multicast socket
  and NSE sequential-packet unpacking used by this runtime.
- [`live/lzo_codec.py`](live/lzo_codec.py) and `live/native/lzo2.dll` provide the
  package-local LZO1Z decompressor.
- [`live/live_dashboard.html`](live/live_dashboard.html) shows the market
  snapshot, every `pred_*` output, call/put/combined IV forecasts, convergence
  diagnostics, current zero-brokerage costs, recent predictions, and the
  predicted one-sigma and approximate 90% SPOT boundaries.

UDP reception, sequential-packet framing, and LZO1Z decompression are local to
this Inference package. The tested 7207 index decoder, 7208/17208 market-update
decoder, 7201/7202/17201/17202 OI decoder, contract selection, IV/Greek helpers,
and two-minute sealing semantics remain shared with `E:\Models\Live Trades`.

In isolated IV/RV mode, the raw feed thread now selects the training-compatible
last spot/option tick and first futures tick for each exchange minute. It sends
one completed-minute sample to a dedicated inference thread. Open paper
positions are marked from relevant option ticks on the raw feed thread, so a
slow model prediction does not delay stop-loss, profit-target, or time-exit
checks. The minute history is retained in the inference thread; it is no
longer copied on every market tick. Health status includes the minute-sample
queue depth and a separate model backlog fault. Raw datagrams remain in the
feed journal for loss detection and replay.

Run a read-only preflight without opening multicast sockets:

```powershell
cd E:\v2\Inference
$env:ERP_AUTH_TOKEN = Read-Host "ERP auth token"
py -3.13 live\live_inference.py --preflight
```

Run live inference:

```powershell
cd E:\v2\Inference
py -3.13 live\live_inference.py
```

The live command is a persistent weekday scheduler. It prepares the contracts
and models five minutes before the session, opens the feeds at **09:00**, closes
them cleanly at **15:40**, and waits for the next weekday, all in
`Asia/Kolkata`. The times and preparation lead are configured by
`session_start`, `session_end`, and `schedule_prepare_seconds` in
[`live/live_config.json`](live/live_config.json). Use `--run-now` only when an
unscheduled diagnostic run is required.

Windows Task Scheduler launches this persistent scheduler through the
`InferencePipeline-PaperTrading` task at user logon. The task uses the Python
3.13 `pythonw.exe`, runs hidden, ignores duplicate launches, and restarts the
process after failures. The internal scheduler—not a second daily Windows
trigger—owns the recurring weekday 09:00–15:40 session window.

At the start of each session, stale `runtime_log`, `prediction_log`, and
`latest_prediction` files are moved to
`logs/prevDayLogs/YYYY-MM-DD/`. The new session then creates fresh active log
files. Restarting during the same day retains and appends to that day's files.

### After-hours historical inference

For an after-market historical replay delivered through the same UDP packet
format, supply the changed FO port. This bypasses the 09:00–15:40 clock while
reusing the same multicast receiver, decoders, feature warmup, models, and
dashboard:

```powershell
cd E:\v2\Inference
py -3.13 live\live_inference.py `
  --historical-replay-port 35001 `
  --historical-date 2026-10-22
```

The normal CM/index port remains active. If the historical CM stream is also
sent on a different port, add `--historical-replay-cm-port 35002`. Stop replay
mode with `Ctrl+C`. Its logs are isolated under
`logs/historicalReplay/YYYY-MM-DD/`, so it cannot append to or rotate the live
session's logs. The dashboard identifies its mode as `historical_udp_replay`.

As a separate offline option, a prebuilt wide feature CSV can be scored without
opening UDP sockets or the dashboard:

```powershell
cd E:\v2\Inference
py -3.13 live\live_inference.py `
  --historical-data E:\path\to\features.csv `
  --historical-date 2026-10-22
```

By default, results are written to
`historical_predictions/YYYY-MM-DD/predictions_HHMMSS.csv`. Use
`--historical-output` to choose a destination. If the model features are split
across the original three tables, also pass `--historical-leg-data` and
`--historical-rv-data`. Omitting `--historical-date` scores every row in the
input file.

Before either command loads contracts, models, or opens UDP sockets,
[`live/contract_bootstrap.py`](live/contract_bootstrap.py) checks for the prior
NSE business day's `NSE_FO_contract_DDMMYYYY.csv.gz` in `live/contracts`. If it
is missing or invalid, the runtime downloads it from the configured ERP API,
validates the gzip and required CSV columns, and checks that usable BANKNIFTY
options and futures exist. Weekends and exchange holidays fall back through
the configured seven-day lookback. The ERP credential is read from the
process-only `ERP_AUTH_TOKEN` environment variable and is never stored or
logged.

The bootstrap writes `live/contracts/banknifty_latest_expiry_tokens.json` with
all CE/PE token numbers for the nearest expiry, the nearest monthly expiry used
by this live strategy, and the front-future token. The selected contract path
is injected into the in-memory configuration before the shared contract
loaders run.

It can also be run independently:

```powershell
cd E:\v2\Inference
$env:ERP_AUTH_TOKEN = Read-Host "ERP auth token"
py -3.13 live\contract_bootstrap.py
```

Then open `http://127.0.0.1:8780`. The runtime is inference-only and contains
no broker or order-placement integration. It writes append-only predictions to
`logs/live_predictions.jsonl` and the latest complete event to
`logs/latest_prediction.json`.

The first complete prediction requires 61 consecutive sealed minutes because
all active saved contracts stop at an exact 60-minute lag. Before that point the
dashboard reports the precise warmup or missing-feed reason.

## Rising-RV long-straddle paper route

The model in `families/rising_rv` is loaded as a separate
probability signal. It requires mean entry call/put IV above trailing 120-minute
spot RV, and it scores 19 training features. Its trailing RV uses the training
model's sample-standard-deviation formula and annualized volatility points.
The score remains unavailable until the 120-minute spot window is present.

The live dashboard shows its entry gate, probability, decision, and an
independent long ATM-straddle paper book. The long book uses the same one-lot,
fixed-strike, three-position cap, target, stop, 120-minute horizon, square-off,
and stronger-signal replacement controls as the existing short book. It has
separate state and trade logs, named from the configured short paths with
`_long` before the file extension. Paper P&L uses the option tick prices and
is gross of charges, like the existing short book. It places no broker orders.

The 29 October 2026 replay and its limitations are documented in
`../../../../analysis/analysis_outputs/rising_rv_120m_2026-10-29/REPORT.md`.
The model's 0.5 threshold is provisional: its research probabilities were not
well calibrated, and the available path feed cannot establish realized
stop/target P&L for the sampled trades.
