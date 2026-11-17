# PCR deployment model card

Deployment: `PCR_60_120_MAX3_TP3000_SL1500`, installed 2026-10-28. Intended use: simulated BANKNIFTY monthly short-straddle paper trading. There is no broker order-submission path.

## Provenance and packaging

The four selected packages were copied unchanged from `E:\PCR\hypothesis testing\output\zero_week_focus_stack_zero_brokerage\models`. Individual model cards, metrics, training/inference code, requirements and component hashes are retained. `model_files.sha256.json` verifies the complete copied artifacts. No retraining, scaler fitting, recalibration or test-period threshold optimization was performed during this integration.

Training metadata reports: algorithm family selection on 2021; final regressors fitted before 2023; final classifiers fitted before 2022 and calibrated on 2022; 2023 used for expiry-group thresholds; historical evaluation from January 2024. Those claims and the source training code remain available for audit. Deployment validation tests formula parity, model-output parity and causal inference behavior; it is not a fresh independent audit of every upstream training/data decision.

The source model cards call 2024 onward a locked test. Strategy alternatives and thresholds have since been repeatedly inspected against that period. Consequently the reported historical strategy performance must be treated as a reviewed research result. Additional unseen paper observations are required to assess prospective performance.

## Prediction meaning

Win probability is calibrated to the source 60-minute profit label. It is **not** directly calibrated to the probability of hitting +INR 3,000 before -INR 1,500. The selected 60/120-minute hold, price-path exits, current lot sizes and current charge rates change realized outcomes. The tail classifier likewise predicts its historical -40-point label, not the INR 1,500 stop event.

Each of the four models takes the same 46 inputs; no model output is a feature of another model. Outputs are combined only by the gate/holding rules in `runtime.py`. All input windows are at most 60 minutes and are lagged to the preceding completed minute. The two-hour horizon is a model target/output, not future feature data.

## Paper execution and limitations

The book holds up to three independent positions, freezing strike, expiry, tokens and quantity at entry. Target/stop decisions use estimated net rupee P&L after itemized costs and two straddle premium points of round-trip slippage. Simulated fills use observed LTPs. There is no bid/ask, queue-position, liquidity or partial-fill simulation, and no assumption that the exact target/stop price is executable.

Historical reference reproduction includes 178 synthetic exits when exact strike paths were absent. Live execution does not interpolate strikes: it waits for exact-contract fresh quotes and flags overdue exits. A feed outage can therefore leave a paper position unresolved after the requested cutoff; it is not reported as an invented fill.

One-second batching and minute feature completion differ from the historical minute cache. Live IV/Greeks are computed from observed LTPs using the source conventions; quote freshness, OI update timing and source data construction may cause distribution differences even though the numeric feature transformations match.

Current live rates and lot sizes are inherited from the existing host's contract/configuration setup. Historical P&L retains historical rates and sizes. No current tax-rate correctness claim is made beyond faithfully using that configured source.

## Validation evidence

`validation/pcr_tests.xml` contains the PCR suite result. `validation/original_app_tests.xml` contains the existing application regression result. The successful synthetic worker test exercises the actual subprocess and actual saved models; controlled model-output fixtures separately force entry/exit gates to verify three-position behavior. No prospective market session was executed as part of installation.

`backtests/installation_reference_check/` reproduces 902 trades, 63.7472283814% wins and INR 98,910.0505345 net, exactly matching the copied reference. Its INR -25,233.7725633 drawdown is measured on daily closes, not intraday equity. A reproduction confirms implementation parity; it does not establish unbiased profitability.

`README.md` documents settings, controls, logs, startup, recorded replay and rollback. Future changes should retain the entry-time configuration, untouched model artifacts, complete decision journals and separate replay output directories.
