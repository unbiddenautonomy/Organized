# 3 and 4 Week OTM Paper Trading

> This document records the original installation and research results.
> Its `E:\` paths and task state are historical. For this portable checkout,
> use the repository root README and `setup/voltr.py`.

This package adds a separate BANKNIFTY paper strategy at `/firstTwoWeeks` to the existing application in `E:\pipelines\engine\Inference`. Its model process, positions, trade ledger, signal history and backtests are independent of the existing IV/RV paper trader.

The selected OTM combination is an **experimental paper candidate**. It did not meet the pre-2024 minimum-trade and coverage requirements. The old document's OTM cascade results do not establish performance under this package's three-position, profit-target and stop-loss rules. Read `otm_models/training_report.json` and the chronological replay's `summary.json` before interpreting performance. No improvement in win rate is established.

The latest verified replay with saved cutoff metadata is `backtests/replay_20261028_204043`; it matches the earlier dashboard-triggered `replay_20261028_200046`. From January 2024 through 8 May 2025 it produced 31 trades, 51.61% wins and **−₹3,210.75 net P&L**, after ₹2,359.50 costs. Of 150 source-observed eligible dates, 20 had trades: **9 green, 11 red, and 130 no-trade days**. Maximum observed concurrency was 3 and no positions remained unresolved. The 3-week bucket contributed 19 trades and −₹2,337.88; the 4-week bucket contributed 12 trades and −₹872.87. These results do not support deployment with real money.

For the earlier quoted 102-trade benchmark, `reference_benchmark.json` preserves the separate original results: 100 available dates, 74 trading dates, 53 green days, 21 red days, and 26 no-trade days. The 3-week bucket had 61 trading days (43 green, 18 red); the 4-week bucket had 13 (10 green, 3 red). Its rules and eligible-date construction differ from the new replay.

## Start and open

The existing `InferencePipeline-PaperTrading` Windows task now starts the independent-process daily service. Preparation begins at 08:55 IST, with a fresh contract download and model loading; live feed processing starts at 09:00. The three workers subscribe independently when ready. The new page uses the same server and port:

`http://127.0.0.1:8780/firstTwoWeeks`

The updated service was restarted and verified on 2026-10-29 with empty paper books. The configured Windows user must remain signed in. ERP authentication is configured locally; credentials are never copied into model artifacts or logs. Runtime details and rollback instructions are in `../engine/Inference/live/ISOLATED_RUNTIME.md`.

The old `http://127.0.0.1:8782/firstTwoWeeks` bookmark now redirects to the actual live page on port 8780. The service owns that compatibility listener; do not launch a duplicate preview process. `preview_dashboard.py` is retained only as a redirect helper and does not run inference. Use `/pipelines` for actual worker and contract readiness. The root fixed-strike page distinguishes feature warmup from live predictions.

## Trading rules

- Monthly expiry, 21 to less than 28 days for 3-week and 28 to less than 35 days for 4-week eligibility, measured to expiry at 15:30.
- At the completed signal snapshot, choose the call **ATM at t +100** and put **ATM at t −100**. Both must use the same expiry. Unexpected strike spacing disables new OTM entries.
- A single consolidated OTM book has at most **3 open or pending positions** across the selected models. The existing trader's separate position limit is unchanged. This is not a combined six-to-three portfolio constraint.
- One contract lot per position, with lot size from the loaded exchange contract file. Historical replay uses the source report's historical lot schedule.
- The selected classifier must pass the saved bucket cutoff, and the regression model must pass the expected-net buffer. Defaults come from frozen pre-2024 selection; operators can change the cutoffs on the dashboard. The OTM Q90 loss estimate must stay under ₹750. All candidate outputs are recorded even when a signal is rejected.
- Hold for at most 120 minutes after execution, subject to **net target +₹1,500**, **net stop −₹750**, or **15:14:30 square-off**. Stops are triggers and may overshoot. A missing/stale quote never fabricates a fill.
- Evaluate from 10:15 and reject entries without time to complete the horizon before square-off. No 90-minute fallback is enabled.
- Features use completed observations and 5/15/30/60-minute lags, allowing at most one minute of backward age for lagged snapshots. At least 13 observed snapshots in the prior hour are required. No cross-date or cross-expiry lags.
- Historical fills wait at least 60 seconds after signal quote availability. Live fills also require both leg quotes at or after completion of the actual inference decision. Quotes older than 60 seconds are rejected. The existing app finalizes bars with its existing two-minute delay; this is retained and recorded.
- Frozen strike paths are logged through the signal horizon even after an early profit/stop exit. Records include source quote times and missing/stale flags. Live trade exits are checked on forwarded quote updates; path logs are sampled once a minute.
- The path viewer begins with the original signal snapshot from the persisted snapshot log, then shows subsequent fixed-contract observations. Final observations never use a quote after the horizon.

Costs deduct both legs' round-trip brokerage, STT, exchange charges, SEBI fees, stamp duty, GST and 2 combined premium points of modeled slippage. Current rates are copied from the existing live configuration, not asserted to be a new tax-rate verification. The `config.json` rates and source-report historical schedules are explicit assumptions. P&L uses full modeled round-trip costs while a position is open and when it closes.

## Models and selection

`otm_models` contains six newly fitted OTM models:

| Model | Output |
|---|---|
| Logistic regression | Calibrated probability of positive exit-adjusted net P&L |
| Extra Trees | Calibrated probability of positive exit-adjusted net P&L |
| LightGBM classifier | Calibrated probability of positive exit-adjusted net P&L |
| Ridge regression | Expected exit-adjusted net premium points |
| LightGBM Huber regressor | Expected exit-adjusted net premium points |
| LightGBM quantile regressor | 90th-percentile maximum observed adverse OTM loss, in premium points |

The selected combination is in `otm_models/selection.json`. Models, preprocessing, estimators, scalers and probability calibrators are saved separately and inside the full packages. Tree models do not require scaling; their `scaler.joblib` deliberately contains `None`.

The fit ends before January 2023. January–June 2023 is used for probability/quantile calibration; July–December 2023 is used for model/threshold selection. Labels crossing split boundaries are purged by their end timestamp. All feature columns are allowlisted. January 2024 onward is excluded from fitting and threshold selection. There is no silent refit using the test period.

The 4-week bucket shares global models because it has few independent days. Threshold candidates keep the 3-week threshold equal to or above the 4-week threshold. The selection seeks precision with positive net P&L, profit factor at least 1.5, at least 20 validation trades and 50% day coverage. If no combination qualifies, the best remaining candidate is visibly labeled experimental. These criteria **were not all met**. Drawdown is reported, not constrained by an unmeasured promise.

`atm_reference_models` contains the seven previous release packages and four tail-veto packages. Their live outputs appear in the separate diagnostic table; **none controls OTM entries**. The original model cards and manifests are preserved. Several reference models were trained after 2024 and must not be represented as pre-2024 OTM test models. Their classifiers refer to ATM outcomes; tail-veto values are risk-ranking scores rather than calibrated loss probabilities.

## Stored output

### Editable entry cutoffs and three simultaneous positions

Open `/firstTwoWeeks` and use **Entry cutoffs & position capacity**. The saved default is **50% for the 3-week bucket and 50% for the 4-week bucket**, from the frozen pre-2024 selection. Enter percentages, then click **Save cutoffs**. For example, a 70% cutoff permits a predicted probability of 70% or greater. The selected classifier remains Logistic regression; outputs of all six OTM models remain visible and saved.

The probability test is one entry gate: expected net edge, predicted OTM loss, quote freshness, session time, and available capacity must also pass. Raising the cutoff does not guarantee a higher realized win rate. Manual changes are labeled `operator_override`; they do not modify fitted models or pretend to be a newly validated threshold. The January 2024 onward period is not used to automatically tune these controls.

The new book allows **up to three simultaneous positions total across both buckets**, including reserved pending entries. It can fill a second and third position while others remain open. A fourth is rejected until capacity is released. The code enforces three even if an old configuration says one or a value above three. One lot per position is separate from the number of concurrent positions. This cap applies to the isolated OTM book; the original paper trader keeps its own independent rules.

Saved changes are read by the running OTM worker within about one second and before each new snapshot. The dashboard distinguishes requested settings from the last policy applied by the worker. Pending entries are rechecked before filling and can be cancelled by a raised cutoff. Open positions keep their original fixed contracts, cost-aware target, stop and horizon. If the settings file is invalid, new entries stop and existing exit processing continues; the dashboard and event log show the error.

Every new signal records predicted probability, cutoff, pass/fail, comparison `>=`, settings revision and occupied capacity. Filled positions and closed trades retain the probability plus both signal-time and fill-time cutoff revisions. Existing historical records are retained as recorded and can show blank fields that did not exist before this update.

| Location | Entry-policy audit |
|---|---|
| `entry_settings.json` | Current operator cutoffs, hard cap, revision and update time |
| `settings_history/<revision>.json` | Before/after policy versions saved before publication |
| `logs/YYYY-MM-DD/settings_changes.jsonl` | Append-only history of saved settings |
| `logs/YYYY-MM-DD/events.jsonl` | Worker policy application, policy errors and cancelled pending entries |
| `runtime/<run>.entry_settings.json` | Frozen policy captured when the dashboard launches a replay |
| `backtests/<run>/entry_settings.json` | Exact policy used by that replay; also included in its summary |

The replay snapshots the saved policy once at launch. Later live edits cannot change a replay already running. New signal exports include cutoff, probability pass/fail and revision; trades retain both signal and entry policy metadata. Previous replays are not overwritten. The settings API is `GET` / `POST /firstTwoWeeks/api/settings`; POST accepts `thresholds` (probabilities from 0 to 1) and `expected_revision`, checks for conflicting updates, and is restricted to local same-origin requests.

Even when a strict cutoff produces zero trades, replay exports include an empty trades file with column headers, every evaluated signal, and a daily P&L row for every eligible date marked `NO_TRADE`. Summary counts include all no-trade days and zero gross P&L/costs; an undefined win rate is shown as unavailable. A replay is marked complete only after its exports and ledger are saved.

After this software update, use the original application's normal restart to load the new Python module. After that, cutoff edits apply without restarting, and the worker continues to start with the normal daily paper-trading session. This update does not modify or restart the original application, its dashboard, configuration, models or other extensions.

| Location | Contents |
|---|---|
| `logs/ledger.sqlite3` | Authoritative live signals, positions, tracks, events and current status; WAL transactions |
| `logs/YYYY-MM-DD/signals.jsonl` | Every evaluated snapshot, features, all OTM outputs, thresholds and rejection/entry decision |
| `logs/YYYY-MM-DD/entries.jsonl` | Every filled paper entry and the exact fixed tokens, strikes, expiry, quantities and prices |
| `logs/YYYY-MM-DD/trades.jsonl` | Every closed trade, exit reason, gross P&L, individual costs and net P&L |
| `logs/YYYY-MM-DD/strike_paths.jsonl` | Fixed CE/PE strike values from t through the horizon, including rejected signals |
| `logs/YYYY-MM-DD/atm_outputs.jsonl` | All available archived ATM model outputs, diagnostic only |
| `logs/YYYY-MM-DD/events.jsonl` | Entries, exits, skipped data, expired requests, recovery and errors |
| `runtime/worker.log` | Worker startup/errors; original application logging is untouched |
| `runtime/bridge_status.json` | Worker process state, queue depth and any dropped-event count |
| `backtests/<run>/` | Separate replay ledger and daily JSONL logs, summary, all-model CSV, trades CSV and daily P&L CSV |
| `data/` | Cached source snapshots, full-chain quotes, training labels and data audit |
| `../../../archive/weeks43/backups/` | Original app source and original configuration/dashboard hashes |

JSONL mirrors are append-only; the SQLite ledger is authoritative for deduplication and recovery. Pending orders are not restored as fills after a restart. Open positions retain their entry contracts and require new fresh quotes. A corrupted/unavailable state database prevents the OTM worker from opening a new book. Feed-queue overflow disables new OTM entries and is shown as an error; the existing paper trader continues.

## Backtests and limits

Use **Backtests & validation → Run historical replay** in the new page or:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' 'E:\pipelines\weeks43\backtest_otm.py'
```

The replay uses the source specified by the dataset-location TXT in the training workspace: `E:\v2\data\icharts_option_banknifty_monthly_atm_5strikes.csv`. Fixed-contract pricing comes from the full chain `E:\v2\data\icharts_option_banknifty.csv`, so an ATM move does not remove the sold contract from the book. The earliest eligible expiry is selected using the current snapshot; a future full-day dominant-expiry statistic is not used.

The dataset is sparse. Training labels require exact entry/end quotes and at least 80% observed intermediate minute snapshots; any unobserved intra-minute peak is unknown. Only 115 pre-2023 labeled observations across 34 days survived these rules. Those are correlated observations, not 115 independent trials. Validation on labeled rows excludes incomplete paths, which can bias it; the chronological test replay does not use future exit availability to decide entries. It reports unresolved positions rather than discarding them. The source-report benchmark has already been examined in earlier research, so this is retrospective evidence rather than a new untouched holdout. Lot sizes and execution conditions can differ from the small training sample.

Day counts are source-observed eligible dates, not all NSE trading sessions. A green/red day is the sum of closed-trade net P&L assigned to the entry date. No-trade days are counted separately. P&L excludes unresolved positions from realized totals and lists those positions explicitly. Minute snapshots cannot reproduce live tick stops or bid/ask fills. Margin, return on capital, order-book depth and hedges are not modeled.

## Integration and rollback

The initial integration used guarded forwarding hooks in `live/live_inference.py`. The subsequent daily-service and dashboard repairs also added independent workers and queues, optimized the LZO decoder, and corrected the root dashboard's status labels and old-port routing. Model artifacts, trading rules, original strategy configuration and ledger paths are preserved. The original runtime's regression tests and focused startup, decoder and dashboard tests were run after the relevant changes.

Set `enabled` to `false` in this package's `config.json` before the next app start to disable its worker while retaining historical viewing. For a full rollback, restore `archive/weeks43/backups/live_inference.before_43weeks.py` to the original `live/live_inference.py` while the application is stopped. Do not overwrite any later edits to that file without reviewing the diff.

The worker uses the configured Python 3.13 executable and one model-computation thread at below-normal Windows priority. It does not install or upgrade packages in the existing app environment. `requirements-frozen.txt` records versions used for the new artifacts.

## Verification

```powershell
Set-Location 'E:\pipelines\weeks43'
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' -m pytest -q test_weeks43.py test_entry_controls.py test_completion_audit.py
```

Tests cover the three-position cap, cost-aware target/stop, fixed-strike preservation, no future/pre-decision fills, stale exits, restart deduplication, continued path logging, backward-only features, pre-test split boundaries, point-in-time OI, and old/new HTTP route isolation.
