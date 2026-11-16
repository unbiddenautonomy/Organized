# PCR paper trading extension

> This document records the original installation. Its `E:\` paths and
> scheduled-service state are historical. For this portable checkout, use
> the repository root README and `setup/voltr.py`.

Installed on 2026-10-28 under `E:\pipelines\PCR`.

The normal paper-trading application now serves **http://127.0.0.1:8780/PCR** alongside its existing dashboard. Its existing daily session startup creates an independent PCR worker using the same received feed and daily contract master. Use the existing launcher as usual; no additional scheduled task or market-data receiver is needed. An already running Python process must reach its next normal launch to load the code change. Installation did not restart the existing trading application or create live paper trades.

## Trading rules

- BANKNIFTY monthly ATM short straddle, one lot per position. The daily contract master supplies expiry, option tokens and lot size.
- **Three simultaneous PCR positions are allowed.** A fourth qualified signal is logged as `capacity_full`. Once a position exits, a subsequent qualified signal can use that slot. Rejected signals are not queued for later execution. Several positions may hold the same strike when successive signals choose it.
- The three-position limit is PCR's own limit. The original strategy retains its own independent positions and limit.
- Entries: 10:15:00 through 13:00:00 IST. A late packet observed at 13:00:01 cannot enter retroactively at 13:00.
- Net target **+INR 3,000** and net stop **-INR 1,500**, per position, after statutory charges and assumed round-trip slippage. The first qualifying observed price is used; gaps can overshoot the target or stop.
- Select 120 minutes if predicted 120-minute net points exceed predicted 60-minute net points by more than 5 points; otherwise select 60 minutes. These are the **two deployed horizons**.
- Exit at the earliest observed target, stop, selected holding-period expiry or 15:14:30 cutoff. A time exit takes priority when both time and price conditions first appear in the same snapshot.
- Strike, expiry, CE/PE tokens, quantity and entry-time exit/cost settings remain fixed for each position throughout its life. A new ATM strike never replaces an existing position's contracts.

## Four model outputs and probability thresholds

| Goal | Selected family | Output |
| --- | --- | --- |
| Straddle decay | Ridge | Expected 60-minute net points |
| Win probability | Logistic regression + saved calibration | Probability of profitable 60-minute historical trade |
| Holding-period selector | Ridge multi-output | Expected 60- and 120-minute net points |
| Tail-risk veto | Logistic regression + saved calibration | Probability that the minimum 60/120-minute net outcome is at most -40 points |

All four models take the same 46 causal features. No model prediction is fed into another model as an input. Their outputs are combined by the decision rules. Each model's existing imputer/scaler/calibrator is reused without fitting on live data.

The initial gates reproduce the selected saved strategy:

| Expiry group | Remaining expiry days | Minimum win probability | Expected 60m points must exceed | Maximum tail probability |
| --- | --- | --- | --- | --- |
| 0 | 0 to under 7 | 59.53519546951542% | 1.5914294141330623 | 12.5% |
| 1 | 7 to under 14 | 48.99177527467293% | 0 | 15% |
| 2 | 14 to under 21 | 51.55104479417535% | 0 | 10% |
| 3-4 | 21 or more; feature week capped at 4 | 49.87736581923341% | 0 | 12.5% |

On `/PCR`, enter the desired percentages under **Win probability cutoffs** and choose **Save cutoffs**. For example, entering 65 requires a calibrated probability of at least 0.65 for that group. The decay and tail gates must also pass, quotes and features must be usable, time must be allowed, and a position slot must be available. Defaults come from the saved research configuration; installation did not search the historical test for a new cutoff.

Changes persist to `config.json`, are picked up on the next signal minute, and appear in the signal record and daily `config_changes.jsonl`. Existing positions retain their entry-time exit rules. The UI rounds the displayed percentage to four decimals but preserves the full stored value if that control is unchanged.

## Costs and execution observations

Brokerage is zero. Live rates are inherited from the host `live_config.json` at session startup and copied into every position: sell-side STT, exchange charges, SEBI charges, buy-side stamp duty and GST on exchange/SEBI charges. Installation did not change the host rates. Two combined-straddle premium points per quantity unit are charged as round-trip slippage, once per position, including in provisional net MTM. Lot size comes from the current contract master.

Quotes are last-traded observations, not guaranteed executable bid/ask fills. Both exact leg prices must be no older than 60 seconds and no more than 10 seconds apart; OI may be no older than 900 seconds. Signal processing delay must be within -1 to +10 seconds in live mode. These limits are saved in `config.json`.

When an exact leg quote is unavailable at an exit time, the position remains open and shows a pending exit until fresh exact quotes arrive. The code does not fabricate a 15:14:30 fill. Any unresolved prior-day position blocks new PCR entries until it is reconciled or receives usable exact-contract prices. Feed, worker and pending-exit status are visible on the page.

## Causal inputs and startup

Features use the preceding completed minute, never later prices or future labels. Raw snapshots are received through a bounded queue into a separate Python process. PCR builds its own minute history because the host's existing feature path has a different finalization delay.

The longest feature lookback is 60 minutes; none of these four packages has a 120-minute input lookback. The 120-minute value is an output/holding horizon. Warmup requires 61 completed valid minute observations, so with uninterrupted data from 09:15 the first eligible inference is approximately 10:16. Starting late or missing minutes can delay it. At most one missing minute is forward-filled in the feature frame; there is no backward fill. Training feature-formula parity and future-data invariance are tested.

The worker checks model hashes before loading. An operating-system lock prevents a second worker from owning the same PCR ledger. Entry/exit journals reconstruct positions after restart; already processed signal IDs prevent duplicate entries. Windows snapshot writes retry transient reader locks while retaining atomic replacement. Queue overflow, transport faults, excessive delay or invalid inputs block new PCR entries and do not change the original strategy.

## Saved files

| Path under PCR | Contents |
| --- | --- |
| `models/straddle_decay_model/` | Complete selected decay package |
| `models/win_probability_model/` | Complete selected calibrated win package |
| `models/holding_period_model/` | Complete selector, separate 60/120m scalers and imputers |
| `models/tail_risk_model/` | Complete calibrated tail-risk package |
| Each model folder | Bundled and standalone pipeline joblibs, scaler/imputer components, calibration where applicable, feature schema, thresholds, metrics, inference/training code, requirements, model card, package README and hash manifest |
| `research/` | Copied training, selection and historical-backtest sources |
| `config.json`, `thresholds.json`, `training_metadata.json` | Live settings, original gates and source training metadata |
| `requirements.lock.txt` | Exact installed ML package versions used for verification |
| `features.py`, `runtime.py`, `worker.py`, `worker_lock.py`, `bridge.py`, `storage.py` | Live features, inference, ledger, feed bridge and persistence |
| `web.py`, `dashboard.html`, `backtest.py` | HTTP integration, dashboard and backtest entry point |
| `reference/` | Original historical candidates and full max-three results |
| `integration_backup/` | Before-change host Python, configs, dashboard and original hashes |
| `validation/` | Test reports, dashboard screenshot and installation audit |
| `deployment_manifest.json` | Installation inventory, file hashes and validation summary |

Daily data is saved under `logs/YYYY-MM-DD/`:

- `session_context.json`: contracts, lot, expiry, charge rates and initial settings.
- `inputs.jsonl`: received normalized snapshots, observed processing delays and settings changes, suitable for replay.
- `minute_rows.jsonl`: completed causal ATM observations.
- `signals.jsonl`: every minute decision including warmup, rejection, all 46 inputs when available, all five numerical outputs from the four models, gates, threshold, selected horizon, strike and entry outcome.
- `signal_intents.jsonl` and `signal_actions.jsonl`: decision and action audit trail.
- `trades.jsonl`: every entry and exit with frozen strike/expiry/tokens, quantities, model outputs and itemized costs.
- `marks.jsonl`: observed fixed-contract gross/net P&L path.
- `config_changes.jsonl`: applied settings changes.

`logs/status.json`, `logs/worker.log` and `bridge_status.json` provide current state and diagnostics. No production logs were seeded with simulated test trades.

## Backtesting

The page offers two separate actions. **Rerun original historical backtest** uses copied frozen model predictions and research code; it does not refit or adopt changed live cutoffs. It requires the raw cache specified in `config.json`, currently `E:\PCR\hypothesis testing\output\win_classifier\all_m1_rows_features_v2.pkl`. That research data dependency is not needed for live inference.

**Replay selected recorded day** runs that day's saved snapshots and original settings in a separate directory, including the recorded delay gate and any recorded transport fault. It requires that day's PCR inputs and context. Any unresolved carry-in positions are restored from the initial session context so replay preserves their risk and blocking behavior. Closed trades from earlier days are not included in that replay's cumulative totals. It is an execution replay, not a search for a better threshold.

All jobs write to their own `backtests/<job_id>/` directories and never place entries in the live PCR book. The historical reference was exactly reproduced during installation in `backtests/installation_reference_check/`:

| Metric | Original reproduced result |
| --- | --- |
| Trades; green / red | 902; 575 / 327 |
| Trade win rate | 63.7472% |
| Gross; total costs; net INR | 158,754.80; 59,844.75; **98,910.05** |
| Market days; active days | 311; 183 |
| Green / red / flat days | 121 / 62 / 128 |
| Maximum daily-close drawdown INR | -25,233.77 |
| 60m / 120m trades | 376 / 526 |
| Target / stop / time exits | 4 / 60 / 838 |
| Capacity rejections | 4,218 |
| Synthetic historical exits | 178 |

This historical result uses 2024 through May 2025 minute data, historical lot sizes/fees and 178 flagged synthetic exit valuations. Live PCR uses observed exact-contract prices and the host's current rates. These are different execution assumptions; the reference is not a prediction of live P&L. The test period has already been inspected repeatedly while comparing strategy settings, so it is a reviewed historical diagnostic, not a fresh untouched evaluation. See `DEPLOYMENT_MODEL_CARD.md` for interpretation.

## Validation and operation

Use the same Python environment as the existing app. Verification used `py -3.13`; the system's unqualified `python` points to a different environment on this computer. No packages were installed or upgraded during integration.

From `E:\pipelines`, run `py -3.13 -m pytest -q PCR/test_pcr.py` for the PCR tests. They test three concurrent positions, fourth-entry rejection and slot reuse, journal recovery, net target/stop and costs, stale cutoff behavior, feature parity/future invariance, probability gating, package hashes/prediction parity, old/new HTTP route coexistence, unchanged host configs, duplicate-worker exclusion and real worker/recorded-replay parity. Test ledgers are temporary and separate from production logs.

Optional offline page inspection: `py -3.13 -m PCR.web --port 8781`, then open `http://127.0.0.1:8781/PCR`. This only serves the page; it does not start inference or consume the live feed. The normal trading application owns the production `/PCR` route on port 8780.

The original `live_config.json`, `config.yaml` and `live_dashboard.html` remain byte-for-byte unchanged. Only `live/live_inference.py` has additive imports, namespaced routes and exception-isolated feed/session hooks. No original model or position state was modified. The original app's 24 regression tests passed.

To disable new PCR entries, set its `config.json` `enabled` to `false`; an active worker picks this up on the next minute and continues managing existing PCR positions. At the next host session startup, a disabled PCR worker does not start. Avoid disabling startup while PCR has unresolved positions. For full removal, after normal shutdown restore only `live/live_inference.py` from `integration_backup/live_inference.py`; retain PCR's artifacts and journals for audit.
