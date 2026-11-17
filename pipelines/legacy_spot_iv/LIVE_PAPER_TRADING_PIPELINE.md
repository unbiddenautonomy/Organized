# Live BANKNIFTY paper-trading pipeline

This document describes the live, paper-only trading code under
`E:\Models\Live Trades`. It covers both executable paper strategies:

1. the default multi-horizon spot/IV/P&L engine started by `main.py`; and
2. the separate 120-minute expanded-hybrid worker started by
   `banknifty_120m_live_inference.py`.

Neither worker contains a broker API. Entries and exits are simulated from the
NSE market-by-price feed and written to append-only JSONL files.

## System map

```text
NSE CM multicast (7207) ──> BANKNIFTY spot ──> one-minute spot features
                                                    │
                                                    ├─> spot horizon models
                                                    │
NSE FO multicast (7208) ──> futures/options books ──┼─> ATM IV features/models
NSE FO OI messages ───────> option OI state ────────┘
                                                    │
                           synchronized spot + IV forecasts
                                                    │
                           HMM regime + P&L/MAE model stack
                                                    │
                           horizon selection and guardrails
                                                    │
                           paper short-straddle position
                                                    │
                           mark at executable asks and exit
```

The two multicast receivers use the configured interface, NSE packet framing,
LZO decompression and the existing 7207/7208 decoders. Exchange timestamps, not
local computer time, drive minute aggregation and trading decisions.

## 1. Default multi-horizon paper engine

### Entry point and components

Run `main.py` to create and connect:

- `SpotPredictionPipeline` from `spot_inference.py`;
- `IvPredictionPipeline` from `iv_inference.py`;
- `LivePnlInferenceAdapter` from `pnl_live_inference.py`;
- `PaperTradingEngine` from `paper_trading/engine.py`;
- the local paper-trading dashboard; and
- `LiveRetrainingCoordinator` for shadow residual models.

`main.py` registers the paper engine and retraining coordinator as listeners on
the spot and IV prediction pipelines. The paper engine considers an entry only
after both pipelines have emitted forecasts for the exact same exchange minute.

### Market-data path

1. Load the BANKNIFTY option/future contract universe and map instrument tokens
   to contract metadata.
2. Join the configured CM and FO multicast groups.
3. Decode BANKNIFTY index messages from CM transaction code 7207.
4. Decode future and option market-by-price updates from FO transaction code
   7208.
5. Keep five bid and five ask levels per tracked instrument in memory.
6. Decode separate OI messages when present and retain their exchange time for
   freshness checks.
7. Seal synchronized one-minute spot, future and option observations.

Replay is supported. If exchange time rewinds beyond the configured tolerance,
the feature histories, quote state, OI freshness and paper-session state are
reset. Output JSONL files remain append-only.

### Current model stack

The source supports several horizons, but the current `config.json` sets
`paper_prediction_horizons` to only `5` and `15` minutes.

#### Spot forecast

For each configured horizon:

- algorithm: `XGBRegressor`;
- selected artifact name: `xgb_absolute`;
- inputs: current spot/future/option state plus training-compatible lagged and
  rolling features;
- output: predicted log return and terminal BANKNIFTY spot.

The current production artifacts are loaded from
`spot_model_dir` (`...\spot_horizon_warmup\fold_05`). The source can emit
5/15/30/45/60-minute forecasts, but only horizons shared with the IV and paper
configuration can reach entry selection.

#### IV forecast

For the 5- and 15-minute horizons:

- selected artifact: `hist_gbdt_median`;
- algorithm: saved preprocessing pipeline plus
  `HistGradientBoostingRegressor`;
- target: change in the current fixed ATM pair's combined implied volatility;
- output: `predicted_iv_change` and `predicted_iv`.

The IV pipeline selects the minute's nearest ATM CE/PE pair, reconstructs
implied volatility and Black-Scholes Greeks, and builds the saved model's exact
feature order. Missing live values are handled by the preprocessing stored with
the artifact.

#### Regime, P&L and adverse-excursion stack

For each configured horizon, `LivePnlInferenceAdapter` combines the synchronized
spot and IV forecasts with current market features and produces:

- a three-state `GaussianHMM` regime posterior and next-state fields;
- LightGBM mean expected P&L;
- LightGBM P&L q10, q50 and q90 estimates;
- a LightGBM profitable-trade classifier;
- LightGBM MAE q50 and q90 adverse-excursion estimates; and
- an `hmm_in_distribution` gate.

The P&L layer therefore does not merely compare predicted IV direction. It
scores the complete short-straddle opportunity using spot, IV, regime and
downside forecasts.

### Entry evaluation

Entry consideration starts only when:

- paper trading is enabled;
- there is no open paper position;
- both spot and IV events exist; and
- the two events have the same exchange minute.

The engine then applies these session-level gates:

| Gate | Current configured value |
|---|---:|
| Entry start | minute 565 = 09:25 |
| Entry end | minute 1260 = 21:00 |
| Daily realized-loss limit | 1% of starting capital |
| Maximum trades/day | 500 |
| Entry cooldown | 5 minutes |
| India VIX required | yes |
| Maximum India VIX | 35 |
| Maximum five-minute VIX rise | 1.0 point |
| IV model warmup | required |

The 21:00 entry end is later than the normal BANKNIFTY session, so it currently
does not constrain ordinary market-hours entries. This is a configuration fact,
not the default used in the Python fallback values.

### Horizon eligibility and ranking

For each of the configured 5- and 15-minute horizons, the engine requires a
spot forecast, IV forecast, P&L forecast and enough time before the configured
forced exit. The horizon is rejected unless all of the following pass:

| Forecast gate | Current configured value |
|---|---:|
| Maximum predicted IV change | `0.00025` |
| Minimum expected P&L | `-150` rupees |
| Minimum profit probability | `0.15` |
| Minimum P&L q10 | `-1100` rupees |
| Maximum MAE q90 | `3000` rupees |
| HMM observation in distribution | required |

Eligible horizons are ranked by:

```text
selection score = expected P&L × profit probability
                  ─────────────────────────────────
                  max(abs(min(P&L q10, 0)), 100)
```

The highest score wins. A tie prefers the shorter horizon.

Because the current minimum expected P&L and probability are permissive
(`-150` and `0.15`), an opportunity does not need positive expected P&L or a
50% profitable-trade probability to be eligible. These are the live configured
values and should not be confused with the more conservative Python defaults.

### Direction, strike and structure

After selecting a horizon:

1. Compute `predicted_spot - current_spot`.
2. If the absolute move is below 50 points, label it `NEUTRAL` and use current
   spot as the strike target.
3. Otherwise label it `UP` or `DOWN` and use predicted terminal spot as the
   strike target.
4. Select the listed strike nearest that target.
5. Sell the CE and PE at that same strike: the structure is always a short
   straddle.

This means a directional forecast does not create a call-only or put-only
trade. It shifts the centre strike of the short straddle toward predicted spot.

### Executability and market guardrails

Before entry, the engine requires:

- a best bid and best ask for both CE and PE;
- each leg's `(ask - bid) / ask` spread to be at most 5%;
- current spot to be strictly inside the selected straddle's break-even range;
- risk budget to allow at least one lot; and
- margin budget to allow at least one lot.

OI freshness is measured and logged. The current configuration has
`paper_require_fresh_oi=false`, so stale or missing OI is advisory rather than a
hard rejection.

### Paper execution and sizing

Entry assumes both option legs are sold at their current best bids:

```text
entry credit = CE best bid + PE best bid
lower break-even = strike - entry credit
upper break-even = strike + entry credit
```

Current sizing inputs are:

| Setting | Value |
|---|---:|
| Starting capital | ₹1,000,000 |
| Lot size | 30 |
| Maximum model lots | 4 |
| Fixed risk budget | ₹6,000 |
| Stop loss per lot | ₹1,500 |
| Maximum margin fraction | 70% |
| Base margin per lot | ₹150,000 |

Risk capacity is `floor(6000 / 1500) = 4` lots before the margin and model
limits. The margin model uses the greater of configured base margin and the
break-even stress estimate with its buffer.

The P&L forecast is considered high confidence only when:

- selection score is at least `1.9229049019`;
- P&L q10 is no worse than `-₹1,500`; and
- MAE q90 is no more than `₹1,500`.

High confidence permits up to four lots; otherwise the model allowance is one
lot. Risk, margin and model allowances are combined by taking their minimum.
When India VIX is at least 22, the configured size multiplier is 0.5 and the
result is floored to a whole number of lots.

Although the code implements staged entry for a 15-minute trade, the current
configuration has `paper_15m_two_stage_entry=false`. Consequently all approved
lots enter immediately. If re-enabled, the first half enters at time `t` and
the reserved lots are reconsidered at `t+5` using an interpolated remaining
10-minute spot/IV forecast and fresh quote/risk checks.

### Position marking and exits

An open short straddle is marked using executable buyback prices:

```text
current debit = CE best ask + PE best ask
unrealized P&L = (entry credit - current debit) × lot size × open lots
```

Exit checks run in this order:

1. configured end-of-day forced exit;
2. fixed loss of ₹1,500 per open lot;
3. spot crossing either break-even boundary;
4. fixed profit target of ₹3,000 per open lot; and
5. selected model-horizon expiry.

At the planned horizon, the engine may extend the position in 15-minute steps,
up to four extensions and no more than 120 total minutes. An extension requires
a fresh stacked forecast with:

- profit probability at least 0.55;
- non-negative expected P&L;
- P&L q10 at least `-₹500`;
- MAE q90 at most `₹3,000`; and
- an in-distribution HMM observation.

If no supporting forecast passes, the position exits at the model horizon.
Paper exits buy both legs at their current best asks. The accounting subtracts
the configured ₹100 round-trip cost per lot.

The current `paper_force_exit_minute` is 1320 (22:00), which is outside the
normal market session. Therefore the configured forced-exit check will not
normally trigger before market data ends. This should be reviewed before using
the runtime as an operational session-close simulator.

### Outputs and dashboard

Default paper-engine outputs:

| Path | Contents |
|---|---|
| `logs/spot_predictions.jsonl` | spot forecasts by horizon |
| `logs/iv_predictions.jsonl` | IV forecasts by horizon |
| `logs/paper_decisions.jsonl` | accepted and rejected entry decisions with reasons |
| `logs/paper_trades.jsonl` | entries, scale-ins, extensions and exits |
| `logs/runtime.log` | receiver and runtime diagnostics |
| `logs/intraday_shadow_predictions.jsonl` | intraday residual-model shadow outputs |
| `logs/retraining_live_outputs.jsonl` | combined base/shadow retraining outputs |

The current dashboard configuration is `http://127.0.0.1:8766`.

### Intraday retraining

The retraining coordinator observes the same spot and IV prediction events as
the paper engine. When a forecast horizon matures, it settles the prediction
against the later observation and writes training rows to SQLite.

Every 120 exchange minutes, a background worker can fit horizon-specific
residual corrections. A chronological holdout is used, and a candidate is
published only when it improves on the unchanged base forecast. Published
artifacts are immutable versions with an atomically updated `CURRENT` pointer.

These intraday candidates run in shadow mode. The large production spot/IV
estimators, P&L models, MAE models and HMM are not refitted or replaced in the
live selector thread.

## 2. Separate 120-minute expanded-hybrid worker

`banknifty_120m_live_inference.py` is independent of `main.py` and the default
paper engine. It has its own dashboard, logs, position class and entry policy.

### Contract and feature construction

The worker loads the nearest non-expired monthly BANKNIFTY expiry, finds the
strike nearest spot and requires a five-pair window from ATM-2 through ATM+2.

For all five strikes it requires:

- both CE and PE quotes;
- quote age no greater than 120 seconds; and
- leg spread no greater than 5%.

It reconstructs IV across the five strikes, smile/skew, Greeks, ATM OI ratios,
spot realized-variance features and intraday time/DTE fields.

### 120-minute model stack

The live decision hierarchy uses three frozen components:

1. **Base realized-variance model**
   - `StandardScaler` followed by `Ridge(alpha=300)`;
   - 16 features;
   - artifact name `Realized_GARCH_extended`;
   - predicts log 120-minute total variance;
   - applies saved clipping and log-bias correction before exponentiation.

2. **Primary expanded-hybrid classifier**
   - `StandardScaler` followed by `LogisticRegression(C=0.01)`;
   - 29 ordered structure/regime/Greeks features;
   - scores a short ATM-2 PE plus ATM+2 CE strangle;
   - profitable-trade threshold `0.6931277592709387`.

3. **Secondary expanded-hybrid classifier**
   - the same scaler/logistic architecture and 29-feature contract;
   - scores an ATM short straddle;
   - profitable-trade threshold `0.7016246301834123`.

### 120-minute entry hierarchy

The first eligible component wins:

#### A. Base RV-versus-IV gate

The worker calculates:

```text
predicted total variance = exp(model output + saved bias correction)
implied total variance   = ATM IV² × 120 / (252 × 375)
variance ratio           = predicted variance / implied variance
theta-decay estimate     = abs(ATM straddle theta/day) × 120 / 1440
expected long P&L points = theta-decay estimate × (variance ratio - 1)
```

It enters a one-lot ATM short straddle when:

```text
expected long-straddle P&L in rupees
    < -2 × estimated one-lot round-trip cost
```

In other words, the base trade requires the variance model to predict that a
long ATM straddle loses more than twice the estimated cost hurdle.

#### B. Primary hybrid fallback

If the base gate does not trade, no position is open and the daily fallback has
not been consumed, enter one lot of:

```text
short ATM-2 PE + short ATM+2 CE
```

when primary probability is at least `0.6931277592709387`.

#### C. Secondary hybrid fallback

If the primary threshold does not pass, enter a one-lot ATM short straddle when
secondary probability is at least `0.7016246301834123`.

The live policy uses the first causal threshold crossing. It does not reproduce
the offline research policy that selected the best score after observing the
entire day.

### 120-minute trade controls and execution

- Session starts at 09:15.
- New entries stop after 13:30.
- Maximum 10 trades per day.
- Only one position can be open.
- At most one hybrid fallback is allowed per day.
- Every trade is one lot.
- Entry sells both legs at their best bids.
- The entry strikes and expiry remain fixed.
- Exit occurs after 120 minutes or at 15:29.
- Exit buys both legs at their best asks.
- Bid/ask execution embeds spread; statutory costs are added separately.

The base model needs 61 consecutive spot minutes and at least six IV minutes.
The hybrid feature builder additionally needs 31 IV minutes and 61 OI-history
observations. If this history is unavailable, the base gate can still operate
but the hybrid classifiers cannot produce a live probability.

### 120-minute outputs

By default the worker writes under `logs\rv120`:

| Path | Contents |
|---|---|
| `rv120_signals.jsonl` | every warmup, wait, no-trade and short decision |
| `rv120_paper_trades.jsonl` | simulated entries and exits |
| `rv120_latest_status.json` | latest dashboard state |
| `runtime.log` | worker diagnostics |

The worker dashboard defaults to port 8770 but can be changed with
`--dashboard-port`.

## 3. Inference-only fixed-strike stack in `E:\v2`

`E:\v2\Inference\live\live_inference.py` is a third, separate runtime. It
constructs the fixed-strike IV/RV model contracts and publishes predictions to
its dashboard and JSONL logs, but it explicitly has no position, entry or broker
integration. Its convergence outputs must not be mistaken for an executed paper
trade.

## Running the systems

### Default multi-horizon engine

```powershell
cd 'E:\Models\Live Trades'
python .\main.py
```

Configuration is read from `config.json`; another file can be supplied with
`--config`.

### 120-minute worker preflight

```powershell
cd 'E:\Models\Live Trades'
python .\banknifty_120m_live_inference.py `
  --contract-file 'E:\Models\seqToseq\contracts\NSE_FO_contract_06102026.csv.gz' `
  --preflight
```

### Start the 120-minute worker

```powershell
cd 'E:\Models\Live Trades'
python .\banknifty_120m_live_inference.py `
  --contract-file 'E:\Models\seqToseq\contracts\NSE_FO_contract_06102026.csv.gz' `
  --dashboard-port 8870
```

## Operational checklist

Before a live paper session:

1. Confirm the multicast interface IP and CM/FO ports in `config.json`.
2. Confirm the contract file/database contains the current BANKNIFTY expiry and
   correct lot size.
3. Run model/contract preflight for the 120-minute worker.
4. Verify all configured artifact directories exist and can be deserialized.
5. Review `paper_entry_end_minute` and `paper_force_exit_minute`; the current
   21:00/22:00 values are outside normal market hours.
6. Confirm whether OI is intended to be advisory or mandatory.
7. Confirm the permissive P&L/probability entry thresholds are intentional.
8. Start only the desired paper engine, or keep their logs and dashboards
   clearly separated if both are run.
9. Verify dashboard health and wait for model warmup before interpreting
   missing signals as `NO_TRADE`.
10. Treat JSONL events as simulations: neither path submits broker orders.

## Source-of-truth files

- `config.json` — active default-engine configuration.
- `main.py` — default runtime wiring and receiver loop.
- `spot_inference.py` — spot feature and prediction pipeline.
- `iv_inference.py` — IV/Greeks feature and prediction pipeline.
- `pnl_live_inference.py` — HMM/P&L/MAE stack.
- `paper_trading/engine.py` — default entry, sizing and exit policy.
- `live_retraining.py` — shadow residual-training lifecycle.
- `banknifty_120m_live_inference.py` — separate expanded 120-minute strategy.
- `rv120_dashboard.html` — 120-minute worker dashboard.

When this document conflicts with executable code or `config.json`, the code and
loaded configuration are authoritative.
