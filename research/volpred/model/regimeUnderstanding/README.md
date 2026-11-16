# regimeUnderstanding

`regimeUnderstanding` learns descriptive regimes for the Greeks of an ATM long
straddle (one ATM call plus one ATM put). It uses only
`data/raw/icharts_option_banknifty_monthly_atm.csv`.

Run it from the repository root with:

```powershell
py -3.13 research/regime_understanding.py
```

The fitted training thresholds are written to
`artifacts/regime_thresholds.json`. Detailed windows, regime episodes, duration
summaries, P&L attribution and the generated report are written to
`outputs/regime_understanding/`.

The model folder also contains two analysis areas:

- `greek_time_evolution/`: unconditional Greek changes across the five horizons
- `iv_rv_scenarios/`: Greek changes conditional on `IV < RV`, `IV = RV`, or
  `IV > RV`
- `trade_policy/`: same-expiry intraday trade equations, walk-forward audit,
  and the resulting paper-only decision policy
- `five_strike_trade_research/`: ATM and symmetric +/-1 and +/-2 strike
  structure research for each predicted IV/RV scene
- `directional_debit_spreads/`: walk-forward tests of ATM long calls/puts,
  one- and two-strike debit spreads, and bought opposite-side safety options
- `hybrid_strategy/`: portfolio-level backtest combining the validated IV/RV
  and directional signal streams with one open trade at a time
- `profit_optimization/`: 2024-only hybrid rule search and frozen 2025 holdout
  evaluation aimed at increasing profit without adding leverage
- `coverage_optimization/`: higher-frequency monthly-expiry rule selected with
  a minimum distinct-trading-day constraint and frozen holdout evaluation
- `retrained_hybrid/`: direct short-strangle P&L model trained on 2019-2022,
  selected on 2023, and combined as a no-signal-day fallback for 2024-2025
- `expanded_signal_hybrid/`: validation-only coverage/profit expansion using a
  wider 120-minute short strangle first and an ATM short straddle only on
  dates still unused by the higher-priority components
- `iv_rv_convergence_backtest/`: cost-stressed test of current IV versus
  trailing current RV, combined with forecast IV convergence toward predicted
  RV; the tested rule failed its 2024 development requirement and was not
  added to live inference
- `iv_rv_transition_profit_optimization/`: keeps the IV/RV transition
  hypothesis mandatory and adds a cost-coverage gate plus aligned observed
  five-minute IV confirmation; the short-vol candidate passes historical
  development and holdout but remains forward-paper-only
- `refined_variance_transition/`: expresses IV/RV convergence in variance
  space, scales near-equality by causal forecast uncertainty, includes gamma,
  theta, vega, and volga in the cost gate, and freezes the 2024-selected rule
  before evaluating the partial-2025 holdout
- `high_coverage_refined_hybrid/`: gives the refined transition rule priority
  and uses previously validated fallbacks only on otherwise-empty dates; it
  also records the rejected 2023 threshold frontier showing why forcing six to
  seven signal months per calendar year was not supported
- `near_daily_short_vol_risk_managed/`: superseded result. Its daily entry
  selection used the maximum score from the completed session and therefore
  contained intraday look-ahead. Do not use its reported profit or policy.
- `near_daily_short_vol_causal/`: leakage-corrected audit of the frozen
  near-daily rule. It enters at the first live-observable threshold crossing,
  keeps the entry strike fixed, applies pathwise exits, and rejects the rule
  because causal 2023 validation is negative.
- `causal_validation_short_vol/`: clean rebuild that trains on 2019-2022,
  calibrates a high-coverage first-crossing threshold on 2023 H1, and performs
  one untouched validation run on 2023 H2. This is the authoritative result;
  the strategy is rejected because validation P&L is negative after costs.
  The same frozen rule was subsequently measured on 2024 through 2025-05-09:
  2024 lost money, partial 2025 made money, and the combined test remained
  negative after costs. This diagnostic test does not reverse the rejection.
- `leakage_audit/`: machine-readable status of reports created before the
  exit-filter and completed-session entry leaks were corrected.
- `causal_profitable_hybrid/`: positive historical walk-forward paper
  candidate. It combines a fixed DTE 0-3 short-ATM sleeve with a pre-2024-
  selected Huber direct-P&L gate for DTE 8-14. Every model row is fit only on
  labels whose exits precede the signal. The 2024 through partial-2025 replay
  is positive after costs, but its bootstrap interval includes zero and the
  period had already been inspected elsewhere, so it is not live-approved.

## Current validity status

The only authoritative strategy result after the leakage repair is
`causal_validation_short_vol/`. Reports in the folders listed by
`leakage_audit/artifact_status.csv` are stale pre-fix artifacts and must be
rebuilt before their P&L may be cited. Source code now treats exit-derived
`stale_any` and `tiny_any` fields as diagnostics only and uses the first
observable threshold crossing rather than the best timestamp of the day.

For the clean validation, 2019-2022 is training, 2023 H1 is threshold
calibration, and 2023 H2 is untouched validation. Later years are not used to
rescue or retune a failed validation rule.

## Train, validation, and test split

| Split | Date range | Purpose |
| --- | --- | --- |
| Train | 2019-01-01 through 2022-12-31 | Fit model parameters and learn regime definitions. |
| Validation | 2023-01-01 through 2023-12-31 | Tune hyperparameters, features, thresholds, and model selection. |
| Test | 2024-01-01 through 2025-12-31 | Perform the final out-of-sample evaluation after all choices are frozen. |

The split must remain chronological. Rows must not be shuffled across periods,
and any rolling feature at a split boundary may use only information available
at or before that row's timestamp. Test data must not influence feature design,
hyperparameter selection, regime labeling rules, or decision thresholds.

If walk-forward refitting is evaluated inside the test period, report it
separately from the fixed-model test and fit each step using only observations
strictly earlier than the prediction timestamp.

## Training-period analysis

The implemented analysis reads only 2019–2022 rows. It covers 15, 30, 60, 120,
and 180-minute horizons and includes gamma, vega, theta, vanna, volga, and
charm. Each combined straddle Greek is assigned a training-sample tercile
regime:

- `low`: value at or below its 33.33rd percentile
- `normal`: value between its 33.33rd and 66.67th percentiles
- `high`: value at or above its 66.67th percentile

A regime duration is a contiguous intraday run of the same state. A new episode
starts when the state changes, the trading date changes, the active expiry
changes, or the data gap exceeds two minutes.

Greek changes follow the ATM row at the future timestamp. P&L attribution is
stricter: it is reported only if the original entry strike is still present as
the ATM row at the exit timestamp, because the ATM-only CSV cannot price a
fixed strike after it leaves ATM. This avoids treating a re-centered option as
the same contract, but biases the P&L sample toward smaller spot moves.

The P&L reference is a long ATM straddle. Short-straddle contributions have the
same magnitude with the opposite sign. Results are gross points and exclude
transaction costs.

The script selects the nearest monthly expiry (`dte_flag = M1`) and recomputes
the Greeks from the CSV's ATM strike, synthetic forward, spot, time to expiry,
and call/put IV. This keeps vega, theta, vanna, volga, and charm on one explicit
unit convention instead of mixing vendor-specific Greek scaling.

## Intraday boundary

All observations are restricted to the BANKNIFTY trading session from 09:15
through 15:30. Both ends of every horizon must occur in the same session, so
the latest permitted entry depends on the horizon:

| Horizon | Latest entry | Latest exit |
| --- | --- | --- |
| 15 minutes | 15:15 | 15:30 |
| 30 minutes | 15:00 | 15:30 |
| 60 minutes | 14:30 | 15:30 |
| 120 minutes | 13:30 | 15:30 |
| 180 minutes | 12:30 | 15:30 |

Overnight observations and windows that would cross the 15:30 close are not
included.

## IV versus RV scenes

IV is converted from calendar-time ATM IV to the same annualized trading-time
clock as forward realized volatility. Exact equality is not stable for
continuous estimates, so `IV = RV` is defined as a difference within plus or
minus 0.5 volatility points. Observations above that band are `IV > RV`, and
observations below it are `IV < RV`.

