# Research-grade split protocol

Status: **FROZEN BEFORE NEW MODEL TRAINING** on 2026-10-13 Asia/Kolkata.
This protocol is intentionally research-grade. The pseudo-holdout has been
exposed to prior repository research and can never support an acceptance claim.

## Dataset and prediction task

- Source: `data/raw/icharts_banknifty_spot_1min.csv` after the deterministic
  cleaning rules in `research/integrity.py`.
- Clock: Asia/Kolkata. A feature row is eligible only when its complete trailing
  window is available at the decision time. No forward fill crosses a bad spot,
  missing bar, session boundary, or gap over 61 seconds.
- Sampling: ordinary weekday sessions, every fifth recorded minute from 09:45
  through 13:30 inclusive. The same timestamp sample is used for all models.
- Primary horizon: 120 trading minutes.
- Target: `rv_fwd_120 = sum(log(S[t+i]/S[t+i-1])^2, i=1..120)`.
  The decision-bar return is excluded. `label_end_at` is the timestamp of the
  final included return and must be no later than 15:30.
- Model target: `log(max(rv_fwd_120, 1e-12))`; forecasts are transformed back
  with a training-only lognormal smearing correction. QLIKE is computed on the
  positive variance scale and log-MSE on the log target.
- Baseline: Ridge HAR using only trailing log RV over 5, 15 and 60 minutes.
  Ridge alpha is selected inside each training fold from `{0.01,0.1,1,10,100}`.

## Immutable calendar split

| Segment | Exact rule | Permitted use |
|---|---|---|
| Development | decision date 2019-10-11 through 2022-12-30 | feature/model design and inner model selection |
| Walk-forward validation | decision date 2023-01-02 through 2024-06-28 | expanding monthly OOS predictive evaluation; no direct parameter optimization |
| Pseudo-holdout, previously exposed | decision date 2024-07-01 through 2025-05-22 | opened once after algorithms, features, hyperparameters and promotion rule are frozen |

If the source lacks one of these boundary dates, the first/last available
eligible session inside the stated interval is used; the written calendar
boundary is never moved. Missing data do not create a replacement interval.

## Purge and embargo invariant

For every fit, each training row must satisfy both:

1. its `label_end_at` is strictly earlier than the first decision timestamp of
   every adjacent test block; and
2. it is outside a one-full-session embargo following every test block.

Rows whose 120-minute labels overlap a test block are purged. Because all
positions are intraday, one full following session is more conservative than
the maximum label overlap. Tests must assert the invariant before fitting.

## Combinatorial purged cross-validation

Model-family and hyperparameter selection uses development data only.
Development dates are divided, in chronological order and without shuffling,
into eight groups with as nearly equal a number of sessions as possible. Every
combination of two groups is the validation set (28 CPCV splits); the other six
groups are training data after purging and embargo. No pseudo-holdout row or
walk-forward-validation outcome participates.

For each candidate, aggregate median fold QLIKE rank first, then median log-MSE
rank. A candidate is frozen only if it beats the HAR baseline in at least 20 of
28 QLIKE folds and its neighboring hyperparameters are within 2% of its median
QLIKE. Otherwise that family is rejected as unstable. Ties choose the simpler
model in this order: fewer features, shallower depth, stronger regularization.

Every candidate/fold fit is a distinct trial record in `TRIALS.csv`, including
failed fits. Fixed random seed: 7319.

## Expanding walk-forward validation

After CPCV selection, each frozen model is evaluated month by month from
2023-01 through 2024-06. For test month `m`, training contains only rows with
`label_end_at < first decision time of m`, followed by the same purge/embargo
guard. Preprocessing, clipping bounds and smearing corrections are refit only
on that training sample. Parameters selected by development CPCV are unchanged.

Predictive promotion requires all of the following on the combined expanding
OOS predictions and separately in 2023 H1, 2023 H2, 2024 H1:

- lower QLIKE than HAR in every subperiod;
- at least 2% aggregate QLIKE improvement;
- HAC Diebold-Mariano one-sided p-value below 0.05 using Newey-West lag 120;
- positive-variance forecasts with no more calibration failures than HAR; and
- no single month contributes more than 35% of total loss improvement.

Only a model satisfying every predictive gate may proceed to strategy P&L.

## Single pseudo-holdout opening

After model code, selected parameters, artifacts, trial log and predictive gate
are frozen and hashed, the pseudo-holdout is scored exactly once. Every output
table and plot must call it **pseudo-holdout, previously exposed**. It is used
only as a research stability check. No refit, threshold change, feature change,
cost change, or model selection is allowed after opening it.

The pseudo-holdout result can yield only:

- `research_stable` — the already-frozen predictive and strategy rules remain
  within their preregistered gates; or
- `research_failed` — they do not.

It can never yield accepted, validated, proven profitable, or equivalent.

## Strategy-stage boundary

PnL simulation is forbidden until the predictive gate passes. If one or more
models pass, the Integrator freezes one economic mapping before PnL: use the
forecast only in the already preregistered H1/H4 variance-hurdle rules, execute
at the next eligible option snapshot, one lot, with ordinary and 2x proxy costs
and the risk limits in `RISK_POLICY.md`. Strategy thresholds may be selected on
development CPCV only, must form a plateau, and are never changed using the
pseudo-holdout.
