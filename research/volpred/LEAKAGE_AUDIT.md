# Independent leakage audit

## Verdict: BLOCK

The Leakage Auditor vetoes use of the current options backtests for strategy
promotion, profitability acceptance, or a tradability claim.  The veto also
applies to treating 2024 or 2025 as an untouched holdout.  Spot-only feature
construction has a passing synthetic prefix-causality check, so continued
outcome-blind engineering is reasonable, but the options signal/fill pipeline
does not meet the causal execution contract in `SCOPE.md` and `DATA_AUDIT.md`.

No quarantined future holdout was located, opened, or inspected for this audit.

## Veto reasons

### 1. Signal snapshots are also used as entry fills

`research/dte4_20_strategy.py` and
`research/frozen_near_expiry_strategy.py` select the entry row at `datetime=t`
and price the entry from that row's `call_mid` and `put_mid`.  The session-end
variant does the same.  There is no order event followed by a strictly later
eligible option snapshot.  `app/trade_puncher.py` likewise scores the latest
chain and immediately emits `PAPER_ORDER_READY` against that same chain.

This conflicts with the frozen rule that information at `t` may generate a
signal but entry must use the first eligible snapshot strictly after the
decision.  Even if the stored value were executable (it is not certified as
such), the present result contains same-bar fill lookahead.

### 2. Option availability time is not implemented

The canonical option extract has `tick_time`, and the data audit requires
`available_at = max(datetime, Date + tick_time)`.  None of the active DTE 4-20
cache builders or the paper puncher reads `tick_time`, constructs
`available_at`, localizes timestamps to Asia/Kolkata, or excludes ambiguous
availability.  Exact `datetime` equality is therefore being treated as both
information time and execution time despite the observed one-second lag.

### 3. Prediction and option-chain clocks are not joined causally

The puncher reads the last prediction record and the maximum chain timestamp,
but `build_intent` does not compare `live_record.bar_timestamp` (or an inference
availability timestamp) with the option snapshot.  A stale or future-dated
prediction can be paired with the chain without rejection.  Research joins
mostly use exact naive `datetime` keys rather than a latest-available backward
as-of join on a common decision clock.

### 4. Exit selection is outcome-dependent and not fully observed

The cache builders inner-join an exact future timestamp.  An entry is absent if
the required exit snapshot is missing, and `collapse_exit_chains` also drops a
trade when fewer than two valid exit-smile IV points are available.  Both are
future-state filters.  When only the fixed strike is missing, the primary
ledger substitutes a Black-Scholes/smile model price.  That is disclosed, but
an observed-only sensitivity is not provided and the primary evidence is not
an observed fixed-strike result as required by scope.

The valid causal choices are to use the first eligible observed snapshot at or
after the predeclared exit time, or to retain a missing-exit event and apply a
predeclared conservative settlement/gap-loss rule.  Silent row loss cannot be
used.

### 5. Purge/embargo is not an enforced invariant

The inspected monthly walk-forward loops train only on earlier timestamps or
days, which is directionally correct, and selected feature columns do not
include the stored future targets.  However, there is no reusable split guard
that asserts `max(training label end) < min(test decision time)` and no explicit
purge/embargo of at least the maximum label horizon.  The repository contains
many parallel research scripts, so review by convention is not sufficient.
Every training row needs a `label_end_at`; split construction must assert the
inequality before fitting.

### 6. Artifact provenance prevents an independent holdout claim

The paper DTE artifacts are refit on all eligible historical rows, and the
model registry states that fitted samples extend through May 2025.  Refitting
for paper inference is acceptable, but those dates cannot then evaluate that
artifact.  The puncher does not enforce an artifact training cutoff, feature
data cutoff, schema/hash compatibility, or a rule that scoring time is later
than `trained_through`.  Registry checksums protect file identity, not temporal
provenance.

Artifact payloads intended for evaluation should record at least the source
snapshot hash, feature-code version, ordered features, training decision-time
range, maximum label end, embargo, tuning period, and immutable policy/trial
identifier.  Inference must reject records at or before the artifact's maximum
label end.

### 7. The named holdouts are already exposed

The repository labels 2024-2025 or 2025 as `holdout` in multiple programs and
reports, while `SCOPE.md` and `DATA_AUDIT.md` establish that these outcomes have
already influenced research.  `frozen_near_expiry_strategy.py` even describes
2024-2025 as untouched in its module docstring, although its generated report
acknowledges that a rule was changed after inspecting that evaluation sample.
These periods are development/pseudo-holdout data only.

## Items that passed or were conditionally acceptable

- The enhanced spot-only RV feature builder passed a synthetic prefix mutation
  test: changing all prices after decision row `t` did not change any enhanced
  feature at `t`.
- The 15-minute realized-variance target was verified to use returns `t+1`
  through `t+15`, excluding the decision-bar return.
- A deliberately leaked `shift(-1)` future-price feature failed the same prefix
  invariant, demonstrating that the negative control is live rather than a
  vacuous test.
- Monthly/daily walk-forward code inspected for the enhanced RV and DTE P&L
  models fits on earlier dates.  This remains **conditional**, not a pass for
  the complete experiment, until label-end purge assertions and artifact
  provenance are enforced.

The automated checks are in `tests/test_leakage_audit.py` and operate only on
synthetic data.  Audit run:

```text
py -3.13 -m pytest -q tests/test_leakage_audit.py
....                                                                     [100%]
4 passed
```

## Conditions to remove the veto

1. Build one timezone-aware event table with source timestamp, `available_at`,
   decision time, signal time, order time, and fill time; test all inequalities.
2. Generate signals only from records with `available_at <= decision_time` and
   fill only from the first eligible snapshot with `available_at > order_time`.
3. Make the prediction/spot/chain join a backward availability join and reject
   stale, future, or asynchronous inputs beyond a frozen tolerance.
4. Preserve every eligible entry through exit accounting.  Report observed
   fixed-strike exits separately and apply a conservative predeclared rule to
   missing exits; do not drop them based on future availability or IV quality.
5. Add `label_end_at` to every supervised sample and assert purging/embargo for
   every split, including tuning and calibration folds.
6. Freeze artifact provenance and make scoring reject dates that overlap
   training/tuning labels.  Do not use refitted artifacts to score their own
   history.
7. Rename all exposed 2024-2025 samples as development or retrospective
   pseudo-holdout.  Keep the future acceptance year outside development paths,
   hidden until code, costs, parameters, and artifact hashes are frozen.

Until all seven conditions are met, the only permitted status is research or
paper engineering; the INR 100,000 acceptance question remains inconclusive.
