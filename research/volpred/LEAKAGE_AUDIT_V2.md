# Research V2 leakage audit

## Current verdict: PASS for exactly one predictive pseudo score; BLOCK PnL and beyond

The common predictive dataset, reusable split guards, and remediated P1/P2
runners are approved for development CPCV and conditional frozen expanding
walk-forward validation. P2's frozen predictive artifact is additionally
authorized for **exactly one** predictive-only score of the pseudo-holdout,
previously exposed, using the reviewed one-shot command and no other access.
Strategy P&L, paper/live promotion, and any acceptance, validation, or
tradability claim remain **BLOCKED**.

P1 and P2 were initially vetoed before execution; their remediations are
recorded below.

No pseudo-holdout target, prediction outcome, strategy result, or P&L was read
for this audit.  The pseudo-holdout remains previously exposed evidence and is
not an acceptance sample under any verdict.

## Split-control review

`research/research_v2_splits.py` implements the frozen protocol without model
or outcome logic:

- exact immutable calendar labels for development, walk-forward validation,
  pseudo-holdout previously exposed, and excluded dates;
- chronological development-session partitioning into eight groups whose sizes
  differ by at most one session;
- all 28 unique CPCV 8-choose-2 held-out group pairs, with six training groups;
- removal of held-out sessions, every predecessor training label whose
  `label_end_at` reaches the first decision in a test block, and the complete
  following session after each contiguous test block;
- assertions that training contains neither a validation session, an
  overlapping predecessor label, nor an embargo session;
- rejection of naive, missing, reversed, or nonfinite temporal intervals;
- fail-closed denial of pseudo-holdout split requests; and
- a restricted Parquet reader that applies the approved split predicate in the
  query, before disallowed rows or columns are materialized in Python.

The split reader permits only `development` and `walk_forward_validation`.
There is deliberately no convenience function to open the pseudo-holdout.
That opening requires a separate post-freeze mechanism and audit decision.

## Automated evidence

`tests/test_research_v2_splits.py` uses synthetic dates and values only.  It
checks exact boundary dates, balanced chronological CPCV groups, all 28 folds,
label overlap removal, a complete following-session embargo, negative controls
that detect both overlap and embargo violations, in-query pseudo-holdout denial,
and the frozen `SPLITS.md` checksum.

Combined audit run:

```text
py -3.13 -m pytest -q tests/test_research_v2_splits.py tests/test_research_v2_data.py
..............                                                           [100%]
14 passed
```

## Mandatory P1/P2 review criteria

A runner receives PASS only if all of the following are evident in code and
artifacts before outcomes are considered:

1. It verifies `SPLITS.md` against `SPLITS.sha256` before reading the dataset.
2. It reads only the registered P0/P1 columns and uses a split predicate at
   Parquet read time; CPCV selection materializes development rows only.
3. It derives the eight groups once from the complete ordered development
   session inventory, never from candidate-specific surviving rows.
4. Every one of the 28 folds calls `purged_embargoed_train` and
   `assert_purge_embargo` before preprocessing or fitting.
5. Imputation, clipping, scaling, smearing, and all other learned transforms fit
   on the purged training rows only.  Validation labels do not affect early
   stopping, feature selection, preprocessing, or calibration outside the
   registered inner rule.
6. Ridge alpha and every candidate parameter are selected only from development
   CPCV.  The walk-forward period is evaluated month by month with frozen
   parameters and expanding prior training only.
7. Each walk-forward fit asserts `max(train.label_end_at) <
   min(test.decision_time)` after purging.  It never trains on a later month or
   on predictions/outcomes from the month being scored.
8. Candidate comparisons aggregate exactly the frozen fold metrics and record
   every fit, including failures, in `TRIALS.csv`; failures cannot disappear
   through complete-case or successful-fit filtering.
9. The stability rule, subperiod gates, HAC DM lag 120, calibration-failure
   comparison, and 35% month-contribution cap are applied exactly as written,
   without looking at pseudo-holdout data.
10. PnL code is unreachable unless every predictive promotion gate passes and
    the separate strategy-stage audits authorize it.

Any direct `pandas.read_parquet` of the complete panel followed by an in-memory
split, use of random row folds, preprocessing before fold construction,
candidate-specific session grouping, omitted failed folds, absent embargo, or
pseudo-holdout read is an automatic BLOCK.

## P2 code review: BLOCK before execution

Reviewed `research/research_v2_p2.py` and
`tests/test_research_v2_p2.py` without running the model or reading any outcome
artifact. The code contains no PnL path, applies a pre-pseudo timestamp
predicate in DuckDB, constructs eight balanced groups and 28 held-out pairs,
disables test-driven early stopping, fits scaling and smearing on training
rows, and gates walk-forward execution on the development CPCV decision.

Execution is nevertheless vetoed for four reasons:

1. The runner does not verify `SPLITS.md` against `SPLITS.sha256` before reading
   data. It accepts an arbitrary `--panel` and records rather than checks its
   hash against the frozen manifest and hash inventory.
2. It reimplements loading and purge/embargo instead of invoking the audited
   shared guards. Its query filters only on timestamp rather than approved
   split labels, so a later source change could bypass the calendar contract.
3. The calibration-failure gate is vacuous. `positive_forecast` replaces every
   invalid raw forecast with `EPS`; the gate later checks only that the replaced
   arrays are finite and positive. `fail0` and `fail2` are calculated but never
   accumulated or compared as required.
4. Candidate/fold fits go to `outputs/research_v2/p2/trials.csv`, while the
   protocol requires every fit, including failures, in root `TRIALS.csv`.

The initial P2 veto required focused negative tests covering checksum mismatch,
disallowed split labels, shared overlap/embargo guards, and unequal calibration
failure counts.

### P2 remediation re-review

The revised runner now pins and verifies the split, manifest, dataset, and hash
inventory; uses the shared restricted reader and split/purge guards; requires
28 successful folds; prevents walk-forward access on a failed development
gate; and accumulates P0/P2 failure counts for comparison. Its focused/shared
suite initially passed 14 tests. A second remediation counts raw NaN, infinite,
overflowing, and underflowing forecasts before replacement and appends each
completed fold/month directly to root `TRIALS.csv` under a byte-range lock with
unique run IDs, exact schema validation, flush, and `fsync`. The expanded suite
passed 16 tests. **P2 now PASSes for predictive-only development CPCV and, only
if that gate passes, frozen walk-forward execution.** Pseudo-holdout access and
strategy/PnL remain blocked.

## P1 code review: BLOCK before execution

Reviewed `research/research_v2_p1.py` and
`tests/test_research_v2_p1.py` without running the model or reading an outcome
artifact. The loader does use a storage-level approved-split predicate, the
feature contract is registered, scaling and smearing are training-only, 28
CPCV folds are constructed, and no pseudo-holdout or PnL path is present.

Execution is vetoed for these reasons:

1. `main()` always calls `run_walk_forward`, even when development CPCV reports
   `cpcv_pass = false`. An unstable family must be rejected without opening the
   walk-forward validation outcomes.
2. `cpcv_pass` requires 20 wins and a plateau but does not require zero failed
   folds or 28 successful folds. A candidate can therefore pass with missing
   fold evidence.
3. The runner does not verify the split checksum or expected dataset hash before
   access and reimplements the audited split/purge logic rather than invoking
   the shared guards.
4. `TrialLogger` writes only an output-local CSV, opens it with mode `w`, and
   erases it on rerun. This is neither the required root `TRIALS.csv` nor an
   append-only record of all attempts.

The initial P1 veto required the walk-forward call to be unreachable on CPCV failure,
successful-fold completeness is part of the gate, shared checksum/split guards
are used, and immutable global trial registration is implemented and tested.

### P1 remediation re-review

The revised runner fixes provenance checks, uses the shared restricted loader
and assertions, requires complete successful CPCV evidence, and returns without
opening walk-forward validation when CPCV fails. Its focused/shared suite
passed 15 tests.

An interrupted pre-remediation run already created 169 successful-fit records.
Those records count as attempted trials because outcomes were opened. They must
remain immutable, be marked `interrupted_pre_remediation` with the superseded
runner version, and enter the multiple-testing inventory. At that review stage
they existed only in the output-local detailed ledger while root `TRIALS.csv`
contained seven registrations. They could not be erased or replaced by a clean
rerun.

The final remediation registered all 169 attempts in root `TRIALS.csv` as
`interrupted_pre_remediation`, preserving their P0/P1 identity and legacy
metadata. Root now contains 176 rows: seven preregistrations plus those 169
attempts. Fresh fits write first to root `TRIALS.csv` under a fail-closed
exclusive lock, validate the exact schema, flush and `fsync`, then append local
detail. The focused/shared suite passed 16 tests. **P1 now PASSes for
predictive-only development CPCV and, only if that gate passes, frozen
walk-forward execution.** Pseudo-holdout access and strategy/PnL remain
blocked.

## Boundary of this PASS

This PASS applies only to the predictive development/CPCV and frozen
walk-forward procedure reviewed here. A failed development gate must leave
walk-forward unopened. Neither P1 nor P2 may access the pseudo-holdout or
trigger strategy/PnL work. Any future code or parameter change requires a new
hash, new trial identity, rerun of the negative controls, and Leakage Auditor
review before execution.

## Frozen P2 and pseudo-holdout review: BLOCK pending one-shot repair

The frozen P2 artifact, metadata, specification, and hash inventory were
reviewed without running the scorer or reading pseudo-holdout outcomes. The
recorded hashes match the model, metadata, runner, tests, split protocol,
dataset, common manifest, model card, gate file, and frozen specification. The
artifact contains the fixed 13-feature order, fixed HistGradientBoosting
parameters, seed 7319, training-only smearing adjustment, and training cutoff
`2024-06-28 15:25:59+05:30`. It was fitted only on development plus authorized
walk-forward rows. The scoring path loads that artifact and calls `predict`; it
contains no refit, transform fitting, parameter mutation, or PnL calculation.

P1 failed its development CPCV plateau gate and therefore correctly left its
walk-forward validation unopened. `SPLITS.md` promotes each model family
against P0/HAR and does not require P2 to beat P1. Consequently P1
disqualification eliminates, rather than leaves pending, any P1 walk-forward
comparison. P2's completed pass versus P0 is sufficient for predictive-stage
eligibility.

The one-shot pseudo scorer is not yet authorized for two reasons:

1. It checks whether the marker/output files exist and then writes the marker
   through `os.replace`. That check-and-replace sequence is not an exclusive
   claim: two concurrent invocations can both observe no marker and both score.
   The STARTED marker must be created with create-new semantics or under an
   exclusive lock, and a failed attempt must remain consumed.
2. The frozen protocol says the pseudo result may be only `research_stable` or
   `research_failed`. The frozen spec contains permitted metrics but no frozen
   stability thresholds, while the scorer emits `SCORED_ONCE`. A deterministic
   stability rule and the allowed result vocabulary must be frozen and hashed
   before any pseudo row is opened.

Negative tests must prove that an existing/in-progress claim prevents data
access, concurrent claims cannot both succeed, and a second invocation cannot
overwrite the marker or either output. Until then, exactly-one predictive
pseudo scoring remains **BLOCKED**; strategy PnL remains blocked independently.

### One-shot remediation and final authorization

The remediated and refrozen implementation closes both vetoes. Before any
pseudo-row query, `score_pseudo_holdout_once()` verifies the complete frozen
hash inventory and atomically creates the permanent marker with
`O_CREAT|O_EXCL`. Existing and concurrent markers fail closed. Any error after
the claim writes `research_failed` and leaves the right permanently consumed;
a successful opening leaves it completed and consumed. The scoring path then
uses only `predict` on the immutable P2 and same-sample P0 artifacts. It does
not call `fit`, refit preprocessing or smearing, change a feature, parameter,
threshold, or cost, or compute strategy P&L.

The frozen rule permits exactly `research_stable` or `research_failed`. It
requires lower P2 QLIKE in each stable chronological half, at least 2% aggregate
QLIKE improvement, one-sided HAC-DM lag-120 p-value below 0.05, P2 log-MSE no
higher than P0, and no more raw P2 calibration failures than P0. Missing,
nonfinite, nonpositive, empty, or post-claim processing failures fail closed to
`research_failed`.

The verified frozen identities are:

- P2 artifact: `ab863e86bb064cac8624e605ad3ad6f32a4302fb3cd51037ef73e627e9314f61`
- P0 artifact: `6210dbebd4739d39b29c7bc4a6b73e140ee292ec2a96a703c364455533beb5cb`
- runner: `8ffdc40ea68d9049b9511672462a4a6f889942e4a745344f59668688cf957722`
- frozen specification: `b9c0be6cd0eb0ab2c434a8671a7fb2df1acfc633850a7b01e942696db5236cde`

At authorization time, the marker, prediction output, and result output were
all absent, and the frozen specification reported `NOT_OPENED`. The focused
P2/shared-split suite passed 21 tests, including existing-marker, concurrent
claim, repeated-claim, exact two-state verdict, fail-closed input, and
unauthorized-fit controls. No pseudo-holdout row, target, prediction, metric, or
outcome was inspected during this review.

**Authorization:** execute exactly once:
`py -3.13 research/research_v2_p2.py --score-pseudo-once`. This authorization
is predictive-only and is consumed by any claimed attempt, whether successful
or failed. It does not authorize a retry, PnL, strategy tuning, artifact refit,
threshold change, or any acceptance claim. P1's development disqualification
means no P1 walk-forward comparison is required by `SPLITS.md`; P2's frozen P0
comparison is the applicable comparator.
