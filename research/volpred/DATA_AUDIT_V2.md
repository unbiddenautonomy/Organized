# Research v2 predictive-data audit

## Gate

**PASS for predictive research only**, limited to development/CPCV and the
predeclared expanding walk-forward validation in `SPLITS.md`.

This gate does **not** authorize inspection or scoring of the
`pseudo_holdout_previously_exposed` targets, strategy P&L, options execution,
paper promotion, or a tradability/acceptance claim. The Data, Leakage and Red
Team blocks in the first research phase remain in force for every strategy
stage.

## Frozen split integrity

The SHA-256 recorded in `SPLITS.sha256` was independently recomputed from
`SPLITS.md`:

```text
expected  4409030241bf656af200c811a50a39387c49a8a4e3a4b01700b06fde1c79b48c
actual    4409030241bf656af200c811a50a39387c49a8a4e3a4b01700b06fde1c79b48c
status    MATCH
```

The builder fails closed on a checksum mismatch.

## Artifact

The reusable panel was built from
`data/raw/icharts_banknifty_spot_1min.csv` using the session, timestamp, gap and
spot-quality primitives in `research/integrity.py`. No model was fitted and no
target outcome was summarized.

| Item | Value |
|---|---|
| Dataset | `outputs/research_v2/common/predictive_dataset_120m.parquet` |
| Rows | 53,121 |
| Dataset SHA-256 | `02850acfebd8442bfb7c518f174abc94402e88181a72975f2db4986d35d8ded0` |
| Manifest | `outputs/research_v2/common/manifest.json` |
| Hash inventory | `outputs/research_v2/common/SHA256SUMS` |
| Source SHA-256 | `0ca1352a5d3a542ab17356cc90e7e89896dfdb5619af0d30bb312fbb7fe650de` |

The Parquet timestamps are stored as timezone-aware timestamps. Its core
columns are `decision_time`, `decision_date`, `label_end_at`, `split`,
`spot_price`, `rv_fwd_120`, and `target`.

`rv_fwd_120` is exactly:

```text
sum(log(S[t+i] / S[t+i-1])^2, i = 1..120)
```

The decision-bar return is excluded. `target` is
`log(max(rv_fwd_120, 1e-12))`. `label_end_at` is the timestamp of the last
included return, is strictly after `decision_time`, and remains on the same
Asia/Kolkata session. A direct artifact check found zero rows violating either
condition.

## Feature contract

P0 contains only trailing log realized variance:

- `p0_log_rv_5`
- `p0_log_rv_15`
- `p0_log_rv_60`

P1 contains P0 plus the registered trailing extension:

- log realized quarticity over 15 and 60 minutes;
- downside and upside realized variance over 15 and 60 minutes;
- positive 60-minute jump variation `max(RV60 - BV60, 0)`;
- current absolute return; and
- fixed sine/cosine time-of-day terms.

All price-derived feature windows end at `decision_time`. They never cross a
session boundary, nonpositive/nonfinite spot, or a gap over 61 seconds. A
future-perturbation test changes prices after a decision and confirms that all
P1 feature values at that decision remain identical while the forward target
changes.

## Data-quality accounting

| Check | Count |
|---|---:|
| Raw spot rows | 533,720 |
| Null timestamp rows | 0 |
| Duplicate timestamp rows | 0 |
| Ordinary primary-session rows | 531,974 |
| Nonpositive/nonfinite spot rows | 5,276 |
| Rows marked after an intraday gap | 2,089 |
| Scheduled 09:45-13:30 five-minute decisions | 65,408 |
| Excluded for insufficient positional history/horizon | 10,367 |
| Candidates with positional feature/label windows | 55,041 |
| Invalid trailing windows | 1,432 |
| Invalid forward windows | 1,626 |
| Nonfinite feature/target inputs | 555 |
| Outside frozen calendar segments | 1,057 |
| Retained rows | 53,121 |

The exclusion categories after positional screening are diagnostic and can
overlap; they must not be summed as if mutually exclusive. No bad spot is
forward-filled. The common source generally omits a distinct 15:29 bar and
ends with 15:30:00, so a scheduled 13:30 row often lacks 120 subsequent
recorded returns and is correctly excluded rather than shortening the target.

## Frozen calendar constructibility

| Segment | Rows | Sessions | First eligible decision | Last eligible decision |
|---|---:|---:|---|---|
| Development | 30,673 | 791 | 2019-10-11 10:20:59+05:30 | 2022-12-30 13:25:59+05:30 |
| Walk-forward validation | 14,006 | 360 | 2023-01-02 10:15:59+05:30 | 2024-06-28 13:25:59+05:30 |
| Pseudo-holdout, previously exposed | 8,442 | 218 | 2024-07-01 10:15:59+05:30 | 2025-05-22 13:25:59+05:30 |

The written boundaries were not moved. On 2019-10-11 the spot source begins
after the ordinary open, so the first eligible decision is the first one with
a complete 60-minute trailing window. Missing data do not create replacement
dates. Row/session counts and timestamp coverage above are data-availability
metadata, not target-outcome summaries.

## Tests

Run:

```powershell
py -3.13 -m pytest -q tests/test_research_v2_data.py
```

Result during this audit:

```text
7 passed in 6.84s
```

The suite verifies:

1. the frozen split checksum and fail-closed mismatch behavior;
2. exact `t+1..t+120` target construction;
3. feature prefix causality under future-price perturbation;
4. immutable split labels at every calendar boundary;
5. same-session, strictly later, purge-ready `label_end_at` values; and
6. exclusion of windows affected by a bad price or gap.

Rebuild the artifact with:

```powershell
py -3.13 research/research_v2_data.py
```

The command verifies `SPLITS.md` before writing the panel, manifest and hashes.

## Conditions of this PASS

- Training code may read only `development` rows for CPCV/model selection and
  may use `walk_forward_validation` only in the frozen expanding OOS procedure.
- The pseudo-holdout target column must remain unqueried until model code,
  hyperparameters, artifacts, trial log and promotion rule are frozen and
  hashed. It is previously exposed research evidence, never final acceptance.
- Every fit must use `label_end_at < first_test_decision_time` and implement the
  one-full-session embargo required by `SPLITS.md`; the panel is purge-ready
  but does not perform fold-specific purging by itself.
- Preprocessing, clipping, imputation, scaling and smearing corrections must be
  learned inside training folds only. Missing rows may not be imputed across a
  gap or session boundary.
- Only the registered P0/P1 feature columns may be used for those hypotheses.
  Any feature change requires a new version, checksum and trial registration.
- No strategy P&L may be computed unless a model passes every predictive gate
  in `SPLITS.md`.

Subject to those conditions, the dataset foundation is approved for the next
predictive-model research step.
