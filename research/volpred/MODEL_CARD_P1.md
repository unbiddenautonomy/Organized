# Model card P1 — HARQ ElasticNet

**Research date:** 2026-10-13  
**Horizon:** 120 trading minutes  
**Seed:** 7319  
**Predictive gate:** **FAIL at development CPCV**  
**Strategy P&L:** not accessed or computed  
**Walk-forward validation:** not opened  
**Pseudo-holdout, previously exposed:** not accessed

## Purpose and permitted interpretation

P1 tests whether the preregistered trailing quarticity, semivariance, jump,
absolute-return, and fixed time-of-day extension improves positive realized-
variance forecasts over the P0 three-feature HAR-Ridge baseline. It is a
predictive research model only. A passing result would not establish strategy
profitability or tradability.

P1 failed the frozen development-selection gate because the selected
hyperparameter cell did not have the required neighboring 2% QLIKE plateau.
Under the leakage reviewer's conditional approval, this failure prohibited
opening the 2023-01 through 2024-06 walk-forward validation. P1 does not advance.

## Frozen inputs and integrity

The runner read only development rows from
`outputs/research_v2/common/predictive_dataset_120m.parquet` for the completed
CPCV evaluation. The storage reader uses predicate pushdown and the audited
guards in `research/research_v2_splits.py`. Before fitting, it verified:

| Artifact | SHA-256 |
|---|---|
| `SPLITS.md` | `4409030241bf656af200c811a50a39387c49a8a4e3a4b01700b06fde1c79b48c` |
| Predictive dataset | `02850acfebd8442bfb7c518f174abc94402e88181a72975f2db4986d35d8ded0` |
| Common manifest | `3778739fa77c6470fbb788788c3ed93571cc7c1b4f5b90ad230a955d82ca18b6` |

The development sample contained 30,673 rows. CPCV used eight chronological
session groups and all 28 two-group validation combinations. For every fold,
the audited splitter removed test sessions, purged training labels overlapping
the first decision of a test block, and embargoed the full following session.
All scalers and lognormal smearing corrections were fitted on training rows
only.

An initial run was interrupted when leakage review requested stronger guards.
Its 169 attempted fits remain in both the local and global registries with
status `interrupted_pre_remediation`; they were not deleted or reused. The
cleared run appended 504 new successful fit records under an exclusive global
registry lock: 140 nested P0 candidates, 28 selected P0 outer fits, and 336 P1
outer candidate fits. There were zero failed fits in the cleared CPCV run.

## Model definitions

P0 is standardized Ridge on:

- trailing log RV over 5, 15, and 60 minutes.

Within every outer fold, P0 alpha was selected using a training-only inner
chronological split from `{0.01, 0.1, 1, 10, 100}`, then refitted on the outer
training rows.

P1 is standardized ElasticNet on the P0 features plus:

- log realized quarticity over 15 and 60 minutes;
- downside and upside RV over 15 and 60 minutes;
- positive 60-minute jump variation;
- current absolute return; and
- fixed sine/cosine time-of-day terms.

The fixed grid was alpha `{0.0001, 0.001, 0.01, 0.1}` by L1 ratio
`{0, 0.25, 0.5}`. Models predict log variance. Positive-scale forecasts use a
training-only mean exponential-residual smearing correction.

## Development CPCV result

The frozen ordering by median fold QLIKE rank, then median log-MSE rank,
selected `alpha=0.01, l1_ratio=0.5`.

| Measure | Selected P1 result |
|---|---:|
| Successful outer folds | 28 / 28 |
| QLIKE wins versus fold-selected P0 | 22 / 28 |
| Required QLIKE wins | 20 / 28 |
| Median QLIKE rank among 12 P1 cells | 4.0 |
| Median fold QLIKE | 0.1488818244 |
| Median log-MSE rank | 9.0 |
| Median fold log-MSE | 0.2864836885 |
| Neighboring 2% plateau | **FAIL** |

The stronger-regularization alpha neighbor `(0.1, 0.5)` had median QLIKE
`0.1579252571`, approximately 6.1% above the selected cell. Therefore not every
adjacent alpha/L1 cell was within 2% of the selected median QLIKE. The favorable
22-fold win count cannot override this preregistered stability failure.

## Walk-forward metrics intentionally unavailable

The following requested diagnostics are **not evaluated**, because doing so
would open walk-forward outcomes after P1 had already failed CPCV:

- aggregate and 2023 H1 / 2023 H2 / 2024 H1 QLIKE and log-MSE;
- HAC Diebold–Mariano test with Newey–West lag 120;
- calibration intercept/slope and failure counts;
- causal low/middle/high volatility regime stability;
- monthly loss-improvement concentration; and
- expanding-month forecast artifacts.

Their absence is a leakage-control result, not missing reporting. The code for
these diagnostics remains tested and can run only when `cpcv_pass=true`.
`predictive_gate.json` records `walk_forward_rows_accessed: 0` and
`pseudo_holdout_accessed: false`.

## Limitations

- CPCV folds share observations across different train/test paths; 28 folds do
  not represent 28 independent market histories.
- The development interval includes structural volatility regimes, and the
  failed plateau shows material regularization sensitivity.
- The common dataset is spot-only. This study says nothing about option-chain
  availability, fills, costs, margin, or economic P&L.
- The pseudo-holdout is already exposed elsewhere in the repository and could
  never provide final acceptance evidence even if it were later opened under a
  separate frozen protocol.

## Artifacts

- `research/research_v2_p1.py` — fail-closed runner.
- `tests/test_research_v2_p1.py` — feature, split, metric, hash, append, and
  registry-lock tests.
- `outputs/research_v2/p1/trials.csv` — append-only detailed fit registry.
- `outputs/research_v2/p1/root_compatible_trials.csv` — root-schema mirror for
  cleared-run fits.
- `outputs/research_v2/p1/cpcv_fold_metrics.csv` — development fold metrics.
- `outputs/research_v2/p1/cpcv_candidate_summary.csv` — candidate aggregation.
- `outputs/research_v2/p1/selected_model.json` — selected cell and CPCV checks.
- `outputs/research_v2/p1/predictive_gate.json` — terminal FAIL verdict.

## Final decision

**FAIL predictive gate. Do not advance P1 to walk-forward validation, strategy
P&L, pseudo-holdout scoring, paper promotion, or live use.** Any modified grid,
plateau definition, feature set, or estimator would be a new hypothesis and
must receive a new preregistration and trial identity.

