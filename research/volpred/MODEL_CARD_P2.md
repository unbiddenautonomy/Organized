# Model card P2 — shallow nonlinear HAR state model

**Final status:** predictive research stable; economic mapping **REJECTED**; never accepted.  
**Research date:** 2026-10-13  
**Horizon:** 120 trading minutes  
**Seed:** 7319

## Decision

P2 passed its P0-relative development and expanding walk-forward gates and the
one-shot **pseudo-holdout, previously exposed** predictive stability check.
This is not an acceptance result: the pseudo sample was previously exposed,
P1 failed before walk-forward, and the frozen S1/S2 economic mapping lost money
under every cost scenario in development and validation. P2 is not approved
for live trading or another pseudo-strategy opening.

## Frozen model

`HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
l2_regularization=1, max_iter=200, early_stopping=False, random_state=7319)`
predicts log 120-minute realized variance from the 13 registered HAR-state
features. Positive forecasts use training-only lognormal smearing. The final
model was fitted on 44,679 development plus walk-forward rows through a maximum
`label_end_at` of `2024-06-28 15:25:59+05:30`.

| Frozen artifact | SHA-256 |
|---|---|
| P2 model | `ab863e86bb064cac8624e605ad3ad6f32a4302fb3cd51037ef73e627e9314f61` |
| P0 comparator | `6210dbebd4739d39b29c7bc4a6b73e140ee292ec2a96a703c364455533beb5cb` |
| Predictive specification | `b9c0be6cd0eb0ab2c434a8671a7fb2df1acfc633850a7b01e942696db5236cde` |
| Common dataset | `02850acfebd8442bfb7c518f174abc94402e88181a72975f2db4986d35d8ded0` |
| Split protocol | `4409030241bf656af200c811a50a39387c49a8a4e3a4b01700b06fde1c79b48c` |

## Predictive evidence

### Development and walk-forward

The selected P2 candidate beat fold-specific P0 in 21/28 CPCV folds and passed
the 2% neighboring-cell plateau. Its median CPCV QLIKE was `0.155609` and
median log-MSE `0.284026`.

| Expanding walk-forward metric, 2023-01 to 2024-06 | P0 | P2 |
|---|---:|---:|
| QLIKE | 0.175972708 | 0.136690661 |
| Log-MSE | 0.419263572 | 0.306445908 |
| Mean actual/forecast | 0.815500 | 0.848417 |
| Raw calibration failures | 0 | 0 |

P2 improved aggregate QLIKE by `22.3228%`; HAC-DM lag 120 was `-10.1975`
with one-sided `p=1.02e-24`. QLIKE improvement was 27.59% in 2023 H1,
21.60% in 2023 H2, and 19.49% in 2024 H1, with all 18 months improving.

### P1 integration

P1 selected ElasticNet `alpha=0.01, l1_ratio=0.5`, won 22/28 CPCV folds
against P0, but failed its preregistered 2% plateau because its adjacent
stronger-regularization cell was about 6.1% worse. P1 therefore failed at
development; its walk-forward and pseudo outcomes were never opened. There is
no honest common walk-forward P2-versus-P1 statistic. P2 advanced because it
passed its own frozen gate while P1 did not—not because a missing P1
walk-forward comparison was silently imputed.

### One-shot predictive pseudo sample

The atomic marker records `COMPLETED_AND_CONSUMED`. The sample label is exactly
**pseudo-holdout, previously exposed**. It contains 8,442 rows and can support
only a research-stability statement.

| Predictive pseudo metric | P0 | P2 |
|---|---:|---:|
| Aggregate QLIKE | 0.223904088 | 0.198055906 |
| First-half QLIKE | 0.252480020 | 0.223265399 |
| Second-half QLIKE | 0.195328155 | 0.172846413 |
| Log-MSE | 0.561828481 | 0.467194329 |
| Raw calibration failures | 0 | 0 |

QLIKE improvement was `11.5443%`; HAC-DM lag 120 statistic was `-4.67416`
with one-sided `p=1.47578e-06`. The frozen verdict was `research_stable`.
Result SHA-256: `8b09a566398cbe6830f0cb615c7e64518f90abe4eec21ceb46614166e8cc760f`.

## Limitations and terminal interpretation

- Predictive success did not survive conversion into positive economic value.
- The predictive pseudo sample was previously exposed and is permanently
  consumed; it is not untouched evidence.
- Historical bid/ask, depth, SPAN margin, historical contract masters, and
  verified lot sizes were unavailable. Prices are LTP-like proxies.
- Monthly five-strike options constrain structure and liquidity inference.
- P2 is research-only, not a verified backtest, not live eligible, and never
  accepted.

Primary evidence: [P2 gate](outputs/research_v2/p2/gate.json),
[predictive pseudo result](outputs/research_v2/p2/pseudo_holdout_once_result.json),
[P1 gate](outputs/research_v2/p1/predictive_gate.json), and
[final economic gates](outputs/research_v2/evaluation/gates.json).
