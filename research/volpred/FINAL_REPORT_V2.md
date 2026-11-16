# Final Research V2 report

## Verdict

# REJECTED — NEVER ACCEPTED

P2 showed genuine predictive improvement over P0, and its one-shot
**pseudo-holdout, previously exposed** predictive check was stable. The frozen
S1/S2 economic mapping nevertheless produced negative gross-to-net economics
in development and validation, failed drawdown and risk-breach gates, had PBO
of 1.0, and lacked sufficient fills or a new quarantined year. Strategy pseudo
P&L was not opened because the development/validation gate failed.

This is the terminal decision for the tested candidate. Predictive accuracy is
not profitability, tradability, paper-capital eligibility, or acceptance.

## Predictive result

Model cards cover P0, P1, P2, and the unexecuted P3 hypothesis; the economic
mapping has its own S1/S2 card.

| Evidence | P0 | P2 | Decision |
|---|---:|---:|---|
| Walk-forward QLIKE | 0.175972708 | 0.136690661 | P2 +22.3228% |
| Walk-forward log-MSE | 0.419263572 | 0.306445908 | P2 lower |
| Predictive pseudo QLIKE | 0.223904088 | 0.198055906 | P2 +11.5443% |
| Predictive pseudo log-MSE | 0.561828481 | 0.467194329 | P2 lower |

P2 beat P0 in 21/28 CPCV folds and passed its parameter plateau. Walk-forward
HAC-DM lag 120 was `-10.1975`, `p=1.02e-24`; predictive pseudo HAC-DM was
`-4.67416`, `p=1.47578e-06`. P1 won 22/28 folds but failed its development
plateau and therefore never opened walk-forward or pseudo outcomes. P2 did not
receive an invented P1 comparison.

P3, the preregistered regime-mixture idea, was not implemented or trained. It
has no trial, score, or outcome evidence and cannot be added after the
predictive pseudo opening. `MODEL_CARD_P3.md` records this protocol-completeness
limitation; it is not treated as either a pass or a failed model result.

The predictive pseudo sample contained 8,442 rows, is permanently consumed,
and is explicitly **pseudo-holdout, previously exposed**. Its
`research_stable` verdict cannot support acceptance.

## Primary economic result, threshold 1.25 historical lot

| Metric (INR except counts/statistics) | Development | Validation |
|---|---:|---:|
| Eligible decisions / sessions | 2,355 / 791 | 1,078 / 360 |
| Filled / rejected / no-trade | 31 / 629 / 1,695 | 17 / 332 / 729 |
| Gross P&L | -9,615.00 | -3,507.75 |
| Ordinary cost / net | 44,371.88 / -53,986.88 | 52,320.19 / -55,827.94 |
| 2x cost / net | 88,743.76 / -98,358.76 | 104,640.37 / -108,148.12 |
| Uncertainty cost / net | 125,506.45 / -135,121.45 | 152,211.34 / -155,719.09 |
| Worst day / ISO week | -9,030.62 / -13,451.44 | -8,204.12 / -22,205.35 |
| Maximum drawdown | 53,986.88 | 55,827.94 |
| Daily loss CVaR95 / bootstrap UCB | 1,349.67 / 2,326.94 | 3,101.55 / 6,224.99 |
| Recorded missing / unrecorded exits | 7 / 0 | 7 / 0 |
| Win rate | 0% | 0% |
| Annualized Sharpe | -1.714971 | -2.399296 |
| PSR / DSR | 7.7267e-13 / 7.8535e-50 | 9.4631e-10 / 9.5916e-39 |
| Event selected / filled / rejected | 66 / 0 / 16 | 36 / 0 / 11 |
| Monetary turnover | 1,118,367.00 | 866,170.50 |
| Maximum modeled margin | 5,791.50 | 1,483.20 |

The exact 14-row threshold/lot/event table is reproduced in
`MODEL_CARD_S1_S2.md` and preserved in the authorized CSV. Constant-15 and
historical-lot results are identical because both evaluation intervals end
before the assumed 2024-12-25 lot-size change; this is not evidence about the
unverified 30-lot regime.

The frozen 1.15/1.25/1.35 plateau passed in both segments, but all neighboring
results had the same negative sign; this is stable loss, not robust profit.
There were no positive months, so profit concentration was unavailable. Both
primary rows breached the INR 50,000 drawdown limit once. Worst day/week,
CVaR, bootstrap UCB, and zero-unrecorded-exit checks passed, but cannot override
negative ordinary and 2x returns or drawdown failure.

CSCV PBO was `1.0` in development and validation using seven variants, eight
blocks, and 70 combinations. DSR used 1,248 substantive trials; 14 completion
records were excluded. Development had 791 sessions but only 31 fills;
validation had 360 sessions and 17 fills. Neither reached 100 fills, validation
did not reach 500 sessions, and no new 12-month quarantined sample exists.

## Red-team disposition

The independent red team confirmed **REJECTED — DO NOT ADVANCE** and added
four important interpretations:

- every central-policy fill was an iron fly, so S2 has no empirical trade
  evidence;
- after removing missing-exit fills, observed-exit ordinary P&L remained
  `-7,694.25` over 24 development trades and `-3,974.51` over 10 validation
  trades;
- drawdown suspension ended effective trading by 2020-01-09 in development
  and 2023-01-25 in validation, followed by 585/321 suspension rejections; and
- final reporting is hash-bound and reproducible, but the evaluator required
  post-outcome bug repair. The current failure is intact; the first economic
  reporting pipeline was not pristine pre-outcome code.

The event diagnostic was identical to the primary trial and filled no event
trade, so it supplies no empirical event-risk evidence.

## Gate disposition

| Gate | Development | Validation |
|---|---|---|
| Absolute decision coverage | PASS | PASS |
| Threshold plateau | PASS, negative sign | PASS, negative sign |
| Positive ordinary and 2x net | FAIL | FAIL |
| Observed tail gate | FAIL | FAIL |
| Historical sample sufficiency | FAIL | FAIL |
| New forward 12-month sample | unavailable | unavailable |

Final machine verdict: `research_failed`. Strategy pseudo P&L remains
unopened (`pseudo_opened=false`). Opening it now cannot rescue the failed
prerequisite and would add avoidable leakage.

## Why this is not tradable evidence

- The options source is a narrow monthly five-strike extract with LTP-like
  prices, not executable bid/ask/depth history.
- Historical fills, partial fills, rejection behavior and market impact are
  modeled, not observed.
- Historical SPAN/exposure margin and broker liquidation behavior are absent;
  reported margin is an unverified proxy.
- Historical contract masters and lot-size evidence are incomplete; lot 15 is
  assumed for the evaluated dates.
- Recorded missing exits were charged adversely, but settlement and gap models
  remain proxies; they affected 23% of development and 41% of validation fills.
- The predictive pseudo sample was previously exposed and consumed; it can
  never provide acceptance evidence.
- Preregistered P3 was not executed and therefore supplies no comparative
  evidence. Adding it now would require a new protocol and new quarantined data.

## Reproducibility and immutable evidence

| Artifact | SHA-256 |
|---|---|
| Evaluation summary | `9803c5f6b3e46af182d6de09f75e1d2cbb33f3f4c8bf9d7e67c18b520421da95` |
| Evaluation gates | `7526b722bdf0dfb3d7bd446ee00bdaa71cecb8870ca074568eab9822d166d0cb` |
| PBO report | `2bc31f19315f9f0ff6f98d6a172cd56f1cbe3d89e187ea5a0464cb91636e149d` |
| Evaluation manifest | `ff6ae530d3455c15f3ffa011c068ede93195414ab34833ea57d5caa2fafb30a5` |
| Evaluation authorization | `0bd6135ea87b5838933cd2e369f60d1473cb3efc237764a3d4fff67d0fcfcc36` |
| Evaluator | `b7985eeac31261c4d5875ea995bda0e53847349593a8566738171f467d6d375d` |
| Evaluation runner | `f370cc6dbcc0d91291dc6c2430a16039b536dc6cbbab97be5a29ae525495d513` |
| Strategy specification | `83b1d87e4fbbd8a6ccfec16dd9783a0b876fa881ed96606d5f12ac6979d6ae42` |
| Risk policy | `3a992923163c970e91387fd7196c40b93a3210fb9dc289e2373214ab0ce0282f` |
| Predictive pseudo result | `8b09a566398cbe6830f0cb615c7e64518f90abe4eec21ceb46614166e8cc760f` |
| Red-team review | `791399d034426871917e7bda714f08846d4ff89bf911063ee33adae0f72ede10` |

Reproduce the authorized evaluation only while its hash-bound inputs remain
unchanged and after archiving the existing output directory under an
independently authorized process:

```powershell
py -3.13 research/run_research_v2_evaluation.py
```

Primary artifacts are the [summary](outputs/research_v2/evaluation/segment_trial_summary.csv),
[gates](outputs/research_v2/evaluation/gates.json),
[PBO report](outputs/research_v2/evaluation/pbo.json), and
[hash inventory](outputs/research_v2/evaluation/SHA256SUMS).

## Forward work

Any future evidence must follow `FROZEN_FORWARD_SHADOW_SPEC_V2.md` and
`DATA_QUARANTINE_PLAN_V2.md`: write-only collection beginning after 2026-10-13,
at least through 2027-10-13, and longer until 500 eligible sessions and 100
fills exist. It cannot change this completed candidate's verdict.

**Research V2 is REJECTED, never accepted.**
