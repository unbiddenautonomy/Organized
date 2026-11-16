# Model card S1/S2 — frozen variance-to-options mapping

**Final verdict: REJECTED. Never accepted.**

S1 is the defined-risk short ATM iron fly used when implied 120-minute
variance divided by P2 forecast variance meets the frozen threshold. S2 is the
long ATM straddle used for the reciprocal signal. The authorized report
evaluates their combined first-signal-per-session policy; it does not provide
honest structure-isolated attribution, so none is inferred here.

## Frozen policy

- Primary threshold: `1.25`; diagnostics: `1.15` and `1.35`.
- Clocks: 10:15, 11:45, and 13:25 Asia/Kolkata; one position per day.
- Horizon: 120 trading minutes; strikes and expiry remain fixed.
- Monthly recorded contracts only; ATM IV must be finite, positive, and converged.
- Ordinary, 2x all-cost, and uncertainty costs are evaluated on identical
  decisions. Recorded missing exits receive the frozen contractual adverse
  treatment and remain in the ledger.
- INR 1,000,000 reference capital; assumed historical lot size 15; no verified
  broker/SPAN margin or executable bid/ask history.

Absolute decision coverage passed: 2,355 decisions per development trial and
1,078 per validation trial, across seven variants in each segment.

## Exact authorized economic table

Amounts are INR. `D` marks the event-counterfactual diagnostic. Constant-15
and historical-lot rows are identical over these dates.

| Segment | Threshold | Lot/role | Fill / reject / no-trade | Gross | Ordinary net / cost | 2x net / cost | Uncertainty net / cost |
|---|---:|---|---:|---:|---:|---:|---:|
| Dev | 1.15 | historical | 30 / 632 / 1693 | -10,511.25 | -54,464.63 / 43,953.38 | -98,418.02 / 87,906.77 | -135,005.80 / 124,494.55 |
| Dev | 1.15 | constant15 | 30 / 632 / 1693 | -10,511.25 | -54,464.63 / 43,953.38 | -98,418.02 / 87,906.77 | -135,005.80 / 124,494.55 |
| Dev | 1.25 | historical | 31 / 629 / 1695 | -9,615.00 | -53,986.88 / 44,371.88 | -98,358.76 / 88,743.76 | -135,121.45 / 125,506.45 |
| Dev | 1.25 | constant15 | 31 / 629 / 1695 | -9,615.00 | -53,986.88 / 44,371.88 | -98,358.76 / 88,743.76 | -135,121.45 / 125,506.45 |
| Dev | 1.25 | historical D | 31 / 629 / 1695 | -9,615.00 | -53,986.88 / 44,371.88 | -98,358.76 / 88,743.76 | -135,121.45 / 125,506.45 |
| Dev | 1.35 | historical | 31 / 625 / 1699 | -9,615.00 | -53,986.88 / 44,371.88 | -98,358.76 / 88,743.76 | -135,121.45 / 125,506.45 |
| Dev | 1.35 | constant15 | 31 / 625 / 1699 | -9,615.00 | -53,986.88 / 44,371.88 | -98,358.76 / 88,743.76 | -135,121.45 / 125,506.45 |
| Val | 1.15 | historical | 16 / 340 / 722 | -2,931.00 | -55,290.98 / 52,359.98 | -107,650.95 / 104,719.95 | -155,431.72 / 152,500.72 |
| Val | 1.15 | constant15 | 16 / 340 / 722 | -2,931.00 | -55,290.98 / 52,359.98 | -107,650.95 / 104,719.95 | -155,431.72 / 152,500.72 |
| Val | 1.25 | historical | 17 / 332 / 729 | -3,507.75 | -55,827.94 / 52,320.19 | -108,148.12 / 104,640.37 | -155,719.09 / 152,211.34 |
| Val | 1.25 | constant15 | 17 / 332 / 729 | -3,507.75 | -55,827.94 / 52,320.19 | -108,148.12 / 104,640.37 | -155,719.09 / 152,211.34 |
| Val | 1.25 | historical D | 17 / 332 / 729 | -3,507.75 | -55,827.94 / 52,320.19 | -108,148.12 / 104,640.37 | -155,719.09 / 152,211.34 |
| Val | 1.35 | historical | 17 / 321 / 740 | -3,507.75 | -55,827.94 / 52,320.19 | -108,148.12 / 104,640.37 | -155,719.09 / 152,211.34 |
| Val | 1.35 | constant15 | 17 / 321 / 740 | -3,507.75 | -55,827.94 / 52,320.19 | -108,148.12 / 104,640.37 | -155,719.09 / 152,211.34 |

## Primary-policy risk and statistics

| Metric | Development | Validation |
|---|---:|---:|
| Eligible sessions | 791 | 360 |
| Filled trades | 31 | 17 |
| Recorded missing exits / unrecorded | 7 / 0 | 7 / 0 |
| Worst day | -9,030.62 | -8,204.12 |
| Worst ISO week | -13,451.44 | -22,205.35 |
| Maximum drawdown | 53,986.88 | 55,827.94 |
| Historical daily loss CVaR95 | 1,349.67 | 3,101.55 |
| Block-bootstrap CVaR95 UCB | 2,326.94 | 6,224.99 |
| Annualized daily Sharpe | -1.714971 | -2.399296 |
| Probabilistic Sharpe | 7.7267e-13 | 9.4631e-10 |
| Deflated Sharpe, 1,248 substantive trials | 7.8535e-50 | 9.5916e-39 |
| Win rate | 0% | 0% |
| Monetary turnover | 1,118,367.00 | 866,170.50 |
| Maximum modeled margin | 5,791.50 | 1,483.20 |

Both maximum drawdowns exceeded INR 50,000 and each ledger recorded one
drawdown breach. Worst day/week and CVaR thresholds passed. Recorded missing
exits were conservatively charged; none was unrecorded. The primary event
policy selected 66/36 event rows in development/validation, filled zero, and
rejected 16/11; event ordinary P&L was zero.

The threshold plateau passed in both segments, but only because all three
thresholds retained the same negative sign and similar loss per fill. Monthly
profit concentration was unavailable because there were no positive months.
CSCV PBO was `1.0` in both segments (7 variants, 8 blocks, 70 combinations).
The registry held 1,248 substantive trials plus 14 administrative completion
records; only substantive trials entered DSR multiplicity.

The independent red team found that every central-policy fill was an iron fly;
the combined policy therefore supplies no empirical S2 long-straddle evidence.
Risk suspension also truncated the effective experiment: fills ended by
2020-01-09 in development and 2023-01-25 in validation, followed by 585 and 321
drawdown-suspension rejections. Removing the seven missing-exit fills in each
segment still left observed-exit ordinary net P&L at `-7,694.25` over 24
development trades and `-3,974.51` over 10 validation trades. The failure is
not solely an adverse missing-exit convention.

## Decision and limitations

Development and validation ordinary and 2x net P&L were negative, tail gates
failed, fewer than 100 fills occurred, and no new quarantined 12-month sample
exists. Strategy pseudo P&L was **not opened** because the prerequisite gate
failed. S1/S2 is rejected and never accepted.

Historical bid/ask, depth, fills, SPAN/exposure margin, contract masters, and
lot sizes are unverified; reconstructed LTP execution cannot establish
tradability. See [authorized summary](outputs/research_v2/evaluation/segment_trial_summary.csv)
and [gates](outputs/research_v2/evaluation/gates.json).
