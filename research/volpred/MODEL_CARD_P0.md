# Model card P0 — HAR-Ridge baseline

**Status: comparator only; outperformed by P2; never accepted.**

P0 forecasts 120-minute realized variance from trailing log realized variance
over 5, 15, and 60 minutes. Within every development CPCV and expanding
walk-forward training sample, Ridge alpha was selected on a training-only
chronological inner split from `{0.01, 0.1, 1, 10, 100}`. Scaling and the
positive-scale lognormal smearing adjustment were also fitted only on the
permitted training sample.

The final comparator refit selected `alpha=0.01` on the 44,679 development plus
walk-forward rows ending at maximum label end `2024-06-28 15:25:59+05:30`.
Frozen artifact SHA-256:
`6210dbebd4739d39b29c7bc4a6b73e140ee292ec2a96a703c364455533beb5cb`.

| Predictive sample | P0 QLIKE | P0 log-MSE | P2 QLIKE | Result |
|---|---:|---:|---:|---|
| Expanding walk-forward, 2023-01 to 2024-06 | 0.175972708 | 0.419263572 | 0.136690661 | P2 improves QLIKE 22.3228% |
| **pseudo-holdout, previously exposed** | 0.223904088 | 0.561828481 | 0.198055906 | P2 improves QLIKE 11.5443% |

P0 had zero raw calibration failures in both samples but an aggregate
actual/forecast ratio of `0.815500` in walk-forward, indicating overforecasting
on average. It is a baseline, not a strategy or profitability claim. The
pseudo sample was previously exposed and permanently consumed, and the frozen
P2-to-options mapping was economically rejected.

Failure modes include persistence breaks, intraday seasonality not represented
by the three features, overforecast calibration, and inability to model
nonlinear quarticity/jump interactions. P0 is not live eligible.
