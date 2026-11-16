# Pre-registered BANKNIFTY volatility hypotheses

Registered on 2026-10-13 before any new-strategy outcomes were computed. The
existing repository's earlier results are prior exposed evidence, not results
of this preregistration. All hypotheses use monthly options actually present in
the source, one historical lot, at most one concurrent position, the causal
availability clock in `DATA_AUDIT.md`, and the cost/margin rules in `SCOPE.md`.
No hypothesis may be modified after its first outcome is opened; a changed rule
receives a new hypothesis and trial identifier.

The existing data may be used only for walk-forward development and validation.
No result can satisfy the final INR 100,000 acceptance test because no untouched
twelve-month period exists.

## Common protocol

- Candidate decision times: 09:45, 11:45 and 13:45, chosen ex ante to span
  intraday seasonality while limiting overlap. A 120-minute position is skipped
  if it cannot close by 15:30. A single time-only version uses 09:45.
- Horizons: 60 and 120 minutes. Fifteen and 30 minutes are model diagnostics,
  not strategy-selection degrees of freedom, because fixed costs dominate and
  prior repository mining already covered them heavily.
- Monthly expiry DTE strata: 0-3, 4-7 and 8-20 calendar days. DTE above 20 is
  excluded from the primary naked-short hypotheses because theta capture per
  unit margin is expected to be weak.
- Forecasts: expanding monthly walk-forward fit; training labels must end before
  the test block and are purged/embargoed by at least 120 minutes. Features are
  strictly timestamp-available. Execution is the next eligible snapshot.
- Selection: development years are evaluated in chronological folds. Candidate
  parameters are fixed grids listed below. A cell advances only if neighboring
  cells form a performance plateau, both halves are positive after cost, and
  the 2x-cost case is not economically destroyed.
- Uncertainty: day-block bootstrap confidence intervals, worst day/week, maximum
  drawdown, 95% CVaR, turnover, cost breakdown, deflated Sharpe using the full
  reconstructed trial count, and PBO where enough comparable folds exist.
- No stop-loss result is promoted because executable intrawindow fixed-strike
  paths are incomplete. Defined-risk structures are preferred where feasible;
  naked results require explicit gap and forced-close stress.

## H0 — supplied live-policy baseline

**Rule.** Sell one ATM straddle when current IV exceeds horizon-matched predicted
RV by a cost-covering margin. Otherwise choose the two-strike OTM short strangle
when its supplied probability is at least 0.693128, then an ATM short straddle
when its supplied probability is at least 0.701625. No long-vol leg.

**Expected sign.** Positive gross theta/variance-premium P&L, weakened materially
by costs and crash days. This is a benchmark, not a candidate to retune.

**Falsifier.** Nonpositive net P&L, negative 2x-cost P&L, tail loss incompatible
with the margin budget, or inability to reconstruct its supplied probabilities
causally.

## H1 — robust variance-risk-premium harvesting

**Economic rationale.** Option-implied variance includes compensation for jump
and variance risk. The compensation should be most harvestable when IV exceeds
an upper predictive distribution for realized variance by more than explicit
theta, execution and tail reserves—not merely when point IV exceeds point RV.

**Rule family.** At a registered decision time, sell an ATM straddle or a
defined-risk iron fly only when
`IV variance - RV predictive quantile - cost reserve - tail reserve > 0`.
The predictive quantile grid is {0.75, 0.85, 0.90}; reserve multipliers are
{1.0, 1.25, 1.5}. Structures are separate trials. Position size is always one
lot and subject to the fixed margin cap.

**Expected sign.** Positive net P&L that increases smoothly with the IV-minus-RV
surplus until trade count becomes sparse; iron-fly tail loss should be smaller
than naked straddle loss at the cost of lower mean P&L.

**Falsifier.** Only one threshold is profitable, the 2x-cost result is negative,
bootstrap lower confidence bound is nonpositive, or profit is concentrated in
one event/day.

## H2 — skew-aware short-strangle selection

**Economic rationale.** When downside IV skew is unusually rich relative to its
own causal history, symmetric ATM selling overpays for central gamma and may
underprice crash asymmetry. Selecting wings using current skew and a conservative
RV forecast may improve P&L per unit tail risk.

**Rule family.** Compare the observed one- and two-strike OTM short strangles at
entry. Trade only when both legs pass IV quality/liquidity proxies and the
minimum leg premium exceeds {5, 10, 15} points. Skew z-score gates are
{0.0, 0.5, 1.0}. Select the structure with the larger entry-time downside-adjusted
score; no trade is an explicit choice. No exit-time availability flag may enter
selection.

**Expected sign.** Positive net P&L and smaller 95% CVaR than H1's naked ATM
straddle, with adjacent premium and skew gates giving similar results.

**Falsifier.** Missing fixed-strike exits drive the result, P&L vanishes when all
model-priced exits are penalized/excluded, or gains come only from the post-2024
lot-size assumption.

## H3 — monthly term-structure relative value

**Economic rationale.** Near-month IV can temporarily richen relative to the
next observed monthly expiry around event and expiry pressure. A same-strike
calendar with bounded net debit may isolate relative-volatility normalization
with less outright variance exposure.

**Rule family.** Only timestamps with observed near and next monthly expiries and
the same absolute ATM strike **at entry** are eligible. Exit availability is
never an eligibility feature: a missing later fixed-strike quote remains in the
ledger and receives the pre-registered conservative missing-exit treatment.
Trade a long-far, short-near ATM calendar when the causal near/far
total-variance spread z-score is in {1.0, 1.5, 2.0}; reverse calendars are not
allowed. Horizons are 60/120 minutes. Net debit and margin must fit the common
limits.

**Expected sign.** Positive spread-convergence P&L with lower spot-delta and
gamma tail sensitivity than naked short volatility.

**Falsifier.** Fewer than 30 independent expiry-cycle observations, performance
depends on model-priced legs, or far-expiry IV/fill uncertainty overwhelms the
gross edge. Sparse cells are reported, not pooled or relaxed.

## H4 — conservative long-volatility trigger

**Economic rationale.** Long volatility should be considered only when a
causal upper-tail RV forecast exceeds market-implied break-even variance by more
than theta and round-trip cost. This is materially different from the rejected
IV/RV convergence rule: it requires an explicit net-payoff hurdle and does not
condition on predicted IV convergence.

**Rule family.** Buy an ATM straddle or one-strike strangle when
`RV predictive quantile - IV break-even variance - theta/cost hurdle > 0`.
RV quantiles are {0.75, 0.85, 0.90}; hurdle multipliers are {1.0, 1.25, 1.5}.
The position is skipped if either observed premium is below 5 points or the next
snapshot cannot fill both legs under the deterministic fill rule.

**Expected sign.** Low-frequency positive convex P&L concentrated in high
forecast-volatility regimes, with positive performance after 2x costs and no
single-day dependence.

**Falsifier.** Negative net expectancy, fewer than 30 independent signals,
failure to outperform an unconditional long-straddle control, or profitability
only at a single quantile/hurdle pair.

## Excluded hypothesis

The earlier rule “long if IV < RV and predicted IV is approximately predicted
RV; short if IV > RV and predicted IV is falling” is not retested or rebranded.
Its prior failure may be analyzed, but it cannot advance as a candidate.

## Research phase predictive hypotheses (registered 2026-10-13)

These hypotheses are governed by the immutable protocol in `SPLITS.md`. No PnL
may be inspected unless a model clears the predictive gate. The July 2024 to
May 2025 evaluation is always labelled “pseudo-holdout, previously exposed.”

### P0 — current HAR-Ridge forecasting baseline

**Features:** trailing log realized variance over 5, 15 and 60 minutes.
**Estimator:** standardized Ridge with alpha in the split-file grid.
**Expected result:** persistence should provide a strong, stable baseline.
This model is a comparator and does not advance by beating itself.

### P1 — HARQ semivariance linear extension

**Rationale:** quarticity proxies forecast uncertainty in realized variance,
while downside/upside semivariance and jump variation distinguish continuous
volatility persistence from asymmetric shocks. A regularized linear extension
may improve QLIKE without the instability of a flexible learner.

**Features:** P0 plus log realized quarticity over 15 and 60 minutes, downside
and upside RV over 15 and 60 minutes, jump proxy `max(RV60-BV60,0)`, current
absolute return, and fixed sine/cosine time-of-day terms. All are trailing.
**Estimator grid:** ElasticNet with alpha `{0.0001,0.001,0.01,0.1}` and l1 ratio
`{0,0.25,0.5}`. **Expected sign:** lower QLIKE than P0, especially after large
downside moves. **Failure:** predictive gate in `SPLITS.md` is not met or the
selected cell lacks a neighboring 2% plateau.

### P2 — shallow nonlinear HAR state model

**Rationale:** interactions among current RV level, quarticity, jump variation
and intraday seasonality may be nonlinear but should be learnable with a small,
strongly regularized tree ensemble.

**Features:** exactly the P1 feature set. **Estimator grid:** histogram gradient
boosting squared-error regressor with max depth `{2,3}`, learning rate
`{0.03,0.05}`, L2 regularization `{1,10}`, and fixed 200 iterations. No early
stopping uses a test fold. **Expected sign:** lower QLIKE than both P0 and P1 in
calm and high-volatility regimes. **Failure:** any predictive gate fails, a
single month supplies over 35% of improvement, or depth/regularization results
form a sharp peak.

### P3 — regime mixture of linear HAR experts

**Rationale:** a single persistence coefficient can be misspecified across
calm and turbulent states. A causal state determined only from trailing RV may
allow two simple Ridge experts without an opaque classifier.

**Features:** P0 features plus the P1 trailing state variables. **State rule:**
training-only rolling percentile of trailing 60-minute RV, with candidate high-
vol thresholds `{0.70,0.80,0.90}`. Fit separate Ridge experts in each state;
Ridge alpha uses the P0 grid. At prediction time the state cutoff comes only
from the current training sample. **Expected sign:** improved high-vol QLIKE
without degrading calm QLIKE. **Failure:** fewer than 250 training rows in an
expert, any predictive gate fails, or only one exact state threshold works.

## Research V2 economic hypotheses (frozen after predictive selection, before P&L)

### S1 — P2-conditioned defined-risk variance-premium sale

Under `STRATEGY_SPEC_V2.md`, a short two-wing iron fly entered when horizon ATM
IV variance is at least 1.25 times frozen P2 predicted RV will have positive net
P&L under ordinary and 2x proxy costs. Expected sign: positive variance-risk
premium capture with bounded contractual loss. It fails if validation is
nonpositive, the 1.15/1.35 plateau fails, any risk gate fails, or profit is
concentrated in one month or event.

### S2 — P2-conditioned long-volatility trigger

A long ATM straddle entered when frozen P2 predicted RV is at least 1.25 times
horizon ATM IV variance will have positive net P&L after ordinary and 2x costs.
This differs from the rejected convergence rule because it uses no IV forecast
or convergence condition. It fails under the same plateau, validation,
concentration and risk rules.
