# Independent Red Team review — Research V2

**Review date:** 2026-10-13  
**Scope:** frozen predictive specifications and artifacts, model cards, strategy
ledgers and evaluation gates, execution/risk/event assumptions, root trial
history, and the archived pre-fix evaluation.  
**Final verdict:** **REJECTED — DO NOT ADVANCE**

## Executive finding

Research V2 establishes a credible *predictive* result for P2 against P0, but
it establishes the opposite of an investable strategy result. The frozen
economic policy loses money before costs in both development and validation,
loses substantially more after ordinary costs, breaches the drawdown gate,
has PBO equal to 1.0, and has far too few fills for tail inference. These are
mandatory failures under `STRATEGY_SPEC_V2.md`; no judgment call, favorable
cost interpretation, or pseudo-holdout result can override them.

The one-shot predictive pseudo result (`research_stable`) is not a strategy
result and is explicitly labeled previously exposed. Strategy pseudo P&L
remains unopened. No pseudo strategy marker, manifest, authorization, or
development/validation strategy-pass gate exists, and this review did not
create or open one.

## Decisive failures

The central historical-lot policy at threshold 1.25 produced:

| Measure | Development | Validation |
|---|---:|---:|
| Eligible decisions | 2,355 | 1,078 |
| Filled trades | 31 | 17 |
| Missing exits | 7 | 7 |
| Gross P&L | INR -9,615.00 | INR -3,507.75 |
| Ordinary-cost net P&L | INR -53,986.88 | INR -55,827.94 |
| 2x-cost net P&L | INR -98,358.76 | INR -108,148.12 |
| Uncertainty-case net P&L | INR -135,121.45 | INR -155,719.09 |
| Maximum drawdown | INR 53,986.88 | INR 55,827.94 |
| Annualized Sharpe | -1.71 | -2.40 |
| Deflated-Sharpe probability | 7.9e-50 | 9.6e-39 |

The frozen gate correctly records `research_failed`. Both segments fail the
required positive ordinary and 2x net-P&L tests, the INR 50,000 drawdown test,
zero-risk-breach test, monthly profit-concentration test, and sample-sufficiency
test. Neither has 100 fills; validation also lacks 500 eligible sessions. The
required new approximately twelve-month quarantined forward sample does not
exist.

This is not merely a conservative missing-exit artifact. Removing all
missing-exit fills as a diagnostic leaves observed-exit gross P&L of only INR
-133.50 over 24 development trades and INR +115.50 over 10 validation trades.
After ordinary costs those same observed-exit subsets lose INR 7,694.25 and
INR 3,974.51, respectively. There is no economically meaningful gross edge
waiting to be recovered by a less punitive settlement assumption.

## Strongest alternative explanations

1. **Predictive accuracy did not translate into an option edge.** P2 improved
   walk-forward QLIKE by 22.32% versus P0 and its already-exposed predictive
   pseudo score by 11.54%, yet all filled central-policy trades were iron flies
   and gross P&L was negative. Better spot realized-variance prediction is not
   evidence that the implied-versus-forecast ratio overcomes option prices,
   smile exposure, path dependence, or execution costs.

2. **The apparent threshold plateau is vacuous.** Thresholds 1.25 and 1.35
   have identical filled counts and P&L in each segment; 1.15 differs by only
   one trade. The plateau passes because all variants have the same negative
   sign and similar losses, not because a profitable region is stable. PBO is
   1.0 in both development and validation, the strongest possible warning
   that variant selection does not generalize across time blocks.

3. **Risk suspension truncates the effective experiment.** Development fills
   occur only from 2019-11-11 through 2020-01-09; validation fills occur only
   from 2023-01-02 through 2023-01-25. After the drawdown limit is reached,
   585 development decisions and 321 validation decisions are rejected for
   drawdown suspension. The nominal multi-year ledgers therefore do not
   demonstrate trading behavior across their advertised regimes; they
   demonstrate rapid early failure followed by indefinite shutdown.

4. **Missing fixed-strike coverage is material.** Seven missing exits represent
   23% of development fills and 41% of validation fills. Their adverse treatment
   is contractually frozen and appropriate, but the frequency itself is a
   tradability/data-coverage failure. It is not reasonable to describe a policy
   with this exit observability as execution-ready.

5. **Costs are modeled, not observed.** Historical bid/ask, depth, fills,
   contract notes, and SPAN/exposure files are absent. Stored prices are
   LTP-like; OI is used only as a capacity proxy. Historical margin and lot
   sizes remain explicitly unverified. Ordinary cost assumptions therefore
   cannot prove exact realized losses—but they also cannot rescue the policy,
   because gross P&L and observed-exit net P&L are already unfavorable.

6. **The sensitivity labels add less evidence than they appear to.** Historical
   and constant-15 results are identical because development and validation end
   before the assumed 2024-12-25 lot-size change. This is an identity check, not
   evidence about the unverified 30-lot regime. The event counterfactual is also
   identical to the primary central trial in both segments; it contains no
   incremental event-day fill and therefore supplies no empirical event-risk
   information.

7. **Multiple testing remains substantial.** The evaluation counts 1,248
   substantive trial records, including remediated and interrupted research
   history. The extremely negative deflated-Sharpe results and PBO=1.0 are
   consistent with the strategy being a product of a broad research search,
   not a robust economic effect. CPCV folds and neighboring thresholds are
   correlated and do not create independent confirmations.

## Protocol deviations and audit-history concerns

- P1 correctly failed its development plateau and never opened walk-forward
  validation. P2 passed versus P0. However, `MODEL_CARD_P2.md` still describes
  the P1 comparison as pending, while the later leakage audit interprets P1's
  disqualification as eliminating that comparison. The later interpretation
  is reasonable, but the frozen documentation is not internally synchronized.

- Economic outcomes were opened before the final evaluator was settled. The
  preserved `evaluation_stale_pre_cvar_fix` archive shows an earlier evaluator
  that could not identify the primary trials and used the pre-fix CVaR path.
  The evaluation code, authorization, and reports were then revised and
  re-run. The current fixes are hash-bound and the final result remains a clear
  failure, so this does not create a false pass. It does mean the economic
  reporting pipeline was not pristine, fully frozen code on its first outcome
  opening; post-outcome bug repair introduced researcher discretion.

- The stale archive's `SHA256SUMS` records paths under the active
  `outputs/research_v2/evaluation/` directory rather than its archive directory.
  The hashes match the archived files only when interpreted by basename; a
  literal verification now checks the current directory and fails for the
  changed reports. The archive is useful audit evidence but is not a cleanly
  relocatable reproduction bundle.

- The current final evaluation is materially better controlled: its exact
  evaluator, runner, tests, registry, authorization, and fourteen ledger hashes
  are bound by `STRATEGY_EVALUATION_AUTHORIZATION_V2.json`; the current hashes
  match that authorization, and all six final output hashes match the active
  `SHA256SUMS`. This establishes integrity of the recorded failure, not market
  validity or live reproducibility.

## Data, tail, and reproducibility caveats

- The option source has narrow moving-strike coverage, no executable quote
  history, and no historical contract master or verified lot schedule. The
  margin proxy cannot model dated SPAN, intraday margin changes, liquidation,
  or broker-specific offsets.
- Only 24 development and 10 validation fills have observed exits in the
  central trial. Historical CVaR and its bootstrap upper bound may be
  numerically estimable because rejected/no-trade sessions are included as
  zero-return ledger days, but this does not provide a credible tail sample of
  actual positions.
- The event calendar contains official dates, but `announced_at` is unknown;
  the unscheduled 2021 halt is correctly retained as incident stress rather
  than retrospectively excluded. No filled event counterfactual means event
  behavior remains untested.
- Re-running the authorized evaluator can reproduce the reports from the
  frozen ledgers, but the underlying economic exercise still depends on
  unavailable bid/ask, fill, margin, and contract evidence. Computational
  reproducibility is not execution reproducibility.
- The predictive pseudo-holdout was already exposed elsewhere and can never
  be an acceptance sample. Opening strategy pseudo P&L after the present
  development/validation failure would have no legitimate promotion purpose
  and would consume another contaminated outcome.

## Final disposition

The Research V2 strategy verdict **must be REJECTED**. It fails the frozen
economic and risk gates in both development and validation, lacks adequate
trade and tail samples, exhibits complete PBO failure, and relies on
unvalidated execution, margin, and lot assumptions. The predictive P2 result
may be retained as a research finding about realized-variance forecasts, but
it must not be presented as evidence of option profitability or tradability.

Do not open pseudo strategy P&L, do not retune thresholds or costs against these
outcomes, and do not promote to paper capital or live capital. Any future work
must be a new, preregistered hypothesis using verified executable-market and
margin evidence and a genuinely new quarantined forward sample.
