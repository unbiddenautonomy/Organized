# Pre-registered risk policy

## Status and purpose

**Status: research/paper only; live capital remains BLOCKED.**  This policy is
frozen before any further strategy outcome is examined.  It defines loss and
operational limits for INR 1,000,000 of cash-equivalent capital; it does not
repair the data, leakage, execution, contract, margin, or untouched-holdout
blocks in `SCOPE.md`, `DATA_AUDIT.md`, or `LEAKAGE_AUDIT.md`.  It must not be
used to claim profitability, optimize a signal, or justify broker submission.

All rupee thresholds are fixed from the initial INR 1,000,000 capital, not from
a rising equity curve.  They may be made smaller before a future test is
opened, but may not be relaxed after outcomes are observed.

## Hard sizing and entry gate

| Control | Frozen limit |
|---|---:|
| Historical position size | At most one exchange lot |
| Concurrent positions | One |
| New positions per session | One |
| Buffered margin | INR 350,000 (35% of capital) |
| Margin buffer | 20% over verified broker/SPAN plus exposure requirement |
| Maximum raw margin before buffer | INR 291,666.67 |
| Pre-trade severe stressed loss | INR 7,500 (0.75% of capital) |

An entry is rejected if any limit is exceeded, any required input is stale or
asynchronous, a position is already open, the session already had an entry, or
the lot size cannot be verified against the contract applicable on that date.
Equality is permitted only for margin and stressed loss; reaching a realized
loss limit is a rejection.  There is no averaging down, pyramiding, margin
reuse, discretionary override, or sizing up to pursue the INR 100,000 research
threshold.

The margin decision uses a broker pre-trade margin response tied to the exact
contracts and quantities.  `1.20 * raw_margin` must be no more than INR
350,000.  A historical proxy may be recorded in simulation, but it is not a
verified pass.  A rejection, unavailable response, expired response, or a
margin increase before acknowledgement means `NO_TRADE`.

For live eligibility, every position must have a contractually bounded maximum
loss no greater than INR 7,500 after conservative costs.  An unhedged short
straddle or strangle has unbounded loss and therefore fails the live gate even
if a finite scenario stress looks small.  Such structures may only be observed
in paper research while the current BLOCK verdicts remain in force.

## Marking and loss limits

Risk P&L is net liquidation P&L: realized P&L plus the conservative cost to
close every open leg, including brokerage, all taxes and fees, adverse bid/ask
crossing, slippage, and settlement STT where relevant.  Missing or stale marks
use the worse of the last valid mark and the current forced-close stress; they
are never treated as zero loss.

| Limit | Trigger | Mandatory response |
|---|---:|---|
| Daily loss | INR 10,000 (1.0%) | Cancel entries, flatten at the first eligible executable quote, and remain off for the session |
| ISO-week loss | INR 25,000 (2.5%) | Flatten and suspend through the end of the ISO week |
| Peak-to-trough drawdown | INR 50,000 (5.0%) | Indefinite suspension; no automatic reset |

Daily P&L resets only at the next ordinary NSE session.  Weekly P&L is measured
from the equity at the start of the ISO week and includes all costs and manual
adjustments.  Drawdown is measured from the high-water mark including the
initial INR 1,000,000 capital.  A limit crossed because of a gap remains a
breach even if the eventual fill improves.

A drawdown suspension can be lifted only after an independent incident review,
reconciliation of every fill and fee, resolution of all data/execution defects,
a newly versioned policy frozen before new outcomes, and at least ten clean
paper sessions.  It cannot be reset merely by starting a new report or model.

## Event-day policy

There is no new entry on a predeclared event day.  Before each calendar month,
the no-entry calendar is frozen from official NSE and official issuer sources
and includes at least Union Budget presentation, RBI policy announcements,
national election result/counting days, scheduled BANKNIFTY methodology or
constituent changes, and exchange special sessions.  Newly announced events are
added when announced, with an audit timestamp; past classifications are never
rewritten.

For an unscheduled exchange halt, circuit event, loss of the spot or option
feed, contract ambiguity, timestamp inversion, or material market-wide news,
no new entry is allowed.  An open position is cancelled/flattened at the first
verified eligible quote according to exchange and broker rules, then the system
enters manual-review suspension.

Event-day exclusions are not a tunable signal.  Event days and rejected orders
remain in the audit ledger so they cannot disappear from denominators.

## Exit, gap, and forced-close stress

The ordinary exit process begins at 15:15 Asia/Kolkata.  All entry strikes and
expiries are immutable.  Any unfilled close is reissued only against a fresh
eligible quote, with escalation by 15:25; lack of a verified close is an
incident, not an assumed flat position.

Before entry, calculate the maximum net loss across all of these joint severe
intraday scenarios:

- underlying shocks of -8%, -5%, +5%, and +8%;
- implied-volatility shocks of +10 and +20 absolute volatility points, applied
  jointly with the underlying shock and without favorable skew assumptions;
- at least 30 minutes of adverse time progression;
- three times the frozen slippage/spread assumption and two times all ordinary
  transaction costs; and
- the worse result from repricing every leg versus its contractually bounded
  intrinsic maximum loss.

The resulting loss, including costs, must not exceed INR 7,500.  Separately,
the ledger must report a forced-close/settlement scenario using underlying
shocks of +/-12%, IV +30 points, five-times slippage, two-times fees, settlement
STT, and no credit for an unavailable exit.  Any position lacking a defensible
price under that scenario is assigned its contractual maximum loss; an
unbounded position fails live eligibility.

These shocks are safeguards, not estimates of quantiles.  They may not be
reduced after seeing results.  A larger shock observed in paper or live data
replaces the relevant minimum prospectively and triggers incident review.

## Tail-risk promotion requirements

These are necessary but not sufficient gates.  They do not override any BLOCK
verdict or prove profitability.  Evaluation uses independent session-level P&L
and non-overlapping ISO weeks, with ordinary costs, two-times costs, and the
forced-close stress reported separately.  The same frozen policy must satisfy
all cases without refitting.

- At least 500 eligible paper sessions and 100 executed paper trades are
  required for a tail review; at least roughly twelve months must come from the
  future quarantined sample specified in `SCOPE.md`.
- Daily 95% loss CVaR must be at most INR 15,000, and its one-sided 95% block-
  bootstrap upper confidence bound must be at most INR 20,000.
- Worst observed day must be no worse than -INR 25,000.
- Worst non-overlapping ISO week must be no worse than -INR 40,000.
- Maximum peak-to-trough drawdown must be at most INR 50,000.
- There must be zero risk-rule overrides, position-limit breaches, unrecorded
  missing exits, or unresolved reconciliation differences.

A daily or weekly hard-stop breach is still an operational failure even where
the aggregate tail statistic passes.  Tail metrics are reported in rupees and
as fractions of the fixed initial capital, with the count of tail observations
and missing/stressed marks disclosed.

## Required audit record

Every decision, including `NO_TRADE`, records the policy version and hash;
decision/order/fill/exit timestamps; source availability timestamps; exact
contracts and historical lot size; verified raw and buffered margin; primary
and forced-close stressed losses; event flag and calendar version; fresh/stale
status; daily and weekly P&L; equity high-water mark and drawdown; rejection
reasons; all fill attempts; fees; and reconciliation state.  Risk checks occur
again immediately before order release because margin and quotes can change.

The pure reference controls are in `research/risk_limits.py`.  They fail closed
on invalid numeric inputs and return every rejection reason rather than only
the first.

## Items explicitly unvalidated

- Historical SPAN/exposure margin and forced-liquidation behavior.
- Historical lot sizes and contract listings without contemporaneous masters.
- Executable bid/ask, depth, partial fills, rejection rates, and market impact.
- Stop and forced-close execution on complete fixed-strike intraday paths.
- Missing-exit settlement, gap behavior, exercise/assignment, and settlement
  STT in the supplied narrow monthly panel.
- The probability or adequacy of the chosen +/-8% and +/-12% shocks.
- CVaR, worst-day/week, and drawdown thresholds on a genuinely untouched year.
- The accuracy and completeness of a reconstructed historical event calendar.
- Any live broker adapter, kill switch, account reconciliation, or operational
  recovery process.

Until these items and the existing audit vetoes are resolved, this document is
a pre-registration for paper research only.
