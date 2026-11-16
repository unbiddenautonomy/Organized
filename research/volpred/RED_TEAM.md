# Independent red-team review

**Review date:** 2026-10-13  
**Decision:** **NO ADVANCE** to a profitability, tradability, capital-allocation,
or final-acceptance claim. The INR 100,000 acceptance question is
**inconclusive**, not passed and not failed.

Outcome-blind engineering and a frozen forward paper/shadow run may continue.
No existing historical result may be called an untouched holdout, and no
current component justifies live broker submission.

This review used `SCOPE.md`, `DATA_AUDIT.md`, `LEAKAGE_AUDIT.md`,
`HYPOTHESES.md`, `TRIALS.csv`, `EXECUTION_MODEL.md`, and the current integrity,
execution-cost, risk-control, paper-intent, and associated test code. It did not
run or recompute strategy P&L and did not inspect a quarantined future holdout.

## Why apparent profitability is not persuasive

Any positive historical result has several plausible non-edge explanations,
none of which the current evidence can eliminate:

1. **Same-bar and availability leakage.** Active options studies price entry
   from the same row used to decide the signal and do not implement
   `available_at = max(datetime, Date + tick_time)`. The current paper puncher
   neither compares prediction time with chain time nor waits for a strictly
   later executable snapshot. A small apparent intraday edge can therefore be
   entirely timing leakage.
2. **Outcome-dependent survival.** Existing cache builders inner-join future
   timestamps, lose entries with missing exits, and sometimes model-price a
   disappeared fixed strike. Narrow moving-strike coverage is most likely to
   lose observations after large underlying moves, exactly where short-gamma
   losses are worst. A profitable retained sample can be survivorship bias.
3. **Researcher and model selection.** This repository contains many previously
   inspected short-volatility, IV/RV, structure, regime, threshold, stop, and
   hybrid variants. The new hypotheses were written after that exposure, so
   preregistration now freezes an exploratory continuation; it does not make
   the old data confirmatory.
4. **Uncounted multiplicity.** `TRIALS.csv` has seven family-level rows, but its
   own grids imply at least 828 primary cells before the supplied baseline,
   alternative single-time versions, neighbor/plateau choices, earlier scripts,
   or reporting choices: 324 H1 cells, 162 H2 cells, 18 H3 cells, and 324 H4
   cells. Treating each family as one trial materially understates selection.
5. **Regime luck and dependence.** Only 68 monthly expiries are present. Signals
   within a day and expiry cycle share volatility, liquidity, and event shocks;
   they are not independent observations. The period spans the pandemic,
   changing weekly/monthly expiry ecology, changing lot size, transaction-tax
   changes, and the end of BANKNIFTY weeklies. A result concentrated in one
   regime, event, year, or transition can mimic stable edge.
6. **Non-executable prices.** Stored values are LTP-like, not certified quotes.
   There are no bid/ask, depth, queue, trade-condition, or fill records. OI is
   not displayed size. The fill model therefore cannot show that an apparent
   premium was simultaneously available to all legs or available after the
   signal.
7. **Margin and tail understatement.** Historical SPAN, broker add-ons,
   expiry-day margin changes, intraday hikes, forced liquidation, and actual
   offsets are absent. Short-volatility returns can look attractive until a
   margin breach or missing-exit tail is represented. A fixed 100 bp gap reserve
   does not establish crash sufficiency.
8. **Acceptance-test misuse.** INR 100,000 is a one-time test on a new hidden
   year, not a target to optimize against and not a threshold that any exposed
   2024/2025 result can satisfy. Scaling lots, selecting a favorable 12-month
   window, relabeling a pseudo-holdout, or changing rules after opening outcomes
   would invalidate the test.

## Design and preregistration defects

### The hypotheses are not yet executable specifications

- H1 mixes IV variance, an RV predictive quantile, a cost reserve, and a tail
  reserve without fixing units, annualization, horizon scaling, or the tail
  stress construction. Multiple implementations could be chosen after results.
- H2 does not define the skew lookback/minimum history, z-score estimator,
  “liquidity proxy,” or “downside-adjusted score.” Choosing the larger score
  also adds an implicit structure-selection degree of freedom.
- H3 declares a timestamp eligible only when the same strike is observed at
  **entry and exit**. Exit observability is future information. Every causal
  entry must remain in accounting; a missing exit needs the frozen penalty or
  settlement path.
- H4 does not fully define “IV break-even variance” or the theta/cost mapping.
  Comparing to an unconditional control after inspecting both can become an
  additional selection channel unless its calculation and decision rule are
  frozen now.
- “Broad plateau,” “economically destroyed,” “controlled drawdown,” “event
  day,” “stale,” and “positive halves” lack numerical, immutable definitions.
  These phrases leave researcher discretion after outcomes are visible.
- The exact final-holdout start/end, minimum coverage, treatment of exchange
  holidays/missing sessions, acceptance drawdown ceiling, minimum independent
  trades/cycles, and treatment of a zero-trade year are not frozen.

`TRIALS.csv` must enumerate every evaluated cell or reference an immutable
machine-readable grid expansion. Failed runs, code-error reruns, alternative
labels, cost variants, and legacy related trials must remain in the denominator.
The exact supplied baseline probabilities also need causal artifact provenance;
otherwise H0 is not reproducible.

## Data, leakage, and provenance vetoes

The independent leakage veto is decisive and remains in force. In particular:

- timestamps are timezone-naive and `tick_time` availability is absent from
  active pipelines;
- signal, order, and fill time are not separate events;
- prediction and chain clocks are not joined by a bounded backward as-of rule;
- purge/embargo is not a reusable enforced invariant with `label_end_at`;
- model artifacts lack sufficient immutable source, feature-code, training,
  tuning, label-end, embargo, and trial provenance;
- current artifacts were fit through exposed history and cannot evaluate that
  same history;
- historical contract masters do not validate expiry, listings, or lot size;
- nonpositive spot sessions, gaps, special sessions, IV failures, incomplete
  snapshots, and absent fixed-strike exits create state-dependent exclusions.

The new `research/integrity.py` functions and synthetic tests are useful unit
primitives, but a repository search finds them imported only by their tests,
not by active research, strategy, or paper-intent code. Passing unit tests do
not cure an unwired production path. The tests also do not cover a complete
signal-to-fill-to-exit event ledger, label-end split enforcement, stale
prediction rejection, multi-leg atomic fills, or retention of every entry
through exit accounting.

## Execution and cost-model challenge

`research/execution_costs.py` is appropriately disclosed as a proxy, but it
cannot validate tradability and is not integrated into active strategy code.
Specific favorable assumptions remain:

- missing OI defaults to one-lot capacity, which converts unknown capacity into
  a modeled fill;
- `floor(OI * 0.001)` treats OI as executable capacity despite OI having no
  price, side, queue, or immediacy information;
- the 15 bp half-spread has no tick-size floor. For a 5-point option it implies
  less than one tick of crossing and can therefore be mechanically too small;
- one volatility ratio is used across strikes, DTE, time of day, and entry/exit;
  cheap wings and event/expiry conditions need separate conservative floors;
- multi-leg synchronization, legging risk, cancellations, stale quotes, price
  limits, and adverse-selection delay are not charged;
- margin carry is manually passed into `execution_cost`, and settlement cost is
  a separate object. Nothing guarantees that callers include either component;
- `CostBreakdown.stressed(2.0)` doubles its fields but does not automatically
  include or stress settlement gap reserve or a margin shortfall/liquidation;
- the 100 bp missing-exit gap reserve is arbitrary, not validated against
  BANKNIFTY event moves, and must not be presented as tail adequacy.

The statutory schedule tests establish arithmetic boundaries, not broker
contract-note reconciliation. Before historical use, compare representative
dates against actual archived contract notes, including tax aggregation and
rounding. Before paper promotion, use captured executable bid/ask/depth and
actual order acknowledgements/fills to calibrate spread, latency, rejections,
and partial fills. Until then, both ordinary and 2x results remain scenarios,
not evidence of execution.

## Risk-control challenge

`research/risk_limits.py` is likewise not imported by active strategy or paper
code. Its defaults (`inputs_fresh=True`, `margin_verified=True`,
`executable_quotes=True`, `defined_max_loss=True`) are fail-open when callers
omit evidence. Paper mode permits unverified margin, non-executable prices, and
unbounded loss. That may be acceptable for pure engineering, but no performance
produced under that mode can support promotion.

Additional issues:

- the docstring says equality at a loss limit rejects, while stressed loss of
  exactly INR 7,500 is allowed and tested as allowed;
- freshness tolerance and event-day calendar are undefined;
- `day_pnl`, `week_pnl`, and `current_drawdown` are not validated as finite and
  nonnegative where applicable, so NaN can bypass comparisons;
- the 0.75% trade stress, 1% day, 2.5% week, and 5% drawdown limits are exposed-
  data-era choices rather than independently validated risk tolerances;
- the code gates new entry but does not demonstrate exit, liquidation, open-risk
  marking, restart recovery, or persistence of daily/weekly/drawdown state.

The execution-margin proxy already shows why enforcement matters: at spot
50,000, one 30-unit uncovered lot produces INR 360,000 buffered margin and
breaches the INR 350,000 cap. Any naked-short result after the lot-size change
must be rejected whenever the frozen proxy breaches; it cannot be kept merely
because historical P&L is attractive.

## Statistical and economic evidence required

Even after causal repair, profitability can still be explained by dependence
and selection unless all of the following are frozen before outcomes:

- expiry-cycle or day-block inference that reflects the true independent unit;
- family-wise multiplicity using the reconstructed legacy-plus-new trial count,
  with uncertainty when that count is only a lower bound;
- deflated Sharpe/PBO only where assumptions are defensible, alongside raw
  trade count and number of independent cycles;
- worst day/week/cycle, event exclusions and inclusions, 95% CVaR, maximum
  drawdown, and concentration by year/regime;
- leave-one-year, leave-one-event, and pre/post structural-regime diagnostics
  specified as robustness checks, never as a way to choose a favorable sample;
- ordinary and 2x all-in costs on the identical frozen trades, observed-exit
  results, and a worst-case ledger retaining all missing exits;
- constant-lot sensitivity until historical contract masters validate sizing;
- no annualization or Sharpe claim from sparse, overlapping intraday trades
  without stating the effective independent sample size.

Positive aggregate P&L, a majority of winning trades, a positive walk-forward
mean, or a profitable pseudo-holdout is insufficient by itself.

## Required remediation and gate sequence

### Gate A — before any new historical outcome is opened

1. Resolve every ambiguous hypothesis definition above and freeze a
   machine-readable policy, grid expansion, risk rules, and acceptance protocol.
2. Expand `TRIALS.csv` to cells and reconstruct a defensible lower-bound legacy
   trial inventory. Never delete failures.
3. Remove H3's future exit-observability eligibility rule. Freeze retention and
   missing-exit accounting for every entry.
4. Freeze the exact acceptance year, coverage requirements, drawdown ceiling,
   minimum independent sample, and zero-trade behavior.

### Gate B — before historical development results are considered

5. Wire `integrity.py`, `execution_costs.py`, and `risk_limits.py` into one
   end-to-end event ledger; do not maintain parallel legacy paths.
6. Enforce timezone-aware availability, bounded backward feature joins,
   strictly later fills, same-contract fixed strikes, causal exits, split
   purging/embargo, and artifact temporal provenance with integration tests.
7. Retain all eligible entries. Report observed fixed-strike exits separately;
   apply the predeclared adverse rule to every missing exit.
8. Make risk and cost evidence fail closed. Couple margin carry, settlement,
   forced liquidation, and both cost scenarios so callers cannot omit them.
9. Add tick-aware, premium/DTE/moneyness/time-of-day liquidity floors and
   multi-leg atomic-fill rules. Label all uncalibrated outputs as proxy-only.
10. Reconcile statutory arithmetic to dated broker contract notes and obtain
    contemporaneous contract masters and historical SPAN where possible.

### Gate C — before paper promotion

11. Complete the predefined dependence-, multiplicity-, regime-, tail-, and
    concentration analyses without changing the policy in response.
12. Require margin feasibility and defined loss for every paper intent. The
    current puncher must reject stale/asynchronous predictions and must not emit
    `PAPER_ORDER_READY` from an LTP-only chain.
13. Run a frozen forward shadow period with captured executable quotes, order
    acknowledgements, partial/rejected fills, margin snapshots, and broker
    contract notes. Calibrate execution only on that evidence; a recalibration
    creates a new policy version and restarts the clean evaluation clock.

### Gate D — final acceptance

14. Acquire roughly twelve months of new data into a quarantined path hidden
    from researchers. Freeze code, parameters, cost/risk policy, artifacts,
    hashes, and the decision manifest before opening it once.
15. Apply the INR 100,000 and controlled-drawdown rules exactly once with one
    historical lot, no post-open changes, no favorable-window choice, and no
    scaling. A data failure yields inconclusive evidence, not an improvised
    replacement window.

## Final verdict

**No advance.** The most favorable defensible interpretation is that the
repository contains hypotheses and engineering components worth testing in a
future shadow program. It does not contain clean evidence that a BANKNIFTY
volatility strategy is profitable after executable costs, feasible under the
fixed margin budget, robust to tails and regime changes, or capable of passing
the final INR 100,000 acceptance test. The leakage veto, exposed outcomes,
unvalidated fills/margin, understated trial count, and absence of a quarantined
year are each independently sufficient to block promotion.

