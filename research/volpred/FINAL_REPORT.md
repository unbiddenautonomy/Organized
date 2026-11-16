# BANKNIFTY volatility research: final integration report

**Integrated:** 2026-10-13  
**Decision:** **NO ADVANCE — BLOCKED**  
**Acceptance-test verdict:** **INCONCLUSIVE**

## Executive conclusion

The repository does not establish that a BANKNIFTY volatility policy can earn
INR 100,000 net on INR 1,000,000 of capital over a genuinely untouched year
with controlled drawdown. It also does not establish that the candidate is
tradable at the stored historical prices. This is not a negative finding about
the economic hypothesis; the required test cannot yet be performed honestly.

The Data Auditor issued **BLOCK**, the Leakage Auditor issued **BLOCK**, and the
Red Team issued **NO ADVANCE**. Each verdict is independently sufficient to
prevent strategy promotion. No strategy P&L was run for this integration, no
candidate outcome was opened, and no existing period was relabelled as a clean
holdout. Live broker submission remains outside scope and prohibited.

Permitted next use is limited to outcome-blind engineering and a fail-closed
paper/shadow program. Any historical result currently in the repository is
retrospective development evidence, not an acceptance result or proof of
tradability.

## Gate summary

| Gate | Status | Integrated finding |
|---|---|---|
| Data foundation | **BLOCK** | Time semantics, bad spot sessions, gaps, IV failures, historical contracts, executable quotes, SPAN and fixed-strike exits are unresolved. |
| Leakage / causality | **BLOCK** | Active paths use same-snapshot signals and fills, omit `tick_time` availability, do not enforce a common causal clock, drop future-missing exits, and lack invariant purge/embargo and artifact cutoffs. |
| Hypothesis registration | Registered, not executable | H0-H4 and grids are recorded with `outcome_opened=false`, but several definitions and promotion terms require further immutable specification. |
| Execution and margin | Proxy only | Fee arithmetic and conservative proxy components exist; they are not calibrated to historical executable quotes, fills, contract notes or SPAN. |
| Risk controls | Paper specification only | Limits are documented and unit primitives now fail closed on omitted evidence, but they are not wired into the active signal-to-exit path. |
| Red-team promotion | **NO ADVANCE** | Multiple testing, regime dependence, unbounded short-vol tails, non-executable prices and missing provenance remain capable of explaining apparent performance. |
| Final holdout | Unavailable | Existing option history ends 2025-05-22 and 2024/2025 outcomes have already been inspected repeatedly. |

## Evidence that is available

The fully scanned canonical monthly extract contains 1,286,387 rows from
2019-10-11 through 2025-05-22, with 210,761 timestamps, 68 monthly expiries and
369 strikes. The spot file contains 533,720 rows through 2025-07-31. Key
uniqueness checks pass, and all distinct option timestamps have an exact spot
timestamp. That establishes mechanical join feasibility, not causal
availability or executable pricing.

The same audit found 5,276 nonpositive spot observations across 23 dates, 60
dates with internal gaps longer than 61 seconds, special/weekend sessions,
out-of-session rows, thousands of nonconverged IV values, and narrow moving
strike coverage. The source is monthly-only. It has LTP-like prices, IV, Greeks
and OI, but no bid/ask, depth, trade condition, fill identifier or historical
SPAN. The only contract master is a 2026 snapshot and cannot prove the earlier
lot size, historical listings or expiry transitions.

The repository contains extensive earlier research across overlapping rules,
thresholds, structures and periods, including programs and artifacts explicitly
calling partial 2025 a holdout. That work supplies hypothesis context, but it
also creates a material and incompletely reconstructed multiple-testing burden.
Its performance outcomes are already exposed and cannot be reused as final
evidence.

New strategy-agnostic primitives cover timezone localization, availability
time, session/spot-quality flags, next-snapshot matching, execution-cost and
margin proxies, and risk limits. On integration, the focused control suite
passed:

```text
26 passed in 3.45s
```

This is evidence that isolated helpers behave as tested. It is not evidence
that an active strategy path is causal, profitable or executable: the Red Team
and Leakage Auditor found that these helpers are not yet wired into one
end-to-end event ledger.

## Results deliberately unavailable

No integrated performance table or strategy chart is presented. In particular,
the following required results are unavailable rather than zero:

- final-holdout net P&L and the pass/fail comparison with INR 100,000;
- gross-to-net cost waterfall under ordinary and 2x all-in costs;
- observed-fixed-strike result and worst-case missing-exit ledger;
- equity curve, drawdown curve and underwater-duration plot;
- maximum drawdown, worst day/week/expiry cycle and 95% CVaR;
- trade count, independent day/expiry-cycle count and turnover;
- confidence interval, day/cycle-block bootstrap, Sharpe, deflated Sharpe and
  probability of backtest overfitting;
- performance and concentration by year, expiry regime, DTE, time of day,
  event day and contract-rule transition;
- margin utilization, rejected/partial-fill rates, liquidation stress and
  capacity; and
- the untouched twelve-month holdout ledger, manifest and one-time verdict.

Producing these now would violate the gate sequence: historical outcomes must
not be opened until the causal ledger and preregistration are complete, and the
final metrics cannot exist until new quarantined data have accumulated.

## Frozen capital and risk context

The research capital base is INR 1,000,000. Unused cash earns nothing in the
primary test. Primary size is at most one contemporaneous exchange lot, with
one open position and one new position per session; scaling to reach the INR
100,000 threshold is forbidden. Historical size is assumed to be 15 through
the December 2024 transition and 30 thereafter, but that assumption remains
unverified until contemporaneous contract masters are obtained.

Buffered margin may not exceed INR 350,000: verified raw broker/SPAN plus
exposure margin receives a 20% buffer, implying a maximum raw requirement of
INR 291,666.67. The severe pre-trade loss cap is INR 7,500. New entries stop at
a net daily loss of INR 10,000 or ISO-week loss of INR 25,000. A peak-to-trough
drawdown of INR 50,000 causes indefinite suspension. Event days are no-entry
days under the frozen risk policy. There is no averaging down, pyramiding,
margin reuse or discretionary override.

Research costs must include dated brokerage, exchange fees, STT, GST, SEBI and
IPFT charges, stamp duty, bid/ask-crossing and impact proxies, rejection and
partial-fill assumptions, margin carry, settlement treatment and a 2x all-in
stress on identical frozen trades. These remain scenarios until reconciled to
dated broker contract notes and executable market data. An unhedged short
straddle or strangle has unbounded loss and therefore cannot pass the live-risk
gate even when a finite scenario stress is small.

## Blocking conditions

1. There is no genuinely untouched twelve-month option period.
2. Active strategy and paper-intent code do not separate signal, order and
   strictly later fill events.
3. `tick_time`, timezone-aware availability and bounded backward joins on a
   common decision clock are not enforced end to end.
4. Entries can disappear based on future exit availability; observed-only and
   worst-case missing-exit accounting are incomplete.
5. Purging/embargo by `label_end_at` and artifact temporal provenance are not
   enforced invariants.
6. Historical bid/ask/depth/fills, SPAN, margin responses and contract notes are
   absent, so execution and margin cannot be validated.
7. Historical contract masters and lot-size/listing evidence are absent; the
   recorded expiry weekdays also conflict with the prose chronology.
8. Nonpositive spot sessions, gaps, special sessions, IV failures and narrow
   moving-strike coverage require frozen fail-closed treatment.
9. The new integrity, cost and risk helpers are not integrated into a single
   production-like path; the legacy paper path can still allow unverified evidence.
10. The trial universe and legacy multiple-testing count are incomplete, while
    several hypothesis terms and final acceptance details remain ambiguous.

## Paper-trading integration plan

### 1. Freeze the specification before outcomes

Resolve units and formulas for every H1-H4 score, all lookbacks and minimum
history, plateau and economic-destruction rules, event calendar, coverage
threshold, minimum independent sample, drawdown acceptance ceiling and
zero-trade behavior. Expand `TRIALS.csv` to every immutable cell and append a
lower-bound inventory of related legacy trials. Never delete failed trials.

### 2. Build one fail-closed event ledger

Replace parallel paths with a single sequence:

```text
raw event -> integrity normalization -> causal feature snapshot -> prediction
-> decision -> risk/margin gate -> paper order -> later simulated fill
-> marked position -> causal exit/forced-close event -> reconciled ledger
```

Each row must carry source and artifact hashes, Asia/Kolkata `available_at`,
decision/order/fill times, exact contract identity, training cutoff, maximum
label end, trial/policy version, costs, margin response, risk decision and
reason codes. Features use a bounded backward as-of join; fills must be from the
first eligible snapshot strictly after the decision. Every accepted entry stays
in the ledger through an observed exit or the frozen adverse missing-exit path.

### 3. Make paper mode evidence-producing, not permissive

The current puncher must reject stale/asynchronous predictions and LTP-only
chains rather than emit `PAPER_ORDER_READY`. Require exact-contract executable
bid/ask/depth capture, atomic multi-leg simulation, broker margin responses,
order acknowledgements, partial/rejected fill records, restart-safe risk state
and forced-close handling. Risk and cost inputs must fail closed. Do not add a
broker order-submission path.

Run a frozen shadow period and reconcile simulated orders against captured
quotes, margin snapshots and broker contract notes. Execution recalibration is
allowed only as a new version; it restarts the clean evaluation clock.

### 4. Quarantine the final year

Acquire approximately twelve months of new data outside all development paths.
Hide outcomes from researchers. Before opening it, freeze and hash the code,
artifacts, features, trial registry, execution model, risk policy, contract and
margin evidence, exact dates, coverage rule and reporting template.

Open the holdout once. Apply one historical lot, the same frozen trades under
ordinary and 2x costs, and the controlled-drawdown criterion without parameter
changes or favorable-window selection. A material data failure yields
**inconclusive**, not a replacement window. Only this evaluation can accept or
reject the INR 100,000 criterion.

## Reproducibility

From the repository root in PowerShell, reproduce the data audit and complete
test suite with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vol_research_gate.ps1
```

For code-only iteration after an unchanged full data audit:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_vol_research_gate.ps1 -SkipFullDataAudit
```

The focused controls validated during this integration were:

```powershell
py -3.13 -m pytest -q tests/test_integrity.py tests/test_execution_costs.py tests/test_risk_limits.py tests/test_leakage_audit.py
```

These commands reproduce integrity checks, not strategy profitability. Passing
them does not lift any audit veto.

## Final disposition

The correct present statement is: **the economic hypotheses remain unproven,
the INR 100,000 acceptance result is unavailable, and the project may advance
only to remediation and controlled paper/shadow data collection.** Historical
profitability, if any, must not be used to override this decision.
