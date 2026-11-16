# Research V2 risk and event review

## Current verdict: PASS for the frozen development and validation inventory only

The original veto below is retained as an audit trail. It was lifted on
2026-10-13 only for development and validation execution authenticated by
`STRATEGY_PNL_AUTHORIZATION_V2.json` and its checksum sidecar. Strategy
pseudo-holdout P&L remains **BLOCKED**. Any changed inventory member, missing
sidecar, different authorization path, or added segment restores the veto
automatically.

`STRATEGY_SPEC_V2.md` is directionally consistent with `RISK_POLICY.md` on one
lot, one position and entry per day, defined-risk structures, INR 7,500 severe
loss, INR 350,000 buffered margin, event-day exclusion, missing-exit stress,
and daily/weekly/drawdown stops. It is not yet sufficiently consistent for
strategy P&L execution. The following items must be resolved in code and frozen
tests without inspecting strategy outcomes.

## Blocking inconsistencies

1. **Margin basis differs.** The strategy specification tests defined maximum
   loss or debit plus 20% against INR 350,000. The risk policy requires a
   verified exact-contract broker/SPAN-plus-exposure response and tests
   `1.20 * raw_margin <= 350000`. Contractual maximum loss is not a substitute
   for portfolio margin, especially for an iron fly. Research may report a
   clearly labeled unverified historical margin proxy, but it cannot record a
   verified margin pass. The runner must keep severe-loss and margin gates as
   separate fail-closed checks.
2. **Exit clocks conflict.** The strategy fixes exit at the first complete
   snapshot at or after decision plus 120 minutes, producing approximately
   12:15, 13:45, or 15:25 exits. The risk policy says the ordinary exit process
   begins at 15:15. For this intraday fixed-horizon research, the 120-minute
   causal exit must govern; 15:15/15:25 must be defined only as end-of-day
   escalation for a still-open or failed-close incident. Until that precedence
   is explicit and tested, the runner can change holding time by signal clock.
3. **Historical lot size is unverified.** The assumed transition from 15 to 30
   after 2024-12-24 is explicitly unverified, while the risk policy rejects an
   entry whose contemporaneous lot cannot be verified. Primary research may
   run only if every result is labeled `unverified_lot_assumption`, the
   constant-15 sensitivity uses identical selections, and no row is presented
   as a verified risk pass or live-eligible trade.
4. **Remaining loss-budget gate must be explicit.** Before entry, the engine
   must reject when current daily or ISO-week net liquidation loss plus the
   proposed position's severe stressed loss would breach INR 10,000 or INR
   25,000, respectively. It must also reject at or beyond the INR 50,000
   drawdown suspension. Post-loss stops alone are insufficient for a new trade.
5. **Unscheduled halt causality must be preserved.** The official calendar
   includes the February 24, 2021 halt, but an all-day retrospective exclusion
   would be lookahead. Without an official, auditable intraday availability
   timestamp, pre-halt decisions must remain in the ledger and the session must
   be treated as an operational-incident stress. Scheduled events remain
   full-day primary exclusions.

## Required fail-closed controls

- Hash-check `STRATEGY_SPEC_V2.md`, `EVENT_CALENDAR_V2.csv`, cost tables,
  predictive ledgers, and the strategy runner before any option-row access.
- Join events on Asia/Kolkata session date only, reduce duplicate event rows to
  one boolean, and retain event type/source metadata without duplicating trades.
- Select the entry using decision-time information only. Exit presence, exit
  price, later OI, later IV, or later contract completeness must never affect
  entry eligibility or structure selection.
- Compute iron-fly contractual loss as strike width times the dated assumed lot,
  less net entry credit, plus conservative close costs; compute long-straddle
  loss as debit plus costs. Apply the joint severe scenario and take the worse
  result. Reject if it exceeds INR 7,500.
- Reject nonfinite or nonpositive IV, incomplete structures, stale/asynchronous
  fields, unknown/negative OI, fewer than 20 trailing eligible sessions, any
  atomic fill failure, unbounded loss, invalid cost input, or unavailable
  margin input. Do not silently drop the row.
- Preserve selected entries with missing exits and charge the frozen adverse
  exit. Never condition entry on a future fixed-strike exit.
- Maintain daily, ISO-week, and high-water-mark state chronologically; record
  every `NO_TRADE`, rejection reason, event flag, breach, and suspension.
- Keep ordinary, 2x-cost, and uncertainty cases on identical trade selections.
  Costs and adverse exits may change P&L, but cannot change which trades were
  selected.

## Tail-risk and promotion boundary

The CVaR, worst-day/week, drawdown, concentration, plateau, and PBO reports are
descriptive research gates only. They cannot authorize live capital or repair
unverified margin, lot, execution, settlement, or event-timestamp evidence.
Strategy pseudo-holdout access and any PnL execution remain blocked until the
five inconsistencies above are remediated, the runner passes leakage/spec tests,
and the Risk/Event Agent explicitly lifts this veto.

No strategy P&L, pseudo option outcome, or event-day counterfactual result was
read or computed for this review.

## Strategy runner static review

Reviewed `research/research_v2_strategy.py` and its synthetic tests without
calling `main()`, reading the real option source, or computing any strategy
outcome. The synthetic suite passed eight tests. The code has several sound
controls: strict next-snapshot entry, fixed recorded strikes and expiry,
atomic-leg fill rejection, retention of selected entries with missing exits,
an explicit future-exit deletion negative control, one qualifying signal per
day, immutable trade selections across cost cases, and no executable pseudo
strategy path.

Execution remains **VETOED** for these additional reasons:

1. `load_event_calendar()` expects `event_date,source,frozen_at`, which is not
   the required and now-frozen schema
   `date,event_type,event_name,source_url,announced_at,primary_no_entry`.
   The authorized calendar therefore cannot be loaded.
2. The runner reads the prediction ledger and the real option file before it
   appends any of the six segment trials. Trial registration must precede the
   first option outcome access, not merely metric computation. An arbitrary
   existing `--leakage-authorization` path is accepted without an approved
   hash, content, scope, or segment check.
3. The option source, event calendar, runner, execution-cost code, prediction
   manifest, and authorization are not verified together against one frozen
   pre-PnL inventory. Development accepts a self-reported manifest hash without
   authenticating the manifest itself.
4. `_fixed_exit()` starts the 120-minute horizon at `entry_time`; the frozen
   specification starts it at `decision_time`. Entry latency must not extend
   the target horizon.
5. `margin_proxy()` explicitly identifies itself as unvalidated historical
   proxy margin, yet `margin.permitted` is treated as a passed buffered-margin
   gate. It may be reported only as `unverified_margin_proxy`; it cannot satisfy
   the verified RISK_POLICY margin requirement.
6. The runner silently drops decisions when no option snapshot, eligible chain,
   valid ATM, or threshold signal exists. The risk policy requires every
   decision and every `NO_TRADE`/rejection to remain in the audit ledger.
7. Weekly entry control tests only whether realized week P&L is already at
   -INR 25,000. It does not reject a proposed INR 7,500 severe loss that would
   consume the remaining weekly budget. The same fail-closed remaining-budget
   rule must be explicit for daily loss and drawdown.
8. The calendar is reduced to a date set, so the unscheduled February 24, 2021
   halt becomes a hindsight-based full-day exclusion. The loader must retain
   event types and apply unscheduled incidents only from a verified observable
   timestamp; with `announced_at=unknown`, it must retain the day and label it
   as an incident stress rather than exclude it.
9. Historical-lot outputs are not labeled as unverified assumptions, and the
   runner has no explicit block against treating proxy-margin/assumed-lot rows
   as verified risk passes or promotion evidence.

No development or validation strategy execution is authorized until these
items and the earlier five policy inconsistencies are remediated and re-reviewed.

## Final remediation review and authorization

**Decision: PASS_HASH_SCOPED_DEVELOPMENT_AND_VALIDATION_ONLY.** Static review
and synthetic testing confirm that every blocker above is resolved in the
following frozen inventory:

- strategy runner: `8c491e2048c85ed4d961f60879ed95e14bec7184c186197d7fee23ab2602d75f`
- strategy tests: `b1e3409a6bc12ae2e01574f645327353d50cca1f05254118fd57b7a057af59c0`
- strategy specification: `83b1d87e4fbbd8a6ccfec16dd9783a0b876fa881ed96606d5f12ac6979d6ae42`
- risk addendum: `abf5349f04e239749ff87336a57eeb05b7de79b98d41a0c10afff493e0b75418`
- event calendar: `35d705b8a16ef3c8f94b4acb2cceae1832a84d4842f586ce47aca4832605fa25`
- execution costs: `e16041e798fcf8153c21192604efab749bf525cc0a79337c7ca6210551897a7e`
- signed authorization JSON: `580f6b94fbe6f071ca86c34aa9cc9918d2fc470aa7ec49769ec3428e5e4bf3a0`

The fixed-name authorization binds those files plus the exact development and
validation predictive artifacts, their provenance files, and the option-source
hash. The runner's authorization verifier passed for both permitted segments
without opening tabular data. Forty-two focused synthetic cost, strategy, and
evaluation tests passed, and the reviewed modules compiled successfully.

The remediation provides: exact event-schema validation and causal incident
treatment; trial preregistration before semantic reads; one unified hash
inventory; decision-plus-120-minute exits; explicitly unverified proxy margin
and lot outputs that cannot support live eligibility or promotion; complete
decision-ledger retention; daily, weekly, and remaining-drawdown entry budgets;
chronological entry scans using only the selected structure's required legs;
fail-closed asynchronous snapshots; immutable terminal completion records; and
a separately claimed pseudo path with its canonical exposed label.

This authorization does not approve pseudo-holdout strategy P&L, live trading,
capital deployment, verified margin, verified historical lots, or promotion.
No real option row, strategy P&L result, or pseudo outcome was opened or
computed during this review.
