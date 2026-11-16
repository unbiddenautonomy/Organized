# BANKNIFTY volatility research data audit

## Gate

**BLOCK** for a final acceptance or tradability claim. Outcome-blind,
model-dependent development may continue only after the cleaning and timestamp
rules below are frozen. The blockers are not waived by a profitable backtest.

The audit is reproducible with:

```powershell
py -3.13 tests/data_audit_scan.py
```

It writes `outputs/data_audit/audit_summary.json` and coverage inventories. The
audit reads market-data fields only; it does not create signals, labels, trades
or P&L.

## Source coverage

| Source | Rows | Period | Dates / timestamps | Key integrity |
|---|---:|---|---:|---|
| One-minute spot | 533,720 | 2019-10-11 to 2025-07-31 | 1,444 dates | no duplicate or null timestamps |
| Monthly up-to-five-strike options | 1,286,387 | 2019-10-11 to 2025-05-22 | 210,761 timestamps, 68 expiries | no duplicate `(datetime, expiry, strike)` keys |
| 2026-10-06 contract master | 2,365 BANKNIFTY rows | one current snapshot | current contracts only | all BANKNIFTY rows show lot size 30 |

The broader 16.2 GB option source is present, as are ATM and five-strike monthly
extracts. This audit treats the monthly up-to-five-strike file as the canonical
structure source because it is the panel used by the current research stack.

All 210,761 distinct option timestamps have an exact timestamp in the spot file,
so mechanical exact alignment is possible. Exact equality does not establish
economic availability; timestamp semantics still require correction below.

## Critical data defects and restrictions

### Information time and timezone

Timestamps are timezone-naive. Most option `datetime` values end in `:59`, and
the source `tick_time` is generally one second later. Another 4,135 option rows
are stamped at `:00`. Treating `datetime` as immediately tradable can introduce
a one-second look-ahead.

The frozen research rule must localize all records to Asia/Kolkata and define
`available_at = max(datetime, Date + tick_time)` when `tick_time` is valid.
Signals may use only rows with `available_at <= decision_time`; entry uses the
first eligible snapshot strictly after the decision time. A timestamp without
an unambiguous availability time is excluded.

### Spot defects and session anomalies

- 5,276 nonpositive spot observations occur across 23 dates, including some
  entire 375-bar sessions. They cannot be forward-filled into features or
  labels; affected windows and any dependent option observations are excluded.
- 60 dates have internal gaps over 61 seconds; the maximum gap is 15,600
  seconds. Windows crossing a gap larger than one expected bar are excluded.
- The sources contain eight weekend or special-session dates.
- 411 spot rows and 1,324 option rows fall outside 09:15-15:30.

The primary study uses recorded exchange sessions intersected with 09:15-15:30.
Weekend/special sessions form a separate diagnostic stratum and are not pooled
with ordinary days. No bar outside the permitted session enters a lag or label.

### Option quality and state-dependent coverage

- Prices are positive and OI is nonnegative in the canonical extract.
- Call IV did not converge on 6,799 rows and put IV on 2,684 rows; 110 rows have
  invalid or zero IV. Nonconverged/invalid IV cannot enter signals. The filter
  is based only on the current row and is logged.
- Of 273,352 `(timestamp, expiry)` snapshots, 233,126 have five distinct strikes
  and 40,226 have only one to four.
- Entry strikes can disappear after large moves because the extract follows a
  narrow moving strike window. Dropping those trades would bias short-gamma
  results upward. Observed fixed-strike exits are the primary evidence.
  Model-priced missing exits must be flagged, assessed separately and cannot
  support a tradability claim.

### Contract and expiry limitations

Every canonical row is marked monthly; no weekly row is present. A weekly
strategy or a comparison against weekly BANKNIFTY contracts is untestable.

Recorded monthly expiries fall on 54 Thursdays, 12 Wednesdays and 2 Tuesdays;
none falls on Monday. This conflicts with a prose assumption that April 2025
monthly contracts had Monday expiry. The recorded `expiry` field is
authoritative for research, but the discrepancy must be reconciled against
contemporaneous NSE contract masters before claiming exchange-rule fidelity.

The only contract master is dated 2026-10-06 and shows lot size 30. It cannot
prove historical listings or the prior lot size 15 / December 2024 transition.
Historical contract masters are required to validate lot size, listed strikes,
expiry and survivorship. Until then, historical sizing is an explicit external
assumption and a separate sensitivity must hold lot size constant.

## Executability, costs and margin

The chain contains price-like values, IV, Greeks and OI, but no bid, ask,
displayed size, trade condition or fill identifier. It also lacks historical
SPAN/exposure-margin snapshots. Consequently:

- historical spread and partial-fill models cannot be calibrated from this data;
- the stored price cannot be certified as an executable quote;
- intrawindow stop execution and size-dependent market impact cannot be validated;
- margin compliance and forced-liquidation paths are proxy-based; and
- modeled slippage, rejected fills, margin carry and a 2x-cost scenario are
  mandatory but do not turn the result into executable evidence.

Any trade that cannot be exited on an observed fixed-strike snapshot is treated
conservatively in risk analysis. It may not be silently discarded.

## Holdout integrity

There is **no fully untouched twelve-month option period** in the repository.
The option history ends on 2025-05-22, while existing programs and reports have
repeatedly inspected 2024 and partial-2025 outcomes, often explicitly calling
2025 a holdout. Those periods are now development data for this research.

Therefore the INR 100,000 acceptance test is necessarily **inconclusive** until
approximately twelve months of new data are quarantined, hidden from all
researchers, and opened once after code and parameters are frozen. A
retrospective pseudo-holdout may be useful for debugging but cannot be used for
acceptance, rejection, threshold selection or a live-capital claim.

## Critical blockers

1. No genuinely untouched year.
2. No historical executable bid/ask quotes or complete fill paths.
3. No historical SPAN/exposure-margin files.
4. No contemporaneous historical contract masters or independently verified
   lot-size history.
5. Timezone-naive timestamps and unresolved `datetime` versus `tick_time`
   availability semantics.
6. Nonpositive spot sessions, long gaps, special-session rows and IV failures
   need deterministic exclusion rules.
7. Monthly-only, narrow strike coverage creates endogenous missing exits.
8. Recorded expiry weekdays conflict with the assumed exchange-rule chronology.

## Gate conditions to resume strategy testing

Exploratory testing may proceed only after code implements and tests the frozen
availability-time, session, bad-spot, gap and IV-quality rules; missing-exit
treatment; and next-snapshot execution. Acceptance testing additionally
requires historical contract/margin evidence, executable quotes or a validated
fill source, and a quarantined new year. The independent Leakage Auditor may
veto any result even after these conditions are met.
