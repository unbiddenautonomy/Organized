# Research V2 risk-policy addendum

Status: **FROZEN BEFORE ANY RESEARCH V2 OPTION P&L** on 2026-10-13.

This addendum resolves two ambiguities between `RISK_POLICY.md` and the more
specific, already-frozen `STRATEGY_SPEC_V2.md`. It does not relax a loss limit,
authorize live trading, validate historical margin, or change any predictive
model, signal, threshold, sizing rule, cost rule, or event exclusion.

## Exit-clock precedence

For the historical Research V2 study only, the ordinary scheduled exit is the
first complete eligible fixed-expiry, fixed-strike snapshot at or after the
signal decision timestamp plus 120 trading minutes, capped at 15:30
Asia/Kolkata. Entry latency does not move this target. This rule applies at all
three frozen decision clocks: 10:15, 11:45, and 13:25.

The 15:15/15:25 language in `RISK_POLICY.md` is a prospective end-of-day
escalation rule for a future live or paper operational program when a scheduled
close remains unfilled. It is not the ordinary exit clock for this historical
120-minute forecasting study. Missing fixed-strike exits in Research V2 retain
the adverse contractual treatment in `STRATEGY_SPEC_V2.md`; they are never
dropped or treated as a free close.

## Historical margin status

The historical margin calculation is an `unverified_margin_proxy` because
dated SPAN/exposure files and broker responses are absent. Research V2 must
retain its raw and 20%-buffered proxy values and may conservatively reject a
trade under that proxy. It must record verified margin status as false or
unknown. A proxy result cannot be described as a verified margin pass, cannot
satisfy live eligibility, and cannot support a tradability or profitability
claim.

All other controls and disclosures in `RISK_POLICY.md` and
`STRATEGY_SPEC_V2.md` remain unchanged. If the two documents conflict on the
two subjects above, this addendum controls for Research V2 only.
