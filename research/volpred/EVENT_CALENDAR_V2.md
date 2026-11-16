# Research V2 official event calendar

## Status and scope

This calendar is frozen before Research V2 strategy P&L. It covers
`2019-10-11` through `2025-05-22` and contains only events supported by official
issuer, regulator, election-authority, or exchange sources. Every listed date is
a primary no-entry date under `STRATEGY_SPEC_V2.md`; event-day trades may appear
only in a separately labeled counterfactual diagnostic.

The machine-readable authority is `EVENT_CALENDAR_V2.csv`. Its fields are:

- `date`: official event/session date in ISO format;
- `event_type`: stable event category;
- `event_name`: concise official-event description;
- `source_url`: official RBI, India Budget, ECI, or NSE/NSE Clearing source;
- `announced_at`: exact public announcement timestamp, or `unknown`; and
- `primary_no_entry`: always `true` for this frozen primary calendar.

## Construction rules

1. Union Budget rows use official Ministry of Finance budget speeches and
   include both the February 2024 interim presentation and July 2024 full
   presentation.
2. RBI rows use the policy-resolution archive or an official resolution/release.
   Scheduled decisions are included, along with the clearly labeled March 2020
   extraordinary, May 2020 advanced, and May 2022 off-cycle decisions because
   they are material policy-announcement days.
3. The national-election row is the official ECI Lok Sabha 2024 counting date.
   State elections are out of scope because the frozen policy names national
   election counting/results.
4. NSE rows include special live sessions, the January 2024 exceptional closure,
   and the February 2021 exchange-system halt. Routine annual public holidays
   are not duplicated: the source has no eligible ordinary trading session on
   those dates. The exceptional closure is retained because it changed the
   previously published trading calendar.
5. `announced_at` is `unknown` throughout. The official documents establish the
   event date, and some establish a publication date, but they do not provide a
   consistently auditable exact public timestamp. No timestamp is inferred.
6. Duplicate dates are intentional where two independent event types coincide,
   such as Union Budget presentation and an NSE Saturday special session. A
   strategy join must reduce the calendar to a boolean per date and must not
   duplicate a signal or trade ledger row.

## Causality limitation for unscheduled incidents

The February 24, 2021 exchange halt is included as a known major halt relevant
to the source, but it was not an ex-ante scheduled event. Its all-day
`primary_no_entry=true` flag must **not** be used to erase or reject a decision
made before the halt became observable. A causal engine may apply the halt only
from its contemporaneously observed activation onward, then flatten/suspend
under `RISK_POLICY.md`. If the runner has no auditable intraday announcement or
halt timestamp, the entire session must remain in the primary ledger and be
reported as an operational-incident stress; retrospective all-day exclusion is
forbidden.

The same principle applies to any future unscheduled halt. Scheduled Budget,
MPC, election-counting, special-session, and predeclared closure dates are
full-session no-entry dates under the frozen research policy.

## Official source families

- Reserve Bank of India monetary-policy archive and releases:
  <https://www.rbi.org.in/Scripts/Annualpolicy.aspx>
- Ministry of Finance Union Budget speeches:
  <https://www.indiabudget.gov.in/bspeech.php>
- Election Commission of India 2024 General Election schedule:
  <https://elections24.eci.gov.in/docs/press-note-no-23.pdf>
- National Stock Exchange circular archive and holidays page:
  <https://www.nseindia.com/resources/exchange-communication-holidays/>

No press article, commercial calendar, broker page, or retrospectively inferred
date was used as calendar authority.

## Integrity

The SHA-256 of `EVENT_CALENDAR_V2.csv` is recorded in
`EVENT_CALENDAR_V2.sha256` as
`35d705b8a16ef3c8f94b4acb2cceae1832a84d4842f586ce47aca4832605fa25`.
Any edit invalidates this review and requires a new hash plus a fresh
risk/leakage review before strategy P&L.
