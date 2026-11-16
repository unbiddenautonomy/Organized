# BANKNIFTY volatility-strategy research scope

## Decision and integrity standard

The research question is whether a volatility-based BANKNIFTY options policy can
produce at least INR 100,000 of net profit over a genuinely untouched period of
about twelve months, after realistic costs, on INR 1,000,000 of capital, with
controlled drawdown.  INR 100,000 is an acceptance test, not a fitting target.
No parameter may be changed after the holdout is opened.

The current repository cannot, by itself, supply that final evidence.  Its
monthly option history ends on 2025-05-22, and existing reports and programs
have already inspected outcomes through that date.  Therefore no interval in
the existing option file can honestly be called a fully untouched final
holdout.  Development and walk-forward validation may use the existing history,
but the acceptance verdict must remain **inconclusive** until roughly twelve
months of newly acquired, quarantined data are evaluated exactly once.  A
diagnostic pseudo-holdout, if reported, must be labelled as previously exposed
and cannot satisfy the acceptance criterion.

## Repository and data inventory

- `data/raw/icharts_banknifty_spot_1min.csv`: one-minute BANKNIFTY spot,
  approximately 2019-10-11 through 2025-07-31.
- `data/raw/icharts_option_banknifty.csv`: 16.2 GB option history.
- Monthly extracts: ATM, ATM five strikes, and up-to-five-strike panels.  The
  broad five-strike monthly panel covers 2019-10-11 through 2025-05-22.
- The chain extracts contain prices/LTP-like values, IV, Greeks and OI, but no
  historical bid, ask, displayed size, trade condition, or fill identifier.
- `data/raw/NSE_FO_contract_06102026.csv.gz`: a single 2026 contract master.  It
  is useful for current metadata, not for reconstructing historical listings.
- Realized-variance research and models include HAR-style Ridge forecasts at
  15, 30, 60 and 120 minutes, plus enhanced RV, IV-change and spot-magnitude
  variants.  The canonical feature family uses trailing 5/15/60-minute RV.
- `app/live_inference.py` is a spot-model paper inference worker.  The options
  paper-intent path is `app/trade_puncher.py`; it does not submit broker orders.
- The active documented options policy is under
  `strategies/dte4_20_conditional_options/`.  Prior research has examined many
  overlapping variants, so its results count toward the multiple-testing burden.

## Fixed operating assumptions

- Capital: INR 1,000,000 cash-equivalent research capital.  This amount is
  fixed before testing.  Unused cash earns nothing in the primary result.
- Margin: at most 35% of capital may be blocked at entry, including a 20%
  buffer over the modeled SPAN plus exposure requirement.  If defensible
  historical SPAN cannot be reconstructed, a conservative broker-style proxy
  must be used and the result remains model-dependent.  No trade is permitted
  if the buffered requirement exceeds the limit.
- Size: at most one historical lot per position in the primary test.  Historical
  lot size is 15 through the December 2024 monthly transition and 30 thereafter,
  subject to validation against contemporaneous contract masters.  Scaling up
  to reach the acceptance amount is forbidden.  Any capacity analysis is
  reported separately.
- Concurrency: one open strategy position at a time.  No averaging down and no
  implicit reuse of blocked margin.
- Session: Asia/Kolkata, 09:15-15:30 on actual NSE trading days.  Orders are
  generated from information timestamped at or before the signal time and are
  filled only from the next available eligible quote/snapshot.
- Frequency: one-minute spot and source option snapshots at their recorded
  timestamps.  Timestamps are not rounded forward.  The common decision clock
  is the latest fully observed timestamp across spot, chain, IV and RV inputs.
- Instruments: only contracts actually present in the historical source at the
  decision time.  The main source is monthly BANKNIFTY options; weekly-contract
  claims cannot be tested from a monthly-only extract.
- Expiry history: BANKNIFTY weekly options ended after 2024-11-13.  Monthly
  expiries in this sample were generally last Thursday, changed to last Monday
  for contracts revised in April 2025, and later moved to Tuesday outside most
  of the option sample.  The recorded contract expiry field, not a hard-coded
  weekday, is authoritative.
- Exits: same listed expiry and same absolute strikes selected at entry.  An
  absent future fixed-strike observation is not silently dropped.  Model-priced
  exits are separately flagged and excluded in an observed-only sensitivity.
- Costs: brokerage, exchange transaction charges, STT at the date-applicable
  rate, GST, SEBI/IPFT fees, stamp duty, bid-ask crossing proxy, volatility- and
  size-sensitive slippage, partial-fill/rejection assumptions, and margin carry.
  The primary and 2x-cost cases are both mandatory.
- Exercise: positions are intended to close intraday before settlement.  If an
  exit cannot be established and exercise/assignment is possible, the relevant
  settlement STT and conservative gap loss are charged rather than assuming a
  free close.

## Baselines and exclusions

The baseline specified for this effort is the supplied current live policy:
sell the ATM straddle when current IV exceeds predicted RV by a cost-covering
margin; otherwise use a two-strike OTM short strangle at probability at least
0.693128, then an ATM short straddle at probability at least 0.701625.  It has
no long-volatility leg.  The repository's documented `dte4_20` policy differs,
so both identities must be kept distinct in reports and code.

The previously rejected IV/RV convergence rule is excluded as a candidate in
its existing form.  It may be used only as a failure analysis or to motivate a
materially different, pre-registered hypothesis.

## Validation protocol

1. The data and leakage auditors must approve the timestamp, contract and cost
   foundations before strategy results are considered.
2. Hypotheses and expected signs are written to `HYPOTHESES.md` before their
   outcomes are computed.  Every evaluated variant is appended to `TRIALS.csv`.
3. Development uses expanding or rolling walk-forward train/validation splits,
   with purging and embargo at least as long as the maximum label horizon.
4. Features at time `t` use only records with source timestamp `<= t`; fills use
   the next available quote.  Automated future-shift and alignment tests are
   required and the leakage auditor has veto authority.
5. Promotion requires a broad parameter plateau, not a point optimum, and must
   survive ordinary and 2x costs, tail/event-day review, and red-team review.
6. Trial count includes prior materially related repository variants where it
   can be reconstructed.  Deflated Sharpe and/or probability of backtest
   overfitting must use that count and disclose uncertainty in the count.
7. A new final holdout is stored outside development paths, its outcomes remain
   hidden until code and parameters are frozen, and it is opened once.  If such
   data are not available, there is no accepted/rejected claim against the INR
   100,000 threshold—only an inconclusive verdict.

## Deliverable boundary

This effort may build a reproducible research framework, audit the existing
history, compare pre-registered hypotheses on walk-forward validation data, and
write a paper-trading integration plan.  It must not enable live broker order
submission.  Because historical executable quotes, complete contract masters,
historical SPAN files, and an untouched year are absent, any profitability
result from the present repository is evidence for further paper trading, not
proof of tradability.
