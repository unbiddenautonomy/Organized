# Data quarantine plan V2

**Status:** future evidence plan only. The completed Research V2 candidate is
**REJECTED, never accepted**; quarantine does not reverse that verdict.

## Purpose and dates

Create genuinely new forward evidence beginning **2026-10-14**, after the
2026-10-13 research freeze. Keep it unreadable to researchers through at least
**2027-10-13** and longer until 500 eligible sessions and 100 filled paper
trades have accumulated. Existing development, validation, and
**pseudo-holdout, previously exposed** data never enter this quarantine.

## Write-only custody

1. A dedicated ingestion identity may append new immutable partitions only.
2. Researchers and model/strategy identities have no list, read, export,
   preview, SQL, object-version, or decryption permission.
3. Encryption keys belong to an independent custodian; audit logs are exported
   to a separate append-only account.
4. Each UTC/India-session partition is content-hashed and entered into a daily
   signed Merkle manifest. Corrections append a superseding record; no object is
   overwritten or deleted.
5. Operational monitoring exposes only feed uptime, record counts, byte counts,
   hash continuity, and schema errors. It must not expose values, charts,
   signals, trades, P&L, or outcome-dependent alerts.

## Required sources

- Timestamped BANKNIFTY spot and complete option chains with bid, ask, depth,
  volume, OI, exchange sequence, receipt time, and source availability time.
- Contemporaneous expiry/strike contract masters, corporate/exchange notices,
  and effective lot sizes.
- Broker order acknowledgements, fills, rejections, cancellations, fees,
  exercise/assignment and settlement records.
- Pretrade broker/SPAN plus exposure margin responses tied to exact contracts,
  quantities, timestamps, and validity windows.
- Frozen official-event calendar snapshots and announcement timestamps.
- Every decision/no-trade, forecast hash, policy hash, risk state, immutable
  strikes/expiry, fill attempts, close/forced-close treatment, and reconciliation.

## Quality and release gates

Ingestion rejects timestamp inversion and records—not repairs—gaps, duplicates,
crossed markets, stale/asynchronous legs, missing masters, and schema drift.
No backfill may masquerade as point-in-time data. A daily manifest lists all
exclusions without revealing outcomes.

Release requires all of the following:

- at least 12 new calendar months after 2026-10-13;
- at least 500 eligible sessions and 100 filled paper trades under one unchanged
  frozen policy;
- complete hash chain and independent access-log review;
- zero research reads before the seal;
- verified bid/ask/depth, contract/lot, broker/SPAN and reconciliation coverage;
- a frozen evaluation package and a separate one-shot authorization.

Failure of any item extends quarantine or terminates the study. It does not
permit partial outcome inspection. The current strategy pseudo-holdout remains
unopened and is not a substitute for this plan.
