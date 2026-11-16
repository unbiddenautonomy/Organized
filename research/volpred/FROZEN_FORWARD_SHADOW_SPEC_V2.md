# Frozen forward shadow specification V2

## Status

**RESEARCH-ONLY OBSERVATION PLAN FOR A REJECTED POLICY.** This document does
not promote, accept, revive, or authorize live use of P2/S1/S2. The completed
development/validation result is **REJECTED**, never accepted. Strategy pseudo
P&L remains unopened.

## Immutable policy under observation

- Predictive model: frozen P2 artifact SHA-256
  `ab863e86bb064cac8624e605ad3ad6f32a4302fb3cd51037ef73e627e9314f61`.
- Primary economic threshold: `1.25`; `1.15` and `1.35` remain diagnostics and
  cannot replace it.
- Structures, clocks, expiry/strike rules, signal formulas, costs, missing-exit
  treatment, INR 1,000,000 reference capital, and risk limits are exactly those
  in `STRATEGY_SPEC_V2.md` (`83b1d87e...9d6ae42`) and `RISK_POLICY.md`
  (`3a992923...ce0282f`).
- No parameter, feature, clock, threshold, structure, cost, lot, margin, event,
  or rejection rule may be changed after quarantine begins. A change creates a
  new hypothesis and cannot inherit this evidence.

## Forward interval and sample requirement

Collection begins no earlier than **2026-10-14**, strictly after the
2026-10-13 freeze. The minimum sealed interval ends no earlier than
**2027-10-13**, providing at least 12 new calendar months. Collection continues
beyond that date until the same frozen primary policy has at least 500 eligible
paper sessions and 100 filled paper trades. Twelve months alone is not assumed
to satisfy those count thresholds.

The data are write-only during collection. Researchers, models, notebooks,
strategy code, and parameter owners receive no read or query access. The
custodian may report only ingestion health, byte counts, partition hashes, and
schema failures—not prices, signals, fills, returns, or outcome aggregates.

## Frozen record and evaluation contract

Every approved decision, including `no_trade`, must carry deterministic
`decision_id`, `eligible_session`, decision-universe count/hash, exact sample
label, causal availability timestamps, contracts, lot provenance, executable
quotes or explicit absence, broker/SPAN response, fill attempts, costs, marks,
exit treatment, reconciliation, and risk state. Missing exits remain recorded
and receive the frozen contractual adverse treatment.

After the minimum interval and counts are satisfied, an independent auditor
seals a manifest and authorizes one evaluation. The report must include
ordinary, 2x and uncertainty economics; worst day/week; drawdown; exact-worst-
5% CVaR and block-bootstrap UCB; fills/rejections/no-trades; events;
concentration; plateau; PBO; and DSR using substantive trials only.

## Interpretation

Even a future pass would be evidence for a new paper-shadow review, not proof
of profitability or automatic live approval. Live eligibility additionally
requires verified executable bid/ask/depth, contemporaneous contract and lot
masters, broker/SPAN margin, reconciliation, operational kill switches, and an
independent acceptance decision. No evidence in this repository meets those
conditions today.
