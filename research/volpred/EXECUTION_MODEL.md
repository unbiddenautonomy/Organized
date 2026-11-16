# BANKNIFTY option execution, cost, and margin proxy

**Frozen/reviewed:** 2026-10-13. **Purpose:** conservative research costing only.
The reusable implementation is `research/execution_costs.py`; it does not
calculate strategy P&L or select/tune strategy parameters.

## Evidence status

The repository has LTP-like option prices, IV, Greeks, and OI, but no historical
bid, ask, displayed size, order-book events, trade conditions, fill IDs, or
historical SPAN/exposure files. Consequently, executable price, market impact,
partial fills, and margin **cannot be validated or reconstructed**. OI is not
displayed liquidity. Every spread, impact, fill, rejection, margin, and carry
number produced here is an intentionally conservative proxy. It cannot support
a tradability claim; it must be replaced or recalibrated when executable quotes,
broker fills, and dated SPAN files are obtained.

## Dated statutory and venue components

Rates are fractions of option-premium turnover unless noted otherwise.

| Component | Effective dates | Frozen rate/rule |
|---|---|---:|
| STT, option sale | through 2023-03-31 | 0.05% of sell premium |
| STT, option sale | 2023-04-01 to 2024-09-30 | 0.0625% of sell premium |
| STT, option sale | 2024-10-01 to 2026-03-31 | 0.10% of sell premium |
| STT, option sale | from 2026-04-01 | 0.15% of sell premium |
| STT, exercise | through 2026-03-31 | 0.125% of intrinsic value, purchaser |
| STT, exercise | from 2026-04-01 | 0.15% of intrinsic value, purchaser |
| NSE transaction fee | before 2023-04-01 | 0.053% proxy |
| NSE transaction fee | 2023-04-01 to 2024-03-31 | 0.050% proxy |
| NSE transaction fee | 2024-04-01 to 2024-09-30 | 0.0495% proxy |
| NSE transaction fee | 2024-10-01 to 2026-02-28 | 0.03503% |
| NSE transaction fee | from 2026-03-01 | 0.03552% |
| NSE IPFT | before 2023-04-01 | INR 0.01/crore (0.0000001%) |
| NSE IPFT | 2023-04-01 to 2026-02-28 | INR 50/crore (0.0005%) |
| NSE IPFT | from 2026-03-01 | INR 0.01/crore (0.0000001%) |
| SEBI turnover fee | all modeled dates | INR 10/crore (0.0001%) |
| Stamp duty | buyer, premium value | 0.003% |
| GST | taxable fees | 18% of brokerage + exchange + SEBI + IPFT |
| Brokerage | each executed leg/order | INR 20 proxy |

The 2019-10-11 to 2023-03-31 exchange fee uses the highest published member
slab as a conservative retail pass-through; actual broker monthly slabs are not
available. National buyer-side stamp duty applies from 2020-07-01. The same
0.003% proxy is deliberately used before that date because the state/broker
history required for exact reconstruction is absent. Brokerage is a broker
assumption, not an exchange tariff. The module does not round taxes per order
because daily contract-note aggregation is broker-dependent.

Primary references:

- [Finance Act 2023: option-sale STT increase effective 2023-04-01](https://incometaxindia.gov.in/Documents/Act/Finance-Act-2023.pdf)
- [Finance (No. 2) Bill 2024 memorandum: STT change from 2024-10-01](https://incometaxindia.gov.in/budgets%20and%20bills/2024/memo-2024.pdf)
- [Income Tax Department 2026 memorandum: option sale/exercise STT to 0.15% from 2026-04-01](https://www.incometaxindia.gov.in/documents/81799/11848482/memo-2026.pdf/fe530cfa-9c49-fc5c-4bfa-fc96fd5e7b7a)
- [NSE statutory levies page, updated 2026-04-17](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies)
- [NSE/FA/56129: 2023 transaction fee and IPFT revision](https://archives.nseindia.com/content/circulars/FA56129.pdf)
- [NSE/FA/73061: fee/IPFT rebalance from 2026-03-01](https://nsearchives.nseindia.com/content/circulars/FA73061.pdf)

## Execution proxy

For each filled order with premium turnover `T`:

```text
half_spread_bps = 15 * sqrt(max(1, volatility_ratio))
impact_bps = 10 + 15 * max(volatility_ratio - 1, 0)
                + 10 * sqrt(max(lots - 1, 0))
crossing = T * half_spread_bps / 10,000
impact = T * impact_bps / 10,000
```

At one lot and normal volatility this is 25 bps of turnover (15 bps crossing
plus 10 bps impact). This is a frozen assumption, not an empirical estimate.
Volatility ratio must be defined before testing (for example current ATM IV over
a trailing, causally available reference) and cannot use future observations.

### Deterministic fill rule

1. No eligible next snapshot or an invalid premium: reject.
2. Capacity is `floor(OI contracts * 0.001)`, capped by the frozen primary limit.
3. Missing OI gives one-lot capacity, explicitly as a proxy.
4. The primary policy caps capacity at one lot and requires the whole position;
   no legging or implicit partial position is allowed.
5. Optional capacity analysis may allow a partial fill only when it is at least
   50% of requested lots. It must be labelled separately from the primary test.

For a multi-leg position, apply the rule to every leg and use the minimum filled
lots across legs; if full-position completion is required, reject all legs when
any one leg fails. OI does not prove executable capacity, so even a modeled fill
remains unvalidated.

## Margin and carry proxy

Historical SPAN is absent. The broker-style substitute is:

- uncovered short: 20% of spot notional (`spot * lot_size * lots`);
- defined-risk structure: maximum loss, including premium debit;
- long-only structure: premium debit;
- then add the scope-mandated 20% buffer;
- reject when buffered requirement exceeds 35% of INR 1,000,000 capital;
- margin carry = buffered requirement × 8% × holding hours / (365 × 24).

This can conservatively reject trades that real SPAN might accept. Conversely,
it cannot model intraday margin hikes, offsets, extreme-loss margin, expiry-day
changes, or broker liquidation. `unvalidated=True` is therefore hard-coded on
every `MarginEstimate`.

## Settlement / failed-exit path

The intended path is an observed fixed-strike close before settlement. If that
cannot be established, do not invent a free close. Long ITM intrinsic value is
charged exercise STT at the dated purchaser rate. Any unclosed short position
also receives a frozen 100 bps underlying-notional gap reserve. This reserve is
not a tax or predicted loss; it is a conservative missing-exit penalty required
because the narrow moving strike window can lose the selected strike. Settlement
payoff itself belongs in a separate position-pricing layer, not this cost module.

## Required scenarios and reporting

Report the component-level primary estimate and `CostBreakdown.stressed(2.0)`.
The 2x scenario doubles every component, including statutory charges, as a
deliberately severe model-risk stress; it is not a claim that tax rates double.
Never change trade selection after seeing the 2x outcome. Reports must retain
the `unvalidated` disclosure above and must not describe stored prices as fills.
