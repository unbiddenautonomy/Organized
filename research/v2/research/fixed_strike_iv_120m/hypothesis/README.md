# Actual-value fixed-strike IV convergence hypothesis

This folder tests the proposed hypothesis without any model predictions.

For the strike `K_t` that is ATM at entry time `t`, the primary ex-post condition is:

1. `IV(K_t,t) > RV(t)`.
2. `IV(K_t,t+120)` is within 2 volatility points of `RV(t+120)`.
3. IV falls and the absolute IV–RV gap becomes smaller by `t+120`.

Straddle IV is the call/put vega-weighted IV. Realized volatility is the annualized variance rate of spot log returns over a trailing 120-minute clock-time window, using 252 trading days and 375 minutes per trading day. For irregular source timestamps, squared returns are divided by their elapsed minutes; gaps above 15 minutes are rejected and at least 60 covered minutes are required. The report also checks 1, 2, 3, and 5 volatility-point tolerances and a stricter definition requiring both legs individually to satisfy all conditions.

## P&L equation

For one short straddle lot entered at actual prices:

```text
PnL(u) = [(Call_t + Put_t) - (Call_u + Put_u)] * historical_lot_size - costs
```

The strike and expiry never change. Two policies are tested:

- `stop_only`: first actual one-minute P&L at or below -₹750; otherwise exit at `t+120`.
- `bracket`: first actual one-minute P&L at or below -₹750 or at or above +₹1,500; otherwise exit at `t+120`.

Because prices are sampled, an executed stop can be worse than exactly -₹750. Bid/ask and fees are not present in the source data; the configurable cost defaults to zero.

## Important limitation

Conditions 2 and 3 require actual `t+120` values. They are unknowable at entry and therefore introduce look-ahead if treated as a trading signal. This analysis tests whether the economic hypothesis is associated with profitable short-straddle outcomes; it does not establish a live entry strategy.

Run:

```powershell
py -3.13 hypothesis_backtest.py --config config.yaml
```
