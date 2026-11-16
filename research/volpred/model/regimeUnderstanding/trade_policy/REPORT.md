# Same-expiry intraday trade policy research

> Superseded by the broader five-strike study in
> `../five_strike_trade_research/REPORT.md`, which corrects the legacy RV-unit
> handling and evaluates ATM through +/-2-strike structures.

## Decision

**No live-capital trade is approved by the present evidence.** The only setup
worth forward paper testing is a **120-minute short ATM straddle when current
trading-time IV exceeds the timestamp-available 120-minute HAR RV forecast**.
It is a candidate, not a deployment rule: after a 25 bps premium-turnover cost
stress its mean was only approximately flat, while its 5% trade tail remained
large.

For `IV < predicted RV`, do not buy the straddle with the current forecast. For
`IV = predicted RV`, do not trade. The apparently attractive long-vol result
when IV is below *realized future* RV is an oracle result and cannot be known at
entry.

## Exact contract and P&L

At entry time `t`, buy or sell the call and put with the same ATM strike `K`
and the same monthly expiry `T`; hold both original contracts to `t+h`. Never
re-center the strike. Define exact gross straddle change in index points as

`G = (C(t+h,K,T) + P(t+h,K,T)) - (C(t,K,T) + P(t,K,T))`.

For positive round-trip cost `c`, `PnL_long = G - c` and
`PnL_short = -G - c`.

The local Greek approximation for position sign `q` (`+1` long, `-1` short) is

`PnL(q) ~= q * [Delta*dS + 0.5*Gamma*dS^2 + Vega*dSigma + Theta*dt + Vanna*dS*dSigma + 0.5*Volga*dSigma^2 + Charm*dS*dt + residual] - costs`.

Here Greeks are the call-plus-put values, `dt=h/1440` calendar days, and
`dSigma` is decimal IV change. At entry, a tradable expected-value gate replaces
unknown `dS^2` and `dSigma` with forecasts:

`E_t[PnL_long] ~= 0.5*Gamma*S^2*vhat_h + Theta*h/1440 + Vega*dSigma_hat + Vanna*E[dS*dSigma] + 0.5*Volga*E[dSigma^2] + Charm*E[dS]*h/1440`.

Enter a short only if `-E_t[PnL_long]` exceeds complete round-trip costs plus a
pre-registered safety buffer. With no directional forecast, set `E[dS]=0` and
do not claim expected delta or charm profit.

## Why 120 minutes is the only paper candidate

- With predicted scenes, 15- and 30-minute short straddles did not overcome the
  25 bps turnover stress; 60 minutes also had negative mean net P&L.
- 120 minutes was the shortest tested horizon with a non-negative mean in both
  2024 development and 2025 holdout, and its median and win rate were positive.
- 180 minutes is excluded because the deployed walk-forward RV model has no
  180-minute forecast artifact. Selecting it from realized-RV hindsight would
  be look-ahead.
- Even at 120 minutes the mean edge is too small relative to the left tail, so
  this remains paper-only.

### Tradable 120-minute short candidate (`IV > predicted RV`)

| split | scene | horizon_min | side | trades | days | mean_net_points | median_net_points | win_rate | p05_net_points | p95_net_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_2024 | IV > RV | 120.0000 | short_atm_straddle | 22677.0000 | 240.0000 | 0.0279 | 8.9495 | 0.6277 | -102.4797 | 59.0802 |
| holdout_2025 | IV > RV | 120.0000 | short_atm_straddle | 8677.0000 | 79.0000 | 0.2576 | 6.1398 | 0.5745 | -85.4040 | 67.2152 |

## Why the other scenes are no-trade

### `IV < RV`

The ex-post 60-minute long straddle looked positive when classified with future
realized RV:

| split | scene | horizon_min | side | trades | days | mean_net_points | median_net_points | win_rate | p05_net_points | p95_net_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_2024_2025 | IV < RV | 60.0000 | long_atm_straddle | 6179.0000 | 160.0000 | 6.0490 | -6.7094 | 0.4300 | -71.1931 | 139.2714 |
| train_2019_2022 | IV < RV | 60.0000 | long_atm_straddle | 4989.0000 | 217.0000 | 21.5930 | 7.0024 | 0.5685 | -57.0319 | 149.5405 |
| validation_2023 | IV < RV | 60.0000 | long_atm_straddle | 5379.0000 | 134.0000 | 6.6786 | -5.7421 | 0.4103 | -42.8646 | 100.3151 |

But using the RV forecast actually available at entry reversed the result:

| split | scene | horizon_min | side | trades | days | mean_net_points | median_net_points | win_rate | p05_net_points | p95_net_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_2024 | IV < RV | 60.0000 | long_atm_straddle | 4248.0000 | 140.0000 | -11.9700 | -15.9941 | 0.3004 | -75.6191 | 67.0979 |
| holdout_2025 | IV < RV | 60.0000 | long_atm_straddle | 850.0000 | 39.0000 | -12.6897 | -16.9375 | 0.2247 | -78.1429 | 69.1348 |

This is forecast error, not a Greek identity failure. Do not trade the oracle
label.

### `IV = RV`

Within the +/-0.5-vol-point equality band, expected gamma
and theta are approximately balanced before IV repricing. Costs make both
directions unattractive, so the prescribed position is zero.

| split | scene | horizon_min | side | trades | days | mean_net_points | median_net_points | win_rate | p05_net_points | p95_net_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_2024 | IV = RV | 15.0000 | no_trade | 2089.0000 | 186.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| development_2024 | IV = RV | 30.0000 | no_trade | 2058.0000 | 179.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| development_2024 | IV = RV | 60.0000 | no_trade | 1737.0000 | 154.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| development_2024 | IV = RV | 120.0000 | no_trade | 1307.0000 | 134.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| holdout_2025 | IV = RV | 15.0000 | no_trade | 705.0000 | 63.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| holdout_2025 | IV = RV | 30.0000 | no_trade | 660.0000 | 62.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 | 0.0000 |
| holdout_2025 | IV = RV | 60.0000 | no_trade | 545.0000 | 47.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| holdout_2025 | IV = RV | 120.0000 | no_trade | 313.0000 | 35.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Research controls

- Entry and exit are in the same 09:15–15:30 session.
- Strike and monthly expiry remain fixed through exit.
- IV and RV use the same annualized trading-time clock.
- The tradable audit uses walk-forward HAR forecasts, not future realized RV.
- Results use a symmetric 25 bps premium-turnover stress because the source has
  no bid/ask quotes. This is not a substitute for exact broker, tax, spread and
  market-impact modeling.
- Overlapping minute-level candidates are descriptive and must not be summed as
  an executable portfolio without a non-overlap and capital-allocation rule.
