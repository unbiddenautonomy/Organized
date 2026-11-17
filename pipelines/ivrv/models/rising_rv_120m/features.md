# Model inputs without 120-minute aggregates

The model takes these 19 features in the order in `features.json`. All are available at entry time `t`. The `current_rv` 120-minute aggregate and its derived `current_gap` were removed from the model input.

| Feature | Meaning |
|---|---|
| `current_iv` | Mean of entry call and put IV, in annualized volatility points |
| `rv_30` | Realized spot volatility over the preceding 30 clock minutes |
| `rv_60` | Realized spot volatility over the preceding 60 clock minutes |
| `iv_skew` | Entry put IV minus call IV, in volatility points |
| `spot_price` | Spot index level at entry |
| `strike` | Strike that was ATM at entry |
| `moneyness` | `(spot_price - strike) / spot_price` |
| `days_to_expiry` | Source-reported days until expiry |
| `minute_of_day` | Entry hour × 60 + minute |
| `weekday` | Monday = 0 through Sunday = 6 |
| `delta_call`, `delta_put` | Entry option deltas |
| `gamma_call`, `gamma_put` | Entry option gammas |
| `vega_call`, `vega_put` | Entry option vegas |
| `call_oi`, `put_oi` | Entry option open interest |
| `future_open` | Source future-open value on the entry row |

The **entry filter** still calculates trailing 120-minute `current_rv` to require `current_iv > current_rv`. The **target** still uses forward 120-minute RV and fixed-entry-strike IV. Neither 120-minute RV value is passed to the fitted model.
