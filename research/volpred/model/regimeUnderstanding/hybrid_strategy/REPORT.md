# Hybrid IV/RV strategy backtest

## Simple trading rules

### IV below predicted RV

1. If the 120-minute signed-return forecast is positive and its complete
   expected Greek P&L is positive after costs, buy the ATM call.
2. If that return forecast is negative, buy the ATM put instead.
3. A 120-minute short two-strike OTM strangle is also eligible when its
   gamma + theta + vega forecast is positive after costs.
4. If both signals occur together, take only the one with the larger expected
   net rupee edge. Keep the entry strikes unchanged and exit after 120 minutes.

### IV above predicted RV

The intuitive trade is to sell the call two strikes above ATM and sell the put
two strikes below ATM for 120 minutes. The backtest does **not** validate this
rule: it lost in 2024. It is included in `forced_two_scene_hybrid` only to show
the effect of forcing a trade in this scene.

### IV approximately equal to predicted RV

Do not trade. No tested structure passed both evaluation periods.

All trades use one monthly expiry, fixed entry strikes, a latest entry of
13:30, one open portfolio position, and an exit no later than 15:30. Results
include the current encoded charge schedule and 25 bps premium-turnover
slippage.

## Results

| strategy                        | period           | trades | net_pnl     | mean_net  | median_net | hit_rate | profit_factor | p05_net    | max_drawdown |
| ------------------------------- | ---------------- | ------ | ----------- | --------- | ---------- | -------- | ------------- | ---------- | ------------ |
| conservative_hybrid             | development_2024 | 94     | 4626.2233   | 49.2151   | -242.0149  | 0.4362   | 1.0600        | -2849.6815 | 13243.1003   |
| conservative_hybrid             | holdout_2025     | 15     | 13123.4892  | 874.8993  | 856.2677   | 0.6000   | 2.1998        | -3556.0088 | 10843.7928   |
| forced_two_scene_hybrid         | development_2024 | 240    | -20690.5266 | -86.2105  | -41.1979   | 0.4833   | 0.8412        | -2511.3907 | 24321.4992   |
| forced_two_scene_hybrid         | holdout_2025     | 112    | 3176.7654   | 28.3640   | 157.4351   | 0.5536   | 1.0426        | -3310.5155 | 14383.9557   |
| iv_lt_rv_short_only             | development_2024 | 25     | 2799.4909   | 111.9796  | 353.3859   | 0.6000   | 1.2838        | -2232.1520 | 3142.0978    |
| iv_lt_rv_short_only             | holdout_2025     | 11     | 4749.1081   | 431.7371  | 856.2677   | 0.8182   | 1.6827        | -3478.0263 | 6861.9483    |
| iv_lt_rv_direction_only         | development_2024 | 89     | 6764.8609   | 76.0097   | -289.6344  | 0.4270   | 1.0924        | -2902.9692 | 13471.6454   |
| iv_lt_rv_direction_only         | holdout_2025     | 10     | 5717.5204   | 571.7520  | -935.8011  | 0.4000   | 1.4481        | -3793.8657 | 10843.7928   |
| iv_gt_rv_short_only_unvalidated | development_2024 | 199    | -39914.9984 | -200.5779 | 6.9288     | 0.5075   | 0.5755        | -2227.0697 | 40276.5648   |
| iv_gt_rv_short_only_unvalidated | holdout_2025     | 107    | 2179.9980   | 20.3738   | 199.4431   | 0.5794   | 1.0333        | -3279.4878 | 19875.7122   |

`conservative_hybrid` uses only the two IV<RV streams.
`forced_two_scene_hybrid` adds the unvalidated IV>RV short strangle.

## Conservative-hybrid trade mix

| strategy            | period           | signal_name               | trades | net_pnl    | mean_net  | hit_rate |
| ------------------- | ---------------- | ------------------------- | ------ | ---------- | --------- | -------- |
| conservative_hybrid | development_2024 | iv_lt_rv_long_atm_call    | 44     | -2580.3089 | -58.6434  | 0.4091   |
| conservative_hybrid | development_2024 | iv_lt_rv_long_atm_put     | 44     | 9840.3925  | 223.6453  | 0.4545   |
| conservative_hybrid | development_2024 | iv_lt_rv_short_2_strangle | 6      | -2633.8603 | -438.9767 | 0.5000   |
| conservative_hybrid | holdout_2025     | iv_lt_rv_long_atm_put     | 8      | 5288.8588  | 661.1073  | 0.3750   |
| conservative_hybrid | holdout_2025     | iv_lt_rv_short_2_strangle | 7      | 7834.6304  | 1119.2329 | 0.8571   |

## Interpretation

The comparison is a portfolio backtest, not a sum of independent strategy
totals. When signals overlap, only the earliest eligible trade is retained and
simultaneous signals compete on expected net edge. The component streams were
previously selected in the same reused 2024-2025 research period, so this is
forward-paper evidence rather than an untouched test.
