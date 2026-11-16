# IV/RV forecast-convergence backtest

## Tested rule

Current realized volatility is the annualized trailing 60-minute volatility:

`RVcurrent = sqrt(sum(last 60 intraday log returns^2) * 252 * 375 / 60)`.

The future forecasts use `IVpred = IVcurrent * exp(predicted log-IV change)`
and the horizon-matched walk-forward `RVpred`. "Nearly equal" means
`abs(IVpred - RVpred) <= 0.005` (0.5 volatility points).

- Long-vol signal: `IVcurrent < RVcurrent` and future IV is nearly equal to
  predicted RV, or IV is forecast to rise toward predicted RV.
- Short-vol signal: `IVcurrent > RVcurrent` and future IV is nearly equal to
  predicted RV, or IV is forecast to fall toward predicted RV.

"Toward" requires the IV forecast to have the correct sign, remain on the
appropriate side of predicted RV (within the equality tolerance is allowed),
and reduce the distance to predicted RV relative to leaving IV unchanged.

Every entry uses only timestamp-available data. It takes the first eligible
signal of the day, keeps the entry strikes fixed, uses one monthly expiry, and
exits after the selected intraday horizon no later than 15:30.

## Data split

- Development/selection: 2024.
- Untouched holdout: 2025 through 9 May, the end of forecast-aligned data.
- The forecasts themselves are walk-forward out-of-sample predictions.

The forecast-aligned panel has no 2023 rows, so the structure and horizon for
this new rule cannot be selected on the repository's usual 2023 validation
period. The 2024-selected hybrid is therefore exploratory, while the 2025
figures are its clean holdout check.

## Condition coverage

| period           | available_days | long_condition_days | short_condition_days | either_condition_days |
| ---------------- | -------------- | ------------------- | -------------------- | --------------------- |
| development_2024 | 235            | 156                 | 217                  | 224                   |
| holdout_2025     | 76             | 39                  | 76                   | 76                    |
| combined         | 311            | 195                 | 293                  | 300                   |

## Best 2024 diagnostics for the requested OR rule

No cell passed the required positive-development-P&L rule. The table therefore
shows the least-bad cell on each side, not an approved strategy.

| side      | structure        | horizon_min | days | net_pnl     | hit_rate | profit_factor | max_drawdown | passed_development_rule |
| --------- | ---------------- | ----------- | ---- | ----------- | -------- | ------------- | ------------ | ----------------------- |
| long_vol  | long_1_strangle  | 120         | 87   | -12616.8910 | 0.2989   | 0.7258        | 20341.6845   | False                   |
| short_vol | short_2_strangle | 120         | 179  | -20002.6376 | 0.5307   | 0.7023        | 19847.3966   | False                   |

## Selected rules in development and holdout

| side      | structure        | horizon_min | period           | days | profitable_days | net_pnl     | mean_net   | hit_rate | profit_factor | max_drawdown |
| --------- | ---------------- | ----------- | ---------------- | ---- | --------------- | ----------- | ---------- | -------- | ------------- | ------------ |
| long_vol  | long_1_strangle  | 120         | development_2024 | 87   | 26              | -12616.8910 | -145.0217  | 0.2989   | 0.7258        | 20341.6845   |
| long_vol  | long_1_strangle  | 120         | holdout_2025     | 18   | 1               | -21646.3759 | -1202.5764 | 0.0556   | 0.0612        | 23058.2302   |
| short_vol | short_2_strangle | 120         | development_2024 | 179  | 95              | -20002.6376 | -111.7466  | 0.5307   | 0.7023        | 19847.3966   |
| short_vol | short_2_strangle | 120         | holdout_2025     | 74   | 43              | 2930.6232   | 39.6030    | 0.5811   | 1.0626        | 11190.7032   |

## Forced one-trade-per-day diagnostic

This combines the two least-bad cells only to quantify the consequence of
forcing the hypothesis into a hybrid. It is not selected for deployment.

| strategy                    | period                     | trades | days | gross_pnl  | cost       | net_pnl     | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | profitable_days |
| --------------------------- | -------------------------- | ------ | ---- | ---------- | ---------- | ----------- | --------- | -------- | ------------- | ------------ | ------------ | --------------- |
| selected_convergence_hybrid | development_2024           | 194    | 194  | 17571.0000 | 41067.6608 | -23496.6608 | -121.1168 | 0.4588   | 0.7018        | -1.8922      | 25599.1776   | 89              |
| selected_convergence_hybrid | holdout_2025               | 75     | 75   | 16833.0000 | 26667.2478 | -9834.2478  | -131.1233 | 0.5067   | 0.8283        | -1.0972      | 18178.9455   | 38              |
| selected_convergence_hybrid | combined_2024_partial_2025 | 269    | 269  | 34404.0000 | 67734.9087 | -33330.9087 | -123.9067 | 0.4721   | 0.7551        | -1.4923      | 38100.7142   | 127             |

## Near-equality versus moving-toward diagnostics

Each row is the best 2024 cell with at least 20 days for that subcondition;
the adjacent 2025 row is untouched holdout performance of the same cell.

| side      | signal_variant | structure        | horizon_min | period           | days | profitable_days | net_pnl     | hit_rate | profit_factor |
| --------- | -------------- | ---------------- | ----------- | ---------------- | ---- | --------------- | ----------- | -------- | ------------- |
| long_vol  | combined_or    | long_1_strangle  | 120         | development_2024 | 87   | 26              | -12616.8910 | 0.2989   | 0.7258        |
| long_vol  | combined_or    | long_1_strangle  | 120         | holdout_2025     | 18   | 1               | -21646.3759 | 0.0556   | 0.0612        |
| long_vol  | near_only      | long_1_strangle  | 120         | development_2024 | 86   | 27              | -10215.0123 | 0.3140   | 0.7742        |
| long_vol  | near_only      | long_1_strangle  | 120         | holdout_2025     | 18   | 1               | -21646.3759 | 0.0556   | 0.0612        |
| long_vol  | toward_only    | long_2_strangle  | 15          | development_2024 | 22   | 4               | -4074.2096  | 0.1818   | 0.2354        |
| long_vol  | toward_only    | long_2_strangle  | 15          | holdout_2025     | 0    | 0               | 0.0000      |          |               |
| short_vol | combined_or    | short_2_strangle | 120         | development_2024 | 179  | 95              | -20002.6376 | 0.5307   | 0.7023        |
| short_vol | combined_or    | short_2_strangle | 120         | holdout_2025     | 74   | 43              | 2930.6232   | 0.5811   | 1.0626        |
| short_vol | near_only      | short_2_strangle | 120         | development_2024 | 43   | 29              | 3197.4399   | 0.6744   | 1.2634        |
| short_vol | near_only      | short_2_strangle | 120         | holdout_2025     | 10   | 6               | -283.8626   | 0.6000   | 0.9608        |
| short_vol | toward_only    | short_2_strangle | 120         | development_2024 | 174  | 93              | -19945.3988 | 0.5345   | 0.6993        |
| short_vol | toward_only    | short_2_strangle | 120         | holdout_2025     | 74   | 43              | 2930.6232   | 0.5811   | 1.0626        |

## Fixed 120-minute ATM benchmarks

| side      | structure          | period           | days | profitable_days | net_pnl     | mean_net   | hit_rate | profit_factor | max_drawdown |
| --------- | ------------------ | ---------------- | ---- | --------------- | ----------- | ---------- | -------- | ------------- | ------------ |
| long_vol  | long_atm_straddle  | development_2024 | 88   | 24              | -19469.4015 | -221.2432  | 0.2727   | 0.6133        | 27183.9028   |
| long_vol  | long_atm_straddle  | holdout_2025     | 18   | 2               | -22136.0127 | -1229.7785 | 0.1111   | 0.0749        | 23433.6053   |
| short_vol | short_atm_straddle | development_2024 | 182  | 94              | -25229.5197 | -138.6237  | 0.5165   | 0.6490        | 25700.7637   |
| short_vol | short_atm_straddle | holdout_2025     | 75   | 45              | 6233.1996   | 83.1093    | 0.6000   | 1.1320        | 11586.9399   |

## Integrity checks

| check                   | violations | passed |
| ----------------------- | ---------- | ------ |
| one_trade_per_day       | 0          | True   |
| cross_day_window        | 0          | True   |
| outside_0915_1530       | 0          | True   |
| holding_period_mismatch | 0          | True   |
| expiry_before_entry     | 0          | True   |

## Interpretation limit

This is a historical, cost-stressed screen, not an instruction to place live
orders. Monthly-expiry short-vol trades retain large gap and tail risk, and
the data use quoted option prices rather than executable bid/ask fills.
