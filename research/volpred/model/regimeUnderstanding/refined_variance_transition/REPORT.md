# Refined variance-space transition backtest

## Rule

The current and predicted IV/RV gaps are squared-volatility gaps. Near equality
means the predicted variance gap lies within one causal forecast-error standard
deviation. Error dispersion at each timestamp uses completed prior dates only
and requires at least 50 observations. A directionally aligned
five-minute observed IV change is mandatory.

Expected structure P&L includes gamma, theta, vega and volga. Expected vanna
and charm are zero because the model has no signed spot-return forecast, so
`E[dS]=0`; no unestimated directional mean is invented. A trade must also pass
the development-selected complete-cost multiple.

## Selected rules

| side      | signal_variant | structure        | horizon_min | cost_multiple | days_2024_H1 | days_2024_H2 | hit_rate_2024_H1 | hit_rate_2024_H2 | net_pnl_2024_H1 | net_pnl_2024_H2 | profit_factor_2024_H1 | profit_factor_2024_H2 | development_days | development_net | worst_half_net | stable_profitable |
| --------- | -------------- | ---------------- | ----------- | ------------- | ------------ | ------------ | ---------------- | ---------------- | --------------- | --------------- | --------------------- | --------------------- | ---------------- | --------------- | -------------- | ----------------- |
| short_vol | combined_or    | short_2_strangle | 120         | 3.0000        | 10           | 9            | 0.8000           | 0.6667           | 7083.0849       | 511.8207        | 3.8279                | 1.3365                | 19               | 7594.9056       | 511.8207       | True              |

## Development and holdout

| side      | signal_variant | structure        | horizon_min | cost_multiple | period       | days | profitable_days | net_pnl   | hit_rate | profit_factor | max_drawdown |
| --------- | -------------- | ---------------- | ----------- | ------------- | ------------ | ---- | --------------- | --------- | -------- | ------------- | ------------ |
| short_vol | combined_or    | short_2_strangle | 120         | 3.0000        | 2024_H1      | 10   | 8               | 7083.0849 | 0.8000   | 3.8279        | 1875.9568    |
| short_vol | combined_or    | short_2_strangle | 120         | 3.0000        | 2024_H2      | 9    | 6               | 511.8207  | 0.6667   | 1.3365        | 1521.0146    |
| short_vol | combined_or    | short_2_strangle | 120         | 3.0000        | 2025_holdout | 18   | 13              | 7882.7767 | 0.7222   | 1.6558        | 9731.5439    |

## Combined

| period       | trades | days | gross_pnl  | cost      | net_pnl    | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | profitable_days |
| ------------ | ------ | ---- | ---------- | --------- | ---------- | -------- | -------- | ------------- | ------------ | ------------ | --------------- |
| 2024_H1      | 10     | 10   | 8991.7500  | 1908.6651 | 7083.0849  | 708.3085 | 0.8000   | 3.8279        | 9.3515       | 1875.9568    | 8               |
| 2024_H2      | 9      | 9    | 1874.2500  | 1362.4293 | 511.8207   | 56.8690  | 0.6667   | 1.3365        | 1.6274       | 1521.0146    | 6               |
| 2025_holdout | 18     | 18   | 13404.0000 | 5521.2233 | 7882.7767  | 437.9320 | 0.7222   | 1.6558        | 2.8174       | 9731.5439    | 13              |
| combined     | 37     | 37   | 24270.0000 | 8792.3177 | 15477.6823 | 418.3157 | 0.7297   | 1.9646        | 3.6220       | 9731.5439    | 27              |

## Day coverage

| period       | start_date          | end_date            | available_days | signal_days | profitable_days | signal_day_coverage |
| ------------ | ------------------- | ------------------- | -------------- | ----------- | --------------- | ------------------- |
| 2024_H1      | 2024-01-03 00:00:00 | 2024-06-28 00:00:00 | 113            | 10          | 8               | 0.0885              |
| 2024_H2      | 2024-07-01 00:00:00 | 2024-12-31 00:00:00 | 122            | 9           | 6               | 0.0738              |
| 2025_holdout | 2025-01-01 00:00:00 | 2025-05-09 00:00:00 | 76             | 18          | 13              | 0.2368              |
| combined     | 2024-01-03 00:00:00 | 2025-05-09 00:00:00 | 311            | 37          | 27              | 0.1190              |

## Forecast Greek attribution

| period       | days | mean_gamma_points | mean_theta_points | mean_vega_points | mean_vanna_points | mean_volga_points | mean_charm_points | mean_total_forecast_points |
| ------------ | ---- | ----------------- | ----------------- | ---------------- | ----------------- | ----------------- | ----------------- | -------------------------- |
| 2024_H1      | 10   | -26.9151          | 33.3169           | 44.5961          | 0.0000            | -0.1178           | 0.0000            | 50.8802                    |
| 2024_H2      | 9    | -17.5214          | 22.4748           | 34.4278          | 0.0000            | -0.1429           | 0.0000            | 39.2383                    |
| 2025_holdout | 18   | -17.6363          | 12.1921           | 52.8781          | 0.0000            | -0.0835           | 0.0000            | 47.3505                    |

## Cost sensitivity

| slippage_bps | period       | days | net_pnl    | hit_rate | profit_factor | max_drawdown |
| ------------ | ------------ | ---- | ---------- | -------- | ------------- | ------------ |
| 0.0000       | 2024_H1      | 10   | 7732.7330  | 0.8000   | 4.3319        | 1706.8730    |
| 0.0000       | 2024_H2      | 9    | 859.1776   | 0.7778   | 1.6087        | 1411.4281    |
| 0.0000       | 2025_holdout | 18   | 10469.4667 | 0.7778   | 1.9332        | 9130.5126    |
| 10.0000      | 2024_H1      | 10   | 7472.8738  | 0.8000   | 4.1210        | 1774.5065    |
| 10.0000      | 2024_H2      | 9    | 720.2348   | 0.6667   | 1.4993        | 1442.5515    |
| 10.0000      | 2025_holdout | 18   | 9434.7907  | 0.7778   | 1.8185        | 9370.9251    |
| 25.0000      | 2024_H1      | 10   | 7083.0849  | 0.8000   | 3.8279        | 1875.9568    |
| 25.0000      | 2024_H2      | 9    | 511.8207   | 0.6667   | 1.3365        | 1521.0146    |
| 25.0000      | 2025_holdout | 18   | 7882.7767  | 0.7222   | 1.6558        | 9731.5439    |
| 50.0000      | 2024_H1      | 10   | 6433.4368  | 0.8000   | 3.3929        | 2045.0405    |
| 50.0000      | 2024_H2      | 9    | 164.4638   | 0.5556   | 1.0971        | 1651.7865    |
| 50.0000      | 2025_holdout | 18   | 5296.0867  | 0.7222   | 1.4087        | 10332.5751   |
| 100.0000     | 2024_H1      | 10   | 5134.1405  | 0.8000   | 2.6799        | 2383.2080    |
| 100.0000     | 2024_H2      | 9    | -530.2499  | 0.4444   | 0.7615        | 1913.3303    |
| 100.0000     | 2025_holdout | 18   | 122.7067   | 0.6667   | 1.0081        | 11534.6376   |

## Bootstrap uncertainty

| period       | days | observed_mean | mean_ci_low | mean_ci_high | bootstrap_probability_mean_positive |
| ------------ | ---- | ------------- | ----------- | ------------ | ----------------------------------- |
| 2024_H1      | 10   | 708.3085      | -43.6119    | 1389.0767    | 0.9668                              |
| 2024_H2      | 9    | 56.8690       | -309.0931   | 366.0570     | 0.6474                              |
| 2025_holdout | 18   | 437.9320      | -665.8851   | 1544.1079    | 0.7988                              |

## Integrity audit

| check                   | violations | passed |
| ----------------------- | ---------- | ------ |
| one_trade_per_day       | 0          | True   |
| cross_day_window        | 0          | True   |
| outside_0915_1530       | 0          | True   |
| holding_period_mismatch | 0          | True   |
| expiry_before_entry     | 0          | True   |
