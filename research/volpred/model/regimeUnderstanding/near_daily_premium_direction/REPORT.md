# Near-daily premium-direction backtest

The model forecasts the 120-minute percentage change in the combined option
premium. It chooses long volatility when the predicted premium increase has
the higher after-cost edge and short volatility otherwise. It then chooses the
best ATM, one-strike, or two-strike straddle/strangle candidate for that day.

Train is 2019-2022. Model and coverage threshold selection use 2023 only. The
2024 and partial-2025 results do not select the rule.

Decision: `REJECT_NO_POSITIVE_TARGET_COVERAGE_RULE`.

## Selected validation rule

| model     | edge_quantile | edge_threshold | days | net_pnl     | mean_net | hit_rate | profit_factor | max_drawdown | H1_net      | H2_net     | target_coverage | stable_positive |
| --------- | ------------- | -------------- | ---- | ----------- | -------- | -------- | ------------- | ------------ | ----------- | ---------- | --------------- | --------------- |
| ridge_100 | 0.1000        | 82.0004        | 214  | -15913.1355 | -74.3604 | 0.5000   | 0.7064        | 19726.9401   | -10193.5985 | -5719.5370 | True            | False           |

## Frozen evaluation

| period               | available_days | days | net_pnl     | mean_net  | hit_rate | profit_factor | max_drawdown | coverage | equivalent_trading_months |
| -------------------- | -------------- | ---- | ----------- | --------- | -------- | ------------- | ------------ | -------- | ------------------------- |
| test_2024            | 235            | 226  | -13421.8371 | -59.3887  | 0.5044   | 0.8044        | 16123.4315   | 0.9617   | 10.7619                   |
| holdout_2025_partial | 76             | 75   | -7749.9354  | -103.3325 | 0.4533   | 0.7902        | 14887.8348   | 0.9868   | 3.5714                    |
| combined             | 311            | 301  | -21171.7725 | -70.3381  | 0.4917   | 0.7994        | 23968.9989   | 0.9678   | 14.3333                   |

## Long/short and structure mix

| year | trade_side | structure_family | days | net_pnl     | hit_rate |
| ---- | ---------- | ---------------- | ---- | ----------- | -------- |
| 2024 | long_vol   | two_strangle     | 55   | -21132.9377 | 0.1636   |
| 2024 | short_vol  | atm_straddle     | 98   | 7739.1584   | 0.5714   |
| 2024 | short_vol  | one_strangle     | 71   | -193.6031   | 0.6761   |
| 2024 | short_vol  | two_strangle     | 2    | 165.5454    | 0.5000   |
| 2025 | long_vol   | one_strangle     | 1    | -414.7257   | 0.0000   |
| 2025 | long_vol   | two_strangle     | 22   | -9091.5665  | 0.3182   |
| 2025 | short_vol  | atm_straddle     | 38   | -2801.5426  | 0.4474   |
| 2025 | short_vol  | one_strangle     | 14   | 4557.8995   | 0.7143   |

## Integrity audit

| check             | violations | passed |
| ----------------- | ---------- | ------ |
| one_trade_per_day | 0          | True   |
| cross_day_window  | 0          | True   |
| outside_0915_1530 | 0          | True   |

The target is rejected unless validation net P&L is positive. A high hit rate
alone is insufficient because option losses are asymmetric and transaction
costs are material. Quotes do not prove fillability; this is research only.
