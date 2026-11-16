# Risk-managed IV/RV transition strategy

The user's long/short transition hypothesis remains a mandatory first gate.
Entries must also forecast enough Gamma + Theta + Vega P&L to cover a selected
multiple of complete costs. The risk overlay checks the unchanged option
strikes after 15, 30, and 60 minutes and otherwise exits after 120 minutes.

Take-profit and stop-loss levels are fractions of the absolute entry premium.
All rule choices use 2024 only and must have at least 8 days plus
positive net P&L in both halves. Partial 2025 is holdout only.

## Selected rules

| side      | signal_variant | structure        | cost_multiple | take_profit | stop_loss | days_2024_H1 | days_2024_H2 | hit_rate_2024_H1 | hit_rate_2024_H2 | max_drawdown_2024_H1 | max_drawdown_2024_H2 | net_pnl_2024_H1 | net_pnl_2024_H2 | profit_factor_2024_H1 | profit_factor_2024_H2 | development_net | worst_half_net | development_days | stable_profitable |
| --------- | -------------- | ---------------- | ------------- | ----------- | --------- | ------------ | ------------ | ---------------- | ---------------- | -------------------- | -------------------- | --------------- | --------------- | --------------------- | --------------------- | --------------- | -------------- | ---------------- | ----------------- |
| long_vol  | near_only      | long_2_strangle  | 1.0000        | 0.5000      | 0.2500    | 9            | 8            | 0.6667           | 0.3750           | 729.2931             | 2358.0475            | 850.9662        | 5752.5219       | 1.9786                | 3.2478                | 6603.4881       | 850.9662       | 17               | True              |
| short_vol | combined_or    | short_2_strangle | 3.0000        | 0.3000      | 0.5000    | 11           | 9            | 0.8182           | 0.5556           | 261.6717             | 1688.3053            | 8263.3696       | 99.2194         | 24.8037               | 1.0533                | 8362.5890       | 99.2194        | 20               | True              |

## Development and holdout

| side      | signal_variant | structure        | cost_multiple | take_profit | stop_loss | period       | days | profitable_days | net_pnl    | hit_rate | profit_factor | max_drawdown | mean_exit_min |
| --------- | -------------- | ---------------- | ------------- | ----------- | --------- | ------------ | ---- | --------------- | ---------- | -------- | ------------- | ------------ | ------------- |
| long_vol  | near_only      | long_2_strangle  | 1.0000        | 0.5000      | 0.2500    | 2024_H1      | 9    | 6               | 850.9662   | 0.6667   | 1.9786        | 729.2931     | 120.0000      |
| long_vol  | near_only      | long_2_strangle  | 1.0000        | 0.5000      | 0.2500    | 2024_H2      | 8    | 3               | 5752.5219  | 0.3750   | 3.2478        | 2358.0475    | 120.0000      |
| long_vol  | near_only      | long_2_strangle  | 1.0000        | 0.5000      | 0.2500    | 2025_holdout | 1    | 0               | -588.5926  | 0.0000   | 0.0000        | 0.0000       | 120.0000      |
| short_vol | combined_or    | short_2_strangle | 3.0000        | 0.3000      | 0.5000    | 2024_H1      | 11   | 9               | 8263.3696  | 0.8182   | 24.8037       | 261.6717     | 111.8182      |
| short_vol | combined_or    | short_2_strangle | 3.0000        | 0.3000      | 0.5000    | 2024_H2      | 9    | 5               | 99.2194    | 0.5556   | 1.0533        | 1688.3053    | 120.0000      |
| short_vol | combined_or    | short_2_strangle | 3.0000        | 0.3000      | 0.5000    | 2025_holdout | 21   | 12              | -1832.7259 | 0.5714   | 0.9109        | 8429.7682    | 115.7143      |

## Combined performance

| period       | trades | days | net_pnl    | mean_net | hit_rate | profit_factor | max_drawdown | profitable_days | mean_exit_min |
| ------------ | ------ | ---- | ---------- | -------- | -------- | ------------- | ------------ | --------------- | ------------- |
| 2024_H1      | 20     | 20   | 9114.3358  | 455.7168 | 0.7500   | 8.4911        | 726.3407     | 15              | 115.5000      |
| 2024_H2      | 17     | 17   | 5851.7413  | 344.2201 | 0.4706   | 2.3240        | 2391.1600    | 8               | 120.0000      |
| 2025_holdout | 21     | 21   | -1757.2136 | -83.6768 | 0.5714   | 0.9143        | 8354.2558    | 12              | 115.7143      |
| combined     | 58     | 58   | 13208.8635 | 227.7390 | 0.6034   | 1.5053        | 8354.2558    | 35              | 116.8966      |

The holdout must be positive by a meaningful margin before considering forward
paper deployment. Short-vol tail exposure remains even with checkpoint stops,
because prices can gap between observations.
