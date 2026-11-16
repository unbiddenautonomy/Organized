# High-coverage refined hybrid

The refined variance-transition strategy has first priority. On dates where it
does not trade, the pre-existing validated expanded hybrid may trade. This
raises coverage without lowering a probability threshold after inspecting
2024 or 2025 outcomes.

## Coverage

| period               | available_days | signal_days | profitable_days | losing_or_flat_days | signal_day_coverage | equivalent_trading_months | profitable_day_rate |
| -------------------- | -------------- | ----------- | --------------- | ------------------- | ------------------- | ------------------------- | ------------------- |
| development_2024     | 235            | 105         | 59              | 46                  | 0.4468              | 5.0000                    | 0.5619              |
| holdout_2025_partial | 76             | 32          | 23              | 9                   | 0.4211              | 1.5238                    | 0.7188              |
| combined             | 311            | 137         | 82              | 55                  | 0.4405              | 6.5238                    | 0.5985              |

Six to seven trading months correspond to approximately 126-147 distinct
trading days. The combined evaluation reaches that absolute range, but the full
2024 calendar year reaches only about five trading months. It is therefore not
valid to describe this as six-to-seven months of coverage per calendar year.

## Net performance after costs

| period               | trades | signal_days | profitable_days | gross_pnl  | cost       | net_pnl    | profit_factor | max_drawdown |
| -------------------- | ------ | ----------- | --------------- | ---------- | ---------- | ---------- | ------------- | ------------ |
| development_2024     | 116    | 105         | 59              | 41758.5000 | 17684.9399 | 24073.5601 | 1.4248        | 9424.6836    |
| holdout_2025_partial | 32     | 32          | 23              | 27513.0000 | 9149.9182  | 18363.0818 | 2.0328        | 9097.0788    |
| combined             | 148    | 137         | 82              | 69271.5000 | 26834.8582 | 42436.6418 | 1.5701        | 9424.6836    |

## Component mix

| year | hybrid_component                    | trades | signal_days | net_pnl    |
| ---- | ----------------------------------- | ------ | ----------- | ---------- |
| 2024 | 120m_short_2_strangle_q75           | 31     | 31          | 878.8911   |
| 2024 | 120m_short_atm_straddle_secondary   | 7      | 7           | 509.2051   |
| 2024 | current_coverage_hybrid             | 59     | 48          | 15090.5584 |
| 2024 | refined_variance_transition_primary | 19     | 19          | 7594.9056  |
| 2025 | 120m_short_2_strangle_q75           | 3      | 3           | 606.7512   |
| 2025 | 120m_short_atm_straddle_secondary   | 1      | 1           | 1069.7626  |
| 2025 | current_coverage_hybrid             | 10     | 10          | 8803.7912  |
| 2025 | refined_variance_transition_primary | 18     | 18          | 7882.7767  |

## Forced-coverage feasibility check

Every tested 120-minute short-volatility structure with 126-147 selected days
lost money in the 2023 validation sample. These rows were rejected rather than
using 2024/2025 results to rescue the threshold.

| structure          | score_quantile | validation_days | equivalent_trading_months | validation_net_pnl | validation_hit_rate | validation_H1_net | validation_H2_net |
| ------------------ | -------------- | --------------- | ------------------------- | ------------------ | ------------------- | ----------------- | ----------------- |
| short_2_strangle   | 0.4000         | 143             | 6.8095                    | -22612.4117        | 0.5315              | -14276.3471       | -8336.0645        |
| short_2_strangle   | 0.4500         | 131             | 6.2381                    | -20720.7806        | 0.5573              | -12677.8493       | -8042.9313        |
| short_1_strangle   | 0.4000         | 143             | 6.8095                    | -20124.8977        | 0.5664              | -12849.9517       | -7274.9459        |
| short_1_strangle   | 0.4500         | 131             | 6.2381                    | -16913.4188        | 0.5649              | -8800.9862        | -8112.4326        |
| short_atm_straddle | 0.4000         | 143             | 6.8095                    | -11194.0018        | 0.5874              | -5262.7044        | -5931.2974        |
| short_atm_straddle | 0.4500         | 131             | 6.2381                    | -11743.8646        | 0.5725              | -4564.4055        | -7179.4590        |

## Bootstrap uncertainty

| period               | days | observed_mean_daily_net | mean_ci_low | mean_ci_high | bootstrap_probability_mean_positive |
| -------------------- | ---- | ----------------------- | ----------- | ------------ | ----------------------------------- |
| development_2024     | 105  | 229.2720                | -88.0371    | 581.5485     | 0.9186                              |
| holdout_2025_partial | 32   | 573.8463                | -163.8401   | 1321.8786    | 0.9360                              |

## Integrity audit

| check                   | violations | passed |
| ----------------------- | ---------- | ------ |
| cross_day_window        | 0          | True   |
| outside_0915_1530       | 0          | True   |
| holding_period_mismatch | 0          | True   |
| overlapping_positions   | 0          | True   |
| expiry_before_entry     | 0          | True   |

## Interpretation

The 2024 result is development evidence because the refined component was
selected using 2024. Only the partial-2025 result is an untouched holdout for
the complete priority policy. Quotes do not establish executable fills, and
short strangles retain unbounded tail risk. The policy remains forward-paper
only.
