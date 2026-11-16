# Expanded signal hybrid

This extension keeps the previously validated hybrid first, relaxes the
120-minute fallback only from the top 20% to the still-profitable top 25% of
2023 daily scores, then adds one independently trained shorter-horizon fallback
only on dates that remain empty. No 2024/2025 result selects a model, horizon,
or threshold.

## Frozen selection

- Primary fallback threshold: `0.693128` (2023 q75).
- Secondary: `120`-minute `short_atm_straddle`.
- Secondary threshold: `0.701625`.
- Validation-day minimum: 40.
- Coverage/profit knee: retain at least 80% of the best eligible 2023 net P&L,
  then maximize validation days. This selection uses no 2024/2025 result.
- Costs: current encoded charges plus 25 bps premium-turnover slippage.
- Same monthly expiry; at most one newly selected fallback per otherwise-empty
  date.

Fallback components are admitted only on a date unused by a higher-priority
component. The pre-existing current component can contain multiple sequential,
non-overlapping trades on one date; the integrity audit confirms zero overlap.

## Results

| strategy                | period           | trades | days | gross_pnl  | cost       | net_pnl    | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade |
| ----------------------- | ---------------- | ------ | ---- | ---------- | ---------- | ---------- | --------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- |
| current_coverage_hybrid | development_2024 | 67     | 54   | 27303.0000 | 8563.7863  | 18739.2137 | 279.6898  | 0.5075   | 1.3753        | 2.3136       | 9362.2720    | 9.0582     | -2841.0144 | 10891.1817 | -6049.2700  |
| current_coverage_hybrid | holdout_2025     | 14     | 14   | 26460.0000 | 3754.8935  | 22705.1065 | 1621.7933 | 0.7857   | 5.6870        | 8.5215       | 4750.1826    | 1272.4216  | -2106.4446 | 9841.6358  | -3270.5805  |
| q75_120m_hybrid         | development_2024 | 105    | 92   | 36308.2500 | 14419.4870 | 21888.7630 | 208.4644  | 0.5429   | 1.3908        | 2.0349       | 10287.0911   | 41.7680    | -2390.5572 | 10891.1817 | -6049.2700  |
| q75_120m_hybrid         | holdout_2025     | 24     | 24   | 30334.5000 | 5691.1764  | 24643.3236 | 1026.8052 | 0.7500   | 4.6773        | 6.7467       | 4864.2609    | 714.7262   | -1395.0098 | 9841.6358  | -3270.5805  |
| expanded_signal_hybrid  | development_2024 | 114    | 101  | 40508.2500 | 16421.0476 | 24087.2024 | 211.2912  | 0.5439   | 1.4196        | 2.1280       | 10338.7579   | 51.3895    | -2376.4861 | 10891.1817 | -6049.2700  |
| expanded_signal_hybrid  | holdout_2025     | 25     | 25   | 31746.0000 | 6032.9138  | 25713.0862 | 1028.5234 | 0.7600   | 4.8369        | 6.9033       | 4864.2609    | 801.5749   | -1366.8123 | 9841.6358  | -3270.5805  |

## Combined 2024 and partial-2025 comparison

| strategy                      | total_available_days | signal_days | profitable_days | signal_day_coverage | profitable_signal_day_rate | trades | net_pnl    | profit_factor | max_drawdown |
| ----------------------------- | -------------------- | ----------- | --------------- | ------------------- | -------------------------- | ------ | ---------- | ------------- | ------------ |
| current_coverage_hybrid       | 311                  | 68          | 38              | 0.2186              | 0.5588                     | 81     | 41444.3202 | 1.7566        | 9362.2720    |
| previous_q80_retrained_hybrid | 311                  | 110         | 64              | 0.3537              | 0.5818                     | 123    | 45611.4662 | 1.7307        | 10287.0911   |
| expanded_signal_hybrid        | 311                  | 126         | 74              | 0.4051              | 0.5873                     | 139    | 49800.2886 | 1.7768        | 10338.7579   |

## Day coverage

| strategy                | period           | total_available_days | signal_days | profitable_days | losing_or_flat_days | signal_day_coverage | profitable_signal_day_rate |
| ----------------------- | ---------------- | -------------------- | ----------- | --------------- | ------------------- | ------------------- | -------------------------- |
| current_coverage_hybrid | development_2024 | 235                  | 54          | 27              | 27                  | 0.2298              | 0.5000                     |
| current_coverage_hybrid | holdout_2025     | 76                   | 14          | 11              | 3                   | 0.1842              | 0.7857                     |
| q75_120m_hybrid         | development_2024 | 235                  | 92          | 50              | 42                  | 0.3915              | 0.5435                     |
| q75_120m_hybrid         | holdout_2025     | 76                   | 24          | 18              | 6                   | 0.3158              | 0.7500                     |
| expanded_signal_hybrid  | development_2024 | 235                  | 101         | 55              | 46                  | 0.4298              | 0.5446                     |
| expanded_signal_hybrid  | holdout_2025     | 76                   | 25          | 19              | 6                   | 0.3289              | 0.7600                     |

## Integrity audit

| check                    | violations | passed |
| ------------------------ | ---------- | ------ |
| cross_day_windows        | 0          | True   |
| outside_0915_1530        | 0          | True   |
| holding_period_mismatch  | 0          | True   |
| overlapping_positions    | 0          | True   |
| expiry_before_trade_date | 0          | True   |

## Guardrails

The short strangle has unbounded tail risk. The 2025 period is partial, quotes
do not prove fillability, and this remains a forward-paper-test policy rather
than a live-trading recommendation.
