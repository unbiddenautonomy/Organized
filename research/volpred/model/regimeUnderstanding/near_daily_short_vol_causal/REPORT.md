# Causal correction of the near-daily short-volatility backtest

Status: `REJECT_CAUSAL_RULE_FAILED_VALIDATION`.

The previously reported result is withdrawn. It selected the highest model
score from every timestamp after the whole trading day was known. A live
system cannot know in advance which later timestamp will be the day's maximum.

This correction freezes the model, thresholds, same-expiry ATM straddle,
180-minute horizon, 50% stop, one/three-lot sizing, fees, and 25 bps slippage.
It changes only the entry selection: take the first timestamp at which
`model_probability >= 0.506968648812`. The fixed ATM strike selected
at entry is retained through the exit path.

The one/three-lot overlay still uses the sizing threshold inherited from the
superseded search, so its P&L is a diagnostic measurement rather than a new
validated strategy. The one-lot table removes that sizing overlay. Both fail
the causal 2023 acceptance gate.

## Corrected performance

| period                     | total_session_days | model_candidate_days | trades | days | gross_pnl   | cost       | net_pnl     | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | signal_coverage_of_sessions | signal_coverage_of_candidates | equivalent_trading_months |
| -------------------------- | ------------------ | -------------------- | ------ | ---- | ----------- | ---------- | ----------- | --------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | --------------------------- | ----------------------------- | ------------------------- |
| validation_2023            | 243                | 234                  | 222    | 222  | 43136.2500  | 47431.8458 | -4295.5958  | -19.3495  | 0.5225   | 0.9353        | -0.2832      | 24326.9459   | 30.9496    | -1338.9454 | 7272.2795  | -6489.0168  | 116             | 0.9136                      | 0.9487                        | 10.5714                   |
| test_2024                  | 246                | 221                  | 211    | 211  | 148470.0000 | 55127.1950 | 93342.8050  | 442.3830  | 0.5972   | 2.4576        | 3.3280       | 7769.5712    | 146.4159   | -1461.6064 | 18303.6039 | -4196.5076  | 126             | 0.8577                      | 0.9548                        | 10.0476                   |
| holdout_2025_partial       | 77                 | 74                   | 71     | 71   | 118476.0000 | 29855.4617 | 88620.5383  | 1248.1766 | 0.5634   | 3.5723        | 3.8114       | 10017.5770   | 250.1264   | -1835.0495 | 34914.7076 | -8582.0986  | 40              | 0.9221                      | 0.9595                        | 3.3810                    |
| combined_2024_2025_partial | 323                | 295                  | 282    | 282  | 266946.0000 | 84982.6567 | 181963.3433 | 645.2601  | 0.5887   | 2.8475        | 3.2100       | 10017.5770   | 150.7181   | -1530.8007 | 34914.7076 | -8582.0986  | 166             | 0.8731                      | 0.9559                        | 13.4286                   |

## Validation stability

| period             | trades | days | gross_pnl  | cost       | net_pnl     | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days |
| ------------------ | ------ | ---- | ---------- | ---------- | ----------- | --------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- |
| validation_2023_H1 | 109    | 109  | 41147.2500 | 23542.9487 | 17604.3013  | 161.5074  | 0.5229   | 1.9122        | 2.5999       | 3935.7630    | 52.8840    | -605.7064  | 7272.2795  | -2677.2311  | 57              |
| validation_2023_H2 | 113    | 113  | 1989.0000  | 23888.8971 | -21899.8971 | -193.8044 | 0.5221   | 0.5349        | -2.6770      | 24326.9459   | 7.8355     | -2452.0377 | 3070.3746  | -6489.0168  | 59              |

The acceptance gate requires positive P&L for all of 2023 and independently
for both halves of 2023. A failed validation gate makes deployable P&L zero,
regardless of a later positive period.

## Profit correction

| version                            | combined_2024_2025_partial_net_pnl |
| ---------------------------------- | ---------------------------------- |
| invalid_daily_max_selection        | 157306.6207                        |
| causal_execution_frozen_old_sizing | 181963.3433                        |
| deployable_after_validation_gate   | 0.0000                             |

## One-lot benchmark

| period                     | total_session_days | model_candidate_days | trades | days | gross_pnl   | cost       | net_pnl    | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | signal_coverage_of_sessions | signal_coverage_of_candidates | equivalent_trading_months |
| -------------------------- | ------------------ | -------------------- | ------ | ---- | ----------- | ---------- | ---------- | -------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | --------------------------- | ----------------------------- | ------------------------- |
| validation_2023            | 243                | 234                  | 222    | 222  | 33233.2500  | 42291.0281 | -9057.7781 | -40.8008 | 0.5135   | 0.7830        | -1.3362      | 15046.3124   | 7.6375     | -903.6738  | 2360.8265  | -2225.9389  | 114             | 0.9136                      | 0.9487                        | 10.5714                   |
| test_2024                  | 246                | 221                  | 211    | 211  | 72694.5000  | 48430.9871 | 24263.5129 | 114.9930 | 0.5877   | 1.4887        | 2.0092       | 7769.5712    | 107.9089   | -1091.0447 | 6038.2680  | -4128.5814  | 124             | 0.8577                      | 0.9548                        | 10.0476                   |
| holdout_2025_partial       | 77                 | 74                   | 71     | 71   | 57813.0000  | 27322.0107 | 30490.9893 | 429.4506 | 0.5634   | 2.1148        | 3.4338       | 10017.5770   | 250.1264   | -1487.5841 | 11575.3025 | -2923.9662  | 40              | 0.9221                      | 0.9595                        | 3.3810                    |
| combined_2024_2025_partial | 323                | 295                  | 282    | 282  | 130507.5000 | 75752.9978 | 54754.5022 | 194.1649 | 0.5816   | 1.7111        | 2.4236       | 10017.5770   | 108.8156   | -1377.8623 | 11575.3025 | -4128.5814  | 164             | 0.8731                      | 0.9559                        | 13.4286                   |

## Entry-time distribution

| entry_time | days |
| ---------- | ---- |
| 09:15      | 140  |
| 10:34      | 1    |
| 10:42      | 1    |
| 10:49      | 3    |
| 10:50      | 1    |
| 10:51      | 1    |
| 10:52      | 1    |
| 10:56      | 1    |
| 10:57      | 2    |
| 10:58      | 1    |
| 11:00      | 1    |
| 11:01      | 2    |
| 11:02      | 4    |
| 11:04      | 2    |
| 11:05      | 2    |
| 11:06      | 1    |
| 11:07      | 1    |
| 11:08      | 2    |
| 11:09      | 3    |
| 11:10      | 3    |
| 11:11      | 1    |
| 11:12      | 5    |
| 11:13      | 2    |
| 11:14      | 1    |
| 11:15      | 3    |
| 11:16      | 3    |
| 11:17      | 1    |
| 11:18      | 5    |
| 11:20      | 3    |
| 11:21      | 4    |
| 11:23      | 1    |
| 11:24      | 1    |
| 11:25      | 6    |
| 11:26      | 4    |
| 11:27      | 2    |
| 11:28      | 5    |
| 11:29      | 5    |
| 11:30      | 2    |
| 11:31      | 3    |
| 11:32      | 5    |
| 11:33      | 4    |
| 11:34      | 4    |
| 11:35      | 4    |
| 11:37      | 2    |
| 11:38      | 5    |
| 11:39      | 2    |
| 11:40      | 7    |
| 11:41      | 5    |
| 11:42      | 2    |
| 11:43      | 3    |
| 11:44      | 7    |
| 11:45      | 6    |
| 11:46      | 5    |
| 11:47      | 5    |
| 11:48      | 4    |
| 11:49      | 9    |
| 11:50      | 4    |
| 11:51      | 6    |
| 11:52      | 7    |
| 11:53      | 9    |
| 11:54      | 6    |
| 11:55      | 4    |
| 11:56      | 6    |
| 11:57      | 4    |
| 11:58      | 9    |
| 11:59      | 11   |
| 12:00      | 9    |
| 12:01      | 2    |
| 12:02      | 6    |
| 12:03      | 4    |
| 12:04      | 4    |
| 12:05      | 5    |
| 12:06      | 6    |
| 12:07      | 8    |
| 12:08      | 5    |
| 12:09      | 7    |
| 12:10      | 10   |
| 12:11      | 6    |
| 12:12      | 4    |
| 12:13      | 5    |
| 12:14      | 3    |
| 12:15      | 11   |
| 12:16      | 4    |
| 12:17      | 5    |
| 12:18      | 3    |
| 12:19      | 5    |
| 12:20      | 3    |
| 12:21      | 6    |
| 12:22      | 2    |
| 12:23      | 2    |
| 12:24      | 1    |
| 12:25      | 1    |
| 12:26      | 2    |

## Exit attribution

| year | exit_reason    | trades | profitable_days | net_pnl     |
| ---- | -------------- | ------ | --------------- | ----------- |
| 2023 | scheduled_exit | 220    | 116             | 7588.4115   |
| 2023 | stop_loss      | 2      | 0               | -11884.0073 |
| 2024 | scheduled_exit | 211    | 126             | 93342.8050  |
| 2025 | scheduled_exit | 71     | 40              | 88620.5383  |

## Cost sensitivity

| slippage_bps | period               | days | net_pnl     | profit_factor | max_drawdown |
| ------------ | -------------------- | ---- | ----------- | ------------- | ------------ |
| 0.0000       | validation_2023      | 222  | 13638.4511  | 1.2332        | 18410.5615   |
| 0.0000       | test_2024            | 211  | 117151.9000 | 3.1409        | 7205.1137    |
| 0.0000       | holdout_2025_partial | 71   | 104265.4408 | 4.7898        | 8266.4911    |
| 10.0000      | validation_2023      | 222  | 6464.8323   | 1.1051        | 20777.1153   |
| 10.0000      | test_2024            | 211  | 107628.2620 | 2.8449        | 7430.8967    |
| 10.0000      | holdout_2025_partial | 71   | 98007.4798  | 4.2446        | 8426.0238    |
| 25.0000      | validation_2023      | 222  | -4295.5958  | 0.9353        | 24326.9459   |
| 25.0000      | test_2024            | 211  | 93342.8050  | 2.4576        | 7769.5712    |
| 25.0000      | holdout_2025_partial | 71   | 88620.5383  | 3.5723        | 10017.5770   |
| 50.0000      | validation_2023      | 222  | -22229.6427 | 0.7048        | 32981.2338   |
| 50.0000      | test_2024            | 211  | 69533.7100  | 1.9326        | 10376.1716   |
| 50.0000      | holdout_2025_partial | 71   | 72975.6358  | 2.7235        | 13029.7385   |
| 100.0000     | validation_2023      | 222  | -58097.7364 | 0.4030        | 64719.4066   |
| 100.0000     | test_2024            | 211  | 21915.5200  | 1.2163        | 24151.7698   |
| 100.0000     | holdout_2025_partial | 71   | 41685.8308  | 1.6784        | 19243.5710   |

## Integrity audit

| check                          | violations | passed |
| ------------------------------ | ---------- | ------ |
| one_trade_per_day              | 0          | True   |
| cross_day_window               | 0          | True   |
| outside_0915_1530              | 0          | True   |
| holding_exceeds_180_minutes    | 0          | True   |
| expiry_before_entry            | 0          | True   |
| not_first_threshold_crossing   | 0          | True   |
| missing_trade_path             | 0          | True   |
| multiple_strikes_in_trade_path | 0          | True   |

The 2025 sample ends on 2025-05-09 and has been examined repeatedly. It must
not be treated as a pristine holdout in subsequent strategy optimization.
Mid-price data and modeled costs also do not establish live fill quality.
