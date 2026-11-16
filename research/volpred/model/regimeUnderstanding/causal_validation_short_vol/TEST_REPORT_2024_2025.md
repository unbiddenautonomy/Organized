# Frozen-rule test on 2024-2025

Status: `TEST_UNPROFITABLE_AND_VALIDATION_REJECTION_UNCHANGED`.

This run applies the model trained on 2019-2022 and the probability threshold
`0.324333406875` selected on 2023 H1. Nothing was retuned with 2024-2025
outcomes. The trade remains a one-lot same-monthly-expiry ATM short straddle,
entered at the first threshold crossing and closed exactly 180 minutes later.

## Test performance

| period                     | total_session_days | model_candidate_days | trades | days | gross_pnl  | cost       | net_pnl     | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | signal_coverage_of_sessions | signal_coverage_of_candidates | equivalent_trading_months |
| -------------------------- | ------------------ | -------------------- | ------ | ---- | ---------- | ---------- | ----------- | --------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | --------------------------- | ----------------------------- | ------------------------- |
| test_2024                  | 246                | 221                  | 221    | 221  | 16035.0000 | 51415.8231 | -35380.8231 | -160.0942 | 0.5430   | 0.6686        | -1.9103      | 35107.5468   | 68.6514    | -2675.9796 | 4515.0370  | -6974.5847  | 120             | 0.8984                      | 1.0000                        | 10.5238                   |
| test_2025_partial          | 77                 | 74                   | 74     | 74   | 49308.0000 | 28980.0893 | 20327.9107  | 274.7015  | 0.5946   | 1.3852        | 1.8216       | 13429.5438   | 399.4696   | -3365.4388 | 11575.3025 | -3746.8767  | 44              | 0.9610                      | 1.0000                        | 3.5238                    |
| combined_2024_2025_partial | 323                | 295                  | 295    | 295  | 65343.0000 | 80395.9124 | -15052.9124 | -51.0268  | 0.5559   | 0.9057        | -0.4856      | 43485.5313   | 79.2533    | -2924.8617 | 11575.3025 | -6974.5847  | 164             | 0.9133                      | 1.0000                        | 14.0476                   |

## Monthly results

| month   | trading_days | profitable_days | gross_pnl  | costs     | net_pnl     | hit_rate |
| ------- | ------------ | --------------- | ---------- | --------- | ----------- | -------- |
| 2024-01 | 14           | 8               | 1237.5000  | 2996.3272 | -1758.8272  | 0.5714   |
| 2024-02 | 19           | 10              | 3651.0000  | 4404.4865 | -753.4865   | 0.5263   |
| 2024-03 | 17           | 11              | 4101.7500  | 3813.1192 | 288.6308    | 0.6471   |
| 2024-04 | 13           | 9               | -2155.5000 | 2602.4589 | -4757.9589  | 0.6923   |
| 2024-05 | 19           | 11              | -1631.2500 | 4586.7335 | -6217.9835  | 0.5789   |
| 2024-06 | 19           | 8               | 113.2500   | 4656.1562 | -4542.9062  | 0.4211   |
| 2024-07 | 21           | 11              | 3857.2500  | 5099.6359 | -1242.3859  | 0.5238   |
| 2024-08 | 20           | 12              | 8357.2500  | 4549.9973 | 3807.2527   | 0.6000   |
| 2024-09 | 20           | 11              | -497.2500  | 4278.0601 | -4775.3101  | 0.5500   |
| 2024-10 | 22           | 10              | -4302.0000 | 4985.4674 | -9287.4674  | 0.4545   |
| 2024-11 | 18           | 10              | 3608.2500  | 4113.9799 | -505.7299   | 0.5556   |
| 2024-12 | 19           | 9               | -305.2500  | 5329.4010 | -5634.6510  | 0.4737   |
| 2025-01 | 23           | 13              | 16867.5000 | 9273.4038 | 7594.0962   | 0.5652   |
| 2025-02 | 20           | 14              | 27075.0000 | 7579.3695 | 19495.6305  | 0.7000   |
| 2025-03 | 17           | 12              | 8137.5000  | 5638.4672 | 2499.0328   | 0.7059   |
| 2025-04 | 12           | 4               | -4867.5000 | 5430.0498 | -10297.5498 | 0.3333   |
| 2025-05 | 2            | 1               | 2095.5000  | 1058.7989 | 1036.7011   | 0.5000   |

## Integrity audit

| check                           | violations | passed |
| ------------------------------- | ---------- | ------ |
| one_trade_per_day               | 0          | True   |
| not_first_threshold_crossing    | 0          | True   |
| cross_day_window                | 0          | True   |
| outside_0915_1530               | 0          | True   |
| holding_not_exactly_180_minutes | 0          | True   |
| expiry_before_entry             | 0          | True   |
| pnl_identity_error_over_1e-8    | 0          | True   |
| outside_requested_test_years    | 0          | True   |

Net P&L equals premium received minus premium repurchased, fees, and 25 bps
slippage. The clean 2023 validation previously failed, so a profitable later
test would not reverse the rejection or authorize live deployment. In
addition, earlier research inspected 2024-2025 repeatedly, so this period is a
diagnostic test rather than a pristine final holdout.
