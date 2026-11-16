# Causal profitable volatility hybrid

Status: `POSITIVE_HISTORICAL_WALK_FORWARD_PAPER_ONLY`.

This strategy combines two non-overlapping one-lot short-ATM-straddle sleeves.
The DTE 0-3 sleeve uses fixed 09:30 and 11:45 entries with 120-minute exits.
The DTE 8-14 sleeve uses a direct after-cost P&L model, refit at each month
using only earlier rows. Model family and threshold quantile were chosen from
pre-2024 walk-forward predictions only.

The pre-2024 selector chose `huber` with the `60%` training
prediction quantile. DTE 4-7 and DTE 15-20 are not traded because they did not
produce an eligible, positive pre-2024 model candidate under the same gate.

## Pre-2024 model selection

| model_name | threshold_quantile | development_signals | development_days | development_net_25bps | development_net_50bps | development_hit_rate | eligible | selected |
| ---------- | ------------------ | ------------------- | ---------------- | --------------------- | --------------------- | -------------------- | -------- | -------- |
| huber      | 0.6000             | 9                   | 9                | 6704.6911             | 5789.1275             | 0.6667               | True     | True     |
| huber      | 0.7000             | 5                   | 5                | 3110.8574             | 2663.0188             | 0.6000               | True     | False    |
| huber      | 0.8000             | 5                   | 5                | 3110.8574             | 2663.0188             | 0.6000               | True     | False    |
| ridge10    | 0.6000             | 8                   | 8                | -8462.6930            | -9374.4523            | 0.5000               | False    | False    |
| ridge10    | 0.7000             | 6                   | 6                | -6837.1295            | -7441.4152            | 0.5000               | False    | False    |
| ridge10    | 0.8000             | 3                   | 3                | 2584.4856             | 2336.7267             | 0.6667               | False    | False    |
| ridge100   | 0.6000             | 3                   | 3                | -745.1140             | -1120.5382            | 0.3333               | False    | False    |
| ridge100   | 0.7000             | 1                   | 1                | 1074.0738             | 926.1455              | 1.0000               | False    | False    |
| ridge100   | 0.8000             | 0                   | 0                | 0.0000                | 0.0000                |                      | False    | False    |

## Cost sensitivity

| half_spread_bps | sleeve         | sample               | trades | days | profitable_days | gross_pnl  | costs     | net_pnl    | hit_rate | profit_factor | max_drawdown | worst_day  | best_day  | bootstrap_ci_low | bootstrap_ci_high | bootstrap_p_net_le_zero |
| --------------- | -------------- | -------------------- | ------ | ---- | --------------- | ---------- | --------- | ---------- | -------- | ------------- | ------------ | ---------- | --------- | ---------------- | ----------------- | ----------------------- |
| 10.0000         | near_expiry    | evaluation_2024_2025 | 25     | 21   | 18              | 13579.7508 | 2682.6278 | 10897.1230 | 0.8571   | 2.3475        | 7647.8543    | -7647.8543 | 4357.4365 | -9758.7981       | 26398.5317        | 0.1328                  |
| 10.0000         | model_dte_8_14 | evaluation_2024_2025 | 21     | 21   | 14              | 8333.2828  | 2718.7444 | 5614.5383  | 0.6667   | 1.4505        | 4918.3986    | -3644.8486 | 5423.0260 | -11757.4817      | 23249.7436        | 0.2614                  |
| 10.0000         | combined       | evaluation_2024_2025 | 46     | 42   | 32              | 21913.0335 | 5401.3723 | 16511.6613 | 0.7619   | 1.8035        | 9364.7376    | -7647.8543 | 5423.0260 | -10330.6019      | 40755.3510        | 0.1041                  |
| 25.0000         | near_expiry    | evaluation_2024_2025 | 25     | 21   | 18              | 13062.3411 | 2682.6262 | 10379.7148 | 0.8571   | 2.2685        | 7687.1696    | -7687.1696 | 4335.3160 | -10350.3389      | 25896.0289        | 0.1442                  |
| 25.0000         | model_dte_8_14 | evaluation_2024_2025 | 21     | 21   | 14              | 7139.4645  | 2718.7540 | 4420.7105  | 0.6667   | 1.3422        | 5207.3278    | -3691.7119 | 5365.5279 | -13010.7920      | 22093.4028        | 0.3092                  |
| 25.0000         | combined       | evaluation_2024_2025 | 46     | 42   | 32              | 20201.8056 | 5401.3802 | 14800.4253 | 0.7619   | 1.7014        | 9527.3718    | -7687.1696 | 5365.5279 | -12147.4605      | 39103.7768        | 0.1290                  |
| 50.0000         | near_expiry    | evaluation_2024_2025 | 25     | 21   | 18              | 12199.9915 | 2682.6236 | 9517.3679  | 0.8571   | 2.1407        | 7752.6952    | -7752.6952 | 4298.4485 | -11256.0964      | 25042.0340        | 0.1626                  |
| 50.0000         | model_dte_8_14 | evaluation_2024_2025 | 21     | 21   | 14              | 5149.7674  | 2718.7699 | 2430.9975  | 0.6667   | 1.1778        | 5688.8766    | -3769.8174 | 5269.6978 | -15068.2215      | 20139.0089        | 0.3954                  |
| 50.0000         | combined       | evaluation_2024_2025 | 46     | 42   | 32              | 17349.7589 | 5401.3935 | 11948.3654 | 0.7619   | 1.5427        | 9798.4289    | -7752.6952 | 5269.6978 | -15160.9767      | 36350.8096        | 0.1792                  |
| 100.0000        | near_expiry    | evaluation_2024_2025 | 25     | 21   | 18              | 10475.2925 | 2681.6184 | 7793.6741  | 0.8571   | 1.8996        | 7883.7464    | -7883.7464 | 4224.7135 | -13114.1983      | 23363.7855        | 0.2010                  |
| 100.0000        | model_dte_8_14 | evaluation_2024_2025 | 21     | 21   | 13              | 1170.3732  | 2717.8017 | -1547.4285 | 0.6190   | 0.8987        | 7160.1706    | -3926.0285 | 5078.0377 | -19267.3852      | 16237.1339        | 0.5700                  |
| 100.0000        | combined       | evaluation_2024_2025 | 46     | 42   | 31              | 11645.6657 | 5399.4201 | 6246.2456  | 0.7381   | 1.2610        | 10340.5430   | -7883.7464 | 5078.0377 | -21112.4300      | 30809.0505        | 0.3111                  |

## Calendar-year evaluation

| year | sleeve                          | sample | trades | days | profitable_days | gross_pnl  | costs     | net_pnl    | hit_rate | profit_factor | max_drawdown | worst_day  | best_day  | bootstrap_ci_low | bootstrap_ci_high | bootstrap_p_net_le_zero |
| ---- | ------------------------------- | ------ | ------ | ---- | --------------- | ---------- | --------- | ---------- | -------- | ------------- | ------------ | ---------- | --------- | ---------------- | ----------------- | ----------------------- |
| 2024 | combined                        | 2024   | 33     | 31   | 25              | 13399.5444 | 3810.7281 | 9588.8163  | 0.8065   | 1.6546        | 9527.3718    | -7687.1696 | 5365.5279 | -13642.9925      | 29189.2615        | 0.1870                  |
| 2024 | frozen_dte_0_3_short_atm        | 2024   | 17     | 15   | 13              | 4489.1961  | 1812.1310 | 2677.0651  | 0.8667   | 1.3469        | 7687.1696    | -7687.1696 | 1378.3925 | -15502.2179      | 13464.3761        | 0.2829                  |
| 2024 | walk_forward_dte_8_14_short_atm | 2024   | 16     | 16   | 12              | 8910.3484  | 1998.5972 | 6911.7512  | 0.7500   | 1.9972        | 3691.7119    | -3691.7119 | 5365.5279 | -6963.7439       | 21418.6619        | 0.1666                  |
| 2025 | combined                        | 2025   | 13     | 11   | 7               | 6802.2611  | 1590.6521 | 5211.6090  | 0.6364   | 1.8077        | 3206.1711    | -3001.1159 | 4335.3160 | -8136.8871       | 18562.9845        | 0.2276                  |
| 2025 | frozen_dte_0_3_short_atm        | 2025   | 8      | 6    | 5               | 8573.1450  | 870.4953  | 7702.6497  | 0.8333   | 17.5365       | 465.7970     | -465.7970  | 4335.3160 | 958.0740         | 15784.5790        | 0.0064                  |
| 2025 | walk_forward_dte_8_14_short_atm | 2025   | 5      | 5    | 2               | -1770.8839 | 720.1568  | -2491.0407 | 0.4000   | 0.5839        | 5207.3278    | -3001.1159 | 3042.2448 | -11646.7965      | 6920.5224         | 0.6676                  |

## Signal coverage

| period                     | total_session_days | signal_days | profitable_signal_days | signal_coverage | equivalent_trading_months |
| -------------------------- | ------------------ | ----------- | ---------------------- | --------------- | ------------------------- |
| 2024                       | 246                | 31          | 25                     | 0.1260          | 1.4762                    |
| 2025                       | 77                 | 11          | 7                      | 0.1429          | 0.5238                    |
| combined_2024_2025_partial | 323                | 42          | 32                     | 0.1300          | 2.0000                    |

## Integrity audit

| check                             | violations | passed |
| --------------------------------- | ---------- | ------ |
| cross_day_window                  | 0          | True   |
| outside_0915_1530                 | 0          | True   |
| expiry_before_entry               | 0          | True   |
| overlapping_positions             | 0          | True   |
| pnl_identity_error_over_1e-8      | 0          | True   |
| training_label_not_strictly_prior | 0          | True   |
| duplicate_sleeve_timestamp        | 0          | True   |

## Missing-strike exit stress

| adverse_imputed_exit_penalty_fraction | imputed_trades | total_trades | imputed_trade_fraction | additional_penalty_rupees | stressed_net_pnl |
| ------------------------------------- | -------------- | ------------ | ---------------------- | ------------------------- | ---------------- |
| 0.0000                                | 14             | 46           | 0.3043                 | 0.0000                    | 14800.4253       |
| 0.0200                                | 14             | 46           | 0.3043                 | 5049.3230                 | 9751.1024        |
| 0.0500                                | 14             | 46           | 0.3043                 | 12623.3074                | 2177.1179        |
| 0.1000                                | 14             | 46           | 0.3043                 | 25246.6148                | -10446.1895      |
| 0.2000                                | 14             | 46           | 0.3043                 | 50493.2296                | -35692.8042      |

Net P&L includes historical lot sizes, brokerage, STT, exchange charges,
SEBI/IPFT charges, GST, stamp duty and the stated half-spread on entry and exit.
Fixed entry strikes are retained. When a fixed strike leaves the five-strike
exit snapshot, its exit value is reconstructed from the contemporaneous smile;
those rows are explicitly marked in the ledger. The separate stress table
penalizes only those reconstructed exit values. A 5% adverse pricing error
leaves a small positive result; a 10% error makes the strategy negative.

The historical mechanism is consistent with variance-risk-premium research,
but positive backtest P&L is not a guarantee. This repository had already
inspected 2024-2025 before the composite was specified, so the result is an
auditable causal replay, not a pristine independent discovery. Promotion
requires new forward paper data.

Sources:

- Carr and Wu, *Variance Risk Premiums*, Review of Financial Studies (2009):
  https://academic.oup.com/rfs/article/22/3/1311/1581057
- Büchner and Kelly, *Option Return Predictability with Machine Learning and
  Big Data*, Review of Financial Studies (2023):
  https://academic.oup.com/rfs/article/36/9/3548/7056660
