# IV/RV transition with pre-2024 profitability gates

The user's short-vol transition remains mandatory. A trade must additionally
pass a profitability-model threshold trained through 2022 and fixed on 2023,
and forecast Gamma + Theta + Vega P&L must cover complete estimated cost.
No 2024 or 2025 result selects these thresholds.

## Components

| structure          | gate                          | probability_threshold | period       | trades | days | gross_pnl  | cost       | net_pnl     | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | profitable_days |
| ------------------ | ----------------------------- | --------------------- | ------------ | ------ | ---- | ---------- | ---------- | ----------- | --------- | -------- | ------------- | ------------ | ------------ | --------------- |
| short_2_strangle   | hypothesis_plus_pre2024_model | 0.6931                | 2024         | 50     | 50   | 1880.2500  | 6862.2500  | -4982.0000  | -99.6400  | 0.5600   | 0.6838        | -2.1314      | 9216.4232    | 28              |
| short_2_strangle   | hypothesis_plus_pre2024_model | 0.6931                | 2025_holdout | 13     | 13   | 6754.5000  | 2400.6566  | 4353.8434   | 334.9110  | 0.8462   | 2.7864        | 6.2073       | 2314.8117    | 11              |
| short_2_strangle   | hypothesis_plus_pre2024_model | 0.6931                | combined     | 63     | 63   | 8634.7500  | 9262.9066  | -628.1566   | -9.9707   | 0.6190   | 0.9655        | -0.2029      | 9216.4232    | 39              |
| short_2_strangle   | hypothesis_model_plus_cost1   | 0.6931                | 2024         | 32     | 32   | -7469.2500 | 4232.4449  | -11701.6949 | -365.6780 | 0.5000   | 0.3665        | -4.3780      | 13977.9045   | 16              |
| short_2_strangle   | hypothesis_model_plus_cost1   | 0.6931                | 2025_holdout | 9      | 9    | 3421.5000  | 1662.0337  | 1759.4663   | 195.4963  | 0.7778   | 1.7219        | 3.1102       | 2314.8117    | 7               |
| short_2_strangle   | hypothesis_model_plus_cost1   | 0.6931                | combined     | 41     | 41   | -4047.7500 | 5894.4787  | -9942.2287  | -242.4934 | 0.5610   | 0.5245        | -3.0272      | 13977.9045   | 23              |
| short_atm_straddle | hypothesis_plus_pre2024_model | 0.7016                | 2024         | 55     | 55   | 5548.5000  | 8783.5800  | -3235.0800  | -58.8196  | 0.6182   | 0.8303        | -1.0344      | 8082.2250    | 34              |
| short_atm_straddle | hypothesis_plus_pre2024_model | 0.7016                | 2025_holdout | 15     | 15   | 10431.0000 | 3321.1905  | 7109.8095   | 473.9873  | 0.8667   | 3.8178        | 8.5104       | 2452.1069    | 13              |
| short_atm_straddle | hypothesis_plus_pre2024_model | 0.7016                | combined     | 70     | 70   | 15979.5000 | 12104.7705 | 3874.7295   | 55.3533   | 0.6714   | 1.1795        | 0.9560       | 8082.2250    | 47              |
| short_atm_straddle | hypothesis_model_plus_cost1   | 0.7016                | 2024         | 35     | 35   | -3333.0000 | 5516.6267  | -8849.6267  | -252.8465 | 0.5429   | 0.5584        | -2.7867      | 13089.0314   | 19              |
| short_atm_straddle | hypothesis_model_plus_cost1   | 0.7016                | 2025_holdout | 12     | 12   | 7992.0000  | 2696.8849  | 5295.1151   | 441.2596  | 0.8333   | 3.0986        | 7.1345       | 2452.1069    | 10              |
| short_atm_straddle | hypothesis_model_plus_cost1   | 0.7016                | combined     | 47     | 47   | 4659.0000  | 8213.5115  | -3554.5115  | -75.6279  | 0.6170   | 0.8425        | -0.8809      | 13089.0314   | 29              |

## Fixed-priority hybrid

| period       | trades | days | gross_pnl  | cost      | net_pnl    | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | profitable_days |
| ------------ | ------ | ---- | ---------- | --------- | ---------- | --------- | -------- | ------------- | ------------ | ------------ | --------------- |
| 2024         | 37     | 37   | -4527.0000 | 5230.1394 | -9757.1394 | -263.7065 | 0.5405   | 0.5013        | -3.2403      | 12758.7111   | 20              |
| 2025_holdout | 12     | 12   | 7195.5000  | 2381.2258 | 4814.2742  | 401.1895  | 0.8333   | 2.9753        | 6.7485       | 2314.8117    | 10              |
| combined     | 49     | 49   | 2668.5000  | 7611.3652 | -4942.8652 | -100.8748 | 0.6122   | 0.7754        | -1.2907      | 12758.7111   | 30              |

The primary trade is the 120-minute short two-strike OTM strangle. The ATM
short straddle is used only on a date without a primary signal. All legs use
one monthly expiry and unchanged entry strikes.
