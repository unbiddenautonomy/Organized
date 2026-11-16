# Hybrid profit optimization

## Frozen improved rule

- Trade only when `IV < predicted RV`.
- If direction is predicted down and expected Greek P&L is positive after
  costs, buy the ATM put.
- The 120-minute short call at ATM+2 and short put at ATM-2 remains eligible
  when its expected Greek P&L is positive after costs.
- Do not take the long-call stream.
- Require at least 4 calendar days to monthly expiry at entry.
- If the put and strangle arrive together, take the one with the larger
  expected net rupee edge.
- Use one lot, one open trade, fixed entry strikes, a 120-minute holding period,
  and enter no later than 13:30.

The rule was selected using 2024 only. The search maximized net P&L subject to
at least 30 development trades and maximum drawdown no
worse than the original hybrid. It searched only four source combinations and
DTE floors (0, 4, 8, 15); it did not tune on 2025.

## Baseline versus optimized

| rule                         | period           | trades | net_pnl    | mean_net  | median_net | hit_rate | profit_factor | p05_net    | max_drawdown |
| ---------------------------- | ---------------- | ------ | ---------- | --------- | ---------- | -------- | ------------- | ---------- | ------------ |
| baseline_conservative_hybrid | development_2024 | 94     | 4626.2233  | 49.2151   | -242.0149  | 0.4362   | 1.0600        | -2849.6815 | 13243.1003   |
| baseline_conservative_hybrid | holdout_2025     | 15     | 13123.4892 | 874.8993  | 856.2677   | 0.6000   | 2.1998        | -3556.0088 | 10843.7928   |
| profit_optimized_hybrid      | development_2024 | 46     | 15299.7619 | 332.6035  | 291.7743   | 0.5435   | 1.3988        | -3379.1100 | 8092.7635    |
| profit_optimized_hybrid      | holdout_2025     | 10     | 16013.6876 | 1601.3688 | 1221.9978  | 0.7000   | 4.3057        | -2464.6402 | 4750.1826    |

## Leading eligible 2024 candidates

| source_rule            | dte_floor | trades | net_pnl    | mean_net | hit_rate | profit_factor | max_drawdown |
| ---------------------- | --------- | ------ | ---------- | -------- | -------- | ------------- | ------------ |
| put_and_short_strangle | 4         | 46     | 15299.7619 | 332.6035 | 0.5435   | 1.3988        | 8092.7635    |
| put_only               | 4         | 37     | 14017.2988 | 378.8459 | 0.5135   | 1.4186        | 7435.7257    |
| put_and_short_strangle | 0         | 58     | 10614.3708 | 183.0064 | 0.5000   | 1.2241        | 11974.7872   |
| put_only               | 0         | 45     | 9345.1698  | 207.6704 | 0.4444   | 1.2337        | 10601.6846   |
| all_validated_streams  | 4         | 66     | 7710.9421  | 116.8325 | 0.4848   | 1.1307        | 11989.7798   |
| put_only               | 8         | 30     | 6492.0842  | 216.4028 | 0.5333   | 1.2254        | 7008.6824    |
| put_and_short_strangle | 8         | 38     | 6061.9507  | 159.5250 | 0.5526   | 1.1800        | 7008.6824    |
| all_validated_streams  | 0         | 94     | 4626.2233  | 49.2151  | 0.4362   | 1.0600        | 13243.1003   |

## Interpretation

The profit increase does not use additional leverage. It comes from removing
the historically losing call stream and avoiding the most expiry-sensitive
0-3 DTE observations. The 2025 holdout contains only ten optimized trades, so
the improvement remains a forward-paper candidate rather than a live strategy.
No IV>RV rule was added because doing so made the 2024 hybrid materially
negative in the prior test.
