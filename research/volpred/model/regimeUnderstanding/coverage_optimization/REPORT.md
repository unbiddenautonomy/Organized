# Hybrid trading-day coverage optimization

## Frozen higher-coverage rule

Trade only in `IV < predicted RV`, with one monthly-expiry position at a time:

- buy the ATM put when the downside forecast and expected-P&L gate agree, but
  only with at least 4 calendar days to expiry;
- sell the ATM-2 put and ATM+2 call when the short-strangle expected-P&L gate
  is positive, including during 0-3 DTE;
- buy the ATM call only during 0-3 DTE when its directional expected-P&L gate
  is positive;
- if signals coincide, choose the highest expected net rupee edge;
- keep entry strikes fixed, hold 120 minutes, and enter no later than 13:30.

The search used 2024 only. A rule needed at least 50 distinct
development trading days, positive net P&L, and drawdown no worse than the
original conservative hybrid. Position size remained one lot.

## Coverage and P&L

| rule                      | period           | trades | days | net_pnl    | mean_net  | hit_rate | profit_factor | max_drawdown | p05_net    |
| ------------------------- | ---------------- | ------ | ---- | ---------- | --------- | -------- | ------------- | ------------ | ---------- |
| profit_optimized_baseline | development_2024 | 46     | 38   | 15299.7619 | 332.6035  | 0.5435   | 1.3988        | 8092.7635    | -3379.1100 |
| profit_optimized_baseline | holdout_2025     | 10     | 10   | 16013.6876 | 1601.3688 | 0.7000   | 4.3057        | 4750.1826    | -2464.6402 |
| coverage_optimized_hybrid | development_2024 | 67     | 54   | 18739.2137 | 279.6898  | 0.5075   | 1.3753        | 9362.2720    | -2841.0144 |
| coverage_optimized_hybrid | holdout_2025     | 14     | 14   | 22705.1065 | 1621.7933 | 0.7857   | 5.6870        | 4750.1826    | -2106.4446 |

## Leading eligible development rules

| call_mode        | put_dte_floor | short_dte_floor | trades | days | net_pnl    | hit_rate | profit_factor | max_drawdown |
| ---------------- | ------------- | --------------- | ------ | ---- | ---------- | -------- | ------------- | ------------ |
| near_expiry_only | 4             | 0               | 67     | 54   | 18739.2137 | 0.5075   | 1.3753        | 9362.2720    |
| near_expiry_only | 4             | 4               | 66     | 53   | 16887.1721 | 0.5000   | 1.3382        | 9362.2720    |
| near_expiry_only | 4             | 15              | 63     | 51   | 16500.3694 | 0.4921   | 1.3421        | 9804.6693    |
| near_expiry_only | 4             | 8               | 65     | 52   | 15174.5755 | 0.4923   | 1.3039        | 9804.6693    |
| near_expiry_only | 0             | 0               | 74     | 59   | 12215.0431 | 0.4595   | 1.2164        | 12300.4613   |
| near_expiry_only | 0             | 4               | 74     | 59   | 12215.0431 | 0.4595   | 1.2164        | 12300.4613   |
| near_expiry_only | 0             | 15              | 71     | 57   | 11828.2404 | 0.4507   | 1.2161        | 13099.5137   |
| all              | 4             | 999             | 81     | 67   | 11436.9899 | 0.4568   | 1.1714        | 10533.4561   |
| all              | 4             | 0               | 87     | 71   | 11150.3939 | 0.4713   | 1.1580        | 10297.7533   |
| near_expiry_only | 8             | 0               | 63     | 50   | 10967.0251 | 0.5079   | 1.2375        | 9644.7912    |

## Limitations

The near-expiry call component added development-period days but generated no
eligible 2025 holdout call trades. Its contribution is therefore not validated
yet. The 2025 sample runs only through the available May data. Forcing trades
in IV=RV or IV>RV was not used because it reduced historical robustness.
