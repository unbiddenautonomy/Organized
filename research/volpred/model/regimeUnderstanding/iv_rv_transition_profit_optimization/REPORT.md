# Cost-aware IV/RV transition strategy

## Constraint

The original hypothesis is mandatory. Long volatility still requires
`IVcurrent < RVcurrent` plus predicted IV becoming equal to or rising toward
predicted RV. Short volatility still requires `IVcurrent > RVcurrent` plus
predicted IV becoming equal to or falling toward predicted RV.

The only added entry requirement is economic sufficiency:

`forecast Greek P&L / estimated complete round-trip cost >= k`.

Forecast Greek P&L is `Gamma + Theta + Vega` for the complete structure and
the selected horizon. The realized option P&L still uses unchanged entry
strikes and observed exit prices. This gate does not redefine the transition;
it rejects transitions whose modeled edge is too small to pay execution cost.

## Selection discipline

- Search dimensions: transition subtype, optional observed five-minute IV
  confirmation, five-strike structure, 15/30/60/120 minute horizon, cost
  multiple, and minimum predicted convergence.
- A candidate needs at least 8 trading days and positive net P&L
  independently in both halves of 2024.
- Selection maximizes the weaker half's net P&L. Partial 2025 is holdout only.
- Complete costs use current encoded charges plus 25 bps
  premium-turnover slippage.

Stable profitable development candidates: 14.

## Selected rules

| side      | signal_variant | confirmation          | structure        | horizon_min | cost_multiple | convergence_floor | days_2024_H1 | days_2024_H2 | hit_rate_2024_H1 | hit_rate_2024_H2 | net_pnl_2024_H1 | net_pnl_2024_H2 | profit_factor_2024_H1 | profit_factor_2024_H2 | development_days | development_net | worst_half_net | stable_profitable |
| --------- | -------------- | --------------------- | ---------------- | ----------- | ------------- | ----------------- | ------------ | ------------ | ---------------- | ---------------- | --------------- | --------------- | --------------------- | --------------------- | ---------------- | --------------- | -------------- | ----------------- |
| short_vol | combined_or    | observed_iv_5_aligned | short_2_strangle | 120         | 3.0000        | 0.0000            | 10           | 9            | 0.7000           | 0.6667           | 5299.5680       | 511.8207        | 2.9157                | 1.3365                | 19               | 5811.3887       | 511.8207       | True              |

## Selected-rule performance

| side      | signal_variant | confirmation          | structure        | horizon_min | cost_multiple | convergence_floor | period       | days | profitable_days | net_pnl   | hit_rate | profit_factor | max_drawdown |
| --------- | -------------- | --------------------- | ---------------- | ----------- | ------------- | ----------------- | ------------ | ---- | --------------- | --------- | -------- | ------------- | ------------ |
| short_vol | combined_or    | observed_iv_5_aligned | short_2_strangle | 120         | 3.0000        | 0.0000            | 2024_H1      | 10   | 7               | 5299.5680 | 0.7000   | 2.9157        | 1875.9568    |
| short_vol | combined_or    | observed_iv_5_aligned | short_2_strangle | 120         | 3.0000        | 0.0000            | 2024_H2      | 9    | 6               | 511.8207  | 0.6667   | 1.3365        | 1521.0146    |
| short_vol | combined_or    | observed_iv_5_aligned | short_2_strangle | 120         | 3.0000        | 0.0000            | 2025_holdout | 18   | 13              | 7882.7767 | 0.7222   | 1.6558        | 9731.5439    |

## Combined hybrid

| period       | trades | days | gross_pnl  | cost      | net_pnl    | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | profitable_days |
| ------------ | ------ | ---- | ---------- | --------- | ---------- | -------- | -------- | ------------- | ------------ | ------------ | --------------- |
| 2024_H1      | 10     | 10   | 7195.5000  | 1895.9320 | 5299.5680  | 529.9568 | 0.7000   | 2.9157        | 7.0302       | 1875.9568    | 7               |
| 2024_H2      | 9      | 9    | 1874.2500  | 1362.4293 | 511.8207   | 56.8690  | 0.6667   | 1.3365        | 1.6274       | 1521.0146    | 6               |
| 2025_holdout | 18     | 18   | 13404.0000 | 5521.2233 | 7882.7767  | 437.9320 | 0.7222   | 1.6558        | 2.8174       | 9731.5439    | 13              |
| combined     | 37     | 37   | 22473.7500 | 8779.5846 | 13694.1654 | 370.1126 | 0.7027   | 1.8398        | 3.2170       | 9731.5439    | 26              |

## Cost sensitivity

| slippage_bps | period       | days | net_pnl    | hit_rate | profit_factor | max_drawdown |
| ------------ | ------------ | ---- | ---------- | -------- | ------------- | ------------ |
| 0.0000       | 2024_H1      | 10   | 5941.7555  | 0.7000   | 3.3084        | 1706.8730    |
| 0.0000       | 2024_H2      | 9    | 859.1776   | 0.7778   | 1.6087        | 1411.4281    |
| 0.0000       | 2025_holdout | 18   | 10469.4667 | 0.7778   | 1.9332        | 9130.5126    |
| 10.0000      | 2024_H1      | 10   | 5684.8805  | 0.7000   | 3.1445        | 1774.5065    |
| 10.0000      | 2024_H2      | 9    | 720.2348   | 0.6667   | 1.4993        | 1442.5515    |
| 10.0000      | 2025_holdout | 18   | 9434.7907  | 0.7778   | 1.8185        | 9370.9251    |
| 25.0000      | 2024_H1      | 10   | 5299.5680  | 0.7000   | 2.9157        | 1875.9568    |
| 25.0000      | 2024_H2      | 9    | 511.8207   | 0.6667   | 1.3365        | 1521.0146    |
| 25.0000      | 2025_holdout | 18   | 7882.7767  | 0.7222   | 1.6558        | 9731.5439    |
| 50.0000      | 2024_H1      | 10   | 4657.3805  | 0.7000   | 2.5741        | 2045.0405    |
| 50.0000      | 2024_H2      | 9    | 164.4638   | 0.5556   | 1.0971        | 1651.7865    |
| 50.0000      | 2025_holdout | 18   | 5296.0867  | 0.7222   | 1.4087        | 10332.5751   |
| 100.0000     | 2024_H1      | 10   | 3373.0055  | 0.7000   | 2.0088        | 2383.2080    |
| 100.0000     | 2024_H2      | 9    | -530.2499  | 0.4444   | 0.7615        | 1913.3303    |
| 100.0000     | 2025_holdout | 18   | 122.7067   | 0.6667   | 1.0081        | 11534.6376   |

## Sampling uncertainty

| period       | days | observed_mean | mean_ci_low | mean_ci_high | bootstrap_probability_mean_positive |
| ------------ | ---- | ------------- | ----------- | ------------ | ----------------------------------- |
| 2024_H1      | 10   | 529.9568      | -212.8432   | 1211.4134    | 0.9234                              |
| 2024_H2      | 9    | 56.8690       | -309.0931   | 366.0570     | 0.6474                              |
| 2025_holdout | 18   | 437.9320      | -665.8851   | 1544.1079    | 0.7988                              |

## Integrity checks

| check                   | violations | passed |
| ----------------------- | ---------- | ------ |
| one_trade_per_day       | 0          | True   |
| cross_day_window        | 0          | True   |
| outside_0915_1530       | 0          | True   |
| holding_period_mismatch | 0          | True   |
| expiry_before_entry     | 0          | True   |

## Research basis

The economic object is a variance-risk-premium comparison: expected future
realized variance versus option-implied variance, with the IV forecast used to
model vega P&L. Carr and Wu (2009) formalize variance risk premium as the
difference between realized variance and an option-implied variance-swap rate.
The present implementation is only an approximation because it uses a
five-strike monthly-expiry structure rather than a model-free variance strip.
The full-cost inequality is necessary because a statistical volatility edge
is not automatically an executable option profit.

## Interpretation

Passing 2024 is necessary but not sufficient. Deployment requires positive
2025 holdout performance and subsequent forward paper trading. A rule that
only compensates costs in development but fails holdout remains rejected.
The bootstrap confidence intervals include zero, so the observed positive P&L
does not yet establish statistical significance.
