# Retrained short-volatility fallback and hybrid

## Research hypothesis

Option returns should be modeled from the conditional physical-versus-implied
volatility gap and state variables rather than from the sign of `IV-RV` alone.
This follows three strands of published evidence:

- [Carr and Wu (2009)](https://academic.oup.com/rfs/article-abstract/22/3/1311/1581057)
  show that option portfolios approximate risk-neutral expected
  variance and define the variance risk premium as realized versus synthetic
  option-implied variance.
- [Goyal and Saretto (2009)](https://www.ruf.rice.edu/~jgsfss/goyal_041808.pdf)
  document option-return predictability from the historical
  realized-volatility minus ATM implied-volatility difference.
- [Corsi (2009)](https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf)
  motivates multiple lagged realized-volatility horizons through HAR-RV.
- [Bali et al. (2023)](https://academic.oup.com/rfs/article/36/9/3548/7056660)
  find that option and stock characteristics can predict delta-hedged option
  returns, with nonlinear machine-learning interactions adding value.

The empirical hypothesis is therefore:

`Pr(net short-strangle P&L > 0 | RV lags, IV, skew, momentum, OI, Greeks, DTE, time)`

can identify profitable 120-minute short-volatility windows even when the
current regime hybrid is inactive.

## Training protocol

- Structure: sell ATM-2 put and ATM+2 call, same monthly expiry.
- Target: realized net P&L after the current encoded charges and 25 bps
  premium-turnover slippage.
- Train: 2019-2022.
- Validation/model and threshold selection: 2023 only.
- Frozen evaluation: 2024 and partial 2025.
- Chosen model: `logit_c0.01`.
- Entry threshold: predicted profitable-trade probability >= 0.725990.
- At most one fallback per day: choose the timestamp with the highest predicted
  probability. The fallback is used only on dates with no current-hybrid trade.

## Results

| strategy                     | period           | trades | days | net_pnl    | mean_net  | hit_rate | profit_factor | max_drawdown | p05_net    |
| ---------------------------- | ---------------- | ------ | ---- | ---------- | --------- | -------- | ------------- | ------------ | ---------- |
| current_coverage_hybrid      | development_2024 | 67     | 54   | 18739.2137 | 279.6898  | 0.5075   | 1.3753        | 9362.2720    | -2841.0144 |
| current_coverage_hybrid      | holdout_2025     | 14     | 14   | 22705.1065 | 1621.7933 | 0.7857   | 5.6870        | 4750.1826    | -2106.4446 |
| retrained_short_vol_fallback | development_2024 | 53     | 53   | 9915.9792  | 187.0939  | 0.6792   | 2.3028        | 2570.9373    | -605.3036  |
| retrained_short_vol_fallback | holdout_2025     | 13     | 13   | 3643.4234  | 280.2633  | 0.7692   | 2.9618        | 915.6532     | -669.3536  |
| combined_retrained_hybrid    | development_2024 | 100    | 87   | 21862.6695 | 218.6267  | 0.5400   | 1.3923        | 10287.0911   | -2401.8461 |
| combined_retrained_hybrid    | holdout_2025     | 23     | 23   | 23748.7967 | 1032.5564 | 0.7391   | 4.5438        | 5300.6071    | -1423.2072 |

## Trading-day coverage

| strategy                     | period           | total_available_days | signal_days | profitable_days | losing_or_flat_days | signal_day_coverage | profitable_signal_day_rate |
| ---------------------------- | ---------------- | -------------------- | ----------- | --------------- | ------------------- | ------------------- | -------------------------- |
| current_coverage_hybrid      | development_2024 | 235                  | 54          | 27              | 27                  | 0.2298              | 0.5000                     |
| current_coverage_hybrid      | holdout_2025     | 76                   | 14          | 11              | 3                   | 0.1842              | 0.7857                     |
| retrained_short_vol_fallback | development_2024 | 235                  | 53          | 36              | 17                  | 0.2255              | 0.6792                     |
| retrained_short_vol_fallback | holdout_2025     | 76                   | 13          | 10              | 3                   | 0.1711              | 0.7692                     |
| combined_retrained_hybrid    | development_2024 | 235                  | 87          | 47              | 40                  | 0.3702              | 0.5402                     |
| combined_retrained_hybrid    | holdout_2025     | 76                   | 23          | 17              | 6                   | 0.3026              | 0.7391                     |

## Entry and P&L equations

`p_t = logistic(beta_0 + beta' x_t)`.

Enter only if `p_t` exceeds the frozen 2023 threshold. For the short strangle:

`PnL = [(P_t(K-2)-P_t+h(K-2)) + (C_t(K+2)-C_t+h(K+2))] * lot - costs`.

## Limitations

The model is fitted to overlapping intraday candidate windows, although only
one daily fallback is executed. The short strangle has unbounded tail risk.
The partial 2025 sample is small, and all results use quote-based rather than
executable bid/ask prices. This remains forward-paper research.

Two hypotheses were rejected before this model was admitted. A retrained
intraday direction classifier produced only about 0.509 evaluation AUC, and the
defined-risk iron-condor model's best qualifying 2023 validation slice lost
approximately INR 13,467. Neither is included in the combined policy.
