# Near-daily risk-managed short-volatility backtest

> **Invalid and superseded.** The entry rule selected the highest model score
> after observing all timestamps in the completed session. This is intraday
> look-ahead. Do not use the profit or PASS decision below. See
> `../near_daily_short_vol_causal/REPORT.md` for the corrected audit.

This test sells one same-expiry ATM straddle on each eligible date. The entry
timestamp is the day's highest score from a model trained on 2019-2022. Stop,
take-profit, score, and DTE settings are selected using 2023 only. Stops use
the first subsequently observed fixed-strike premium satisfying the rule;
there is no end-of-window look-ahead.

Decision: `PASS_BACKTEST_FORWARD_PAPER_ONLY`.

## Trading equations

The entry model estimates

`p_t = Pr(net short-straddle P&L > 0 | IV, RV lags, skew, returns, OI, Greeks, DTE, time)`.

For each day, use the candidate timestamp with maximum `p_t` and trade only
when `p_t >= 0.506969`. Position size is

`lots_t = 3 if p_t >= 0.638222, otherwise 1`.

For a fixed ATM strike and the same monthly expiry,

`net P&L = lots_t * contract_lot * [(C_entry + P_entry) - (C_exit + P_exit)] - all costs`.

Exit at the first fixed-strike quote where combined premium is at least 150%
of entry premium; otherwise exit after 180 minutes. The selected take-profit
parameter is 100%, which is effectively disabled because option premium cannot
fall below zero.

## Selected 2023 rule

| horizon_min | stop_fraction | take_fraction | score_quantile | score_threshold | dte_floor | trades | days | gross_pnl  | cost       | net_pnl    | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net   | best_trade | worst_trade | profitable_days | H1_net     | H2_net   | target_coverage | stable_positive |
| ----------- | ------------- | ------------- | -------------- | --------------- | --------- | ------ | ---- | ---------- | ---------- | ---------- | -------- | -------- | ------------- | ------------ | ------------ | ---------- | --------- | ---------- | ----------- | --------------- | ---------- | -------- | --------------- | --------------- |
| 180         | 0.5000        | 1.0000        | 0.0500         | 0.5070          | 0         | 222    | 222  | 57564.7500 | 42103.6275 | 15461.1225 | 69.6447  | 0.6171   | 1.4382        | 2.0082       | 8444.2896    | 94.2216    | -837.4835 | 2601.9711  | -1325.3875  | 137             | 15222.9452 | 238.1773 | True            | True            |

## Frozen evaluation

| period               | available_days | trades | days | gross_pnl   | cost        | net_pnl     | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | coverage | equivalent_trading_months |
| -------------------- | -------------- | ------ | ---- | ----------- | ----------- | ----------- | -------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | -------- | ------------------------- |
| test_2024            | 235            | 211    | 211  | 185339.2500 | 64215.7677  | 121123.4823 | 574.0449 | 0.6209   | 2.6050        | 3.8354       | 11738.5582   | 188.5797   | -1686.4193 | 18317.8829 | -7251.8333  | 131             | 0.8979   | 10.0476                   |
| holdout_2025_partial | 76             | 71     | 71   | 72298.5000  | 36115.3616  | 36183.1384  | 509.6217 | 0.5211   | 2.0007        | 3.4606       | 13013.7243   | 58.0392    | -1766.7166 | 7881.1369  | -5279.4363  | 37              | 0.9342   | 3.3810                    |
| combined             | 311            | 282    | 282  | 257637.7500 | 100331.1293 | 157306.6207 | 557.8249 | 0.5957   | 2.4093        | 3.7484       | 13013.7243   | 171.9724   | -1713.7742 | 18317.8829 | -7251.8333  | 168             | 0.9068   | 13.4286                   |

## One-lot benchmark

| period               | available_days | trades | days | gross_pnl   | cost       | net_pnl    | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | coverage | equivalent_trading_months |
| -------------------- | -------------- | ------ | ---- | ----------- | ---------- | ---------- | -------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | -------- | ------------------------- |
| test_2024            | 235            | 211    | 211  | 72576.7500  | 48078.8361 | 24497.9139 | 116.1039 | 0.6114   | 1.5129        | 2.0226       | 7543.5507    | 144.1793   | -1179.3109 | 6043.3610  | -4035.1052  | 129             | 0.8979   | 10.0476                   |
| holdout_2025_partial | 76             | 71     | 71   | 30799.5000  | 27051.6072 | 3747.8928  | 52.7872  | 0.5211   | 1.1418        | 0.8117       | 9622.3803    | 58.0392    | -1544.4408 | 2623.2009  | -1954.0299  | 37              | 0.9342   | 3.3810                    |
| combined             | 311            | 282    | 282  | 103376.2500 | 75130.4432 | 28245.8068 | 100.1624 | 0.5887   | 1.3807        | 1.6884       | 9622.3803    | 121.2227   | -1285.2690 | 6043.3610  | -4035.1052  | 166             | 0.9068   | 13.4286                   |

## Selected confidence sizing

| size_quantile | size_threshold | high_confidence_lots | trades | days | gross_pnl   | cost       | net_pnl    | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | H1_net     | H2_net     | stable_positive |
| ------------- | -------------- | -------------------- | ------ | ---- | ----------- | ---------- | ---------- | -------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | ---------- | ---------- | --------------- |
| 0.6000        | 0.6382         | 3                    | 222    | 222  | 124976.2500 | 52447.9781 | 72528.2719 | 326.7039 | 0.6306   | 2.3153        | 3.6761       | 12378.6077   | 160.5846   | -1151.2349 | 7994.7133  | -3788.3625  | 140             | 57510.7448 | 15017.5271 | True            |

## Exit attribution

| year | exit_reason    | trades | net_pnl     | hit_rate |
| ---- | -------------- | ------ | ----------- | -------- |
| 2024 | scheduled_exit | 210    | 128375.3156 | 0.6238   |
| 2024 | stop_loss      | 1      | -7251.8333  | 0.0000   |
| 2025 | scheduled_exit | 71     | 36183.1384  | 0.5211   |

## Integrity audit

| check                       | violations | passed |
| --------------------------- | ---------- | ------ |
| one_trade_per_day           | 0          | True   |
| cross_day_window            | 0          | True   |
| outside_0915_1530           | 0          | True   |
| holding_exceeds_180_minutes | 0          | True   |
| expiry_before_entry         | 0          | True   |

## Cost sensitivity

| slippage_bps | period               | days | net_pnl     | profit_factor | max_drawdown |
| ------------ | -------------------- | ---- | ----------- | ------------- | ------------ |
| 0.0000       | test_2024            | 211  | 151079.9842 | 3.3212        | 10106.2320   |
| 0.0000       | holdout_2025_partial | 71   | 56101.8597  | 3.0833        | 10023.1079   |
| 10.0000      | test_2024            | 211  | 139097.3834 | 3.0157        | 10759.1625   |
| 10.0000      | holdout_2025_partial | 71   | 48134.3712  | 2.5743        | 11214.7694   |
| 25.0000      | test_2024            | 211  | 121123.4823 | 2.6050        | 11738.5582   |
| 25.0000      | holdout_2025_partial | 71   | 36183.1384  | 2.0007        | 13013.7243   |
| 50.0000      | test_2024            | 211  | 91166.9804  | 2.0462        | 13370.8845   |
| 50.0000      | holdout_2025_partial | 71   | 16264.4172  | 1.3490        | 17475.5380   |
| 100.0000     | test_2024            | 211  | 31253.9767  | 1.2640        | 26849.7617   |
| 100.0000     | holdout_2025_partial | 71   | -23573.0253 | 0.6781        | 37585.0192   |

## Bootstrap uncertainty

| period               | days | observed_mean | mean_ci_low | mean_ci_high | bootstrap_probability_mean_positive |
| -------------------- | ---- | ------------- | ----------- | ------------ | ----------------------------------- |
| test_2024            | 211  | 574.0449      | 269.3620    | 913.2240     | 0.9998                              |
| holdout_2025_partial | 71   | 509.6217      | -11.3508    | 1068.0561    | 0.9724                              |

The rule is rejected if it cannot produce positive net P&L independently in
both halves of 2023 at 210-231 signal days. Mid-price paths and modeled
slippage do not guarantee executable fills. Short straddles retain gap and
tail risk even with an intraday stop.

The higher net P&L uses up to three lots and therefore requires materially
more margin and tail-risk capacity. Historical margin data is unavailable, so
the report does not claim a return-on-margin improvement from raw rupee P&L.

## Sizing exposure

| period               | one_lot_days | three_lot_days | lot_equivalents | net_pnl     | net_pnl_per_lot_equivalent |
| -------------------- | ------------ | -------------- | --------------- | ----------- | -------------------------- |
| test_2024            | 117          | 94             | 399             | 121123.4823 | 303.5676                   |
| holdout_2025_partial | 47           | 24             | 119             | 36183.1384  | 304.0600                   |

## Research basis

- Carr and Wu (2009), *Variance Risk Premiums*, motivate measuring the wedge
  between risk-neutral option-implied variance and expected physical variance.
  https://academic.oup.com/rfs/article-abstract/22/3/1311/1581057
- Goyal and Saretto (2009), *Cross-Section of Option Returns and Volatility*,
  motivate conditioning long/short option returns on realized versus implied
  volatility. https://doi.org/10.1016/j.jfineco.2009.01.001
- Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*,
  motivates using realized-volatility signals at several horizons rather than
  one lag. https://doi.org/10.1093/jjfinec/nbp001
- Bali et al. (2023), *Option Return Predictability with Machine Learning and
  Big Data*, motivate combining option and underlying characteristics while
  testing nonlinearities out of sample.
  https://academic.oup.com/rfs/article/36/9/3548/7056660
