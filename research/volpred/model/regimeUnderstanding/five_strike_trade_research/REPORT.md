# Five-strike IV/RV structure research

## Decision

The expanded search changes the `IV < predicted RV` conclusion. The preferred
forward-paper candidate is a **120-minute short two-strike OTM strangle**:

- sell the put two strikes below entry ATM;
- sell the call two strikes above entry ATM;
- use the same monthly expiry for both legs;
- retain both original strikes through exit;
- enter no later than 13:30 and exit after 120 minutes;
- enter only when the predicted Greek P&L, including the separate IV-change
  forecast, exceeds estimated complete costs.

This is not a contradiction. `IV < predicted RV` makes the short position
negative-gamma, but the selected observations also forecast a sufficiently
large IV decline. For the selected short-two-strangle sample, mean expected
gamma, theta and vega contributions were approximately -28.44, +10.63 and
+60.16 points respectively before costs. Positive forecast vega P&L outweighed
negative gamma.

The `IV = predicted RV` scene remains no-trade: no straddle, strangle, iron fly,
iron condor or reverse structure passed the same development/holdout rule, even
after DTE and time-of-day subdivision. `IV > predicted RV` also produced no
cell passing this stricter complete-cost rule.

Selected cell:

| scene   | structure        | horizon_min | robust_positive | trades_development_2024 | mean_net_development_2024 | hit_rate_development_2024 | trades_holdout_2025 | mean_net_holdout_2025 | hit_rate_holdout_2025 |
| ------- | ---------------- | ----------- | --------------- | ----------------------- | ------------------------- | ------------------------- | ------------------- | --------------------- | --------------------- |
| IV < RV | short_2_strangle | 120         | True            | 25                      | 111.9796                  | 0.6000                    | 11                  | 431.7371              | 0.8182                |

## Scope

- Source structure universe: ATM, one strike above/below, and two strikes
  above/below ATM from the five-strike dataset.
- All legs use the same monthly expiry and retain their entry strikes.
- Horizons: 15, 30, 60, and 120 minutes, with exits no later than 15:30.
- IV/RV scenes use the timestamp-available RV forecast, not future RV.
- `IV = RV` is the +/-0.5-vol-point band.
- Primary execution stress: current charge schedule encoded by the repository
  plus 25 bps of premium-turnover slippage.
- Trades are non-overlapping within each structure, horizon, and scene.

## Structures

- Straddles use the ATM call and put.
- Strangles use symmetric +/-1 or +/-2 OTM legs.
- Iron flies buy/sell the ATM straddle and take the opposite position in
  symmetric wings.
- Iron condors use +/-1 short legs and +/-2 protective wings; reverse iron
  condors invert every leg.
- Directional vertical spreads and risk reversals are excluded because no
  signed spot-return forecast is used.

## Entry equation

For leg weights `w_i`, all at the same expiry,

`E[PnL] = sum_i w_i * (0.5*Gamma_i*S^2*vhat_h + Theta_i*h/1440 + Vega_i*dIV_hat) - costs`.

`vhat_h` is cumulative forecast variance over the horizon. It is annualized
only for the IV/RV scene comparison as
`predicted_RV = sqrt(vhat_h * 252 * 375 / h)`.

The Greek-edge gate requires this expected net P&L to be positive before an
entry is eligible. Exact exit P&L uses the unchanged strikes:

`PnL = sum_i w_i * (OptionPrice_i(t+h) - OptionPrice_i(t)) * lot - costs`.

Vanna, volga, charm, higher-order terms and forecast error remain in realized
P&L even though they are not used as independently forecast entry terms.

## Robustness rule

A cell passes only if it has at least 20 non-overlapping 2024 trades, at least
10 non-overlapping 2025 holdout trades, and positive mean net P&L in both
periods after the primary cost stress.

## Passing cells

| scene   | structure        | horizon_min | robust_positive | trades_development_2024 | mean_net_development_2024 | hit_rate_development_2024 | trades_holdout_2025 | mean_net_holdout_2025 | hit_rate_holdout_2025 |
| ------- | ---------------- | ----------- | --------------- | ----------------------- | ------------------------- | ------------------------- | ------------------- | --------------------- | --------------------- |
| IV < RV | short_2_strangle | 120         | True            | 25                      | 111.9796                  | 0.6000                    | 11                  | 431.7371              | 0.8182                |
| IV < RV | short_1_strangle | 120         | True            | 21                      | 207.9927                  | 0.6190                    | 10                  | 331.2775              | 0.7000                |

## Leading cells by scene

| scene   | structure               | horizon_min | robust_positive | trades_development_2024 | mean_net_development_2024 | hit_rate_development_2024 | trades_holdout_2025 | mean_net_holdout_2025 | hit_rate_holdout_2025 |
| ------- | ----------------------- | ----------- | --------------- | ----------------------- | ------------------------- | ------------------------- | ------------------- | --------------------- | --------------------- |
| IV < RV | short_2_strangle        | 120         | True            | 25                      | 111.9796                  | 0.6000                    | 11                  | 431.7371              | 0.8182                |
| IV < RV | short_1_strangle        | 120         | True            | 21                      | 207.9927                  | 0.6190                    | 10                  | 331.2775              | 0.7000                |
| IV < RV | short_atm_straddle      | 120         | False           | 21                      | -64.6864                  | 0.5238                    | 9                   | 487.8499              | 0.7778                |
| IV < RV | short_1_strangle        | 60          | False           | 16                      | -365.6741                 | 0.3125                    | 16                  | 183.5747              | 0.6250                |
| IV < RV | short_2_strangle        | 60          | False           | 23                      | -202.4262                 | 0.4348                    | 16                  | 137.8949              | 0.6250                |
| IV < RV | short_atm_straddle      | 60          | False           | 15                      | -480.5063                 | 0.4000                    | 14                  | -55.2010              | 0.5000                |
| IV < RV | reverse_iron_condor_1_2 | 30          | False           | 0                       |                           |                           | 1                   | -57.6095              | 0.0000                |
| IV < RV | reverse_iron_fly_2      | 60          | False           | 2                       | -466.0191                 | 0.0000                    | 1                   | -72.8409              | 0.0000                |
| IV = RV | short_atm_straddle      | 120         | False           | 37                      | -775.3899                 | 0.4865                    | 16                  | 613.8162              | 0.6250                |
| IV = RV | short_1_strangle        | 15          | False           | 3                       | 53.3653                   | 0.6667                    | 1                   | 466.9420              | 1.0000                |
| IV = RV | short_1_strangle        | 120         | False           | 37                      | -623.6051                 | 0.5676                    | 16                  | 323.3853              | 0.5625                |
| IV = RV | short_atm_straddle      | 15          | False           | 3                       | -385.5036                 | 0.3333                    | 1                   | 279.9761              | 1.0000                |
| IV = RV | short_2_strangle        | 120         | False           | 38                      | -473.6825                 | 0.6316                    | 15                  | 203.9237              | 0.6000                |
| IV = RV | short_2_strangle        | 15          | False           | 2                       | 74.5798                   | 0.5000                    | 2                   | 143.7837              | 1.0000                |
| IV = RV | short_1_strangle        | 60          | False           | 23                      | -818.2919                 | 0.3913                    | 21                  | 41.7287               | 0.6190                |
| IV = RV | short_atm_straddle      | 60          | False           | 17                      | -317.8672                 | 0.4706                    | 18                  | 14.5854               | 0.5556                |
| IV > RV | short_1_strangle        | 120         | False           | 192                     | -231.0335                 | 0.5052                    | 106                 | 33.8626               | 0.5849                |
| IV > RV | short_atm_straddle      | 30          | False           | 135                     | -182.1528                 | 0.4074                    | 78                  | 32.6338               | 0.4615                |
| IV > RV | short_2_strangle        | 120         | False           | 199                     | -200.5779                 | 0.5075                    | 107                 | 20.3738               | 0.5794                |
| IV > RV | short_1_strangle        | 30          | False           | 148                     | -258.4470                 | 0.3986                    | 83                  | 9.1398                | 0.4578                |
| IV > RV | short_atm_straddle      | 120         | False           | 185                     | -268.5039                 | 0.5189                    | 104                 | 8.3551                | 0.5769                |
| IV > RV | short_2_strangle        | 30          | False           | 152                     | -202.3052                 | 0.4079                    | 93                  | -7.2876               | 0.4409                |
| IV > RV | short_atm_straddle      | 15          | False           | 34                      | -166.5561                 | 0.2353                    | 20                  | -16.0587              | 0.5000                |
| IV > RV | short_1_strangle        | 15          | False           | 39                      | -402.6996                 | 0.1795                    | 27                  | -31.7096              | 0.5185                |

## Guardrails

This is a historical research screen, not an instruction to place a live
order. A positive cell still requires tail-risk limits, independent forward
paper trading, executable bid/ask data, and a single portfolio-level
non-overlap rule across competing structures.

Coverage: 961,606 clean structure-window candidates from
2024-01-03 15:05:59 through 2025-05-09 15:09:59.
