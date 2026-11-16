# Leakage-controlled validation backtest

Status: `REJECT_FAILED_UNTOUCHED_VALIDATION`.

The classifier is trained only on 2019-2022. The probability threshold is
selected only on January-June 2023 subject to at least
80% signal coverage. July-December 2023 is then
evaluated once as the untouched validation period. The validation result does
not participate in model, horizon, structure, threshold, sizing, stop, or
take-profit selection.

Every trade is a one-lot, same-monthly-expiry ATM short straddle. It enters at
the first live-observable probability crossing and exits exactly 180 minutes
later within the same NSE session. Net P&L is:

`net P&L = short option premium received - option premium repurchased - fees - slippage`.

## Performance

| period              | total_session_days | model_candidate_days | trades | days | gross_pnl  | cost       | net_pnl     | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | signal_coverage_of_sessions | signal_coverage_of_candidates | equivalent_trading_months |
| ------------------- | ------------------ | -------------------- | ------ | ---- | ---------- | ---------- | ----------- | --------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | --------------------------- | ----------------------------- | ------------------------- |
| calibration_2023_H1 | 120                | 117                  | 117    | 117  | 15339.0000 | 23305.2960 | -7966.2960  | -68.0880  | 0.4615   | 0.7174        | -1.7068      | 13798.0765   | -31.4206   | -1137.4000 | 2360.8265  | -2521.6710  | 54              | 0.9750                      | 1.0000                        | 5.5714                    |
| validation_2023_H2  | 122                | 118                  | 118    | 118  | -4225.5000 | 21975.5749 | -26201.0749 | -222.0430 | 0.4661   | 0.3242        | -5.4792      | 25921.6049   | -39.6850   | -1691.1856 | 896.6021   | -2611.2197  | 55              | 0.9672                      | 1.0000                        | 5.6190                    |

## Frozen threshold

Selected probability threshold: `0.324333406875`.

| probability_quantile | probability_threshold | candidate_days | coverage | trades | days | gross_pnl  | cost       | net_pnl     | mean_net | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days | selected |
| -------------------- | --------------------- | -------------- | -------- | ------ | ---- | ---------- | ---------- | ----------- | -------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- | -------- |
| 0.0000               | 0.3243                | 117            | 1.0000   | 117    | 117  | 15339.0000 | 23305.2960 | -7966.2960  | -68.0880 | 0.4615   | 0.7174        | -1.7068      | 13798.0765   | -31.4206   | -1137.4000 | 2360.8265  | -2521.6710  | 54              | True     |
| 0.1000               | 0.4037                | 117            | 1.0000   | 117    | 117  | 13816.5000 | 23284.7008 | -9468.2008  | -80.9248 | 0.4615   | 0.6650        | -2.0923      | 13362.0931   | -40.8702   | -1137.4000 | 2360.8265  | -2521.6710  | 54              | False    |
| 0.2000               | 0.4403                | 117            | 1.0000   | 117    | 117  | 14347.5000 | 23270.2271 | -8922.7271  | -76.2626 | 0.4615   | 0.6682        | -2.0392      | 12192.7605   | -40.8702   | -1117.0866 | 2360.8265  | -2108.2849  | 54              | False    |
| 0.3000               | 0.4672                | 117            | 0.9915   | 116    | 116  | 13473.7500 | 23017.1147 | -9543.3647  | -82.2704 | 0.4310   | 0.6452        | -2.1503      | 13133.0825   | -43.7895   | -1118.2961 | 2360.8265  | -2671.3163  | 50              | False    |
| 0.4000               | 0.4920                | 117            | 0.9829   | 115    | 115  | 12030.0000 | 22767.7409 | -10737.7409 | -93.3717 | 0.4435   | 0.6163        | -2.1507      | 15790.9559   | -41.1774   | -887.9700  | 2360.8265  | -4572.5326  | 51              | False    |
| 0.5000               | 0.5131                | 117            | 0.9316   | 109    | 109  | 12202.5000 | 21382.7965 | -9180.2965  | -84.2229 | 0.4679   | 0.6718        | -2.0759      | 14954.8590   | -46.7087   | -1184.0894 | 2360.8265  | -2362.9707  | 51              | False    |
| 0.6000               | 0.5428                | 117            | 0.7350   | 86     | 86   | 7898.2500  | 16057.2889 | -8159.0389  | -94.8725 | 0.5233   | 0.6588        | -2.1895      | 12425.5796   | 19.9463    | -1504.6763 | 2360.8265  | -2362.9707  | 45              | False    |
| 0.7000               | 0.5885                | 117            | 0.4872   | 57     | 57   | 10452.0000 | 9607.3447  | 844.6553    | 14.8185  | 0.6316   | 1.0661        | 0.3052       | 8120.7406    | 148.3115   | -1468.8576 | 2360.8265  | -3322.0489  | 36              | False    |
| 0.8000               | 0.6609                | 117            | 0.2821   | 33     | 33   | 7848.7500  | 4781.5978  | 3067.1522   | 92.9440  | 0.6364   | 1.5127        | 2.2768       | 1551.1058    | 127.9480   | -1047.2078 | 2360.8265  | -1264.4326  | 21              | False    |
| 0.9000               | 0.8001                | 117            | 0.1368   | 16     | 16   | 8761.5000  | 2053.7614  | 6707.7386   | 419.2337 | 0.8125   | 6.9511        | 10.1636      | 306.2184     | 326.3522   | -376.1774  | 2360.8265  | -586.0545   | 13              | False    |

## Post-validation sensitivity (diagnostic only)

These rows apply every calibration-defined threshold to the untouched period
after the primary rule was frozen. They do not change the selected policy.

| calibration_probability_quantile | frozen_probability_threshold | was_selected_before_validation | trades | days | gross_pnl  | cost       | net_pnl     | mean_net  | hit_rate | profit_factor | daily_sharpe | max_drawdown | median_net | p05_net    | best_trade | worst_trade | profitable_days |
| -------------------------------- | ---------------------------- | ------------------------------ | ------ | ---- | ---------- | ---------- | ----------- | --------- | -------- | ------------- | ------------ | ------------ | ---------- | ---------- | ---------- | ----------- | --------------- |
| 0.0000                           | 0.3243                       | True                           | 118    | 118  | -4225.5000 | 21975.5749 | -26201.0749 | -222.0430 | 0.4661   | 0.3242        | -5.4792      | 25921.6049   | -39.6850   | -1691.1856 | 896.6021   | -2611.2197  | 55              |
| 0.1000                           | 0.4037                       | False                          | 118    | 118  | -5452.5000 | 21972.8443 | -27425.3443 | -232.4182 | 0.4576   | 0.3134        | -5.6447      | 27145.8742   | -48.6967   | -1691.1856 | 896.6021   | -2611.2197  | 54              |
| 0.2000                           | 0.4403                       | False                          | 118    | 118  | -3016.5000 | 21966.6654 | -24983.1654 | -211.7217 | 0.4746   | 0.3479        | -5.2174      | 24703.6953   | -20.9898   | -1627.8216 | 896.6021   | -2611.2197  | 56              |
| 0.3000                           | 0.4672                       | False                          | 118    | 118  | -5624.2500 | 21972.0966 | -27596.3466 | -233.8673 | 0.5085   | 0.3353        | -4.8577      | 27393.5414   | 7.9453     | -1954.5080 | 896.6021   | -4563.7084  | 60              |
| 0.4000                           | 0.4920                       | False                          | 117    | 117  | 140.2500   | 21723.9525 | -21583.7025 | -184.4761 | 0.5043   | 0.4070        | -4.4614      | 22013.5823   | 1.0597     | -1675.5812 | 793.9638   | -2611.2197  | 59              |
| 0.5000                           | 0.5131                       | False                          | 111    | 111  | -670.5000  | 20398.6353 | -21069.1353 | -189.8120 | 0.4955   | 0.4082        | -4.4968      | 22573.1448   | -8.4821    | -1771.6966 | 835.7184   | -2611.2197  | 55              |
| 0.6000                           | 0.5428                       | False                          | 97     | 97   | -3306.0000 | 17278.8114 | -20584.8114 | -212.2146 | 0.4845   | 0.3651        | -4.9384      | 20885.9372   | -8.4821    | -1959.8127 | 793.9638   | -2611.2197  | 47              |
| 0.7000                           | 0.5885                       | False                          | 75     | 75   | -9264.7500 | 12673.6352 | -21938.3852 | -292.5118 | 0.4800   | 0.3146        | -5.2864      | 22871.3514   | -38.1498   | -2261.5268 | 793.9638   | -3776.7559  | 36              |
| 0.8000                           | 0.6609                       | False                          | 41     | 41   | -7338.7500 | 6143.0726  | -13481.8226 | -328.8249 | 0.5610   | 0.3419        | -5.1771      | 13582.2078   | 41.2401    | -2277.1936 | 1183.6969  | -3485.9303  | 23              |
| 0.9000                           | 0.8001                       | False                          | 19     | 19   | -9618.0000 | 2516.9511  | -12134.9511 | -638.6816 | 0.3684   | 0.1821        | -8.4409      | 13488.9341   | -218.6318  | -2672.4036 | 793.9638   | -3223.0586  | 7               |

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
| exit_quality_used_as_entry_gate | 0          | True   |

Passing requires positive calibration P&L, positive untouched-validation P&L,
validation profit factor above one, and zero integrity violations. A pass is
still research evidence only; quote data do not prove executable fills.
