# Directional call, put and debit-spread test

## Decision

At least one forecast-driven configuration passed the stated robustness screen. It remains a paper-trading candidate because the direction model and execution assumptions require live validation.

The defined-risk variants are:

- bullish: buy the entry ATM call and sell the call one or two strikes above ATM;
- bearish: buy the entry ATM put and sell the put one or two strikes below ATM.

The farther option is sold, not bought. It reduces the initial debit and caps
maximum profit. Both the naked long option and the debit spread already have a
maximum loss equal to the debit plus costs. The `bought_safety_1` and
`bought_safety_2` families test the user's literal bought-safety alternative:
an ATM call plus a one- or two-strike OTM put for a bullish signal, and an ATM
put plus a one- or two-strike OTM call for a bearish signal. This adds reversal
protection but also adds premium and negative theta.

## P&L equations

For a bull call spread with lower strike `K0` and upper strike `K1`:

`PnL_T = [(C_T(K0)-C_0(K0)) - (C_T(K1)-C_0(K1))] * lot - costs`.

At expiry, before costs: `max(S_T-K0,0) - max(S_T-K1,0) - net_debit`.
Maximum loss is `net_debit`; maximum profit is `(K1-K0)-net_debit`.

For a bear put spread with upper strike `K0` and lower strike `K-1`:

`PnL_T = [(P_T(K0)-P_0(K0)) - (P_T(K-1)-P_0(K-1))] * lot - costs`.

At expiry, before costs: `max(K0-S_T,0) - max(K-1-S_T,0) - net_debit`.
Maximum loss is `net_debit`; maximum profit is `(K0-K-1)-net_debit`.

The timestamp entry estimate is:

`E[PnL] = Delta*dS_hat + 0.5*Gamma*S^2*vhat_h + Theta*h/1440 + Vega*dIV_hat - costs`.

Here `dS_hat = S*(exp(rhat_h)-1)`. `vhat_h` is cumulative forecast variance,
and `dIV_hat` is the separate IV-change forecast.

## Direction tests

- `oracle_sign` uses the realized future sign only. It is a structural ceiling,
  not a tradable result.
- `forecast_sign` uses a Ridge signed-return forecast. Alpha is selected on
  2023, then the model is refit monthly using only the preceding 60 sessions.
- `sign_only` trades every nonzero signal. `sign_plus_expected_edge` also
  requires forecast Greek P&L to exceed complete estimated costs.
- Results are shown for all timestamps and separately for `IV < predicted RV`,
  `IV = predicted RV`, and `IV > predicted RV`; equality is the same
  +/-0.5-vol-point band used in the regime study.
- Costs use the repository's current charge schedule plus 25
  bps slippage on premium turnover.
- All legs use one monthly expiry, strikes stay fixed, and each 15
  to 120 minute exit remains inside 09:15-15:30.

## Direction-model diagnostics

| horizon_min | year | observations | alpha_selected_on_2023 | validation_mse | r2_vs_zero | direction_hit_rate | return_correlation | mae_return |
| ----------- | ---- | ------------ | ---------------------- | -------------- | ---------- | ------------------ | ------------------ | ---------- |
| 15          | 2024 | 40500        | 1000.0000              | 0.0000         | -0.0200    | 0.5048             | -0.0247            | 0.0011     |
| 15          | 2025 | 13359        | 1000.0000              | 0.0000         | -0.0140    | 0.4852             | -0.0390            | 0.0012     |
| 30          | 2024 | 38707        | 1000.0000              | 0.0000         | -0.0329    | 0.4971             | -0.0364            | 0.0016     |
| 30          | 2025 | 12786        | 1000.0000              | 0.0000         | -0.0144    | 0.4827             | -0.0328            | 0.0017     |
| 60          | 2024 | 35318        | 1000.0000              | 0.0000         | -0.0533    | 0.5040             | -0.0473            | 0.0023     |
| 60          | 2025 | 11739        | 1000.0000              | 0.0000         | -0.0189    | 0.4996             | -0.0143            | 0.0023     |
| 120         | 2024 | 28599        | 1000.0000              | 0.0000         | -0.0774    | 0.4925             | -0.0404            | 0.0033     |
| 120         | 2025 | 9506         | 1000.0000              | 0.0000         | -0.0297    | 0.5110             | 0.0015             | 0.0032     |

## Configurations passing the robustness rule

| scene   | family    | horizon_min | signal_mode   | gate                    | robust_positive | trades_development_2024 | mean_net_development_2024 | hit_rate_development_2024 | trades_holdout_2025 | mean_net_holdout_2025 | hit_rate_holdout_2025 | p05_net_holdout_2025 |
| ------- | --------- | ----------- | ------------- | ----------------------- | --------------- | ----------------------- | ------------------------- | ------------------------- | ------------------- | --------------------- | --------------------- | -------------------- |
| IV < RV | naked_atm | 120         | forecast_sign | sign_plus_expected_edge | True            | 89                      | 76.0097                   | 0.4270                    | 10                  | 571.7520              | 0.4000                | -3793.8657           |
| IV < RV | naked_atm | 120         | forecast_sign | sign_only               | True            | 96                      | 68.1318                   | 0.4167                    | 15                  | 142.7143              | 0.4667                | -4412.3014           |
| ALL     | naked_atm | 120         | oracle_sign   | sign_only               | True            | 497                     | 1072.3264                 | 0.7626                    | 159                 | 2096.8319             | 0.7736                | -887.2667            |
| ALL     | naked_atm | 60          | oracle_sign   | sign_only               | True            | 1104                    | 714.0592                  | 0.7292                    | 371                 | 1305.6587             | 0.7898                | -522.9971            |
| ALL     | naked_atm | 120         | oracle_sign   | sign_plus_expected_edge | True            | 342                     | 844.4275                  | 0.6725                    | 82                  | 1118.3061             | 0.6341                | -1079.8476           |
| ALL     | naked_atm | 30          | oracle_sign   | sign_only               | True            | 2162                    | 468.5123                  | 0.6947                    | 734                 | 1001.1776             | 0.7657                | -437.8972            |
| ALL     | naked_atm | 60          | oracle_sign   | sign_plus_expected_edge | True            | 604                     | 533.5203                  | 0.6308                    | 88                  | 852.8565              | 0.5795                | -780.8691            |
| ALL     | naked_atm | 30          | oracle_sign   | sign_plus_expected_edge | True            | 652                     | 342.4522                  | 0.6258                    | 110                 | 830.6991              | 0.6455                | -554.8987            |
| ALL     | naked_atm | 15          | oracle_sign   | sign_only               | True            | 3837                    | 299.5105                  | 0.6427                    | 1322                | 636.0947              | 0.7163                | -401.6773            |
| ALL     | naked_atm | 15          | oracle_sign   | sign_plus_expected_edge | True            | 400                     | 255.1748                  | 0.6100                    | 59                  | 445.3337              | 0.6102                | -475.1733            |
| IV < RV | naked_atm | 60          | oracle_sign   | sign_only               | True            | 188                     | 1031.1472                 | 0.7713                    | 32                  | 2269.2639             | 0.8438                | -444.7936            |
| IV < RV | naked_atm | 30          | oracle_sign   | sign_plus_expected_edge | True            | 156                     | 535.4138                  | 0.7115                    | 19                  | 1719.8560             | 0.6842                | -692.0795            |
| IV < RV | naked_atm | 30          | oracle_sign   | sign_only               | True            | 363                     | 755.2537                  | 0.7658                    | 68                  | 1713.2118             | 0.8235                | -486.4308            |
| IV < RV | naked_atm | 120         | oracle_sign   | sign_only               | True            | 97                      | 1381.5424                 | 0.7216                    | 15                  | 1571.6333             | 0.6000                | -1393.2399           |
| IV < RV | naked_atm | 15          | oracle_sign   | sign_only               | True            | 660                     | 588.3687                  | 0.7455                    | 123                 | 1118.7505             | 0.7967                | -409.3434            |
| IV < RV | naked_atm | 60          | oracle_sign   | sign_plus_expected_edge | True            | 113                     | 805.9749                  | 0.6195                    | 11                  | 797.6118              | 0.4545                | -1204.7190           |
| IV < RV | naked_atm | 15          | oracle_sign   | sign_plus_expected_edge | True            | 157                     | 364.3976                  | 0.6369                    | 19                  | 470.6825              | 0.6842                | -485.6576            |
| IV = RV | naked_atm | 120         | oracle_sign   | sign_only               | True            | 120                     | 1595.9409                 | 0.7917                    | 23                  | 1991.4060             | 0.8261                | -1088.6079           |
| IV = RV | naked_atm | 60          | oracle_sign   | sign_only               | True            | 184                     | 1022.7273                 | 0.7500                    | 44                  | 1776.0290             | 0.7045                | -865.7033            |
| IV = RV | naked_atm | 30          | oracle_sign   | sign_only               | True            | 281                     | 725.5162                  | 0.7367                    | 83                  | 1310.6843             | 0.8193                | -333.4192            |
| IV = RV | naked_atm | 120         | oracle_sign   | sign_plus_expected_edge | True            | 72                      | 1043.7880                 | 0.6806                    | 11                  | 1090.7679             | 0.6364                | -1370.8420           |
| IV = RV | naked_atm | 15          | oracle_sign   | sign_only               | True            | 390                     | 369.8773                  | 0.6795                    | 103                 | 737.8889              | 0.7379                | -443.9158            |
| IV = RV | naked_atm | 30          | oracle_sign   | sign_plus_expected_edge | True            | 84                      | 380.9930                  | 0.6429                    | 15                  | 698.5556              | 0.6000                | -600.6445            |
| IV > RV | naked_atm | 120         | oracle_sign   | sign_only               | True            | 469                     | 1116.6158                 | 0.7889                    | 155                 | 2010.6481             | 0.7742                | -822.3354            |
| IV > RV | naked_atm | 60          | oracle_sign   | sign_only               | True            | 1009                    | 676.3390                  | 0.7374                    | 357                 | 1331.0131             | 0.7871                | -554.3394            |
| IV > RV | naked_atm | 120         | oracle_sign   | sign_plus_expected_edge | True            | 307                     | 811.2238                  | 0.6971                    | 76                  | 1107.2269             | 0.6053                | -789.4640            |
| IV > RV | naked_atm | 30          | oracle_sign   | sign_only               | True            | 1872                    | 437.0145                  | 0.6971                    | 684                 | 949.3899              | 0.7588                | -446.7124            |
| IV > RV | naked_atm | 60          | oracle_sign   | sign_plus_expected_edge | True            | 509                     | 500.1635                  | 0.6424                    | 73                  | 773.2042              | 0.6027                | -763.4116            |
| IV > RV | naked_atm | 30          | oracle_sign   | sign_plus_expected_edge | True            | 487                     | 299.3573                  | 0.6160                    | 84                  | 660.8264              | 0.6310                | -457.1446            |
| IV > RV | naked_atm | 15          | oracle_sign   | sign_only               | True            | 3204                    | 248.8485                  | 0.6289                    | 1180                | 582.2122              | 0.7017                | -399.8341            |
| IV > RV | naked_atm | 15          | oracle_sign   | sign_plus_expected_edge | True            | 225                     | 203.6512                  | 0.5911                    | 34                  | 502.5974              | 0.6176                | -314.8784            |

The rule requires at least 20 non-overlapping 2024 trades, at least 10 2025
holdout trades, and positive mean net P&L in both periods.

## Leading configurations

| scene   | family         | horizon_min | signal_mode   | gate                    | robust_positive | trades_development_2024 | mean_net_development_2024 | hit_rate_development_2024 | trades_holdout_2025 | mean_net_holdout_2025 | hit_rate_holdout_2025 | p05_net_holdout_2025 |
| ------- | -------------- | ----------- | ------------- | ----------------------- | --------------- | ----------------------- | ------------------------- | ------------------------- | ------------------- | --------------------- | --------------------- | -------------------- |
| ALL     | naked_atm      | 120         | forecast_sign | sign_plus_expected_edge | False           | 427                     | -116.7191                 | 0.3794                    | 96                  | 74.1362               | 0.4375                | -4039.7021           |
| ALL     | debit_spread_1 | 15          | forecast_sign | sign_plus_expected_edge | False           | 3                       | 461.2879                  | 0.6667                    | 1                   | -75.9548              | 0.0000                | -75.9548             |
| ALL     | debit_spread_1 | 30          | forecast_sign | sign_plus_expected_edge | False           | 15                      | -20.1804                  | 0.1333                    | 8                   | -166.2794             | 0.2500                | -458.6280            |
| ALL     | naked_atm      | 15          | forecast_sign | sign_only               | False           | 3836                    | -106.3177                 | 0.3478                    | 1321                | -187.1124             | 0.3656                | -2035.2796           |
| ALL     | naked_atm      | 30          | forecast_sign | sign_plus_expected_edge | False           | 968                     | -127.2131                 | 0.3616                    | 170                 | -188.8018             | 0.3529                | -2996.8644           |
| IV < RV | naked_atm      | 120         | forecast_sign | sign_plus_expected_edge | True            | 89                      | 76.0097                   | 0.4270                    | 10                  | 571.7520              | 0.4000                | -3793.8657           |
| IV < RV | naked_atm      | 120         | forecast_sign | sign_only               | True            | 96                      | 68.1318                   | 0.4167                    | 15                  | 142.7143              | 0.4667                | -4412.3014           |
| IV < RV | naked_atm      | 30          | forecast_sign | sign_plus_expected_edge | False           | 226                     | -108.9192                 | 0.3761                    | 27                  | 303.5643              | 0.3704                | -3051.8685           |
| IV < RV | debit_spread_1 | 30          | forecast_sign | sign_plus_expected_edge | False           | 10                      | 13.8984                   | 0.2000                    | 1                   | 145.3112              | 1.0000                | 145.3112             |
| IV < RV | naked_atm      | 15          | forecast_sign | sign_only               | False           | 659                     | -107.4068                 | 0.3915                    | 123                 | -65.4807              | 0.4146                | -2314.1835           |
| IV = RV | naked_atm      | 120         | forecast_sign | sign_only               | False           | 119                     | -301.4693                 | 0.3613                    | 23                  | 147.9236              | 0.4783                | -4705.9416           |
| IV = RV | naked_atm      | 120         | forecast_sign | sign_plus_expected_edge | False           | 102                     | -202.5003                 | 0.3824                    | 15                  | 33.8323               | 0.4667                | -3145.7895           |
| IV = RV | naked_atm      | 15          | forecast_sign | sign_only               | False           | 389                     | -96.4253                  | 0.3882                    | 104                 | -264.8895             | 0.3269                | -2327.5581           |
| IV = RV | debit_spread_1 | 30          | forecast_sign | sign_plus_expected_edge | False           | 7                       | -112.6501                 | 0.0000                    | 4                   | -264.9786             | 0.2500                | -479.8958            |
| IV = RV | naked_atm      | 30          | forecast_sign | sign_plus_expected_edge | False           | 142                     | -138.1349                 | 0.3803                    | 24                  | -292.4806             | 0.3750                | -2954.4760           |
| IV > RV | naked_atm      | 120         | forecast_sign | sign_plus_expected_edge | False           | 391                     | -109.6856                 | 0.4041                    | 92                  | -45.5919              | 0.4022                | -4056.8860           |
| IV > RV | debit_spread_1 | 30          | forecast_sign | sign_plus_expected_edge | False           | 6                       | -142.1539                 | 0.0000                    | 4                   | -89.2025              | 0.2500                | -198.9984            |
| IV > RV | debit_spread_2 | 120         | forecast_sign | sign_plus_expected_edge | False           | 44                      | -289.2949                 | 0.3182                    | 5                   | -190.1514             | 0.4000                | -798.3237            |
| IV > RV | naked_atm      | 15          | forecast_sign | sign_only               | False           | 3204                    | -117.6055                 | 0.3358                    | 1179                | -190.7432             | 0.3681                | -1988.4798           |
| IV > RV | naked_atm      | 120         | forecast_sign | sign_only               | False           | 468                     | -194.3632                 | 0.3889                    | 155                 | -193.4189             | 0.4000                | -4535.6654           |
| ALL     | naked_atm      | 120         | oracle_sign   | sign_only               | True            | 497                     | 1072.3264                 | 0.7626                    | 159                 | 2096.8319             | 0.7736                | -887.2667            |
| ALL     | naked_atm      | 60          | oracle_sign   | sign_only               | True            | 1104                    | 714.0592                  | 0.7292                    | 371                 | 1305.6587             | 0.7898                | -522.9971            |
| ALL     | naked_atm      | 120         | oracle_sign   | sign_plus_expected_edge | True            | 342                     | 844.4275                  | 0.6725                    | 82                  | 1118.3061             | 0.6341                | -1079.8476           |
| ALL     | naked_atm      | 30          | oracle_sign   | sign_only               | True            | 2162                    | 468.5123                  | 0.6947                    | 734                 | 1001.1776             | 0.7657                | -437.8972            |
| ALL     | naked_atm      | 60          | oracle_sign   | sign_plus_expected_edge | True            | 604                     | 533.5203                  | 0.6308                    | 88                  | 852.8565              | 0.5795                | -780.8691            |
| IV < RV | naked_atm      | 60          | oracle_sign   | sign_only               | True            | 188                     | 1031.1472                 | 0.7713                    | 32                  | 2269.2639             | 0.8438                | -444.7936            |
| IV < RV | naked_atm      | 30          | oracle_sign   | sign_plus_expected_edge | True            | 156                     | 535.4138                  | 0.7115                    | 19                  | 1719.8560             | 0.6842                | -692.0795            |
| IV < RV | naked_atm      | 30          | oracle_sign   | sign_only               | True            | 363                     | 755.2537                  | 0.7658                    | 68                  | 1713.2118             | 0.8235                | -486.4308            |
| IV < RV | naked_atm      | 120         | oracle_sign   | sign_only               | True            | 97                      | 1381.5424                 | 0.7216                    | 15                  | 1571.6333             | 0.6000                | -1393.2399           |
| IV < RV | naked_atm      | 15          | oracle_sign   | sign_only               | True            | 660                     | 588.3687                  | 0.7455                    | 123                 | 1118.7505             | 0.7967                | -409.3434            |
| IV = RV | naked_atm      | 120         | oracle_sign   | sign_only               | True            | 120                     | 1595.9409                 | 0.7917                    | 23                  | 1991.4060             | 0.8261                | -1088.6079           |
| IV = RV | naked_atm      | 60          | oracle_sign   | sign_only               | True            | 184                     | 1022.7273                 | 0.7500                    | 44                  | 1776.0290             | 0.7045                | -865.7033            |
| IV = RV | naked_atm      | 30          | oracle_sign   | sign_only               | True            | 281                     | 725.5162                  | 0.7367                    | 83                  | 1310.6843             | 0.8193                | -333.4192            |
| IV = RV | naked_atm      | 120         | oracle_sign   | sign_plus_expected_edge | True            | 72                      | 1043.7880                 | 0.6806                    | 11                  | 1090.7679             | 0.6364                | -1370.8420           |
| IV = RV | naked_atm      | 15          | oracle_sign   | sign_only               | True            | 390                     | 369.8773                  | 0.6795                    | 103                 | 737.8889              | 0.7379                | -443.9158            |
| IV > RV | naked_atm      | 120         | oracle_sign   | sign_only               | True            | 469                     | 1116.6158                 | 0.7889                    | 155                 | 2010.6481             | 0.7742                | -822.3354            |
| IV > RV | naked_atm      | 60          | oracle_sign   | sign_only               | True            | 1009                    | 676.3390                  | 0.7374                    | 357                 | 1331.0131             | 0.7871                | -554.3394            |
| IV > RV | naked_atm      | 120         | oracle_sign   | sign_plus_expected_edge | True            | 307                     | 811.2238                  | 0.6971                    | 76                  | 1107.2269             | 0.6053                | -789.4640            |
| IV > RV | naked_atm      | 30          | oracle_sign   | sign_only               | True            | 1872                    | 437.0145                  | 0.6971                    | 684                 | 949.3899              | 0.7588                | -446.7124            |
| IV > RV | naked_atm      | 60          | oracle_sign   | sign_plus_expected_edge | True            | 509                     | 500.1635                  | 0.6424                    | 73                  | 773.2042              | 0.6027                | -763.4116            |

## Limitations

The 2024-2025 period is a reused research evaluation period, not an untouched
test. Quote data do not provide executable bid/ask depth. The direction model
uses overlapping targets during fitting, while evaluated trades are forced to
be non-overlapping. Results therefore remain forward-paper evidence only.

Coverage: 833,276 clean directional structure-windows from
2024-01-03 15:05:59 through 2025-05-09 15:09:59.
