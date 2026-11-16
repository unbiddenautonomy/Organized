# Greek changes conditional on IV versus RV

This uses the 2019–2022 training sample and intraday windows that finish by
15:30. Greek values come from the ATM-only option CSV. Forward RV is computed
from the underlying one-minute spot path over the matching 15, 30, 60, 120, or
180-minute horizon.

Because exact equality is not meaningful for continuous estimates, `IV = RV`
means absolute IV minus RV is at most 0.5
volatility points. Values above that band are `IV > RV`; values below it are
`IV < RV`. IV is converted to the same trading-time annualized clock as RV.

## IV < RV

Mean Greek change:

| greek | 15 | 30 | 60 | 120 | 180 |
| --- | --- | --- | --- | --- | --- |
| gamma | 0.000028 | 0.000023 | 0.000037 | 0.000127 | 0.000152 |
| vega | -4.096750 | -8.855334 | -19.208947 | -39.578646 | -51.901522 |
| theta | -0.589233 | -5.403075 | -13.422744 | -26.215236 | -38.262922 |
| vanna | -0.000844 | -0.003775 | 0.008374 | 0.004354 | 0.023935 |
| volga | -1.308978 | 1.717691 | -6.951430 | -11.394935 | -24.143646 |
| charm | -0.000231 | 0.013477 | -0.001002 | 0.002054 | 0.003855 |

Available observations:

| horizon_min | n |
| --- | --- |
| 15 | 2975 |
| 30 | 2527 |
| 60 | 1935 |
| 120 | 1413 |
| 180 | 1034 |

## IV = RV

Mean Greek change:

| greek | 15 | 30 | 60 | 120 | 180 |
| --- | --- | --- | --- | --- | --- |
| gamma | 0.000011 | 0.000021 | 0.000041 | 0.000087 | 0.000284 |
| vega | -2.850890 | -6.517696 | -15.617696 | -32.273333 | -62.263600 |
| theta | -1.052497 | -2.934687 | -8.834297 | -35.196045 | -34.777336 |
| vanna | 0.016240 | -0.009837 | 0.010377 | -0.000358 | -0.036134 |
| volga | -6.771208 | 13.580568 | -7.467066 | 1.380906 | -3.509722 |
| charm | -0.007060 | 0.004812 | -0.001940 | -0.009234 | 0.043073 |

Available observations:

| horizon_min | n |
| --- | --- |
| 15 | 655 |
| 30 | 625 |
| 60 | 569 |
| 120 | 355 |
| 180 | 261 |

## IV > RV

Mean Greek change:

| greek | 15 | 30 | 60 | 120 | 180 |
| --- | --- | --- | --- | --- | --- |
| gamma | 0.000006 | 0.000009 | 0.000020 | 0.000039 | 0.000051 |
| vega | -2.143854 | -4.128878 | -8.414354 | -17.593359 | -25.306234 |
| theta | -0.707410 | -1.291787 | -2.603638 | -5.234494 | -6.604745 |
| vanna | -0.000825 | -0.003527 | -0.008826 | -0.013090 | -0.016034 |
| volga | 0.745941 | 1.716431 | 5.407253 | 10.398731 | 12.242594 |
| charm | -0.001555 | -0.000917 | 0.000057 | 0.002371 | 0.002981 |

Available observations:

| horizon_min | n |
| --- | --- |
| 15 | 37258 |
| 30 | 33692 |
| 60 | 30207 |
| 120 | 23925 |
| 180 | 18658 |
