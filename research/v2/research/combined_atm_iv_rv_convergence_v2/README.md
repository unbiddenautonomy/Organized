# V2 combined ATM IV to RV convergence

## Goal

At entry time `t`, select the exact strike identified by the source as ATM. Combine its call and put implied volatilities into one ATM-IV observation, keep that strike fixed, and predict its combined IV at exactly `t + 120` minutes and whether its distance from an out-of-sample 120-minute RV forecast contracts.

The exit target is the IV of the original entry strike. It is not the IV of the strike that becomes ATM at the exit timestamp.

## Combined IV definition

The primary combined IV is the arithmetic mean:

`combined_iv = (iv_call + iv_put) / 2`

The source contains option prices, open interest, IV, Greeks, and vega, but no bid/ask spread or traded volume. At exact ATM rows, call vega weight is effectively 50% for nearly all observations, so vega weighting is retained only as a diagnostic feature. The signed and absolute call-put IV differences are retained as entry-time features and quality checks.

## Leakage controls

- The entry strike must equal the source `atm_strike`.
- The exit joins on exact timestamp, DTE flag, expiry, and the unchanged entry strike.
- The spot RV label requires a complete 121-price, exact-clock 120-minute window.
- Future labels, exit values, availability flags, and eligibility flags are forbidden as model features.
- RV-gap features may be added only from predictions tagged `oof`, `walk_forward`, or `held_out`.
- Chronological splitting keeps timestamp groups together and purges labels that cross train/validation boundaries.

## Build

```powershell
cd E:\v2\research\combined_atm_iv_rv_convergence_v2
py -3.13 build_features.py --config config.yaml
```

The builder writes the feature CSV, an explicit column-role manifest, and a data-audit JSON file under `artifacts/`.
