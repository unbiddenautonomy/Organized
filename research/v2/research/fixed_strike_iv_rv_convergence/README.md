# Fixed-strike IV to RV convergence: feature construction

This project builds the leakage-safe feature table for testing whether the strike that is ATM at entry moves toward predicted realized volatility over the next 120 clock minutes.

The entry strike is selected from the source `atm_strike`. That exact `(expiry, dte_flag, strike)` is looked up at `t + 120 minutes`; the exit is never replaced with the new ATM strike.

Realized volatility is constructed from the dedicated one-minute BANKNIFTY spot file. A 120-minute label is accepted only when the spot series contains the entry price and every following minute through `t+120`—121 consecutive prices and 120 one-minute returns.

## Build features

```powershell
cd E:\v2\research\fixed_strike_iv_rv_convergence
py -3.13 build_features.py --config config.yaml
```

For a smoke test:

```powershell
py -3.13 build_features.py --config config.yaml --max-days 20 --output artifacts/smoke_features.csv
```

## Output

The table contains:

- current call, put, and vega-weighted ATM-strike IV;
- separate five-strike call/put ATM IV, skew, smile curvature, and fit RMSE;
- exact 5/15/30/60/120-minute IV, skew, smile, straddle, OI, futures, and spot lag features;
- exact historical spot returns and realized-volatility features;
- exact forward 120-minute realized volatility and spot return labels;
- same-strike call/put IV and prices at `t+120`;
- fixed-strike IV-change and convergence-to-actual-RV research labels;
- entry-only quality flags in `quality_entry_ok`;
- future label availability in the separate `label_available` flag;
- `training_row_eligible`, which is only for historical row selection and is never a feature;
- a generated JSON manifest containing the explicit leakage-safe feature whitelist.

`RV_pred` is intentionally not manufactured here. It must be produced out of sample by an RV model. After generating out-of-fold, walk-forward, or held-out RV predictions, call `add_rv_prediction_convergence()`. The function now requires an `rv_prediction_provenance` column containing `oof`, `walk_forward`, or `held_out`; it rejects predictions without that declaration.

Use `purged_chronological_split()` for modeling. It splits whole timestamp groups and removes training or validation rows whose 120-minute labels cross into the next period. Fit imputers, scalers, selectors, and estimators independently inside each training fold.

The builder validates every real-data batch and fails if:

- `entry_strike != atm_strike`;
- the matched exit strike differs from the entry strike;
- the matched exit timestamp is not exactly 120 minutes after entry.

For side `s`:

```text
gap_entry_s = entry_iv_s - RV_pred
gap_exit_s  = exit_fixed_iv_s - RV_pred
converged_s = |gap_exit_s| < |gap_entry_s|
fraction_closed_s = (entry_iv_s - exit_fixed_iv_s) / gap_entry_s
```

Only interpret `fraction_closed` when the absolute entry gap exceeds the configured material-gap threshold (default one volatility point).

