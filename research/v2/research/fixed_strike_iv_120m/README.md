# Fixed-entry-strike IV after 120 minutes

This project creates the dataset for predicting the future IV of the strike that is ATM at entry time.

For each entry timestamp `t`:

```text
K_t = atm_strike at t
exit time = t + 120 minutes
```

The label lookup uses the full option chain and requires the exact same:

```text
exit timestamp, expiry, dte_flag, and strike K_t
```

Separate call and put targets are produced:

```text
label_iv_change_call = IV_call(K_t, t+120) - IV_call(K_t, t)
label_iv_change_put  = IV_put(K_t, t+120)  - IV_put(K_t, t)
```

The pipeline also creates the future fixed-strike IV levels, exact fixed-strike option prices, and actual short-straddle P&L. Future fields are prefixed with `exit_` or `label_`; they must never be used as model inputs.

## Data-source findings

- `icharts_option_banknifty_monthly_atm_5strikes.csv` is the entry surface source. It runs from October 2019 through May 2025 and contains five monthly strikes around ATM with IVs and Greeks.
- `icharts_option_banknifty.csv` is the full-chain exit source. It contains exact strike/expiry IVs and prices, is about 16 GB, and runs from 27 March 2020 through July 2025.
- Consequently, entries before 27 March 2020 cannot be assigned exact fixed-strike exit labels from the available full-chain file.
- The full chain is streamed in chunks; it is never loaded entirely into memory.
- Surface moneyness uses the common `future_open` for the timestamp. It does not use per-strike synthetic futures, which can differ because of quote noise.

## Run

From this directory:

```powershell
py -3.13 run_pipeline.py all --config config.yaml
```

Stages can be run independently:

```powershell
py -3.13 run_pipeline.py entries --config config.yaml
py -3.13 run_pipeline.py exits --config config.yaml
py -3.13 run_pipeline.py validate --config config.yaml
```

Quick smoke test around the beginning of full-chain coverage:

```powershell
py -3.13 run_pipeline.py all --config config.yaml --start-date 2020-03-27 --max-days 2
```

The full output contains entry-only features, exact fixed-strike labels, lagged entry information, quality flags, and a JSON coverage report.

Leakage protection is enforced rather than documented only:

- `model_features.csv` physically contains entry-time features only.
- `model_targets.csv` contains future labels, audit keys, quality flags, and split assignments.
- `feature_manifest.json` is the authoritative feature whitelist; `row_id` is only a join key.
- Train/validation and validation/test boundaries are purged using the complete 120-minute label horizon.
- Validation fails if the entry strike differs from entry ATM, the matched exit strike differs from the entry strike, or the source exit timestamp is not exactly `t+120`.

## Full-history validation result

The full pipeline was executed over 233,126 entry surfaces and 24,730,574 full-chain rows:

- 233,126 / 233,126 entries use the declared ATM strike.
- 135,410 exact call-and-put exits were found.
- 135,410 / 135,410 matched exits retain the entry strike.
- 135,410 / 135,410 matched source timestamps equal entry time plus exactly 120 minutes.
- 133,989 rows pass all entry/exit quality requirements.
- The model-ready feature file contains 169 approved entry-time features and zero future-dependent columns.
- 137 otherwise trainable rows are purged around validation/test boundaries; no training label crosses into validation and no validation label crosses into testing.

