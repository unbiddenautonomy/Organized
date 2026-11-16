# Data sources reviewed for this bundle

All files under `data/` were inventoried. Headers and representative rows were
read from every non-empty raw file; the existing full-source audit reports were
used for large-file row counts and integrity findings. No raw data was changed.

| File | Size | Role / observation |
|---|---:|---|
| `data/raw/banknifty_spot_1min.csv` | 0 B | Empty placeholder; never use as a feature source. |
| `data/raw/icharts_banknifty_spot_1min.csv` | 29.4 MB | Canonical one-minute spot source for trailing realized-variance features. |
| `data/raw/icharts_option_banknifty_monthly_atm.csv` | 183.3 MB | Monthly ATM option quotes and IV/Greeks/OI fields. |
| `data/raw/icharts_option_banknifty_monthly_atm_5strikes.csv` | 779.4 MB | Fixed five-strike monthly extract. |
| `data/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv` | 866.0 MB | Variable-coverage monthly five-strike extract. |
| `data/raw/icharts_option_banknifty_sample_4rows.csv` | 3.3 KB | Four-row option schema/sample fixture. |
| `data/raw/icharts_option_banknifty.csv` | 16.2 GB | Broad option source; not read directly by live inference. |
| `data/raw/NSE_FO_contract_06102026.csv.gz` | 1.8 MB | Current contract-master snapshot, not historical training evidence. |

The audited spot source has 533,720 rows from 2019-10-11 through 2025-07-31.
The audited monthly up-to-five-strike source has 1,286,387 rows through
2025-05-22. These dates explain the model warning in the bundle README.

## Feature implications

- Trailing RV features use only same-session, contiguous one-minute spot bars.
- A minimum of 61 prices is required to produce the 60-return feature.
- Nonpositive prices, cross-gap windows and cross-session windows are rejected.
- Option timestamps are timezone-naive in the raw files. Historical research
  treats availability as `max(datetime, Date + tick_time)` and never assumes a
  quote is tradable before its `tick_time`.
- The option gate uses the current ATM quote's premium, IV, DTE and OI together
  with lagged spot/RV features. Invalid IV blocks a decision; it is not imputed.
- Historical contract sizing cannot be inferred from the 2026-10-06 contract
  snapshot. The bundle therefore contains prediction models only, not sizing or
  broker-execution authority.

See the repository-level `DATA_AUDIT.md` and `DATA_AUDIT_V2.md` for the complete
integrity evidence and exclusions.

