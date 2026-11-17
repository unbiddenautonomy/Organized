# VolTr transfer: external files and setup

Updated 12 November 2026. This checklist accompanies the refactored source
archive `VolTr-source-refactored.zip`. It does not replace the source archive.
The archive and its Desktop clone are no longer present on this PC; the ZIP was
reported as copied to the new machine. Verify the transferred ZIP before setup.

## Verify the source ZIP

The verified refactored ZIP had SHA-256:

```text
be6ad575bc76a7fe7c435fae8d7eb4d4790e9bf5f4545a59c756b3b96c404f2c
```

It contained the refactored `pipelines/`, `research/`, `simulations/` and
`setup/` layout, 117 deployed `.joblib` models, the LZO DLL, and
`setup/DEPENDENCIES_AND_RETRAINING.md`. If the new machine's ZIP has a
different checksum or lacks that guide, do not assume it is the validated
archive. A source ZIP is enough to initialize a **new** Git repository; it
does not carry the old `.git` history, local Python environments, test output,
historical data or credentials.

For fresh IVRV, PCR, 43Weeks and standalone 120-minute RV/hybrid **paper
inference**, no historical CSV or `.feed` journal is required. Install Python
3.13 and the packaged requirements, configure the new machine's FO/CM network
interface and ERP token, and supply a current FO contract master or enable the
configured daily download. Do not copy old machine credentials/config files
unchanged. Real feed and contract-service access remain external to Git.

Paths below use `<repo>` for the extracted source root and `<data-root>` for
one data directory on the destination. These are placeholders, not hardcoded
drive letters. Preserve each basename and the specified relative structure.

## Shared raw historical data

Copy these five files into one `<data-root>/raw/` directory. Then, from
`<repo>`, run `setup/link_research_raw.ps1 -RawDataDirectory <data-root>/raw`.
That creates local links at `research/v2/data/` and
`research/volpred/data/raw/` without duplicating the files.

| Source on this PC | MiB | Destination basename under `<data-root>/raw/` | Main users |
|---|---:|---|---|
| `E:\v2\data\icharts_option_banknifty.csv` | 15,445.24 | `icharts_option_banknifty.csv` | Full option chain; 43Weeks training and VolPred structure work |
| `E:\v2\data\icharts_option_banknifty_monthly_atm_5strikes.csv` | 743.28 | `icharts_option_banknifty_monthly_atm_5strikes.csv` | IVRV features, 43Weeks entry snapshots, PCR training |
| `E:\v2\data\icharts_option_banknifty_monthly_atm_up_to_5strikes.csv` | 825.91 | `icharts_option_banknifty_monthly_atm_up_to_5strikes.csv` | Rising-RV and five-strike research |
| `E:\v2\data\icharts_option_banknifty_monthly_atm.csv` | 174.83 | `icharts_option_banknifty_monthly_atm.csv` | VolPred ATM IV/RV research |
| `E:\v2\data\icharts_banknifty_spot_1min.csv` | 28.03 | `icharts_banknifty_spot_1min.csv` | Spot and RV features |

`E:\Letsss Start Again\data\icharts_option_banknifty_monthly_atm_up_to_5strikes.csv`
is byte-identical to the third file above (SHA-256
`8d6e25662c7e1a3a117a95af8c8ab81ceb70377cca2540ac53d01231ef6949d3`).
Copy just one and pass its new path to the rising-RV trainer's `--source`.

Set `VOLTR_43W_ENTRY_SOURCE` to the five-strike CSV and
`VOLTR_43W_FULL_CHAIN_SOURCE` to the full option-chain CSV when running the
43Weeks main training route. The raw data totals about 16.8 GiB.

## Other known inputs for historical routes

These can live anywhere on the new machine; set the indicated environment
variable to the copied file's absolute path. The suggested destination keeps
them separate from deployed models.

| Source on this PC | MiB | Suggested destination | Consumer |
|---|---:|---|---|
| `E:\PCR\hypothesis testing\output\win_classifier\all_m1_rows_features_v2.pkl` | 212.95 | `<data-root>/derived/pcr/all_m1_rows_features_v2.pkl` | PCR backtest via `VOLTR_PCR_RAW_CACHE`; rebuildable from five-strike CSV |
| `E:\v2\2025Testing\banknifty_options_model_input_2026.csv` | 73.14 | `<data-root>/derived/weeks43/banknifty_options_model_input_2026.csv` | 2026 retrain via `VOLTR_43W_2026_ENTRY_SOURCE` |
| `E:\v2\data\raw\rissin_nse_options_intraday\BANKNIFTY_2026.parquet` | 156.44 | `<data-root>/derived/weeks43/BANKNIFTY_2026.parquet` | 2026 retrain via `VOLTR_43W_2026_CHAIN_SOURCE` |

## IVRV prebuilt feature tables

Copy each table **with its adjacent manifest** if you want to retrain without
rebuilding features. Restore the source path after `E:\v2\research\` under
`<repo>/research/v2/research/`, or preserve this subdirectory structure in an
external feature root and run
`setup/prepare_v2_training_configs.py --feature-root <feature-root>`.

```text
E:\v2\research\combined_atm_iv_rv_convergence_v2\artifacts_fullchain\combined_atm_iv_convergence_v2_features_fullchain.csv
E:\v2\research\combined_atm_iv_rv_convergence_v2\artifacts_fullchain\combined_atm_iv_convergence_v2_features_fullchain_column_manifest.json
E:\v2\research\combined_atm_iv_rv_convergence_v2\artifacts\combined_atm_iv_convergence_v2_features.csv
E:\v2\research\combined_atm_iv_rv_convergence_v2\artifacts\combined_atm_iv_convergence_v2_features_column_manifest.json
E:\v2\research\fixed_strike_iv_rv_convergence\artifacts\fixed_strike_convergence_features.csv
E:\v2\research\fixed_strike_iv_rv_convergence\artifacts\fixed_strike_convergence_features_column_manifest.json
E:\v2\research\fixed_strike_iv_120m\artifacts\model_features.csv
E:\v2\research\fixed_strike_iv_120m\artifacts\model_targets.csv
E:\v2\research\fixed_strike_iv_120m\artifacts\feature_manifest.json
```

The four large CSVs above are approximately 433.3, 435.3, 424.9 and
181.9 MiB; `model_targets.csv` is approximately 32.8 MiB. They are derived
inputs and can be regenerated from the raw data with the copied builders.

## VolPred RV and expanded-hybrid retraining panels

Restore each file at the same relative path under `<repo>/research/volpred/`.
The first three support the VolPred RV trainer. The remaining files are known
expanded-hybrid inputs; the spot feature panel is shared.

```text
E:\volpred\outputs\spot_magnitude_models\spot_magnitude_feature_panel.parquet
E:\volpred\outputs\volatility_specific_models\volatility_feature_extensions.parquet
E:\volpred\outputs\rv_prediction_models\five_strike_smile_features.parquet
E:\volpred\outputs\intraday_structure_attribution\_fixed_strike_exit_prices.parquet
E:\volpred\outputs\structure_model_backtest\_fixed_strike_exit_prices_120m.parquet
E:\volpred\outputs\atm_iv_rv_analysis\cleaned_atm_iv_rv_panel.csv
E:\volpred\outputs\rv_prediction_models\intraday_walk_forward_predictions.parquet
E:\volpred\outputs\log_iv_change_models\compact_five_feature\walk_forward_predictions.parquet
E:\volpred\model\regimeUnderstanding\coverage_optimization\coverage_trade_ledger.parquet
E:\volpred\model\regimeUnderstanding\five_strike_trade_research\candidate_panel.parquet
E:\volpred\model\regimeUnderstanding\retrained_hybrid\combined_trade_ledger.parquet
```

The source and these known panels do **not** guarantee a successful new
expanded-hybrid model: its historical selection gate failed in the earlier
validation. Other experimental VolPred studies may require additional large
derived panels, including
`E:\volpred\model\regimeUnderstanding\refined_variance_transition\refined_candidate_panel.parquet`,
`E:\volpred\outputs\five_strike_spot_analysis\five_strike_feature_panel.csv`,
`E:\volpred\outputs\intraday_structure_attribution\primary_trade_windows.csv`,
`E:\volpred\outputs\log_iv_change_models\log_iv_feature_panel.parquet`, and
`E:\volpred\outputs\short_straddle_extended_analysis\short_straddle_windows.parquet`.
Preserve those only for the studies that consume them.

## Saved-date simulations

For the prepared 2 and 3 November replays, copy both
`banknifty_futures_1min.csv` and `banknifty_option_chain_1min.csv` from each
of these directories, maintaining the date beneath `<repo>/datasets/banknifty/`
or passing the copied directory with `--data`:

```text
E:\data\datasets\banknifty\2026-11-02\
E:\data\datasets\banknifty\2026-11-03\
```

Old `.feed` journals are not required for these prepared one-minute replays.
The mock dashboard cannot reproduce every tick's stop/target timing from
one-minute prepared quotes.

## Limits that files on this PC cannot resolve

- The older spot/IV standalone engine requires four model directories under
  `<repo>/pipelines/families/ivrv/standalone/models/legacy_spot_iv/`: `spot/`,
  `iv/`, `pnl/`, and `hmm/`. Their last configured source paths were
  `E:\Models\NNmodels\modelWeights\spot_horizon_warmup\fold_05`,
  `E:\Models\NNmodels\modelWeights\iv_horizon_warmup\fold_05`,
  `E:\Models\spot_iv_hmm_pnl\artifacts_live_v2`, and
  `E:\Models\spot_iv_hmm_pnl\artifacts\hmm_horizon_warmup`. **All four are
  absent on this PC** and must come from another backup or owner.
- A current live session on the new machine requires its own market feed,
  daily FO contract/service, network interface selection, ERP token and
  installed dependencies. A previously downloaded contract becomes stale.
- The copied source has been checked offline, but a real market-hours session
  and every experimental trainer have not been validated on the new machine.

Do not copy entire junctions from the deleted Desktop clone or rely on the
transferred ZIP for any of the external historical files listed above.
