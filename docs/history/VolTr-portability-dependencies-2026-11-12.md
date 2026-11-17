# VolTr transfer and data layout

Updated 12 November 2026. This guide accompanies the **refactored**
`VolTr-source-refactored.zip`. The archive and its former Desktop clone are no
longer on this PC; verify the transferred ZIP has SHA-256
`be6ad575bc76a7fe7c435fae8d7eb4d4790e9bf5f4545a59c756b3b96c404f2c`.
The verified archive contained `setup/DEPENDENCIES_AND_RETRAINING.md` inside
the `setup/` directory. If your copy has another hash or lacks this file,
check the archive before relying on it.

`<REPO>` below means the extracted ZIP root. `<DATA_ROOT>` means **one chosen
data directory on the new PC**, on any drive with enough space. These are
placeholders, not fixed drive letters. Preserve the following structure;
there is no need to scatter files or copy the 15-GiB chain repeatedly.

```text
<REPO>/
  pipelines/              # code and deployed model packages: already in ZIP
  research/v2/             # feature builders and trainers: already in ZIP
  research/volpred/        # VolPred code and model packages: already in ZIP
  simulations/             # replay and mock dashboard tools: already in ZIP
  setup/                   # setup, preflight, manifests and guides: already in ZIP

<DATA_ROOT>/
  raw/                     # one shared copy of the five historical CSVs
  derived/
    pcr/                   # PCR feature cache
    weeks43/               # 2026 holdout inputs
    v2/research/            # IVRV feature tables and adjacent manifests
    volpred/outputs/        # generated VolPred feature/prediction panels
  replays/banknifty/        # optional prepared historical dates
    2026-11-02/
    2026-11-03/
```

For **fresh paper inference** of IVRV, PCR, 43Weeks and the independent
120-minute RV/hybrid engine, the ZIP supplies saved models and the LZO DLL.
No historical CSV or `.feed` journal below is required. Install Python 3.13
and dependencies, configure the new machine's FO/CM network interface and ERP
token, and obtain a **current** FO contract from the configured service or a
current local master. Do not copy machine-specific credentials unchanged.

## 1. Shared raw data: copy once to `<DATA_ROOT>/raw/`

| Source file on this PC | Destination on new PC | MiB | Used by |
|---|---|---:|---|
| `E:\v2\data\icharts_option_banknifty.csv` | `<DATA_ROOT>/raw/icharts_option_banknifty.csv` | 15,445.24 | 43Weeks full chain; VolPred exits/structures |
| `E:\v2\data\icharts_option_banknifty_monthly_atm_5strikes.csv` | `<DATA_ROOT>/raw/icharts_option_banknifty_monthly_atm_5strikes.csv` | 743.28 | IVRV features, 43Weeks entries, PCR training |
| `E:\v2\data\icharts_option_banknifty_monthly_atm_up_to_5strikes.csv` | `<DATA_ROOT>/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv` | 825.91 | Rising-RV and five-strike research |
| `E:\v2\data\icharts_option_banknifty_monthly_atm.csv` | `<DATA_ROOT>/raw/icharts_option_banknifty_monthly_atm.csv` | 174.83 | VolPred ATM IV/RV research |
| `E:\v2\data\icharts_banknifty_spot_1min.csv` | `<DATA_ROOT>/raw/icharts_banknifty_spot_1min.csv` | 28.03 | Spot and RV features |

The third source is byte-identical to
`E:\Letsss Start Again\data\icharts_option_banknifty_monthly_atm_up_to_5strikes.csv`
(SHA-256 `8d6e25662c7e1a3a117a95af8c8ab81ceb70377cca2540ac53d01231ef6949d3`).
Copy it only once. From `<REPO>`, run
`setup/link_research_raw.ps1 -RawDataDirectory <DATA_ROOT>/raw` to create
ignored local links at `research/v2/data/` and
`research/volpred/data/raw/`. Pass the same raw file to rising-RV `train.py`
with `--source`; it does not need a second physical copy.

## 2. PCR and 43Weeks derived inputs

| Source file on this PC | Destination on new PC | Set for the consumer |
|---|---|---|
| `E:\PCR\hypothesis testing\output\win_classifier\all_m1_rows_features_v2.pkl` | `<DATA_ROOT>/derived/pcr/all_m1_rows_features_v2.pkl` | `VOLTR_PCR_RAW_CACHE` |
| `E:\v2\2025Testing\banknifty_options_model_input_2026.csv` | `<DATA_ROOT>/derived/weeks43/banknifty_options_model_input_2026.csv` | `VOLTR_43W_2026_ENTRY_SOURCE` |
| `E:\v2\data\raw\rissin_nse_options_intraday\BANKNIFTY_2026.parquet` | `<DATA_ROOT>/derived/weeks43/BANKNIFTY_2026.parquet` | `VOLTR_43W_2026_CHAIN_SOURCE` |

Set `VOLTR_43W_ENTRY_SOURCE` to the five-strike CSV in `raw/` and
`VOLTR_43W_FULL_CHAIN_SOURCE` to the full-chain CSV in `raw/`. The PCR cache
can be rebuilt from the five-strike CSV if you do not transfer it.

## 3. IVRV prebuilt training tables

Preserve the following paths after `E:\v2\research\` under
`<DATA_ROOT>/derived/v2/research/`. Keep each CSV beside its JSON manifest.
From `<REPO>`, use
`setup/prepare_v2_training_configs.py --feature-root <DATA_ROOT>/derived/v2/research`.
This produces local ignored training configs that point at the external
tables, without copying gigabyte feature sets into the repo.

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

These are derived inputs, not deployed models. The four feature CSVs are
approximately 433.3, 435.3, 424.9 and 181.9 MiB; the targets file is
32.8 MiB. Their builders can regenerate them from the shared raw data.

## 4. VolPred RV and expanded-hybrid panels

Copy files that begin `E:\volpred\outputs\` to the matching path under
`<DATA_ROOT>/derived/volpred/outputs/`. A fresh ZIP has no generated
`<REPO>/research/volpred/outputs/` content. If the trainer expects that path,
make it a local directory junction to `<DATA_ROOT>/derived/volpred/outputs/`
after extraction; do not copy the panel bytes twice. The first three paths
support the RV trainer; the remaining output paths are known hybrid inputs.

```text
E:\volpred\outputs\spot_magnitude_models\spot_magnitude_feature_panel.parquet
E:\volpred\outputs\volatility_specific_models\volatility_feature_extensions.parquet
E:\volpred\outputs\rv_prediction_models\five_strike_smile_features.parquet
E:\volpred\outputs\intraday_structure_attribution\_fixed_strike_exit_prices.parquet
E:\volpred\outputs\structure_model_backtest\_fixed_strike_exit_prices_120m.parquet
E:\volpred\outputs\atm_iv_rv_analysis\cleaned_atm_iv_rv_panel.csv
E:\volpred\outputs\rv_prediction_models\intraday_walk_forward_predictions.parquet
E:\volpred\outputs\log_iv_change_models\compact_five_feature\walk_forward_predictions.parquet
```

Three small-to-medium hybrid panel files live under `model/`, alongside
tracked model/source directories. Restore these at their **exact** paths
under `<REPO>/research/volpred/model/`; do not replace the whole model tree
with a junction:

```text
E:\volpred\model\regimeUnderstanding\coverage_optimization\coverage_trade_ledger.parquet
E:\volpred\model\regimeUnderstanding\five_strike_trade_research\candidate_panel.parquet
E:\volpred\model\regimeUnderstanding\retrained_hybrid\combined_trade_ledger.parquet
```

Other experimental VolPred studies can use additional generated files. Keep
their relative paths under the same `derived/volpred/outputs/` or
`<REPO>/research/volpred/model/` convention only if you plan to run them:

```text
E:\volpred\model\regimeUnderstanding\refined_variance_transition\refined_candidate_panel.parquet
E:\volpred\outputs\five_strike_spot_analysis\five_strike_feature_panel.csv
E:\volpred\outputs\intraday_structure_attribution\primary_trade_windows.csv
E:\volpred\outputs\log_iv_change_models\log_iv_feature_panel.parquet
E:\volpred\outputs\short_straddle_extended_analysis\short_straddle_windows.parquet
```

The expanded-hybrid retrain reached its historical selection gate in the
earlier validation but did not produce a new selected model. Moving these
files does not resolve that model-selection failure.

## 5. Prepared replay dates

For the 2 and 3 November one-minute simulations, copy these two files from
each source date directory into the matching directory under
`<DATA_ROOT>/replays/banknifty/YYYY-MM-DD/`:

```text
E:\data\datasets\banknifty\2026-11-02\banknifty_futures_1min.csv
E:\data\datasets\banknifty\2026-11-02\banknifty_option_chain_1min.csv
E:\data\datasets\banknifty\2026-11-03\banknifty_futures_1min.csv
E:\data\datasets\banknifty\2026-11-03\banknifty_option_chain_1min.csv
```

Pass the chosen date directory to `simulations/run_replay_20261102.py` or
`simulations/mock_session_20261103.py` with `--data`. The mock dashboard also
needs the matching saved `inference.jsonl` through `--inference`; copy that
dated file only for the mock session you intend to replay. Raw `.feed`
journals are not required for these prepared one-minute replays.

## Missing items and new-machine checks

- The older spot/IV standalone engine still needs `spot/`, `iv/`, `pnl/`
  and `hmm/` under
  `<REPO>/pipelines/families/ivrv/standalone/models/legacy_spot_iv/`.
  Their configured sources
  `E:\Models\NNmodels\modelWeights\spot_horizon_warmup\fold_05`,
  `E:\Models\NNmodels\modelWeights\iv_horizon_warmup\fold_05`,
  `E:\Models\spot_iv_hmm_pnl\artifacts_live_v2` and
  `E:\Models\spot_iv_hmm_pnl\artifacts\hmm_horizon_warmup`
  are **all absent on this PC**. No copy plan can supply them from here.
- On the destination, run the ZIP's `setup/setup.ps1` and
  `setup/voltr.py preflight --pipeline all`. The older engine will remain a
  known preflight failure until its four packages are recovered. Run
  `setup/setup_research.ps1` and `setup/validate_research.ps1` before
  retraining. Confirm a real market-hours paper session separately.
- Saved models and source files in the ZIP are not historical market data.
  Python packages, current FO contracts, ERP credentials and multicast
  access must be obtained on the new machine.
