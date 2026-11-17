# Research source kept with the paper engines

The original portable clone contained the deployed IVRV and 120-minute
RV/hybrid model bytes but omitted their trainer projects. On 9 November 2026,
their source was copied **read-only from `E:` into this Desktop clone** so a
future machine can update models. No deployed model or live service was
replaced by this import.

| Project here | Original source | Included |
|---|---|---|
| `v2/` | `E:\v2\models/` and `E:\v2\research/` relevant IVRV directories | Combined-IV/RV, forward-RV, sharp-RV event, and fixed-strike call/put-IV feature builders, trainers, configs, manifests, tests and small reports. Generated large feature CSVs are excluded. |
| `volpred/` | `E:\volpred` | Research Python source, tests, separate spot live app/API/dashboard, DTE 4–20 paper intent workflow, small model packages and metadata, plus the 8.29-MiB frozen research-v2 approval inputs. Large generated panels and logs are excluded. |

`E:\volpred` has its own Git directory, but much of its current research and
model work is untracked there. That is why Git access to that separate
repository would not by itself recover this source. The source copy here is
the handoff snapshot. `E:\v2` was not a Git repository.

## What VolPred does

VolPred is part of the same BANKNIFTY research family, with several separate
branches of work:

- `research/` builds ATM IV/RV, spot magnitude, volatility-extension,
  structure-attribution, realized-variance and hybrid feature panels; fits and
  evaluates models; and contains additional strategy research. The
  `rv_prediction_complete.py` and `expanded_signal_hybrid.py` trainers
  produced the model families used by the standalone 120-minute RV/hybrid
  paper engine in `pipelines/families/ivrv/standalone/`.
- `model/` holds its local model registry and small serialized estimators.
  The active standalone engine already has a separate deployed copy of its
  two selected models in `standalone/models/rv120/`.
- `app/` is another paper-only spot inference/API/dashboard path. It reads an
  updating one-minute spot feed or HTTP source; it is not the scheduled IVRV
  supervisor. `app/trade_puncher.py` emits paper intents using the DTE 4–20
  model family; it has no broker submission.
- `strategies/dte4_20_conditional_options/` documents and runs that separate
  paper-intent policy. `tests/` covers research invariants and the frozen
  research-v2 approval inputs.

## Python environments

The scheduled inference and copied `v2` research use the root `.venv`
installed by `setup/setup.ps1`. The fixed-strike call/put-IV training suite
also imports `catboost`; install the separate
[`v2/requirements-training.txt`](v2/requirements-training.txt) into that
environment before running its trainer. VolPred needs its own ignored
`.venv-volpred` because its frozen research suite requires pandas 3 while the
deployed inference environment pins pandas 2. Run from the repository root:

```powershell
py -3.13 -m venv .venv-volpred
& '.venv-volpred\Scripts\python.exe' -m pip install -r research/volpred/requirements-vol-research.txt
```

The VolPred requirements include its LightGBM dependency. Run its tests with
the **VolPred project as the working directory** and both the project and
`research/` on `PYTHONPATH`; [the validation report](../docs/history/VALIDATION_2026-11-09.md)
records the passing run.

## Raw data and derived panels

Place one shared copy of the five raw CSVs on a data drive and run
`setup/link_research_raw.ps1 -RawDataDirectory <directory>` from the repository
root. It creates two ignored Windows directory junctions:

- `research/v2/data/`
- `research/volpred/data/raw/`

Both point to the same data directory. The 15.1-GiB full option chain, monthly
ATM files, and spot one-minute file therefore need not be copied into Git or
duplicated in the working tree. The script refuses to replace an existing
directory or link. Do not make a literal ZIP of this working folder while the
junctions are present; use the tracked-file `git archive` recipe in
[`setup/TRANSFER_AND_RESTORE.md`](../setup/TRANSFER_AND_RESTORE.md).

The copied `v2` trainer configs expect feature CSVs and JSON manifests under
their own `research/*/artifacts*/` directories. The large CSVs are absent from
Git. To use them at another location, run
`setup/prepare_v2_training_configs.py --feature-root <research-directory>`;
it writes four local YAML configs under ignored `runtime/v2-training-configs/`
without modifying the source configs or deployed models. The feature root
must contain the three directories
`combined_atm_iv_rv_convergence_v2/`,
`fixed_strike_iv_rv_convergence/`, and `fixed_strike_iv_120m/` with their
original `artifacts/` or `artifacts_fullchain/` subdirectories.

VolPred training also uses derived `outputs/*.parquet` panels and
`model/regimeUnderstanding/*.parquet` candidate/ledger panels. The root
[large-file inventory](../docs/history/LARGE_LOCAL_FILES_2026-11-09.md) lists every input
over 100 MiB by original path; the
[validation report](../docs/history/VALIDATION_2026-11-09.md) lists smaller inputs
required by the tested training routes. Raw-file rebuilding of every
historical panel has not yet been verified end to end. Training outputs should
be written under ignored `runtime/` or VolPred `outputs/`; promote a selected
model into `pipelines/families/` only after comparing its feature
contract and offline/live preflight results.
