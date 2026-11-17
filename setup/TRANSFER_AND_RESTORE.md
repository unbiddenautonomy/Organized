# Transfer this checkout to another Windows machine

This guide applies to branch `codex/portable-structure-20261111`. Run the commands
from the Desktop clone at `C:\Users\Admin\Desktop\VolTr-portable-clean-20261109`.
The clean source snapshot contains the paper engines, all 117 deployed
`.joblib` files, the LZO DLL, and the V2/VolPred research source. The current `origin` returns “Repository not
found”, so publishing requires a valid new repository URL or restored access.

## Choose a transfer format

**For a new Git repository, export the tracked and hash-checked snapshot:**

```powershell
$source = 'C:\Users\Admin\Desktop\VolTr-portable-clean-20261109'
git -C $source status --short
& "$source\.venv\Scripts\python.exe" "$source\setup\export_source.py" --output 'C:\Users\Admin\Desktop\VolTr-source-refactored.zip'
```

Transfer `VolTr-source-refactored.zip` and its `.sha256.txt` to the new machine.
The exporter requires a clean commit, verifies the ZIP entries and CRC, and
checks every model against `setup/model_manifest.json`. `git archive` keeps the
repository-relative directory layout and excludes `.git`, `.venv`, test output,
local contracts, credentials, and other ignored files. Extract it into an
empty directory, such as `D:\VolTr`:

```powershell
Expand-Archive 'D:\Transfer\VolTr-source-refactored.zip' -DestinationPath 'D:\VolTr'
Set-Location 'D:\VolTr'
git init -b main
git add -f -A
git commit -m 'Import portable VolTr snapshot'
git remote add origin '<URL of your new empty repository>'
git push -u origin main
```

Verify `Get-FileHash 'D:\Transfer\VolTr-source-refactored.zip' -Algorithm SHA256`
against the digest in the sidecar before extracting. The forced add
above is intentional: some archived backtest and frozen
research-approval files in the snapshot match ignore rules. The ZIP contains
only tracked files, so forcing the initial add preserves them without adding
local test output or secrets.

The largest tracked file is a 43Weeks model of 99,585,290 bytes. It is below
[GitHub's 100-MB regular-Git file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
though GitHub may warn about files over 50 MiB. The source snapshot is about
687 MB uncompressed, below [GitHub's 2-GB push-size limit](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit). No Git LFS setup is
needed for the current snapshot; historical commits are a different matter.

The existing branch name and earlier history are not carried into that new
repository; its first commit contains the validated snapshot. If you need to
continue the **existing branch with its history**, create a Git bundle instead:

```powershell
git -C $source bundle create 'C:\Users\Admin\Desktop\VolTr-branch.bundle' codex/portable-structure-20261111
```

On the other machine, run `git clone D:\Transfer\VolTr-branch.bundle D:\VolTr`,
then set the clone's `origin` to an accessible repository URL and push
`codex/portable-structure-20261111`. A bundle carries history, so it is larger
than the clean source snapshot. Historical commits may contain LFS pointers;
the current snapshot contains ordinary model blobs. Pushing a branch into an
existing remote requires access to that remote and may require its earlier
history to be present.

A literal ZIP of this entire working folder carries the more than 2-GiB Git
history, two local Python environments, ignored training output, machine
configuration and raw-data junctions. It can be far larger than the clean
source payload and may include local credentials. Use the tracked-file
`git archive` above. Keep any one-off archive outside the folder being zipped.

## Directory layout and files to restore

Do not flatten or rename the extracted `pipelines/` tree.
The saved models, model metadata, configs, and DLL already sit in their
expected relative directories. Keep `setup/`, `README.md`, and
`docs/history/LARGE_LOCAL_FILES_2026-11-09.md` and `research/` at their shipped
repository-relative locations.

For **fresh IVRV paper inference**, no historical CSV or feed journal is
needed. The daily FO contract is downloaded to the ignored
`runtime/contracts/` directory. Supply the contract-service URL,
`ERP_AUTH_TOKEN`, multicast feed access, and a usable network interface on the
new machine. The contract master may also be supplied locally for offline
preflight. The 120-minute RV/hybrid engine needs a current master passed as
`--contract-file` or set in its ignored `pipelines/rv120/config.json`.

For **historical preparation, training, or replay**, restore only the inputs
for the workflow you intend to run:

| Input absent from Git | Suggested clone-relative path, or use the option shown |
|---|---|
| Five-strike CSV (743.3 MiB) | `pipelines/weeks43/data/icharts_option_banknifty_monthly_atm_5strikes.csv`; alternatively `VOLTR_43W_ENTRY_SOURCE`. PCR trainer accepts `--source <path>`. |
| Full option-chain CSV (15,445.2 MiB) | `pipelines/weeks43/data/icharts_option_banknifty.csv`; alternatively `VOLTR_43W_FULL_CHAIN_SOURCE`. |
| PCR raw feature cache (212.9 MiB) | `pipelines/pcr/data/all_m1_rows_features_v2.pkl`; alternatively `VOLTR_PCR_RAW_CACHE`. It can be rebuilt from the five-strike CSV. |
| Rising-RV source CSV (825.9 MiB) | `pipelines/ivrv/models/rising_rv_120m/data/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv`; alternatively pass `--source <path>` to its `train.py`. |
| 43Weeks 2026 entry CSV (73.1 MiB) | `pipelines/weeks43/data/banknifty_options_model_input_2026.csv`; alternatively set `VOLTR_43W_2026_ENTRY_SOURCE`. |
| 43Weeks 2026 chain Parquet (156.4 MiB) | `pipelines/weeks43/data/BANKNIFTY_2026.parquet`; alternatively set `VOLTR_43W_2026_CHAIN_SOURCE`. |
| Dated replay inputs | `datasets/banknifty/YYYY-MM-DD/` if replaying that date, or pass the replay tool's `--data <path>`. |

These `data/`, `datasets/`, and `contracts/` paths are ignored by Git. Large
inputs may stay on a separate drive; use the environment variables or CLI
paths to avoid duplicating them. The
[large-file inventory](../docs/history/LARGE_LOCAL_FILES_2026-11-09.md) gives exact original
paths, sizes, uses, and restore paths for all 528 excluded workspace files
over 100 MiB, plus sixteen external training/research inputs over 100 MiB. The four
missing older spot/IV model packages must eventually be restored under
`pipelines/legacy_spot_iv/models/legacy_spot_iv/`
as `spot/`, `iv/`, `pnl/`, and `hmm/`; see that directory's README.

The V2 and VolPred trainer source is now in `research/v2/` and
`research/volpred/`. Its large data and intermediate panels remain
external. After the normal setup, run `setup/setup_research.ps1`; use
`setup/link_research_raw.ps1 -RawDataDirectory <path>` to share one raw-data
directory with both projects. Use `setup/prepare_v2_training_configs.py
--feature-root <path>` to write ignored local feature-table configs for the
four IVRV model routes. The
[dependency and retraining map](DEPENDENCIES_AND_RETRAINING.md) lists the
required files and the tested commands. The older spot/IV packages are still
missing and must be recovered from another backup or owner.

## Setup after extraction

1. Install Python 3.13 and ensure PyPI or an equivalent wheel cache is
   reachable. In the extracted repository root, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File setup/setup.ps1 -FoInterface auto -CmInterface auto
   ```

   This creates an ignored `.venv`, installs the shared requirements, writes
   local machine configuration, and checks IVRV model loading. Supply
   `-FoInterface` and `-CmInterface` explicitly if `auto` cannot choose a
   single active adapter.
2. Configure the contract service with
   `setup/setup.ps1 -ContractUrl <URL>` (or the `setup/voltr.py configure`
   command in the root README). Set `ERP_AUTH_TOKEN` for the account that will
   run the task. The token and market service are not in Git.
3. Run `& '.venv\Scripts\python.exe' setup/voltr.py preflight --pipeline ivrv`.
   `--pipeline all` checks every engine but currently exits unsuccessfully
   for the four missing older spot/IV packages. Inspect the
   [refactor validation report](../docs/history/VALIDATION_2026-11-11.md) for the tests already run.
4. If you plan to retrain models, run `setup/setup_research.ps1`, then
   `setup/validate_research.ps1`. Place the five raw CSVs together on a data
   drive and call `setup/link_research_raw.ps1 -RawDataDirectory <path>`.
   For the IVRV feature tables, call
   `& '.venv\Scripts\python.exe' setup/prepare_v2_training_configs.py
   --feature-root <path-to-research-directory>`. The
   [dependency map](DEPENDENCIES_AND_RETRAINING.md) gives each expected file.
5. During market hours, verify a real contract download, multicast feed,
   dashboard, and paper P&L on the new machine. Only after that, inspect
   `setup/install_task.ps1` without `-Apply`, then use `-Apply` to register the
   separate 08:55 IVRV paper task. Check for any existing receiver using the
   same feed sockets or dashboard port.

This setup makes the default IVRV inference service portable. It does not
claim that all historical training, the older spot/IV engine, or a real live
session has already been validated on the new machine.
