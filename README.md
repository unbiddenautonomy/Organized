# BANKNIFTY paper-trading workspace

This checkout keeps each executable paper engine with its canonical code and
saved models under `pipelines/families/`. The duplicate
`models/families/` package was removed after comparing its files with the
canonical tree. All 117 deployed `.joblib` files and the LZO DLL are ordinary
Git blobs. A checkout of this branch does not need Git LFS to obtain the deployed model
bytes. `setup/model_manifest.json` records their SHA-256 hashes. Historical
commits may still contain LFS pointers.

## Layout

| Path | Purpose |
|---|---|
| `pipelines/families/ivrv/engine/Inference/` | Scheduled IVRV worker, short and long straddle paper books, feed supervisor and live dashboard |
| `pipelines/families/ivrv/standalone/` | Shared market decoder plus two older standalone paper engines |
| `pipelines/families/rising_rv/` | Rising-RV model route used by IVRV |
| `pipelines/families/PCR/` | PCR worker, dashboard and model packages |
| `pipelines/families/weeks43/` | 43Weeks worker, dashboard and model packages |
| `simulations/` | Historical replay, tick repricing, and mock-session dashboard tools |
| `docs/ARCHITECTURE.md` | Strategy routing, path conventions, and new-strategy checklist |
| `setup/` | Machine configuration, model manifest, offline preflight and Windows task installer |
| `research/v2/` | IVRV feature builders, trainer source, configs and tests imported from `E:\v2` |
| `research/volpred/` | RV/hybrid and other BANKNIFTY research, model packages, separate paper app and tests imported from `E:\volpred` |
| `datasets/`, family `logs/`, `reports/`, `runtime/`, `data/` | Ignored dated inputs and generated outputs |

IVRV includes its sharp-RV-decline event model and rising-RV route. The live
service's `live/service_config.json` enables **only IVRV** by default. PCR and
43Weeks can be enabled deliberately. The older `main.py` and separate 120
minute RV/hybrid engine are launched independently; they are not supervisor
workers. All engines produce paper decisions only.

## Install on another Windows machine

For ZIP transfer and creation of a new Git repository or continuation of this
branch, follow [the transfer and restore guide](setup/TRANSFER_AND_RESTORE.md).

1. Clone this branch to any local path. Install Python 3.13. A normal network
   package install also needs access to PyPI or an equivalent wheel cache.
2. From the repository root, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File setup/setup.ps1 -FoInterface auto -CmInterface auto
   ```

   The script creates ignored `.venv/`, installs
   `pipelines/requirements.txt`, creates ignored standalone
   `pipelines/families/ivrv/standalone/config.json`, writes optional ignored IVRV
   `pipelines/families/ivrv/engine/Inference/live/machine.local.json`, and runs an **offline** IVRV preflight. It does
   not start or schedule a receiver. Re-running it preserves local credentials
   and strategy thresholds. To use installed packages without reinstalling,
   pass `-SkipPackages`.
3. If there is more than one active network adapter, supply its name or IPv4
   address through `-FoInterface` and `-CmInterface`, or use the
   `VOLTR_FO_INTERFACE` and `VOLTR_CM_INTERFACE` environment variables.
   `auto` succeeds only when exactly one active non-loopback IPv4 adapter is
   available. Paths in the shipped JSON/YAML resolve relative to their config
   files; the repository can move after setup.
4. Configure the contract service for the new network. Use
   `setup/setup.ps1 -ContractUrl <URL>` or
   `py -3.13 setup/voltr.py configure --contract-url <URL>`. Set
   `ERP_AUTH_TOKEN` as a persistent user environment variable for the account
   that will run the scheduled task. Confirm access to the contract endpoint,
   multicast feed and dashboard port. The daily FO contract is downloaded by
   the scheduled IVRV service. These services and the token are external to
   Git.
5. Review `& '.venv\Scripts\python.exe' setup/voltr.py preflight --pipeline all`. The command
   checks file hashes and saved-model loading without opening market sockets.
   It will report the older spot/IV engine as incomplete until its four
   missing model directories are recovered. To check only the default live
   engine, use `--pipeline ivrv`.
6. After verifying a real contract/feed session, inspect the proposed task:

   ```powershell
   powershell -ExecutionPolicy Bypass -File setup/install_task.ps1
   ```

   Add `-Apply` to register the separate `VolTr-PaperTrading-Clean` task at
   08:55 for this checkout. It does not replace or start an existing task.
   Ensure the task's user has the persistent ERP token. The dashboard is at
   `http://127.0.0.1:8790/` when the service runs.

The offline preflight proves local files and model loading. It does not prove
contract download, market-feed reception, or a successful trading session.
Avoid running the new task simultaneously with another service on the same
feed sockets or dashboard port.

## Paper engines and independent validation

The [11 November refactor validation](docs/history/VALIDATION_2026-11-11.md)
records the checks on this layout. The [9 November validation](docs/history/VALIDATION_2026-11-09.md)
records the dependency installation, offline tests, historical training and
replays, temporary external data paths, and remaining blockers for each engine.
The [research dependency map](setup/DEPENDENCIES_AND_RETRAINING.md) explains
the imported IVRV and VolPred trainers and their external data requirements.
Run `setup/setup_research.ps1` and `setup/validate_research.ps1` after the
normal setup when preparing a machine for future model updates.

| Engine | Entry point and requirement | Offline check |
|---|---|---|
| Scheduled IVRV | `pipelines/families/ivrv/engine/Inference/live/live_inference.py`; 70% entry gate, live tick marks, fixed-strike paper books | `--pipeline ivrv` |
| PCR | `pipelines/families/PCR/worker.py`; packaged models, optional supervisor worker | `--pipeline pcr` |
| 43Weeks | `pipelines/families/weeks43/weeks43_worker.py`; packaged models, optional supervisor worker | `--pipeline 43weeks` |
| 120 minute RV/hybrid | `pipelines/families/ivrv/standalone/banknifty_120m_live_inference.py`; two small models in `models/rv120/`; pass a current FO master with `--contract-file` or set one in ignored `config.json` | `--pipeline rv120` |
| Older spot/IV | `pipelines/families/ivrv/standalone/main.py`; four model packages absent from the source PC; needs a current FO master or PostgreSQL contracts | `--pipeline legacy-spot-iv` |

The older engine's exact recovery locations are in
[`legacy_spot_iv/README.md`](<pipelines/families/ivrv/standalone/models/legacy_spot_iv/README.md>).
Its local database password can be set with `VOLTR_DB_PASSWORD`. The shared
setup also installs its `hmmlearn` dependency. The RV/hybrid engine's
two model files are included in Git, but it still needs a current contract
master and feed access to run live.

## Historical data and generated dashboards

[The large-file inventory](docs/history/LARGE_LOCAL_FILES_2026-11-09.md) lists **every one
of the 528 files over 100 MiB** excluded from Git, with an exact restore path
and usage. This includes 517 feed journals (84.72 GiB), four 43Weeks ledgers
(0.65 GiB), four PCR/43Weeks logs (0.55 GiB), and three transfer archives
(1.67 GiB). The inventory also names sixteen external files over 100 MiB used
for historical training and research: source CSVs, IVRV feature tables, and
VolPred intermediate panels.
None is needed for a fresh IVRV live session. For historical workflows, set
`VOLTR_43W_ENTRY_SOURCE`, `VOLTR_43W_FULL_CHAIN_SOURCE`, and
`VOLTR_PCR_RAW_CACHE`, or put those files at the family-local `data/` paths
in the configs. Smaller ignored datasets and logs must likewise be supplied
for a replay of their dates.

Reusable IVRV and older spot/IV dashboard builders and templates are tracked.
Run the relevant `build_inference_dashboard.py` or `build_logs_dashboard.py`
with `--logs`, optional `--date YYYY-MM-DD`, and optional `--output`.
Generated HTML goes to ignored `reports/<date>/` by default. The mock
dashboard can replay a prepared date with
`& '.venv\Scripts\python.exe' simulations/mock_session_20261103.py --date YYYY-MM-DD --data <prepared-date-dir> --inference <inference.jsonl>`.
It uses one observed quote per minute, so its P&L cannot reproduce every live
tick. Sixteen other saved reports in `analysis_outputs/` describe one historic
date and remain historical evidence, not live dashboards.
See [simulation commands](simulations/README.md) for input and output paths.

## Portability and storage limits

The 117 deployed model files total about 589 MB; additional small VolPred
research model packages are under `research/volpred/model/`. The largest original Git blob is
99,585,290 bytes. This branch was built in a separate Desktop clone so the
running `E:\data` checkout and its Windows task remain unchanged. A normal
clone needs enough space for a Git object database, a checked-out model tree,
and a Python environment; allow at least 3 GiB plus room for live feed
journals. The source feed journal has exceeded 10 GiB during a session.
Full offline setup also requires the Python installer and dependency wheels;
Git stores code, configuration, saved models and the DLL, not those external
installers or the market/contract services.
