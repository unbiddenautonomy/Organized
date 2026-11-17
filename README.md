# VolTr: BANKNIFTY options paper trading

VolTr receives the NSE FO/CM multicast feed, builds one-minute option features,
scores them with saved models and keeps **paper** positions. Nothing here
places orders. One daily service runs every enabled pipeline in its own worker
process; IVRV is the only pipeline enabled by default.

## Where things are

```text
config/            the daily service's settings (start here when something runs differently)
  live_config.json   session: network, contract download, dashboard, IVRV paper rules
  service.json       which pipelines exist, which are enabled, worker limits, schedule
  feed.json          multicast addresses and feed buffer sizes
  machine.local.json (ignored) this PC's adapter IPs, contract URL, dashboard port
voltr/             shared code; pipelines import it, it never imports a pipeline
  core/              market clock, machine settings, crash-safe JSON files and locks
  feed/              UDP receiver, LZO decoder (+ native DLL), message decoders, order book, raw journal
  instruments/       FO contract master: option chains, front future, daily ERP download
  market/            bars, implied volatility and greeks, the shared ATM feature stream
  service/           daily supervisor, one-pipeline worker, feed fan-out, pipeline registry
  web/               signal and position sounds
pipelines/         one package per strategy, same shape each (README, pipeline.py, models/, tests/)
  ivrv/              IV-to-RV convergence short straddles (live, enabled)
  pcr/               PCR short straddles (packaged, disabled)
  weeks43/           3/4-week OTM short straddles (packaged, disabled)
  rv120/             120-minute RV/expanded-hybrid engine (independent, own entry point)
  legacy_spot_iv/    original spot/IV app (not runnable: four model packages missing)
  _template/         copy this to start a new pipeline
research/          trainers, studies and research models (ivrv/, pcr/, weeks43/, v2/, volpred/)
tests/             tests for voltr/ and the service
tools/
  parity/            proves a code change leaves behaviour identical (replays, live feed, dashboard)
  replay/            historical replay, tick repricing and the mock-session dashboard
setup/             machine setup, offline preflight, model manifest, scheduled-task installer
docs/              ARCHITECTURE.md, ADDING_A_PIPELINE.md, reference PDF, history/
archive/           old backups kept for audit; never imported
datasets/ runtime/ (ignored) inputs you supply, outputs the code writes
```

How the pieces connect is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). To add
a strategy, follow [docs/ADDING_A_PIPELINE.md](docs/ADDING_A_PIPELINE.md).

## Everyday commands

Run everything from the repository root with the project's Python.

| Task | Command |
|---|---|
| All tests | `python -m pytest` |
| Offline check of models and config | `python setup/voltr.py preflight --pipeline all` |
| Start the daily service now | `python -m pipelines.ivrv.live --config config/live_config.json` |
| Prove a change is behaviour-neutral | `python tools/parity/run.py check --label my-change --against baseline` |
| Replay a prepared day (IVRV, PCR, 43Weeks) | `python tools/replay/run_replay_20261102.py --data <prepared-day> --output runtime/replays/<name>` |
| RV120 engine | `python -m pipelines.rv120.engine --config pipelines/rv120/config.json` |

`preflight` reports the legacy spot/IV engine as incomplete until its missing
model packages are recovered; that is expected.

## Install on another Windows machine

1. Install Python 3.13 and clone or copy this repository anywhere.
2. From the repository root run
   `powershell -ExecutionPolicy Bypass -File setup/setup.ps1 -FoInterface auto -CmInterface auto`.
   It creates `.venv/`, installs `requirements.txt`, writes the ignored
   `config/machine.local.json` and the RV120/legacy `config.json` files, and runs
   the offline IVRV preflight. It never starts or schedules the service.
3. With more than one network adapter, pass its name or IPv4 through
   `-FoInterface`/`-CmInterface` or set `VOLTR_FO_INTERFACE`/`VOLTR_CM_INTERFACE`.
4. Point the contract download at this network with `-ContractUrl <URL>` and set
   `ERP_AUTH_TOKEN` as a persistent user environment variable.
5. After a real contract and feed check, inspect the scheduled task with
   `powershell -ExecutionPolicy Bypass -File setup/install_task.ps1` and add
   `-Apply` to register `VolTr-PaperTrading-Clean` at 08:55. The dashboard is
   at `http://127.0.0.1:8790/` while the service runs.

[setup/TRANSFER_AND_RESTORE.md](setup/TRANSFER_AND_RESTORE.md) covers moving a
checkout with its data; [setup/DEPENDENCIES_AND_RETRAINING.md](setup/DEPENDENCIES_AND_RETRAINING.md)
covers training inputs.

## Data and what Git holds

Git holds code, configuration, the 117 deployed model files (about 589 MB) and
the LZO DLL; `setup/model_manifest.json` records their hashes. It does not hold
credentials, `machine.local.json`, the RV120/legacy `config.json`, daily contract
masters (`runtime/contracts/`), logs, feed journals or training data. The
historical file inventory and validation reports are in [docs/history/](docs/history/).
For historical workflows set `VOLTR_43W_ENTRY_SOURCE`, `VOLTR_43W_FULL_CHAIN_SOURCE`
and `VOLTR_PCR_RAW_CACHE`, or put those files in each pipeline's ignored `data/`.
Allow at least 3 GiB for a checkout and environment, plus room for feed journals
(a busy session has exceeded 10 GiB).
