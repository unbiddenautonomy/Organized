# Portable workspace architecture

The repository root is resolved from each entry point or its config file. Run
the commands in the root README from the extracted root; it can be on any drive.
Generated data, credentials, contracts, logs and local virtual environments are
ignored. `setup/model_manifest.json` records the deployed model bytes.

| Directory | Role |
|---|---|
| `pipelines/families/ivrv/engine/Inference/` | Scheduled IVRV model, paper books, supervisor, feed fanout and live dashboard. `live/service_config.json` enables IVRV alone by default. |
| `pipelines/families/PCR/`, `pipelines/families/weeks43/` | Optional supervisor workers, their packaged models, historical backtests and dashboards. |
| `pipelines/families/rising_rv/` | Rising-RV model called by the IVRV family. |
| `pipelines/families/ivrv/standalone/` | Market decoder, independent 120-minute RV/hybrid engine, and older spot/IV engine. The older engine needs four missing model packages. |
| `research/v2/`, `research/volpred/` | Trainers, feature builders, tests, separate VolPred spot app and saved research models. Large raw and derived panels remain external. |
| `simulations/` | Historical three-worker replay, quote repricing and mock dashboard. Outputs go under ignored `runtime/`. |
| `setup/` | Machine setup, preflight, training input configuration, external-data links and ZIP export. |

## Data and execution boundaries

The supervisor takes every market tick from the feed and fans it out to the
enabled workers. The IVRV model receives a completed one-minute sample; open
paper positions are marked and stop/target exits are checked as ticks arrive.
The one-minute mock dashboard is an approximation because its prepared files
have one observed quote per minute. PCR and 43Weeks are packaged and can be
enabled deliberately; this branch ships `enabled_pipelines: ["ivrv"]`.
VolPred's spot app and the 120-minute RV/hybrid engine are independent entry
points, not supervisor workers. All trade paths here are paper only.

Default machine configuration is local and ignored. Setup leaves strategy
thresholds intact and writes `machine.local.json` only for explicit machine
settings. Contract downloads and feed access are external services. For
historical training, keep one raw-data copy on a separate drive and link
`research/v2/data/` and `research/volpred/data/raw/` to it with
`setup/link_research_raw.ps1`. The optional feature tables and their restore
paths are in `setup/DEPENDENCIES_AND_RETRAINING.md`.

## Adding a strategy

1. Put the executable family, selected saved models and its own tests under
   `pipelines/families/<family>/`. Put feature generation and trainers under
   `research/`, with generated outputs in ignored directories.
2. Define the market input, one-minute feature contract, model feature order,
   paper decision record, and tick-time valuation/exit behavior. Use paths
   relative to a config or `__file__`; expose external sources as arguments or
   environment variables.
3. For a supervisor worker, register its adapter in
   `pipelines/families/ivrv/engine/Inference/live/pipeline_adapters.py` and
   worker dispatch in `pipeline_worker.py`. Add it to `service_config.json`
   only after offline validation. Independent engines keep their own entry
   point and dashboard.
4. Add deployed model hashes to `setup/model_manifest.json` and preflight
   coverage to `setup/voltr.py`. Add training-only inputs to the dependency
   map and large-file inventory; never commit raw feed journals or credentials.
5. Run model preflight, family tests, historical backtest/replay, dashboard
   HTTP checks and a real paper session on the destination machine before
   scheduling it. A saved model that loads is only the first check.

The structure groups existing code by purpose; it does not auto-discover
strategies. The four legacy spot/IV packages and expanded-hybrid retraining
gate remain outstanding, as described in the validation report.
