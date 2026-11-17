# Architecture

## Layers

```text
pipelines/<name>/   strategy code: features it adds, models, entry/exit rules, paper book, dashboard
        │ imports
        ▼
voltr/              shared, strategy-neutral code
  service/   supervisor ─ worker ─ fanout ─ registry      (the daily process tree)
  market/    bars, implied volatility, greeks, AtmFeatureStream
  instruments/  contract master, daily download
  feed/      UDP receiver, LZO, message decoders, order book, journal
  core/      clock, machine settings, JSON files, locks
```

Two rules keep this navigable:

1. `voltr` never imports a pipeline by name. The one place that names pipelines
   is `config/service.json`, read by `voltr/service/registry.py`.
2. A pipeline never imports another pipeline. Anything two pipelines need moves
   into `voltr`.

## The daily service

```text
Task Scheduler 08:55 → python -m pipelines.ivrv.live --config config/live_config.json
                        └ isolate_pipelines=true → voltr.service.supervisor.run_service
UDP FO/CM multicast ──► supervisor (receiver only, no model code)
                         ├─ FeedJournal: 65-minute raw datagram journal for replay after a restart
                         └─ Fanout ──► one FeedChannel per enabled pipeline
                                        └─ python -m voltr.service.worker --name <pipeline>
                                             decode packets (voltr.feed) → adapter.index / market / oi
```

* **Schedule.** 09:00–15:40 IST every day; contracts are downloaded first
  (`voltr.instruments.download`) and each worker gets the same session context.
* **Isolation.** A worker that crashes, stalls or falls behind is restarted or
  blocked from new entries on its own; the others keep receiving the feed.
* **Registry.** `config/service.json` lists every pipeline:
  `adapter` (`package.module:Class`), `lock` (one owner per ledger) and an optional
  `status_file` for the pipeline's own dashboard, and optional `web` (dashboard routes served by the
  main dashboard) and `bridge` (used only by the older single-process session). `enabled_pipelines` chooses which run.
  `host` names the module that provides the session config loader and the main
  dashboard (IVRV today); `runtime_dir` is where service status and journals go.
* **Adapter contract.** `__init__(config, contracts, lot_size, expiry, future_token, source)`,
  then `index(message)`, `market(update)`, `oi(message)`, `pulse()`, `fault(reason)`,
  `status() -> dict`, `close()`; optional `recent()`, `note_feed_gap()`,
  `feed_gap_recovered()`. `source` maps names to the shared decoders and maths
  (see `pipelines/ivrv/live.py: install_source_imports`).

## Configuration

`config/live_config.json` is loaded by `pipelines.ivrv.config.load_live_config`,
which layers, in order: `feed.json` → `pipelines/ivrv/config.yaml` (`live_udp`,
signal horizons) → `live_config.json` → `machine.local.json` (only interface IPs,
contract URL and dashboard port) → `service.json`. Relative paths resolve against
the file that names them.

## Pipelines

| Pipeline | Runs as | Data it adds | Paper book | Cost formula |
|---|---|---|---|---|
| ivrv | service worker | `AtmFeatureStream` row + lags, 120-minute RV/IV models | fixed-strike short straddles, target/stop/horizon, replacements | statutory, no brokerage (`strategy.current_costs`) |
| pcr | service worker | per-second batches, PCR features | max 3 positions, target 3000 / stop 1500 | statutory + 2 points slippage (`runtime.costs`) |
| weeks43 | service worker | own `AtmFeatureStream` + OTM chain | 3/4-week OTM straddles | brokerage + dated historical rates (`features.costs`) |
| rv120 | own process | quotes and order book | RV/hybrid straddles | Zerodha schedule (`engine.charge_components`) |
| legacy_spot_iv | own process (not runnable) | spot/IV/PnL models | original paper engine | its own |

The four cost formulas differ on purpose and stay inside their pipelines.

## Behaviour checks

`tools/parity/run.py` replays both prepared days through IVRV, PCR and 43Weeks,
drives a synthetic NSE feed (real LZO packets) through the live IVRV path,
runs the model preflight and every dashboard route, and hashes every ignored
file. A refactor is accepted only when the result matches the baseline.
