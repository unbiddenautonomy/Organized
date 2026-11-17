# Daily paper service and feed isolation

The normal `live_inference.py --config live/live_config.json` command now uses the independent-process service when `service_config.json` has `isolate_pipelines: true`. Historical CSV/UDP modes and explicit `--preflight` retain their existing paths. Trading configuration, model files and strategy dashboards are unchanged.

## Daily operation

- Prepare at **08:55 Asia/Kolkata**. Download and validate a fresh contract master before loading workers.
- Start the **09:00–15:40** session every calendar day, including weekends. No market data means no quote-based entries. Existing strategy entry windows, warmup, exits and square-off rules still apply.
- Force a network download each session; a cached file alone cannot satisfy readiness. The source normally names its file for the prior business day. The downloaded file date, retrieval time, SHA-256, token manifest and selected expiry are recorded per session.
- Retry contract preparation every 60 seconds on failure. All workers use the same validated daily file. Missing credentials or unavailable contracts are shown as `WAITING_FOR_CONTRACTS`; stale files are not silently substituted.
- The Windows task has daily preparation and logon triggers, ignores duplicate launches, wakes the computer where supported, and restarts the service on failure. The in-process scheduler owns the 09:00 session time.

The PC must be powered on or able to wake, the configured Windows account must be signed in for the existing interactive task, and the ERP/feed network must be available. A powered-off computer cannot be started by Task Scheduler.

The installed task runs at normal process priority (4); model children run below normal. Windows denied the attempted S4U registration on 2026-10-28, so the existing interactive principal was retained. Logged-out startup has not been enabled. The task definition and verification results are saved in `logs/isolated_service/`.

## Credential setup

The configured ERP endpoint requires an `auth-token`. Put the token in the Windows user's **ERP_AUTH_TOKEN** environment variable, or add `contract_auth_token_file` to `service_config.json` pointing to a local UTF-8 token file. Do not put the token in chat, a command argument, a repository, logs or this JSON configuration. The downloader reads the named user environment variable on each attempt, so setting it does not require restarting a waiting service. Token-file contents are read only when downloading and are never saved to a session context.

At initial deployment on 2026-10-28, this credential was absent. It was configured for the Windows user on 2026-10-29; the production service successfully downloaded and validated the 2026-10-28 contract master on 2026-10-29. No credential value is recorded here. The health page refreshes credential readiness every 30 seconds.

## Three independent processes

| Process | Existing strategy and output |
|---|---|
| `ivrv` | Original IV/RV models and paper engine; original logs and `/` dashboard |
| `pcr` | PCR models, minute history and paper book; `PCR/logs`, `/PCR` |
| `43weeks` | One-strike OTM models, own bars/OI, ATM diagnostics and paper book; `weeks43/logs`, `/firstTwoWeeks` |

The receiver only receives raw FO/CM datagrams and offers immutable packet bytes to each bounded queue using `put_nowait`. It never runs a model, feature builder, decoder, strategy log write or worker pipe write. Each queue has its own pipe-writer thread. Each worker decodes independently and owns its model inference, feature history and paper positions. IV/RV failure no longer prevents PCR or 43Weeks from building features.

A separate supervisor handles startup, health checks, restarts and diagnostics. A slow or crashed worker cannot stop another queue. After a queue gap or stale backlog, **only the affected pipeline stops new entries**. Fresh observations can still mark and exit its existing positions. Automatic restarts after a feed interruption retain that entry block until the next daily session, to avoid trading with a partial feature history. Workers get up to three independent restart attempts, then remain visibly failed while the others continue.

Workers subscribe only after their model-loading readiness handshake. Pre-readiness packets are counted separately and are not queued into an uninitialized worker. Each worker then collects its required feature history. The queue capacity is 50,000 datagrams per worker; the 10-second stale-data guard still applies. LZO decompression uses thread-local output buffers and direct byte copying; 100 captured messages matched the prior decoder byte for byte, with about 6.3 times higher throughput in the local 5,000-call decoder benchmark.

The supervisor retains the last valid heartbeat for the current worker PID and session. A transient Windows sharing error while reading the status file must not revert a running worker to its startup timeout. A genuinely unchanged heartbeat still expires after 60 seconds; a process exit or broken transport is detected independently. The dashboard also rejects status/history belonging to an earlier worker or session.

Heavy numerical libraries are limited to one thread per worker; workers run below the receiver's Windows priority. Bounded queues protect memory. UDP is not a guaranteed-delivery protocol: network/kernel drops, a failed common receiver, disk exhaustion or a powered-off host cannot be eliminated by process isolation. Local queue-drop counts, stale-frame counts and worker errors are exposed.

## Monitoring and audit

- `http://127.0.0.1:8780/pipelines` — worker health, queues, drops, restarts and contract preparation.
- `http://127.0.0.1:8782/` — compatibility redirect to the actual service on port 8780. This listener starts with the scheduled service; the old standalone preview no longer impersonates a live dashboard. Paths such as `/firstTwoWeeks` are preserved.
- `/` distinguishes loading, contract blockage, stale data, feature warmup and current predictions. HTTP success alone is never called live inference. Fixed-strike predictions require 61 consecutive valid feature minutes and the existing completed-bar delay.
- `/api/pipelines` — complete service status.
- `logs/isolated_service/service.log` — scheduler and download errors.
- `logs/isolated_service/status.json` — latest service state.
- `logs/isolated_service/sessions/YYYY-MM-DD/` — daily contract context, per-worker contexts/status/logs and supervisor events.

No credentials are copied into worker contexts. A process lock prevents duplicate services and separate locks prevent two writers for each paper ledger. Daily archival retains unresolved original IV/RV positions instead of discarding them.

`test_isolated_runtime.py` injects queue backpressure, worker failure, startup failure and malformed framing. `test_pipeline_workers.py` loads all three real model packages against an archived contract **fixture**, sends empty synthetic packets without opening market sockets, kills IV/RV, and verifies PCR and 43Weeks continue receiving. These tests write isolated logs and never use an old contract master for live operation.

Set `isolate_pipelines` to `false` in `service_config.json` only while the service is stopped to return to the legacy runtime. The pre-change source, configuration and Windows task definition are under `archive/ivrv/live_backups/before_isolated_scheduler_20261028/` (repository root).
