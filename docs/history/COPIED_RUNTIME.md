# Copied daily inference runtime

> Historical installation record. The Windows task and drive paths below
> describe the source PC at the time of copying. For this portable branch,
> use the repository root README and `setup/`; this clone does not register
> or alter that existing task.

All files under `E:\pipelines`, including code, strategy/model packages,
dashboards, historical logs, backups, and transfer archives, were copied here.
The decoder source under `E:\Models\Live Trades` was copied to `standalone`.
Existing copied files with the worker and recovery fixes were preserved. The
source installation was not modified. The copied service uses
dashboard port 8790 (compatibility alias 8792), so it does not claim the source
dashboard ports.

Start from `families\ivrv\engine\Inference` with:

```
py -3.13 -m live.live_inference --config live/live_config.json
```

The existing daily 09:00–15:40 Asia/Kolkata scheduler is in the service code.
Windows Task Scheduler runs this copy as `InferencePipeline-PaperTrading-Data`
with an 08:55 daily trigger and a logon trigger. The former
`InferencePipeline-PaperTrading` task pointing to `E:\pipelines` was
removed on 2026-11-02. Its task definition is preserved in
`engine/Inference/logs/isolated_service/previous_scheduler_task.xml`.
The task runs under the Admin interactive account and uses that user's
`ERP_AUTH_TOKEN` environment variable for the daily contract download.

## Recovery and isolation

- The UDP receiver appends every datagram once to a shared, minute-segmented
  `logs/isolated_service/sessions/YYYY-MM-DD/feed_journal/`. It retains 65
  minutes, pruned once per minute. Worker readers have independent cursors and
  do not use the bounded memory queue in this mode.
- A worker subscribes after loading, replays the retained data from the oldest
  minute, and continues into the live tail. A failed worker restarts on its own;
  the other worker readers continue. A replaying worker blocks new paper entries
  until its processing delay returns below the configured backlog threshold.
- `state_history/` contains per-worker status/dashboard snapshots every minute
  for the preceding hour. The raw journal, rather than a dashboard snapshot,
  rebuilds feature state on restart.
- `transport`, `queue_depth` (journal records awaiting delivery), `delivered`,
  worker `processing_delay_seconds`, and worker `fault` appear in
  `/api/pipelines` and `/pipelines`.
- Each qualified trade signal in IV/RV, PCR, or 43Weeks plays
  `live/assets/peshaan-ravi-kishan.mp3`. Each committed paper position open
  **and** close plays `live/assets/koteswara-ravi-kishan_U5lot3o.mp3`.
  Playback uses Windows MCI in a separate process. Errors go to
  `logs/position_alert_errors.log` and do not interrupt a worker. Set
  `POSITION_ALERT_DISABLED=1` only when intentionally running without audio,
  such as automated tests.
- IV/RV requires convergence probability above 70% for every new paper position. The IV-above-RV route also requires its convergence rule; near/below-RV routes also require the RV-decline event score. When its three paper slots are full,
  a new qualified signal can replace the weakest open position on the same route
  if its confidence is at least 10 percentage points higher. Replacements have
  a 10-minute cooldown and a limit of three per session date. The displaced
  position is closed using fresh prices for its own fixed-strike legs, then the new position opens.
  With a free slot, the new position opens without forcing an exit. No exit is
  fabricated when the old legs lack fresh quotes. Other model, timing, and
  risk gates remain in force.

At the observed source rate on 2026-11-02 (17.8 GB over about two hours), the
journal may need roughly 10 GB for 65 minutes. Monitor free disk space and
worker lag. If a worker cannot process as fast as the incoming feed, its delay
will keep growing and replay cannot catch it up; adding CPU capacity or reducing
per-packet strategy work is then required. UDP sender or kernel packet loss
cannot be recovered from this local journal.
