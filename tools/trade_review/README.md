# Trade review

A read-only review of one IVRV paper-trading day. It is built from the saved logs in
`pipelines/ivrv/logs` and does not change the trading engine or its files.

| File | Does |
|---|---|
| `serve_dashboard.py` | Live review at `http://127.0.0.1:8795/`. Routes: `/market`, `/executions`, `/cutoff-marks`, `/positions` (the demo-position ledger) |
| `generate_daily.py` | Writes `runtime/trade_reviews/Trade review <date>.html`, a self-contained snapshot of one day |
| `replay_stop.py` | Review-only what-if: for trades stopped above -1,500 INR, continues the fixed-entry path to a -1,500 loss limit |
| `daily_snapshot_template.html` | The page both scripts fill |
| `extract_replay_ticks.c` | Optional fast tick extractor for the feed journals that `replay_stop.py` reads |

```powershell
python tools/trade_review/serve_dashboard.py
python tools/trade_review/generate_daily.py --date 2026-11-19
```

## Tick extractor

The extractor is a Windows program built from `extract_replay_ticks.c`. The binary is not in git.
Put it at `runtime/trade_review/extract_replay_ticks.exe`, or point `VOLTR_TICK_EXTRACTOR` at it.
It reads the session's feed journals under `pipelines/ivrv/logs/isolated_service/sessions/<date>/`
with `voltr/feed/native/lzo2.dll`.

Build it with MinGW from the repository root:

```powershell
gcc -O2 -o runtime/trade_review/extract_replay_ticks.exe tools/trade_review/extract_replay_ticks.c
```

Without it, continuations use the minute closes in `fixed_strike_120m_paths.jsonl` and are labelled
"Minute-close estimate".
