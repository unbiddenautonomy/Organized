# Historical simulation tools

Run these from the repository root after `setup/setup.ps1`. Inputs can stay on
a separate data drive; pass their absolute paths. Outputs should stay under
ignored `runtime/` in the clone. The scripts never start the scheduled feed.

| Tool | Purpose and required input |
|---|---|
| `run_replay_20261102.py` | Replays IVRV, PCR and 43Weeks from a prepared one-minute BANKNIFTY date directory. Pass `--data <dir> --output runtime/replays/<name>`. It writes paper events and summaries. |
| `mock_session_20261103.py` | Serves a paced IVRV paper dashboard from `--date`, `--data <prepared-date-dir>` and `--inference <saved-inference.jsonl>`. Pass `--output runtime/mock_sessions/<name> --port <free-port>`. |
| `build_banknifty_chain.py` | Builds a prepared historical chain from market source files. Check `--help` for input and output paths. |
| `reprice_entered_trades_ticks.py`, `calculate_entered_trade_tick_pnl.py` | Reprice previously entered paper trades from available tick history. These require the corresponding saved entry and feed data. |
| `mock_session_dashboard.html` | Static template served by the mock-session script. |

Example:

```powershell
& '.venv\Scripts\python.exe' tools/replay/run_replay_20261102.py --data 'D:\VolTrData\prepared\2026-11-02' --output 'runtime\replays\2026-11-02-check'
& '.venv\Scripts\python.exe' tools/replay/mock_session_20261103.py --date 2026-11-03 --data 'D:\VolTrData\prepared\2026-11-03' --inference 'D:\VolTrData\inference\2026-11-03\inference.jsonl' --output 'runtime\mock_sessions\2026-11-03-check' --port 8891
```

The prepared date directories and saved inference logs are excluded from Git.
The mock dashboard has one quote observation per minute, so its P&L and
stop/target timing are less precise than the live tick-driven worker. The
three-worker replay evaluates each worker independently; the default live
supervisor enables IVRV alone.
