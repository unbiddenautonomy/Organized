# Refactor parity checks

Proves that a refactor step leaves behaviour unchanged. Outputs go to the
gitignored `runtime/parity/<label>/`; only this code is tracked.

```powershell
python tools/parity/run.py capture --label baseline        # once, on unchanged code
python tools/parity/run.py check --label step03 --against baseline
python tools/parity/run.py compare baseline step03          # re-compare saved runs
```

| Check | Covers |
|---|---|
| A tests | IVRV, PCR, 43Weeks (file by file), standalone suites; a failed file is rerun once (43Weeks `test_completion_audit` races a subprocess) |
| B replay | `tools/replay/run_replay_20261102.py` on the prepared 2026-11-02 and 2026-11-03 days (IVRV, PCR, 43Weeks) |
| C livefeed | `feedgen.py` turns those days into NSE 7207/17208/7202 packets, LZO1Z-compressed with `lzo2.dll`; `live_feed.py` runs them through the production unpacker, both LZO decoders, message decoders, `LiveFeatureInference` and the IVRV paper books |
| D preflight | `setup/voltr.py preflight --pipeline all` (legacy-spot-iv fails by design) |
| E dashboard | `dashboard_probe.py` GETs every dashboard route; JSON shape and stable values, HTML hashes |
| F inventory | `inventory.py` hashes every gitignored file, since git does not carry them |

A run also fails loudly if the repo changed outside `runtime/parity` while it ran.
Wall-clock fields are dropped before hashing (`VOLATILE_KEYS` in `run.py`,
`VOLATILE` in `dashboard_probe.py`). When a refactor step moves code, update
only the location-specific parts: `SUITES`, `REPLAY_SCRIPT`, `PREFLIGHT_SCRIPT`
and the `load_runtime()` functions.
