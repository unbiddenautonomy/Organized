# Portable structure validation — 11 November 2026

All mutations and generated outputs in this pass were confined to the Desktop
clone on branch `codex/portable-structure-20261111`. Historical input files on
`E:` were read only. No scheduled task, receiver, live configuration or
`E:\data` checkout was changed. The earlier training experiments and their
limits are recorded in [the 9 November report](VALIDATION_2026-11-09.md).

| Route | Result in refactored clone |
|---|---|
| Setup and IVRV preflight | `setup/setup.ps1 -SkipPackages -FoInterface auto -CmInterface auto` passed; 19 model/DLL files loaded and checked. |
| PCR, 43Weeks, RV/hybrid preflight | `setup/voltr.py preflight --pipeline pcr`, `43weeks`, `rv120` passed; 20, 77 and 2 model files respectively. |
| Older spot/IV preflight | Still blocked by its four absent `legacy_spot_iv/{spot,iv,pnl,hmm}` model directories. Component tests pass, but complete startup is unavailable. |
| Inference tests | 149 tests passed: IVRV 62, PCR 14, 43Weeks 41, older standalone 32. The Node dashboard health suite passed 13 scenarios. |
| Research tests | `setup/validate_research.ps1` passed: V2 19 and VolPred 128 tests. Research source imports after the top-level move. |
| Three-worker historical replay | Completed on 2 November prepared data; IVRV had four closed positions with ₹1,142.853 net. PCR and 43Weeks had no entries on that date. Result matched the pre-refactor replay. |
| Chain and tick simulation tools | `build_banknifty_chain.py` built 1,125 futures and 3,610 chain rows from a three-tick synthetic feed and real contract master. `reprice_entered_trades_ticks.py` extracted all 115,918 saved token ticks from a locally generated feed fixture. `calculate_entered_trade_tick_pnl.py` repriced 11 historical 50% threshold entries, closing all 11 with ₹558.927 net. Their original 60-GiB raw feed was not reread. |
| PCR historical backtest | Completed 902 trades, ₹98,910.05 net; summary CSV matched the pre-refactor output byte for byte. Input cache was supplied by `VOLTR_PCR_RAW_CACHE`. |
| 43Weeks historical backtest | Completed 150 eligible days, 31 trades, −₹3,210.75 net; the saved selection still reports experimental validation constraints unmet. |
| RV/hybrid standalone | Loaded two deployed models and the 6 November FO contract master; 382 calls and 382 puts available. No market socket opened. |
| VolPred spot inference/API | Offline one-shot forecast loaded four models and produced 15/30/60/120-minute forecasts. API returned HTTP 200 for `/`, `/health`, `/latest` and `/predictions?limit=1`. |
| Mock paper dashboard | 3 November prepared date completed 361/361 minutes. `/`, `/api/status`, `/api/events` returned HTTP 200; 100 events and zero entries under the 70% gate. |
| Other dashboards | Older spot/IV dashboard endpoints `/`, `/retraining.html`, `/api/state` returned 200 with synthetic state. RV120 dashboard `/`, `/api/status`, `/api/signals`, `/api/trades`, `/api/health` returned 200. 43Weeks dashboard preview redirected as configured. IVRV historical and older logs HTML builders produced files in clone runtime. |

The replay and mock session have minute-resolution historical observations.
They do not prove live tick timing, multicast access, contract-service access,
or broker connectivity. Live execution remains paper only. A real market-hours
session on the destination machine is still required before enabling a task.

All 117 deployed `.joblib` files and the LZO DLL are covered by the manifest.
The four absent legacy packages remain outside the ZIP. The VolPred
expanded-hybrid retrain reached its historical selection gate in the earlier
validation but did not produce a new selected model. Full end-to-end training
for every experimental strategy has not been established; large historical
inputs are listed in `LARGE_LOCAL_FILES_2026-11-09.md` and the dependency map.

The tracked-source archive is generated only after committing this refactor
with `setup/export_source.py`. It verifies the exact tracked-file set, ZIP CRC
and all model hashes before reporting a SHA-256 checksum.
