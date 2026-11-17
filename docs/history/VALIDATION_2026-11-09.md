# Standalone clone validation — 9 November 2026

This report tests the Desktop clone on branch `codex/portable-clean-20261109`.
It describes the pre-refactor layout. For the current `pipelines/`,
`research/`, and `simulations/` tree, use [the 11 November report](VALIDATION_2026-11-11.md).
All source data was read from `E:`; all generated test output was written to
ignored paths inside this clone. The active `E:\data` checkout, task, and live
configuration were not changed. The tests were performed outside market hours,
so no real multicast feed or contract-service session was claimed.

## Result by engine

| Engine | Saved-model load and offline startup | Historical testing | Retraining from this clone | Real live paper session |
|---|---|---|---|---|
| Scheduled IVRV | Pass: 18 model artifacts, 363-feature contract, current FO master, LZO decoder | Pass: three-worker replay and 361-minute mock dashboard | Trainer source is now in `research_projects/v2/`. Combined-IV and sharp-RV full training passed; forward RV passed a quick run; fixed-strike call/put IV completed an isolated end-to-end training check. Rising-RV training passed earlier. Large inputs remain external. | Not exercised: needs current contract service/token, market feed, and trading hours. |
| PCR | Pass: packaged model preflight and isolated worker startup | Pass: 902-trade backtest, plus shared replay | Pass: raw cache rebuilt from the five-strike CSV, then full 30/60/120/180-minute training with output in ignored clone runtime | Not exercised on a real feed; worker startup with synthetic frames passed. |
| 43Weeks | Pass: packaged model preflight and isolated worker startup | Pass: 150-day backtest, 31 trades | Pass: external entry/full-chain CSVs rebuilt training data and models; selection and model hashes match the shipped artifacts | Not exercised on a real feed; worker startup with synthetic frames passed. |
| 120-minute RV/hybrid | Pass: both saved models and a real local FO master loaded | Component tests passed in the older-engine suite; no independent historical trade replay was run | VolPred source is now in `research_projects/volpred/`. Full RV training reproduced all eight original model files and manifest byte-for-byte. The expanded-hybrid retrain stops at its current validation gate (details below). | Not exercised on a real feed. |
| Older spot/IV `main.py` | Blocked: four model package directories absent on both the clone and source PC | Its generic component suite passed; full strategy replay was not possible | Blocked by missing model packages and trainer provenance | Blocked until model packages and a current FO master or configured contracts database are supplied. |

**Conclusion:** the clone is sufficient for offline inference checks of IVRV,
PCR, 43Weeks, and RV/hybrid after Python dependencies are installed. V2 and
VolPred trainer source has since been imported, but large historical inputs
remain external and the expanded-hybrid validation gate blocks its current
retrain. A real live session still needs external services. The older spot/IV
engine cannot start because its four model packages are missing.

## Dependencies and temporary data links

The clone's `setup/setup.ps1 -FoInterface auto -CmInterface auto` created an
ignored `.venv` and passed its offline IVRV check. The shared requirements now
include `hmmlearn==0.3.3`, required by the older engine; `pip check` found no
broken requirements. The copied VolPred research requirements request
`pandas==3.0.5`, while inference/V2 uses `pandas==2.3.3`. The separate
`.venv-volpred` was installed and passed `pip check`.

Temporary external inputs were passed by CLI option or environment variable,
without editing the active source checkout or copying the large raw files:

| Input | Temporary link or placement | Use |
|---|---|---|
| `E:\v2\data\icharts_option_banknifty_monthly_atm_5strikes.csv` | `VOLTR_43W_ENTRY_SOURCE`; PCR `--source` | 43Weeks snapshots and PCR cache build |
| `E:\v2\data\icharts_option_banknifty.csv` | `VOLTR_43W_FULL_CHAIN_SOURCE` | 43Weeks option-chain data preparation |
| `E:\PCR\hypothesis testing\output\win_classifier\all_m1_rows_features_v2.pkl` | `VOLTR_PCR_RAW_CACHE` | PCR backtest; independently rebuilt from five-strike CSV |
| `E:\Letsss Start Again\data\icharts_option_banknifty_monthly_atm_up_to_5strikes.csv` | rising-RV trainer `--source` | Rising-RV route training |
| `E:\data\datasets\banknifty\2026-11-02` and `2026-11-03` | replay and mock `--data` | Historical one-minute observations and inference |
| `E:\data\inference Pipeline\families\ivrv\Inference_full_2026-10-24\Inference\live\contracts\NSE_FO_contract_06112026.csv.gz` | Small 1.96-MB copy to ignored clone `live/contracts/` | Offline IVRV and RV/hybrid contract preflight |

The [large-file inventory](../LARGE_LOCAL_FILES_2026-11-09.md) lists all 528
source-workspace files over 100 MiB excluded from Git, their exact restore
locations, and fifteen additional external training files over 100 MiB. All 528
were verified present under `E:\data` and absent from the Desktop clone at the
time of the inventory. The 517 feed journals (84.72 GiB) are optional history
for a fresh live paper session. Other smaller ignored dated data, logs, a
current daily contract master, and external market services are also absent
from Git. The four missing older-engine directories are named in
[`legacy_spot_iv/README.md`](<../inference Pipeline/families/ivrv/Live Trades source/models/legacy_spot_iv/README.md>).

## Checks actually run

- Python: 62 IVRV, 14 PCR, 41 43Weeks, and 32 older-engine tests passed (149
  total). The IVRV worker suite started all three workers with generated FO
  contracts and synthetic feed frames; the older contract tests now generate
  their own fixture rather than relying on this PC's PostgreSQL service.
- Node dashboard health suite: 13 scenarios passed.
- `setup/voltr.py preflight --pipeline all`: IVRV (19 files), PCR (20),
  43Weeks (77), and RV120 (2) passed. Legacy spot/IV reported exactly four
  missing model directories, as expected.
- Actual IVRV `live.live_inference --preflight --config live/live_config.json`
  passed using the copied current FO master; no socket was opened. Actual
  RV/hybrid `banknifty_120m_live_inference.py --preflight --contract-file`
  passed with 382 calls, 382 puts, front expiry, and both saved models.
- `43Weeksmodels/backtest_otm.py` replayed 150 eligible days and 31 trades
  (net -₹3,210.75). `PCR/backtest.py` replayed 902 trades (net ₹98,910.05).
  The root three-worker replay for 2 November accepted and closed four IVRV
  trades (net ₹1,142.85); PCR and 43Weeks made no entries on that one day.
- The 3 November mock session processed all 361 minute observations, served
  the dashboard over HTTP 200 with `MOCK_REPLAY/COMPLETE`, and emitted 100
  events. It entered zero trades because no signal cleared the configured 70%
  gate on that date. It is a minute-sample replay, not a full tick-by-tick
  market reproduction.
- 43Weeks preparation produced 50,392 snapshots, 4,211 feature rows and
  1,574 labels. The feature and training rows matched the source cached
  outputs byte-for-byte; retrained model files and selection matched too.
- PCR raw-cache rebuild from 1,150,955 one-minute rows matched the source
  cache byte-for-byte (SHA-256
  `8BC44F6ED7DB46A527B4B4192F4C34913B66360AB45EF7B24C2ACB84F99B04D6`).
  A full four-horizon training run produced models in ignored clone runtime.
- The rising-RV route trained on 1,426 events with test AUC about 0.9152;
  its selected model hash matched the shipped model exactly.

These results verify that the training and test commands can work with
temporary external paths. They do not imply that the excluded raw inputs are
stored in Git. A new machine must supply those
inputs for historical retraining, install Python and wheels, configure the
contract token/feed, and recover the four older model packages if that engine
is required. The clone has no validation of a full live market session, socket
competition, contract download, or tick-by-tick paper P&L under live feed.

## Research source import and validation, same day

The relevant `E:\v2` source trees and the active `E:\volpred` Python source,
tests, small model packages, and frozen research-v2 approval inputs were copied
into `research_projects/`. No active Python file was omitted from the selected
source trees. All 161 imported Python files compiled. The new
`setup/validate_research.ps1` gate passed 19 V2 tests and 128 VolPred tests in
their respective Python environments. VolPred requires pandas 3 in its own
`.venv-volpred`; the inference/V2 `.venv` remains on pandas 2. Both
environments passed `pip check`.

The copied VolPred `rv_prediction_complete.py` trained 15/30/60/120-minute
intraday and 09:15 RV models using three external feature panels placed in
ignored clone `outputs/`. All eight resulting `.joblib` files and
`model_manifest.csv` were byte identical to `E:\volpred` outputs. The VolPred
spot app loaded four saved enhanced-RV models and produced an offline forecast
from a contiguous 61-minute historical sample; its real feed/API/dashboard
were not exercised.

The copied V2 combined-IV trainer completed all nine stages using the
433.3-MiB full-chain feature table, producing held-out predictions and model
artifacts under ignored clone `runtime/`. The sharp-RV-decline event trainer
completed using its 435.3-MiB table (held-out ROC AUC about 0.8402). The
forward-RV trainer completed its `--quick --exclude-over-60m` route using the
424.9-MiB fixed-strike feature table. A two-day fixed-strike IV raw-data run
read the shared CSV junction without copying its 15-GiB chain: 164
exact-horizon entries, 92 matched same-strike exits, 169 feature columns, and
zero leaked feature columns. Outputs stayed inside ignored clone paths.

The combined-IV/RV builder also ran on the full 743.3-MiB five-strike input
and 28-MiB spot input through the raw-data junction. Its 456,426,839-byte
feature CSV and column manifest matched the original `E:\v2` artifacts
byte-for-byte (SHA-256 `4FA095A347E6D768FBE6AD2266A3E6E1B0960D4C06C3F688C8D4E0F058D130A0`
for the CSV). The full fixed-strike IV trainer reached its candidate search
and completed three call-side families before its long forest search was
stopped. An isolated 370-row end-to-end run completed both call and put
searches, producing all eight model families per side plus quantile models, a
leaderboard, and test predictions under ignored `runtime/`. Its small sample
is an execution check, not a model-quality result; no deployed artifact was
overwritten.

The fixed-strike IV/RV feature builder also completed on the full linked raw
inputs. Its 445,568,262-byte forward-RV feature CSV and column manifest were
byte identical to the original `E:\v2` artifacts (CSV SHA-256
`163E230888F824522EB96ECE498658C6999D17FFF375EDEF0725426B61495363`).

The copied expanded-hybrid trainer imported and reached its primary
short-volatility validation gate, then refused to train a replacement. On the
new entry-safe panel, the best positive 2023 candidate produced ₹1,928.92 over
32 signal days, below its required 40 days; the 64-day candidate had
-₹862.78. The previous model-selection file was based on a different
historical panel and had eligible candidates. This is an unresolved research
data/rule reconciliation, not a missing Python package. The saved deployed
hybrid model was not changed, and the gate was not relaxed.
