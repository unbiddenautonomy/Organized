# Large local files excluded from Git

Inventory snapshot: 2026-11-09, refreshed after the live feed rotation. Cutoff: strictly larger than 100 MiB (104,857,600 bytes).
The 528 workspace paths below are relative to the repository root. If a historical run needs one, restore it at that exact path under the new clone. A separate section at the end lists large training inputs referenced outside this repository by IVRV, PCR, 43Weeks, and VolPred research.
None of these files is required to start a new IVRV live paper session. They are local history, output, or transfer packages.
Daily contracts are downloaded at runtime; ERP credentials and market-feed access must be configured on the new machine.

| Category | Files | Total GiB | Usage |
|---|---:|---:|---|
| Raw IVRV market feed journal | 517 | 84.72 | Optional tick replay and incident audit; not needed for a new live session. |
| 43Weeks backtest ledger | 4 | 0.65 | Saved backtest results; not needed to load the deployed models. |
| PCR/43Weeks historical log | 4 | 0.55 | Saved input or strike-path audit; not needed for new live inference. |
| Transfer archive | 3 | 1.67 | Previous copy or transfer package; not needed after a Git checkout. |

The inventory is a dated snapshot: live journals and logs can change while the service runs. Smaller ignored datasets and logs are also absent from Git; they are not needed for a fresh live session.

The outside-repository inputs are **not** included in the 528-file / 87.60-GiB workspace total. Their current absolute paths are machine-specific. On a new machine, keep them on an existing data drive and point local backtest configuration at them; copying them into a second checkout would waste space.

## Raw IVRV market feed journal (517 files)

Optional tick replay and incident audit; not needed for a new live session.

| Restore under repository root | MiB |
|---|---:|
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894939.feed` | 170.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894940.feed` | 187.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894941.feed` | 173.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894942.feed` | 177.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894943.feed` | 168.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894944.feed` | 172.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894945.feed` | 170.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894946.feed` | 180.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894947.feed` | 173.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894948.feed` | 166.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894949.feed` | 175.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894950.feed` | 174.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894951.feed` | 184.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894952.feed` | 189.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894953.feed` | 197.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894954.feed` | 179.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894955.feed` | 183.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894956.feed` | 178.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894957.feed` | 173.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894958.feed` | 171.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894959.feed` | 170.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894960.feed` | 171.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894961.feed` | 170.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894962.feed` | 179.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894963.feed` | 185.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894964.feed` | 185.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894965.feed` | 169.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894966.feed` | 172.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894967.feed` | 169.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894968.feed` | 173.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894969.feed` | 180.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/4bec5da222e041b188eb09fbcc518574/000029894970.feed` | 200.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894862.feed` | 168.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894863.feed` | 165.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894864.feed` | 159.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894865.feed` | 159.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894866.feed` | 171.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894867.feed` | 168.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894868.feed` | 165.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894869.feed` | 168.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894870.feed` | 168.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894871.feed` | 169.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894872.feed` | 167.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894873.feed` | 163.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894874.feed` | 160.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894875.feed` | 159.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894876.feed` | 159.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894877.feed` | 166.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894878.feed` | 164.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894879.feed` | 163.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894880.feed` | 168.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894881.feed` | 164.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894882.feed` | 162.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894883.feed` | 160.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894884.feed` | 167.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894885.feed` | 161.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894886.feed` | 168.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894887.feed` | 164.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894888.feed` | 161.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894889.feed` | 171.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894890.feed` | 167.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894891.feed` | 168.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894892.feed` | 162.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894893.feed` | 169.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894894.feed` | 166.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894895.feed` | 161.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894896.feed` | 162.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894897.feed` | 162.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894898.feed` | 168.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894899.feed` | 170.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894900.feed` | 172.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894901.feed` | 175.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894902.feed` | 173.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894903.feed` | 164.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894904.feed` | 169.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894905.feed` | 165.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894906.feed` | 163.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894907.feed` | 168.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894908.feed` | 167.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894909.feed` | 174.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894910.feed` | 177.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894911.feed` | 172.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894912.feed` | 178.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894913.feed` | 184.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894914.feed` | 181.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894915.feed` | 177.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894916.feed` | 172.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894917.feed` | 166.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894918.feed` | 162.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894919.feed` | 169.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894920.feed` | 163.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894921.feed` | 166.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894922.feed` | 161.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894923.feed` | 158.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894924.feed` | 170.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894925.feed` | 166.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/c0e60b5f5fab4ed19bbdb90e704574b0/000029894926.feed` | 130.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894974.feed` | 105.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894975.feed` | 191.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894976.feed` | 208.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894977.feed` | 206.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894978.feed` | 188.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894979.feed` | 187.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894980.feed` | 185.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894981.feed` | 179.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894982.feed` | 185.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894983.feed` | 181.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894984.feed` | 188.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894985.feed` | 172.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894986.feed` | 163.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894987.feed` | 170.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894988.feed` | 151.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894989.feed` | 156.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894990.feed` | 184.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894991.feed` | 169.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894992.feed` | 164.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894993.feed` | 158.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894994.feed` | 158.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894995.feed` | 172.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894996.feed` | 175.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894997.feed` | 169.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894998.feed` | 144.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029894999.feed` | 160.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895000.feed` | 175.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895001.feed` | 165.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895002.feed` | 147.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895003.feed` | 142.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895004.feed` | 133.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895005.feed` | 130.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895006.feed` | 127.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895007.feed` | 126.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895008.feed` | 134.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-03/feed_journal_runs/faa07e08953346d7b3533e8d2db5b796/000029895009.feed` | 132.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896387.feed` | 123.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896388.feed` | 164.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896389.feed` | 165.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896390.feed` | 169.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896391.feed` | 171.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896392.feed` | 169.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896393.feed` | 169.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896394.feed` | 172.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896395.feed` | 176.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896396.feed` | 163.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896397.feed` | 165.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896398.feed` | 171.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896399.feed` | 185.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896400.feed` | 183.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896401.feed` | 171.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896402.feed` | 169.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896403.feed` | 171.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896404.feed` | 167.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896405.feed` | 179.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896406.feed` | 185.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896407.feed` | 189.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896408.feed` | 180.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896409.feed` | 183.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896410.feed` | 207.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896411.feed` | 196.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896412.feed` | 189.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896413.feed` | 188.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896414.feed` | 179.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896415.feed` | 183.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896416.feed` | 178.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896417.feed` | 175.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896418.feed` | 174.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896419.feed` | 176.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896420.feed` | 183.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896421.feed` | 179.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896422.feed` | 184.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896423.feed` | 181.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896424.feed` | 187.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896425.feed` | 168.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896426.feed` | 150.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896427.feed` | 155.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896428.feed` | 145.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896429.feed` | 153.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896430.feed` | 184.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896431.feed` | 171.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896432.feed` | 161.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896433.feed` | 164.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896434.feed` | 171.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896435.feed` | 178.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896436.feed` | 168.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896437.feed` | 165.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896438.feed` | 147.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896439.feed` | 170.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896440.feed` | 169.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896441.feed` | 153.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896442.feed` | 136.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896443.feed` | 137.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896444.feed` | 135.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896445.feed` | 129.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896446.feed` | 129.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896447.feed` | 129.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896448.feed` | 136.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/9a2d73f05f0c417893f74c0eec2686f4/000029896449.feed` | 136.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896234.feed` | 174.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896235.feed` | 181.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896236.feed` | 171.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896237.feed` | 154.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896238.feed` | 167.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896239.feed` | 171.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896240.feed` | 171.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896241.feed` | 169.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896242.feed` | 169.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896243.feed` | 168.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896244.feed` | 161.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896245.feed` | 169.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896246.feed` | 167.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896247.feed` | 113.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896248.feed` | 162.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896249.feed` | 118.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896250.feed` | 166.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896251.feed` | 162.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896252.feed` | 164.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896253.feed` | 159.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896254.feed` | 152.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896255.feed` | 158.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896256.feed` | 169.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896257.feed` | 165.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896258.feed` | 174.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896259.feed` | 152.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896260.feed` | 168.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896261.feed` | 175.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896262.feed` | 176.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896263.feed` | 165.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896264.feed` | 175.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896265.feed` | 172.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896266.feed` | 167.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896267.feed` | 171.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896268.feed` | 166.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896269.feed` | 171.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896270.feed` | 172.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896271.feed` | 170.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896272.feed` | 166.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896273.feed` | 162.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896274.feed` | 160.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896275.feed` | 162.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896276.feed` | 160.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896277.feed` | 160.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896278.feed` | 161.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896279.feed` | 164.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896280.feed` | 166.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896281.feed` | 163.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896282.feed` | 158.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896283.feed` | 153.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896284.feed` | 157.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896285.feed` | 160.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896286.feed` | 164.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896287.feed` | 160.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896288.feed` | 162.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896289.feed` | 159.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896290.feed` | 170.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896291.feed` | 164.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896292.feed` | 161.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896293.feed` | 161.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896294.feed` | 160.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896295.feed` | 144.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896296.feed` | 164.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896297.feed` | 159.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-04/feed_journal_runs/bc46545643c84d88ad54987250a7a2c5/000029896298.feed` | 126.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897826.feed` | 182.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897827.feed` | 184.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897828.feed` | 186.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897829.feed` | 184.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897830.feed` | 181.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897831.feed` | 185.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897832.feed` | 178.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897833.feed` | 181.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897834.feed` | 181.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897835.feed` | 185.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897836.feed` | 178.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897837.feed` | 180.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897838.feed` | 188.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897839.feed` | 185.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897840.feed` | 188.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897841.feed` | 187.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897842.feed` | 186.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897843.feed` | 188.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897844.feed` | 190.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897845.feed` | 188.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897846.feed` | 193.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897847.feed` | 187.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897848.feed` | 190.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897849.feed` | 192.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897850.feed` | 216.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897851.feed` | 203.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897852.feed` | 198.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897853.feed` | 193.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897854.feed` | 196.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897855.feed` | 199.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897856.feed` | 196.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897857.feed` | 195.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897858.feed` | 192.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897859.feed` | 194.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897860.feed` | 200.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897861.feed` | 197.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897862.feed` | 200.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897863.feed` | 198.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897864.feed` | 202.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897865.feed` | 178.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897866.feed` | 164.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897867.feed` | 164.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897868.feed` | 160.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897869.feed` | 164.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897870.feed` | 197.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897871.feed` | 180.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897872.feed` | 170.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897873.feed` | 169.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897874.feed` | 167.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897875.feed` | 179.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897876.feed` | 175.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897877.feed` | 173.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897878.feed` | 166.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897879.feed` | 175.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897880.feed` | 178.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897881.feed` | 156.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897882.feed` | 147.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897883.feed` | 145.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897884.feed` | 146.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897885.feed` | 143.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897886.feed` | 141.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897887.feed` | 138.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897888.feed` | 156.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-05/feed_journal_runs/753d265dbf5441d1adaed6b70eb19d55/000029897889.feed` | 158.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899266.feed` | 197.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899267.feed` | 197.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899268.feed` | 195.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899269.feed` | 196.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899270.feed` | 197.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899271.feed` | 197.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899272.feed` | 202.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899273.feed` | 192.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899274.feed` | 195.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899275.feed` | 198.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899276.feed` | 192.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899277.feed` | 190.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899278.feed` | 195.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899279.feed` | 202.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899280.feed` | 200.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899281.feed` | 206.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899282.feed` | 203.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899283.feed` | 201.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899284.feed` | 197.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899285.feed` | 199.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899286.feed` | 199.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899287.feed` | 190.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899288.feed` | 197.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899289.feed` | 200.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899290.feed` | 216.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899291.feed` | 203.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899292.feed` | 200.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899293.feed` | 205.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899294.feed` | 203.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899295.feed` | 207.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899296.feed` | 209.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899297.feed` | 208.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899298.feed` | 216.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899299.feed` | 220.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899300.feed` | 215.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899301.feed` | 209.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899302.feed` | 212.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899303.feed` | 206.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899304.feed` | 215.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899305.feed` | 187.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899306.feed` | 178.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899307.feed` | 176.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899308.feed` | 177.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899309.feed` | 177.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899310.feed` | 205.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899311.feed` | 191.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899312.feed` | 188.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899313.feed` | 179.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899314.feed` | 175.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899315.feed` | 182.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899316.feed` | 185.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899317.feed` | 180.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899318.feed` | 165.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899319.feed` | 185.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899320.feed` | 175.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899321.feed` | 162.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899322.feed` | 153.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899323.feed` | 151.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899324.feed` | 152.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899325.feed` | 147.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899326.feed` | 142.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899327.feed` | 142.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899328.feed` | 153.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-06/feed_journal_runs/bf58761fc3c44beeb3972bdf23932cff/000029899329.feed` | 152.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902146.feed` | 150.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902147.feed` | 148.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902148.feed` | 149.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902149.feed` | 150.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902150.feed` | 147.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902151.feed` | 146.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902152.feed` | 146.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902153.feed` | 146.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902154.feed` | 147.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902155.feed` | 148.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902156.feed` | 148.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902157.feed` | 146.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902158.feed` | 143.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902159.feed` | 151.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902160.feed` | 151.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902161.feed` | 147.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902162.feed` | 144.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902163.feed` | 143.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902164.feed` | 140.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902165.feed` | 141.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902166.feed` | 144.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902167.feed` | 143.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902168.feed` | 143.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902169.feed` | 144.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902170.feed` | 147.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902171.feed` | 143.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902172.feed` | 143.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902173.feed` | 143.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902174.feed` | 140.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902175.feed` | 147.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902176.feed` | 151.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902177.feed` | 145.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902178.feed` | 146.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902179.feed` | 146.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902180.feed` | 155.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902181.feed` | 152.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902182.feed` | 166.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902183.feed` | 161.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902184.feed` | 163.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902185.feed` | 144.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902186.feed` | 140.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902187.feed` | 136.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902188.feed` | 133.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902189.feed` | 142.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902190.feed` | 157.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902191.feed` | 144.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902192.feed` | 139.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902193.feed` | 137.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902194.feed` | 147.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902195.feed` | 152.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902196.feed` | 151.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902197.feed` | 152.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902198.feed` | 134.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902199.feed` | 133.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902200.feed` | 131.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902201.feed` | 112.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902202.feed` | 113.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902203.feed` | 119.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902204.feed` | 116.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902205.feed` | 111.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902206.feed` | 112.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902207.feed` | 124.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902208.feed` | 130.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-08/feed_journal_runs/dde40495ad7a4108b61df94d22f34c9c/000029902209.feed` | 119.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903586.feed` | 174.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903587.feed` | 179.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903588.feed` | 179.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903589.feed` | 184.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903590.feed` | 175.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903591.feed` | 177.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903592.feed` | 175.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903593.feed` | 168.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903594.feed` | 162.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903595.feed` | 175.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903596.feed` | 153.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903597.feed` | 156.6 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903598.feed` | 159.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903599.feed` | 162.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903600.feed` | 171.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903601.feed` | 162.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903602.feed` | 155.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903603.feed` | 154.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903604.feed` | 170.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903605.feed` | 180.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903606.feed` | 181.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903607.feed` | 180.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903608.feed` | 178.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903609.feed` | 171.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903610.feed` | 198.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903611.feed` | 179.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903612.feed` | 156.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903613.feed` | 175.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903614.feed` | 177.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903615.feed` | 191.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903616.feed` | 163.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903617.feed` | 163.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903618.feed` | 178.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903619.feed` | 172.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903620.feed` | 161.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903621.feed` | 176.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903622.feed` | 176.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903623.feed` | 169.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903624.feed` | 167.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903625.feed` | 167.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903626.feed` | 156.0 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903627.feed` | 161.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903628.feed` | 148.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903629.feed` | 152.3 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903630.feed` | 174.2 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903631.feed` | 176.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903632.feed` | 145.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903633.feed` | 154.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903634.feed` | 154.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903635.feed` | 155.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903636.feed` | 162.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903637.feed` | 162.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903638.feed` | 133.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903639.feed` | 142.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903640.feed` | 152.8 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903641.feed` | 133.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903642.feed` | 133.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903643.feed` | 137.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903644.feed` | 124.5 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903645.feed` | 124.9 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903646.feed` | 122.7 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903647.feed` | 122.1 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903648.feed` | 130.4 |
| `pipelines/families/ivrv/engine/Inference/logs/isolated_service/sessions/2026-11-09/feed_journal_runs/83810dd2c8f247b398c86b3a250b3876/000029903649.feed` | 133.4 |

## 43Weeks backtest ledger (4 files)

Saved backtest results; not needed to load the deployed models.

| Restore under repository root | MiB |
|---|---:|
| `pipelines/families/weeks43/backtests/initial_2024_test/ledger.sqlite3` | 175.9 |
| `pipelines/families/weeks43/backtests/replay_20261028_200046/ledger.sqlite3` | 189.3 |
| `pipelines/families/weeks43/backtests/replay_20261028_203517/ledger.sqlite3` | 121.6 |
| `pipelines/families/weeks43/backtests/replay_20261028_204043/ledger.sqlite3` | 177.7 |

## PCR/43Weeks historical log (4 files)

Saved input or strike-path audit; not needed for new live inference.

| Restore under repository root | MiB |
|---|---:|
| `pipelines/families/weeks43/logs/2026-10-29/strike_paths.jsonl` | 105.9 |
| `pipelines/families/weeks43/logs/2026-10-30/strike_paths.jsonl` | 118.5 |
| `pipelines/families/PCR/logs/2026-10-29/inputs.jsonl` | 160.1 |
| `pipelines/families/PCR/logs/2026-10-30/inputs.jsonl` | 182.9 |

## Transfer archive (3 files)

Previous copy or transfer package; not needed after a Git checkout.

| Restore under repository root | MiB |
|---|---:|
| `data-trimmed-2026-11-04.zip` | 977.5 |
| `pipelines/archives/pipelines-transfer-2026-11-02.zip` | 485.2 |
| `inference-pipeline-no-datasets-2026-11-05.zip` | 250.7 |

## Additional large inputs referenced outside this Git workspace

These sixteen existing files are used for historical preparation, model training, or research. Fifteen were checked on 9 November 2026; the 2026 chain Parquet was added to this inventory on 11 November. They are not needed to start a new IVRV paper session. Preserve them on a data drive and pass their paths to training commands. 43Weeks and PCR accept `VOLTR_43W_ENTRY_SOURCE`, `VOLTR_43W_FULL_CHAIN_SOURCE`, `VOLTR_43W_2026_ENTRY_SOURCE`, `VOLTR_43W_2026_CHAIN_SOURCE`, and `VOLTR_PCR_RAW_CACHE`. The PCR cache can be rebuilt byte-for-byte from the 743.3-MiB five-strike CSV. The raw option and spot CSVs can be shared with both copied research projects through `setup/link_research_raw.ps1` without duplicating them.

| Original path on this PC | MiB | Consumer and usage |
|---|---:|---|
| `E:\v2\data\icharts_option_banknifty_monthly_atm_5strikes.csv` | 743.3 | 43Weeks entry snapshots and PCR raw-cache source |
| `E:\v2\data\icharts_option_banknifty_monthly_atm.csv` | 174.8 | VolPred ATM IV/RV feature-building input |
| `E:\v2\data\icharts_option_banknifty.csv` | 15445.2 | `weeks43/prepare_otm_data.py` full option chain (`config.json` `full_chain_source`) |
| `E:\v2\data\raw\rissin_nse_options_intraday\BANKNIFTY_2026.parquet` | 156.4 | 43Weeks 2026 holdout retraining; restore to `pipelines/families/weeks43/data/BANKNIFTY_2026.parquet` or set `VOLTR_43W_2026_CHAIN_SOURCE` |
| `E:\PCR\hypothesis testing\output\win_classifier\all_m1_rows_features_v2.pkl` | 212.9 | `PCR/backtest.py` raw feature cache (`config.json` `reference_raw_cache`) |
| `E:\v2\research\combined_atm_iv_rv_convergence_v2\artifacts_fullchain\combined_atm_iv_convergence_v2_features_fullchain.csv` | 433.3 | Combined-IV full-chain model training from copied `research/v2` source |
| `E:\v2\research\combined_atm_iv_rv_convergence_v2\artifacts\combined_atm_iv_convergence_v2_features.csv` | 435.3 | Sharp RV-decline event model training from copied source |
| `E:\v2\research\fixed_strike_iv_rv_convergence\artifacts\fixed_strike_convergence_features.csv` | 424.9 | Forward RV model training from copied source |
| `E:\v2\research\fixed_strike_iv_120m\artifacts\model_features.csv` | 181.9 | Fixed-strike call/put IV model training from copied source; also needs its 32.8-MiB target table |
| `E:\Letsss Start Again\data\icharts_option_banknifty_monthly_atm_up_to_5strikes.csv` | 825.9 | Rising-RV route training (`rising_rv/train.py`) |
| `E:\volpred\outputs\spot_magnitude_models\spot_magnitude_feature_panel.parquet` | 103.5 | 120-minute RV/hybrid model training from copied `research/volpred` source |
| `E:\volpred\model\regimeUnderstanding\refined_variance_transition\refined_candidate_panel.parquet` | 107.5 | VolPred refined-variance research candidate panel |
| `E:\volpred\outputs\five_strike_spot_analysis\five_strike_feature_panel.csv` | 241.5 | VolPred five-strike spot research feature panel |
| `E:\volpred\outputs\intraday_structure_attribution\primary_trade_windows.csv` | 593.9 | VolPred structure-attribution research windows |
| `E:\volpred\outputs\log_iv_change_models\log_iv_feature_panel.parquet` | 128.5 | VolPred log-IV research feature panel |
| `E:\volpred\outputs\short_straddle_extended_analysis\short_straddle_windows.parquet` | 129.4 | VolPred extended short-straddle research windows |

The older `standalone/main.py` needs four model directories that are missing on this PC. Their original and restore paths are in [`legacy_spot_iv/README.md`](<pipelines/families/ivrv/standalone/models/legacy_spot_iv/README.md>). They must be recovered and sized before that engine can meet a Git-only handoff. The two small 120-minute RV/hybrid models are now included in Git under `standalone/models/rv120/`.

The companion 43Weeks 2026 entry file is 73.1 MiB, below this inventory's
100-MiB cutoff: `E:\v2\2025Testing\banknifty_options_model_input_2026.csv`.
Restore it to `pipelines/families/weeks43/data/banknifty_options_model_input_2026.csv`
or set `VOLTR_43W_2026_ENTRY_SOURCE`.
