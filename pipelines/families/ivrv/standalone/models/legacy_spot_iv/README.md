# Older spot/IV engine models to recover

The standalone `main.py` engine cannot start from this checkout yet. Its four
model directories were absent on the source PC during the 9 November 2026
inventory. Restore their complete package contents here, preserving model
metadata and feature schemas:

| Directory here | Previously configured location |
|---|---|
| `spot/` | `E:\Models\NNmodels\modelWeights\spot_horizon_warmup\fold_05` |
| `iv/` | `E:\Models\NNmodels\modelWeights\iv_horizon_warmup\fold_05` |
| `pnl/` | `E:\Models\spot_iv_hmm_pnl\artifacts_live_v2` |
| `hmm/` | `E:\Models\spot_iv_hmm_pnl\artifacts\hmm_horizon_warmup` |

The paths in `../config.example.json` already point to these clone-local
locations. The full packages must be recovered and validated before claiming
this engine is portable. It also needs a current FO contract master or the
configured PostgreSQL contract table. Its database password can be supplied
through `VOLTR_DB_PASSWORD` instead of the ignored `../config.json`.

Run `py -3.13 setup/voltr.py preflight --pipeline legacy-spot-iv` from the
repository root to see which directories remain missing. The preflight only
checks package presence; a full engine launch must still validate their
contents, database and market feed.
