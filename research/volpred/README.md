# VolPred

BANKNIFTY volatility research, deployable model artifacts, and live paper
inference.

## Repository layout

- `app/` — live inference worker, local JSON API, and browser dashboard
- `data/raw/` — immutable source market datasets and contract data
- `model/` — canonical deployment models, extracted scalers, and inventory
- `outputs/` — generated research panels, reports, backtests, and dashboards
- `research/` — offline feature engineering, training, and evaluation programs
- `runtime/` — live predictions, process IDs, and service logs
- `scripts/` — maintenance utilities such as the model registry builder
- `strategies/dte4_20_conditional_options/` ? active conditional options strategy, workflow, status and runner

## Live services

The short-straddle dashboard runs at `http://localhost:8770`. Its JSON API runs
at `http://localhost:8765`, with `/health`, `/latest`, and `/predictions` routes.

Start the components from the repository root with:

```powershell
py -3.13 app/live_inference.py --source yahoo-banknifty --poll-seconds 30
py -3.13 app/inference_api.py --port 8765
py -3.13 -m http.server 8770 --bind 127.0.0.1 --directory app/live_dashboard
```

Refresh the organized model registry after retraining with:

```powershell
py -3.13 scripts/organize_models.py
```

## Corrected short-straddle research gate

The clock-consistent 120-minute method and backtest are in
`research/corrected_short_straddle_backtest.py`. It removes expected overnight
variance, allocates intraday IV with a frozen time-of-day profile, uses paired
RV/IV/spot scenarios and nonlinear option repricing, and applies cost and tail
risk gates. Run it with:

```powershell
py -3.13 research/corrected_short_straddle_backtest.py
```

The detailed specification and results are written under
`outputs/corrected_short_straddle/`. This research gate is not connected to
live order placement.

## DTE 4-20 paper trade intents

Start with `strategies/dte4_20_conditional_options/README.md`. The current policy
uses an ATM gate for DTE 4-14 and a conditional strangle selector for DTE 15-20;
all bands remain paper-only. Create an auditable paper intent from an updating
option-chain file and the live prediction log with:

```powershell
py -3.13 app/trade_puncher.py --chain path/to/live_chain.csv --prediction runtime/predictions.jsonl
```

The puncher writes `runtime/trade_intents.jsonl`; it has no broker submission
code. The API exposes the latest decision at `/trade-intent`.
