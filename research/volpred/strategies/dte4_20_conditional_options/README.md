# DTE 4–20 conditional options strategy

This folder is the single entry point for the active BANKNIFTY monthly-options
paper strategy. Model binaries, research programs and generated ledgers remain
in their canonical repository locations; this package links them together so
there is only one copy of each artifact.

## Current policy

| DTE | Decision universe | Gate | Status |
|---|---|---|---|
| 4–7 | ATM short straddle / no trade | Ridge forecast above its training-only 80th percentile | paper only |
| 8–14 | ATM short straddle / no trade | band-specific Huber forecast above its training-only 60th percentile | experimental paper only |
| 15–20 | one-strike short strangle / two-strike short strangle / no trade | highest positive downside-adjusted Huber score | experimental paper only |

Entry is between 09:45 and 09:59. Every position uses the nearest monthly
expiry and closes the same fixed strikes at or after 15:15. The system writes
paper intents and never calls a broker.

## Start here

1. Read [WORKFLOW.md](WORKFLOW.md) for the complete decision sequence.
2. Read [STATUS.md](STATUS.md) for the latest walk-forward evidence.
3. Inspect [strategy.json](strategy.json) for the machine-readable policy.
4. Run `py -3.13 strategies/dte4_20_conditional_options/run.py --validate` for
   the fast validation suite.
5. Run `py -3.13 strategies/dte4_20_conditional_options/run.py --research` to
   rebuild the structure comparison and selector. This is the slower workflow.

To produce a paper decision from live inputs:

```powershell
py -3.13 app/trade_puncher.py `
  --chain path/to/live_chain.csv `
  --prediction runtime/predictions.jsonl
```

The latest intent is written to `runtime/trade_intents.jsonl` and exposed by the
local API at `http://localhost:8765/trade-intent` when the API is running.

## Canonical files

- Puncher: [`app/trade_puncher.py`](../../app/trade_puncher.py)
- 4–7 model: [`model/dte4_20_paper/dte4_7_pnl_gate.joblib`](../../model/dte4_20_paper/dte4_7_pnl_gate.joblib)
- 8–14 model: [`model/dte4_20_paper/dte8_20_huber_gate.joblib`](../../model/dte4_20_paper/dte8_20_huber_gate.joblib)
- 15–20 selector: [`model/dte4_20_paper/multi_structure_selector.joblib`](../../model/dte4_20_paper/multi_structure_selector.joblib)
- Multi-structure report: [`results/dte4_20_multi_structure/REPORT.md`](../../results/dte4_20_multi_structure/REPORT.md)
- Active policy: [`results/dte4_20/punching_policy.csv`](../../results/dte4_20/punching_policy.csv)

