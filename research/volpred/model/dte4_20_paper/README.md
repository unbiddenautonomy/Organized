# DTE 4–20 paper-trade models

| Artifact | DTE | Model | Status |
|---|---:|---|---|
| `dte4_7_pnl_gate.joblib` | 4–7 | scaled Ridge direct-P&L gate | paper only |
| `dte8_20_huber_gate.joblib` | 8–14, 15–20 | separate robust-scaled Huber direct-P&L gates | experimental paper only |
| `multi_structure_selector.joblib` | 15–20 | downside-aware Huber selector across seven structures | experimental paper only |

Both artifacts contain their preprocessing pipeline, ordered feature contract,
threshold, entry window, exit rule, training end date and deployment status.
Inference must call the stored pipeline directly; no external scaler is required.

The DTE 8–20 artifact is a research candidate. Its historical walk-forward P&L
is positive at the primary 25 bp assumption, but the bootstrap confidence
interval includes zero and 2025 is negative. Do not connect it to broker order
submission until the promotion gate in `results/dte8_20_robust_gate/REPORT.md`
passes on new paper observations.

For 15–20 DTE, the puncher uses `multi_structure_selector.joblib`. The older
ATM-only Huber model remains the 8–14 DTE model and a research benchmark for
15–20 DTE.
