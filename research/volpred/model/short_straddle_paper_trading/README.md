# Short-straddle paper-trading model bundle

This folder is the self-contained model bundle for the BANKNIFTY paper-only
short-straddle path.

It contains:

- the four enhanced-RV artifacts loaded by `app/live_inference.py`;
- the 4-7 DTE Ridge gate used for an ATM short straddle;
- the 8-14 DTE Huber gate used for an ATM short straddle;
- an extracted scaler for every fitted pipeline in those artifacts; and
- one model-specific feature-engineering module per copied model artifact.

The model artifacts still contain their fitted scalers. Production inference
should call the stored pipeline directly. The files in `scalers/` are audit and
integration copies and must not be applied before calling the full pipeline,
or the input will be scaled twice.

## Scope boundary

`model/dte4_20_paper/multi_structure_selector.joblib` is intentionally not in
this bundle. The active 15-20 DTE policy selects one- or two-strike short
strangles, not an ATM short straddle. The unused 15-20 submodel embedded in
`dte8_20_huber_gate.joblib` is retained because it is part of that immutable
source artifact, and its scaler is extracted for completeness, but the live
short-straddle route uses only its `8-14` submodel.

All artifacts are paper-only. They are not approved for broker submission or
live capital. The fitted samples end in May 2025 and require current forward
validation before any promotion.

## Layout

- `models/`: byte-for-byte copies of the canonical joblib artifacts.
- `scalers/`: separately extracted `StandardScaler` or `RobustScaler` objects.
- `feature_engineering/`: one importable module for each model artifact.
- `manifest.json`: source paths, SHA-256 hashes, feature contracts and scaler
  mappings.
- `DATA_SOURCES.md`: the raw-data inventory and data-quality constraints used
  to define these feature contracts.
- `build_bundle.py`: reproducibly rebuilds binary copies and the manifest.
- `verify_bundle.py`: validates hashes, feature contracts and scaler equality.

Rebuild and verify from the repository root:

```powershell
py -3.13 model/short_straddle_paper_trading/build_bundle.py
py -3.13 model/short_straddle_paper_trading/verify_bundle.py
```

Model-specific modules expose `FEATURE_NAMES` and `build_features(...)`.
Enhanced-RV modules additionally expose `prepare_for_model(...)`, which applies
the artifact's frozen clipping bounds before the embedded pipeline is called.

