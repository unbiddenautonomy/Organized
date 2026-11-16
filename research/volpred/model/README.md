# Model registry

This directory is the canonical deployment registry. Run `py -3.13 scripts/organize_models.py`
from the repository root after retraining to refresh it.

`inventory.csv` is the complete machine-readable list. It records the family,
canonical artifact path, original research output, horizon, estimator type,
feature contract, scaler location, deployment status, and SHA-256 checksum.

Each family has this layout:

- `artifacts/`: self-contained joblib payloads with the model, ordered feature
  names, horizon, and any target conversion metadata.
- `scalers/`: fitted `StandardScaler` objects extracted from pipeline models for
  inspection and explicit reuse.
- `*manifest*.csv`: the training manifest copied from the research output.

The scaler remains embedded in every pipeline artifact. Inference should call the
pipeline directly so preprocessing and prediction cannot drift. The extracted
copy exists to make the scaler easy to audit and fulfills deployments that require
it as a separate file. Tree models do not use a scaler and are marked
`not_required` in the inventory.

There are 72 model artifacts across eight organized families. Of these, 60 contain
an embedded scaler and have an extracted scaler copy. The `enhanced_rv` family is
the current spot-only live specification. HAR is selected at all four horizons;
the 120-minute enhanced candidate remains archived because the full daily-unit
validation suite classified its evidence as inconclusive. The other families
remain available for research and option-chain-backed deployment.

The evaluated Enhanced 120-minute specification is frozen under `candidates/`.
It is excluded from the live registry and governed by
`outputs/enhanced_rv_models/validation_120/preregistered_forward_test.md`.

The fitted samples end in May 2025. Before capital is exposed, retrain and validate
on data through the current deployment date.
