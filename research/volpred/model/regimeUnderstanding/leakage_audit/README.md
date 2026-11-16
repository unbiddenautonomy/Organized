# Leakage audit status

The October 2026 audit found two forms of look-ahead in earlier research:

1. Some entry filters used `stale_any` or `tiny_any`, fields computed with the
   future exit quote.
2. Some daily selectors inspected every timestamp in the completed session and
   chose the day's maximum model score.

The source has been changed so entry eligibility uses entry-time data and the
first observable threshold crossing. Existing generated reports are not
silently rewritten; `artifact_status.csv` marks them stale until rebuilt.

`../causal_validation_short_vol/` is a fresh, independent validation run. Its
threshold was chosen on 2023 H1 and frozen before 2023 H2. The strategy failed
that validation, so its deployable P&L is zero and it must not be activated in
live inference.
