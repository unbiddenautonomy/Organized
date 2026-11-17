# Template pipeline

Copy this folder to `pipelines/<name>/`, rename `TemplateAdapter`, then follow
[docs/ADDING_A_PIPELINE.md](../../docs/ADDING_A_PIPELINE.md).

| File | Holds |
|---|---|
| `pipeline.py` | the adapter the service worker drives: feed in, paper decisions out |
| `models/` | deployed artifacts and their manifest |
| `tests/` | this pipeline's tests (run with `python -m pytest pipelines/<name>`) |

Add `strategy.py` (entry/exit rules and costs), `book.py` (paper positions) and
`dashboard/` as the strategy grows; `pipelines/ivrv/` shows the full shape.
