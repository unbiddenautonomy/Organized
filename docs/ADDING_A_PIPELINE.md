# Adding a pipeline

A new hypothesis is one new folder under `pipelines/` and one entry in
`config/service.json`. Nothing in `voltr/` or another pipeline changes.

1. **Copy the template.** `pipelines/_template/` → `pipelines/<name>/` and rename
   the classes. Its test passes as soon as it is copied.
2. **Build features.** In `pipeline.py`, start from `TemplateAdapter`. For the
   standard five-strike ATM row (IV per leg, greeks, surfaces, OI, lags), subclass
   `voltr.market.atm_features.AtmFeatureStream` and implement `_finalize_minute`
   (see `pipelines/weeks43/pipeline.py: OtmFeatureStream`). Otherwise use the
   decoded updates directly.
3. **Score and trade on paper.** Keep models in `models/` with a manifest, entry
   rules in a `strategy.py`, positions in a `book.py`. Write ledgers with
   `voltr.core.io` (`atomic_write_json`, `append_jsonl`) under the pipeline's
   ignored `logs/` folder. Keep your own cost formula with your rules.
4. **Keep research apart.** Trainers, notebooks and studies go in
   `research/<name>/`; only deployed artifacts live in `pipelines/<name>/models/`.
   Add the deployed files' hashes to `setup/model_manifest.json` and a preflight
   check to `setup/voltr.py`.
5. **Register it.** Add an entry to `config/service.json`:

   ```json
   "my_pipeline": {"adapter": "pipelines.my_pipeline.pipeline:MyAdapter",
                   "lock": "pipelines/my_pipeline/logs/worker.lock"}
   ```

   It is loaded only when its name is in `enabled_pipelines`.
6. **Prove it.** `python -m pytest pipelines/<name>`, a replay of a past day,
   `python setup/voltr.py preflight --pipeline all`, then a live paper session
   before enabling it in the scheduled service.
