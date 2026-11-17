# RV120: 120-minute realised-volatility / expanded-hybrid engine

An independent paper engine with its own process, dashboard (port 8770) and
config; it is not a worker of the daily service.

| File | Holds |
|---|---|
| `engine.py` | feed handling, strategy (`LiveRvStrategy`), costs (`charge_components`), dashboard server, CLI |
| `models/` | `rv_intraday_120m_model.joblib`, `expanded_hybrid_models.joblib` |
| `config.example.json` | the settings it reads; copy to the ignored `config.json` (`setup/voltr.py configure` does this) |
| `dashboard.html` | its dashboard |
| `tests/` | its tests |

Run from the repository root with a current FO master:

```powershell
python -m pipelines.rv120.engine --config pipelines/rv120/config.json --contract-file <NSE_FO_contract_DDMMYYYY.csv.gz>
```

Add `--preflight` to load contracts and models without opening sockets.
Training code is in `research/volpred/` (enhanced RV models).
