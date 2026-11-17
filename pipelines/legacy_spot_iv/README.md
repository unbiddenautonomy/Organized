# NSE FO UDP 7208 decoder

> This is the original decoder and standalone-engine guide. The `E:\` launch
> paths below are historical. Use the repository root README and
> `setup/voltr.py` for this portable checkout; the older spot/IV engine
> still requires the missing model packages listed in `models/legacy_spot_iv/`.

This receiver joins multicast `233.1.2.5:34330`, unpacks NSE sequential
broadcast packets, decompresses LZO1Z messages with the copied LZO 2.10 source,
filters transaction code 7208 (`BCAST_ONLY_MBP`), and prints one JSON line per
security update.

The copied Drop Copy 1.9 PDF documents a different TCP protocol and transaction
code 2222. UDP/LZO/7208 framing and field offsets are therefore implemented from
the NSE FO broadcast protocol reference already present in the ScanRisk source.

## Run

From PowerShell:

```powershell
cd 'E:\Models\Live Trades'
.\build_lzo.ps1
python .\main.py
```

Stop with Ctrl+C. If the multicast feed is delivered through a specific network
adapter, replace `interface_ip` in `config.json` with that adapter's IPv4 address.

Every received UDP datagram (raw hex), ignored message, decoder warning, and
decoded 7208 JSON update is appended to `logs/feed.log`.

At startup, the application queries `public.nse_fo_contract` for BANKNIFTY CE
and PE contracts belonging to the nearest non-expired expiry. An in-memory
order book is keyed by integer `fininstrmid` for O(1) updates. Every matching
book update, enriched with its contract metadata, is appended to
`logs/banknifty_orderbook.jsonl`.

For each contract, the in-memory book retains exactly five bid levels and five
ask levels. NSE 7208 defines depth positions 1-5 as buys and positions 6-10 as
sells; the `BbBuySellFlag` field is currently zero and is not used for routing.

The book also watches every unique `undrlygfininstrmid` referenced by the loaded
contracts. Feed updates for an underlying token are retained as a separate
top-five book and persisted with `event_type: "underlying"`; option updates use
`event_type: "option"`.

`spot_inference.py` is the dedicated BANKNIFTY spot prediction pipeline. It
decodes BANK NIFTY from message 7207, tracks the nearest FUTIDX token from the
database, creates synchronized one-minute bars, builds the training-compatible
11 base and 200 engineered features, applies the saved fold-5 preprocessors,
and emits 5/15/30/45/60-minute predictions to
`logs/spot_predictions.jsonl`. Approximately 120 uninterrupted market minutes
are required before the first prediction.

Prices are emitted both as their raw integer value and divided by 100. Each JSON
record includes the token, LTP, volume, ten MBP depth levels, aggregate buy/sell
quantities, OHLC prices, trading status, and broadcast sequence number.
# Live IV inference

`iv_inference.py` consumes BANK NIFTY 7207 spot plus 7208 future and nearest-expiry
option updates. It selects the minute's nearest ATM CE/PE pair, reconstructs IV and
Black-Scholes Greeks, builds the training-compatible 1/5/15/30/60-minute features,
and runs fold 5 of `iv_expanding_window_v3`. Predictions are appended to
`logs/iv_predictions.jsonl`.

The paper engine evaluates the 5/15/30/45/60-minute spot and IV forecasts at
each synchronized decision point. It selects the eligible horizon with the
strongest predicted IV contraction per square-root minute, excludes horizons
that would extend beyond the forced exit, and applies the configured
horizon-specific combined-premium stop. OI freshness is recorded but is
advisory in paper mode; executable quotes, spread, VIX, IV, risk, margin,
daily-loss and market-time controls remain mandatory.

A supportive 15-minute opportunity is prioritized and entered in two stages
when sizing permits. Half the planned lots enter at time `t`. At `t+5`, a
remaining-ten-minute estimate is formed from the midpoint of the new 5m and
15m forecasts. The reserved lots enter on the original straddle only when its
spot direction matches the original `t+15` direction and IV, quotes, spread,
risk and margin checks still pass. The full comparison and outcome are written
to `logs/paper_decisions.jsonl`.

Every accepted or rejected entry decision is appended to
`logs/paper_decisions.jsonl`, including human-readable reasons and the market,
forecast, quote, OI, risk and margin values used by the applicable checks.

Replay feeds are supported while the process remains running. If embedded
exchange time rewinds by more than five minutes, spot/IV aggregations, cached
quotes, OI freshness and paper-session state reset automatically; JSONL files
remain append-only and receive the new replay pass once each pipeline warms up.

The 7208 MBP record does not contain contract open interest. `call_oi` and `put_oi`
are therefore represented as missing values; saved model pipelines apply their
training-time median imputers. Each output event records this limitation.

## Expanded 120-minute hybrid paper inference

`banknifty_120m_live_inference.py` loads the frozen RV artifact and the expanded
hybrid artifact from `E:\volpred`. It consumes the existing CM/FO multicast
feeds, selects the nearest monthly expiry, builds five fixed strike pairs, and
evaluates this causal live hierarchy:

1. the existing 120-minute RV-versus-IV short ATM-straddle cost gate;
2. a one-lot ATM-2 put plus ATM+2 call short strangle when its frozen profitable-
   trade probability reaches `0.693128`;
3. a one-lot ATM short straddle when its probability reaches `0.701625`.

Only the first eligible component is entered, positions cannot overlap, entry
ends at 13:30, and every position exits after 120 minutes or by the session
close. This worker has no broker API and writes only paper signals/trades under
`logs/rv120`.

The offline research chose the highest score observed over a full day and knew
whether the base component traded anywhere on that date. Those facts are not
available causally. Live paper inference therefore uses the first threshold
crossing and at most one fallback per day. Its live results must be evaluated
separately from the offline backtest.

Preflight:

```powershell
python .\banknifty_120m_live_inference.py `
  --contract-file 'E:\Models\seqToseq\contracts\NSE_FO_contract_06102026.csv.gz' `
  --preflight
```

Start the paper worker and dashboard:

```powershell
python .\banknifty_120m_live_inference.py `
  --contract-file 'E:\Models\seqToseq\contracts\NSE_FO_contract_06102026.csv.gz'
```

The current worker is started with `--dashboard-port 8870` because port 8770
is already occupied. Dashboard: `http://127.0.0.1:8870`.

## Live intraday retraining

`live_retraining.py` is attached directly to the spot and IV prediction listeners.
Those predictions originate from the CM/FO UDP multicast streams after LZO
decompression, decoding, minute aggregation, and the production feature pipelines.
Each later observation settles only forecasts whose horizon has matured.

Every 120 exchange minutes, a background worker fits horizon-specific residual
correction models for spot and IV. The UDP selector thread performs only SQLite
inserts and never fits a model. Training uses a chronological holdout, publishes
only candidates that improve on the unchanged base forecast, writes immutable
version directories under `artifacts/intraday_live`, and atomically updates the
`CURRENT` pointer. New predictions are written in shadow mode to
`logs/intraday_shadow_predictions.jsonl`.

The retrainer deliberately keeps the large production spot/IV models and their
scalers frozen during market hours. Refitting those entire estimators from only
two hours of observations would be unstable. Full base-model, scaler, P&L, MAE,
and HMM parameter retraining remains an EOD operation over the complete historical
dataset. The dashboard exposes ledger counts, worker state, and loaded intraday
candidates under `intraday_retraining`.
