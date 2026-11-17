# Intraday/EOD model orchestration

## Decision

Run both candidates on every eligible feature row. Only the selected champion's
forecast reaches the trading policy; the other remains a shadow forecast. Keep
champions independently for each target and horizon because a model that wins at
5 minutes can lose at 15 minutes.

```text
market data -> point-in-time features -> intraday model ----\
                                      -> EOD model ----------+-> forecast ledger
                                                             |       |
realized value (after horizon expires) ----------------------+-> paired scoring
                                                                     |
                                                  guarded selector -> champion map
                                                                     |
                                                        trading/risk engine
```

## Update semantics

- The intraday trainer runs every 120 minutes, uses data only through its declared
  `trained_through` watermark, and publishes an immutable version. An online
  partial-fit implementation may replace this trainer later without changing the
  orchestrator contract.
- The EOD trainer runs after 15:45 IST against the completed session and publishes
  an immutable version after validation.
- Training never overwrites an artifact in use. Publish to a versioned directory,
  validate load + schema + a smoke prediction, then register it.
- New versions start in shadow. Registration is not promotion.

## Selection semantics

Comparison is paired: both models must have predicted the same target, horizon,
and timestamp before the realized label arrived. The default selector uses the
latest 500 paired outcomes and requires at least 100, at least 2% lower MAE, no
loss in direction accuracy, and a 60-minute switch cooldown. The first healthy
EOD model is the safe bootstrap champion. A stale champion falls back to a fresh
healthy model.

This prevents look-ahead, unequal-sample comparisons, noisy one-trade switches,
and a single global winner masking horizon-specific weakness. For trading P&L,
add a second gate based on net P&L/drawdown only after enough actual fills exist;
forecast accuracy alone is not a profitability guarantee.

## Integration contract

1. Trainers publish an immutable artifact and call `ModelRegistry.register`.
2. Live inference loads both registered candidates and calls `record_prediction`
   for each output.
3. When an observed spot/IV value arrives, call `settle` with its timestamp.
4. Call `evaluate` after settlement; call `route` before passing a forecast into
   the existing paper/live engine.
5. Persist `model_id`, training watermark, feature schema version, prediction,
   selected champion, and switch reason with every decision.

Do not enable automatic live-capital switching initially. Run the full mechanism
in paper/shadow mode for multiple market regimes, then enable promotion with a
manual kill switch and the existing risk controls.
