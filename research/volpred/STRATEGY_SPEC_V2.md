# Frozen Research V2 economic mapping

Status: **FROZEN BEFORE RESEARCH V2 STRATEGY P&L** on 2026-10-13.
This is research-grade only. The pseudo-holdout is previously exposed and can
never support an acceptance or proof-of-profitability claim.

## Inputs and prediction provenance

- Horizon: 120 trading minutes.
- Development signal: for each development row, use the mean of the seven P2
  CPCV forecasts produced when its one-of-eight group is held out. Every
  contributing model excludes that row's group, applies label-end purging and
  one-session embargo, and uses training-only smearing.
- Walk-forward validation signal: frozen monthly P2 predictions already written
  by the cleared Research V2 runner for 2023-01 through 2024-06.
- Pseudo-holdout signal: the single consumed prediction ledger produced by the
  frozen P2 artifact. It may be joined only after all strategy code, thresholds,
  costs, risk rules and hashes are frozen.
- Model identities are those in `FROZEN_PREDICTIVE_SPEC_V2.json`. No refit is
  permitted inside the strategy engine.

## Option universe and causal clock

- Source: `data/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv`.
- Only recorded monthly contracts are eligible. At each decision, select the
  nearest recorded nonexpired expiry; never infer an expiry weekday.
- Candidate decision clocks are exactly 10:15, 11:45 and 13:25 Asia/Kolkata.
  Use at most the first qualifying signal per session. The 09:45 diagnostic is
  unavailable because the frozen 60-minute feature window is incomplete; it is
  not replaced by an optimized time.
- `available_at = max(datetime, Date + tick_time)`. Entry is the first complete
  eligible option snapshot strictly after the decision. Exit is the first
  complete eligible snapshot at or after decision plus 120 minutes, no later
  than 15:30. Expiry and absolute strikes never change.
- Future exit availability never selects a trade. Every signal remains in the
  ledger through rejection, observed exit, or the frozen adverse missing exit.

## Variance signal

```text
atm_iv = mean(iv_call, iv_put)
iv_variance_120 = atm_iv^2 * 120 / (252 * 375)
short_ratio = iv_variance_120 / P2_RV_forecast_120
long_ratio  = P2_RV_forecast_120 / iv_variance_120
```

ATM call and put IV must be positive, finite and converged. No predicted IV or
future IV enters.

## Structures and thresholds

Three sensitivity levels are separate trials: `{1.15, 1.25, 1.35}`. The central
`1.25` rule is the frozen primary policy; neighbors are robustness diagnostics
and can never replace it after outcomes are opened.

1. If `short_ratio >= threshold`, sell an ATM iron fly: short the ATM call and
   put, buy the call two recorded strike positions above and put two positions
   below. All four legs and their current quality fields must exist.
2. Else if `long_ratio >= threshold`, buy the ATM call and put.
3. Else no trade.

There is no naked short, discretionary structure selection, stop-loss, delta
hedge, averaging, pyramiding or overlapping position.

## Fill, costs and margin

- One historical lot: assumed 15 through 2024-12-24 and 30 afterward, explicitly
  unverified. A constant-15 sensitivity is mandatory.
- Stored prices are LTP-like and never called executable fills. Apply
  `research/execution_costs.py` with dated charges, INR 20/order, spread/impact,
  margin carry and settlement handling.
- `volatility_ratio = max(1, atm_iv / trailing_median_atm_iv)`, using only the
  previous 20 eligible sessions. Fewer than 20 sessions rejects the fill.
- OI must be present and nonnegative. Require full atomic completion of every
  leg under `FillPolicy(max_primary_lots=1, oi_participation=0.001,
  unknown_oi_capacity_lots=0)`; any leg failure rejects the whole position.
- Report ordinary, 2x all-component costs, and an uncertainty case with 3x
  spread/impact plus 2x statutory/brokerage fees on identical trades.
- Defined maximum loss/debit plus the 20% buffer must not exceed INR 350,000.
  Margin carry is 8% annualized for actual holding time.

## Missing exits and risk

- Missing/invalid fixed-strike exit never removes a selected entry.
- Iron fly missing exit: charge contractual maximum loss plus stressed close
  costs. Long straddle missing exit: charge the full entry debit plus stressed
  close/settlement costs.
- Apply `RISK_POLICY.md`: one position/day, INR 7,500 pretrade severe-loss cap,
  INR 10,000 daily stop, INR 25,000 weekly stop and INR 50,000 drawdown stop.
- Predeclared official event dates are no-entry in the primary policy. Event-day
  trades are retained only as a labeled counterfactual diagnostic.

## Plateau and promotion rule

No threshold is optimized: 1.25 is primary. Plateau passes only if 1.15 and
1.35 have the same sign of total ordinary-cost net P&L as 1.25 and each has net
P&L per filled trade within 25% of primary. This can reject but cannot select a
neighbor.

For development, validation and the once-opened strategy pseudo-holdout, report
gross/net P&L under all costs; fills/rejections/missing exits; worst day/week,
drawdown, 95% loss CVaR, win rate, turnover, margin and cost components; event
behavior and concentration; deflated Sharpe using full `TRIALS.csv`; and PBO
when estimable.

`CANDIDATE FOR FORWARD SHADOW TEST` requires positive ordinary and 2x net P&L
in development and validation, plateau and all risk gates, no month above 35%
of profit, and a passing one-shot strategy pseudo-holdout. Otherwise the verdict
is `REJECTED`. Neither outcome is acceptance.
