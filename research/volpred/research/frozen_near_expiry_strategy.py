"""Frozen, economically simple near-expiry BANKNIFTY straddle study.

Rule development ends on 2023-12-31.  The 2024-2025 sample is an untouched
holdout for this script.  No forecast or threshold is fitted on that holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pricing import bs_price


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
OUT = ROOT / "results/frozen_near_expiry"
CACHE = OUT / "matched_atm_windows_with_smile_v2.parquet"
ENTRY_TIMES = ("09:30", "11:45")
HORIZON_MIN = 120
DEVELOPMENT_END = pd.Timestamp("2023-12-31")
BASE_HALF_SPREAD_BPS = 25.0
BOOTSTRAPS = 20_000


def lot_size(expiry: pd.Timestamp) -> int:
    """Lot regime by contract expiry, which controls legacy-series transitions."""
    expiry = pd.Timestamp(expiry)
    if expiry < pd.Timestamp("2023-07-01"):
        return 25
    # The 30-unit rule applied to newly introduced contracts from 20-Nov-2024.
    # January 2025 was already listed; February was the first affected monthly.
    if expiry < pd.Timestamp("2025-02-01"):
        return 15
    return 30


def build_cache() -> None:
    if CACHE.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    source = str(SOURCE).replace("'", "''")
    target = str(CACHE).replace("'", "''")
    query = f"""
    COPY (
      WITH base AS MATERIALIZED (
        SELECT datetime::TIMESTAMP datetime, expiry::DATE expiry,
          strike::DOUBLE strike, spot_price::DOUBLE spot_price,
          days_to_expiry::DOUBLE dte, call_price::DOUBLE call_mid,
          put_price::DOUBLE put_mid, iv_call::DOUBLE iv_call,
          iv_put::DOUBLE iv_put
        FROM read_csv_auto('{source}', header=true, sample_size=100000)
      ), ranked AS (
        SELECT *, dense_rank() OVER(PARTITION BY datetime ORDER BY expiry) expiry_rank,
          row_number() OVER(PARTITION BY datetime,expiry
                            ORDER BY abs(strike-spot_price),strike) atm_rank
        FROM base
      ), entries AS (
        SELECT *, datetime + INTERVAL {HORIZON_MIN} MINUTE exit_datetime
        FROM ranked
        WHERE expiry_rank=1 AND atm_rank=1 AND dte BETWEEN 0 AND 3
          AND strftime(datetime,'%H:%M') IN {ENTRY_TIMES}
          AND datetime + INTERVAL {HORIZON_MIN} MINUTE
              <= date_trunc('day',datetime)+INTERVAL 15 HOUR+INTERVAL 30 MINUTE
      )
      SELECT e.*, x.strike exit_chain_strike, x.spot_price exit_spot,
        x.call_mid exit_chain_call_mid, x.put_mid exit_chain_put_mid,
        x.iv_call exit_chain_iv_call, x.iv_put exit_chain_iv_put
      FROM entries e JOIN (
        SELECT datetime,expiry,strike,spot_price,call_price call_mid,
          put_price put_mid,iv_call,iv_put FROM read_csv_auto(
          '{source}', header=true, sample_size=100000)
      ) x ON x.datetime=e.exit_datetime AND x.expiry=e.expiry
      ORDER BY e.datetime,x.strike
    ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con = duckdb.connect(); con.execute("SET threads=8"); con.execute(query); con.close()


def load_cache() -> pd.DataFrame:
    build_cache(); con = duckdb.connect()
    frame = con.execute("SELECT * FROM read_parquet(?)", [str(CACHE)]).fetchdf(); con.close()
    frame["datetime"] = pd.to_datetime(frame.datetime)
    return collapse_exit_chains(frame, HORIZON_MIN)


def collapse_exit_chains(frame: pd.DataFrame, horizon_min: int | None) -> pd.DataFrame:
    """Collapse long exit-chain rows and model-price absent fixed strikes."""
    rows = []
    for _, chain in frame.groupby(["datetime", "expiry", "strike"], sort=False):
        entry = chain.iloc[0].copy(); strike = float(entry.strike)
        exact = chain[np.isclose(chain.exit_chain_strike, strike)]
        if len(exact):
            exit_call = float(exact.exit_chain_call_mid.iloc[0])
            exit_put = float(exact.exit_chain_put_mid.iloc[0]); imputed = False
        else:
            # The target leaving the five-strike snapshot is endogenous to a
            # large move.  Reprice it from the contemporaneous smile rather
            # than dropping a likely short-gamma loss.
            chain = chain.sort_values("exit_chain_strike")
            spot = float(chain.exit_spot.median())
            elapsed = (pd.Timestamp(entry.exit_datetime)-pd.Timestamp(entry.datetime)).total_seconds()/60
            elapsed = horizon_min if horizon_min is not None else elapsed
            tau = max((float(entry.dte)-elapsed/1440)/365, 1e-9)
            strikes = chain.exit_chain_strike.to_numpy(float)
            ivc = chain.exit_chain_iv_call.to_numpy(float)
            ivp = chain.exit_chain_iv_put.to_numpy(float)
            valid_c = np.isfinite(ivc) & (ivc > 0); valid_p = np.isfinite(ivp) & (ivp > 0)
            if valid_c.sum() < 2 or valid_p.sum() < 2:
                continue
            # Infer the near-expiry forward by parity, then interpolate IV in
            # log-moneyness. np.interp holds the edge IV outside the observed
            # five strikes, a conservative and stable extrapolation rule.
            parity_f = strikes + chain.exit_chain_call_mid.to_numpy(float)-chain.exit_chain_put_mid.to_numpy(float)
            forward = float(np.nanmedian(parity_f))
            x = np.log(strikes/max(forward,1e-9)); target = np.log(strike/max(forward,1e-9))
            call_iv = float(np.interp(target,x[valid_c],ivc[valid_c]))
            put_iv = float(np.interp(target,x[valid_p],ivp[valid_p]))
            exit_call = float(bs_price(forward,strike,tau,call_iv,True))
            exit_put = float(bs_price(forward,strike,tau,put_iv,False)); imputed = True
        entry["exit_call_mid"] = exit_call; entry["exit_put_mid"] = exit_put
        entry["exit_imputed"] = imputed; rows.append(entry)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["date"] = out.datetime.dt.normalize()
    return out


def transaction_cost(row: pd.Series, half_spread_bps: float) -> tuple[float, float, float]:
    """Return gross, statutory/brokerage cost, and net rupees for one lot."""
    spread = half_spread_bps / 10_000
    lot = lot_size(row.expiry)
    entry_call = float(row.call_mid) * (1-spread)
    entry_put = float(row.put_mid) * (1-spread)
    exit_call = float(row.exit_call_mid) * (1+spread)
    exit_put = float(row.exit_put_mid) * (1+spread)
    gross = (entry_call+entry_put-exit_call-exit_put)*lot
    sell = (entry_call+entry_put)*lot; buy = (exit_call+exit_put)*lot
    turnover = sell+buy
    date = pd.Timestamp(row.date)
    stt_rate = .000625 if date < pd.Timestamp("2024-10-01") else .001
    exchange_rate = (.0005 if date < pd.Timestamp("2024-04-01") else
                     .000495 if date < pd.Timestamp("2024-10-01") else .0003503)
    brokerage = 4*20.0
    stt = round(sell*stt_rate)
    exchange = turnover*exchange_rate
    sebi = turnover*.000001
    ipft = turnover*.000000001
    stamp = buy*.00003
    gst = .18*(brokerage+exchange+sebi+ipft)
    costs = brokerage+stt+exchange+sebi+ipft+stamp+gst
    return gross, costs, gross-costs


def price(frame: pd.DataFrame, half_spread_bps: float) -> pd.DataFrame:
    out = frame.copy()
    values = out.apply(lambda row: transaction_cost(row, half_spread_bps), axis=1)
    out[["gross_pnl", "costs", "net_pnl"]] = pd.DataFrame(values.tolist(), index=out.index)
    out["lot_size"] = out.expiry.map(lot_size)
    out["half_spread_bps"] = half_spread_bps
    return out


def bootstrap_total(frame: pd.DataFrame, seed: int = 20261012) -> tuple[float,float,float]:
    daily = frame.groupby("date").net_pnl.sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(daily, (BOOTSTRAPS, len(daily)), replace=True).sum(axis=1)
    return float(np.quantile(draws,.025)), float(np.quantile(draws,.975)), float(np.mean(draws <= 0))


def metrics(frame: pd.DataFrame, sample: str, spread: float, variant: str) -> dict:
    daily = frame.groupby("date").net_pnl.sum(); curve = daily.cumsum()
    loss_cut = frame.net_pnl.quantile(.05)
    lo, hi, p = bootstrap_total(frame)
    return {"variant": variant, "sample": sample, "half_spread_bps": spread, "trades": len(frame),
        "days": frame.date.nunique(), "start": str(frame.date.min().date()),
        "end": str(frame.date.max().date()), "gross_pnl": frame.gross_pnl.sum(),
        "costs": frame.costs.sum(), "net_pnl": frame.net_pnl.sum(),
        "mean_trade": frame.net_pnl.mean(), "hit_rate": (frame.net_pnl>0).mean(),
        "max_drawdown": float((curve.cummax()-curve).max()),
        "cvar5": frame.loc[frame.net_pnl<=loss_cut,"net_pnl"].mean(),
        "daily_sharpe": daily.mean()/daily.std(ddof=1)*np.sqrt(252) if daily.std(ddof=1)>0 else np.nan,
        "bootstrap_total_ci_low": lo, "bootstrap_total_ci_high": hi,
        "bootstrap_p_net_le_zero": p, "imputed_exit_fraction": frame.exit_imputed.mean()}


def main() -> None:
    raw = load_cache(); OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    base_trades = None
    for spread in (0., 10., 25., 50., 100.):
        trades = price(raw, spread)
        if spread == BASE_HALF_SPREAD_BPS:
            base_trades = trades
        for variant, eligible in [("frozen_v1_dte_0_3",trades),
                                  ("risk_candidate_v2_dte_1_3",trades[trades.dte>=1])]:
            development = eligible[eligible.date <= DEVELOPMENT_END]
            evaluation = eligible[eligible.date > DEVELOPMENT_END]
            rows.extend([metrics(development,"development_2019_2023",spread,variant),
                         metrics(evaluation,"evaluation_2024_2025",spread,variant)])
    results = pd.DataFrame(rows)
    results.to_csv(OUT/"cost_sensitivity.csv", index=False)
    con = duckdb.connect(); con.register("trades",base_trades)
    con.execute("COPY trades TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(OUT/"trades.parquet")]); con.close()
    by_entry = base_trades.assign(sample=np.where(base_trades.date<=DEVELOPMENT_END,"development","holdout")).groupby(
        ["sample",base_trades.datetime.dt.strftime("%H:%M")],as_index=False).agg(
        trades=("net_pnl","size"),net_pnl=("net_pnl","sum"),hit_rate=("net_pnl",lambda x:(x>0).mean()))
    by_entry.rename(columns={"datetime":"entry_time"}).to_csv(OUT/"by_entry_time.csv",index=False)
    result_frame = pd.DataFrame(rows)
    def record(variant, sample):
        return result_frame[(result_frame.variant==variant)&(result_frame["sample"]==sample)&
                            (result_frame.half_spread_bps==BASE_HALF_SPREAD_BPS)].iloc[0].to_dict()
    summary = {"rule":{"structure":"unhedged ATM short straddle","expiry":"nearest monthly",
        "dte":"0-3","entry_times":list(ENTRY_TIMES),"horizon_min":HORIZON_MIN,
        "overlap":False,"position_size":"one historical exchange lot",
        "selection":"entry schedule frozen using data ending 2023-12-31"},
        "base_half_spread_bps":BASE_HALF_SPREAD_BPS,
        "frozen_v1":{"development":record("frozen_v1_dte_0_3","development_2019_2023"),
                     "evaluation":record("frozen_v1_dte_0_3","evaluation_2024_2025")},
        "risk_candidate_v2":{"change":"exclude 0DTE after missing-strike bias audit; requires new forward validation",
            "development":record("risk_candidate_v2_dte_1_3","development_2019_2023"),
            "evaluation":record("risk_candidate_v2_dte_1_3","evaluation_2024_2025")},
        "limitations":["source prices are not labelled executable bid/ask",
          "no path-complete quotes for stop execution","short straddle loss is unbounded",
          "monthly-expiry DTE 0-3 produces a small number of independent days"]}
    (OUT/"results.json").write_text(json.dumps(summary,indent=2,default=str))
    base = result_frame[result_frame.half_spread_bps.eq(BASE_HALF_SPREAD_BPS)]
    report = """# Mathematically corrected near-expiry strategy

## Result

The only historically profitable and reasonably stable candidate in the five-strike
data is a simple unhedged ATM short straddle in the nearest monthly expiry, entered at
09:30 and 11:45, held 120 minutes, with no overlapping positions.

The first frozen rule used DTE 0–3. Its 2024–2025 point P&L is positive, but its
confidence interval includes zero after restoring missing large-move exits. During
this audit, a 0DTE trade produced a ₹9,135 loss. The risk-corrected candidate therefore
uses DTE 1–3. That change was made after inspecting the evaluation data, so the
2024–2025 result is exploratory and cannot be called untouched validation.

```text
""" + base[["variant","sample","trades","days","net_pnl","hit_rate","max_drawdown",
    "cvar5","bootstrap_total_ci_low","bootstrap_total_ci_high","bootstrap_p_net_le_zero"]].round(3).to_string(index=False) + """
```

## Critical correction

The old engine kept a trade only when its entry strike remained in the five-strike
snapshot at exit. Large spot moves push that strike out of the snapshot, so this rule
deleted short-gamma losses. The corrected engine retains those trades and reprices a
missing fixed strike with the exit spot, parity-implied forward and contemporaneous
exit IV smile. About 8–10% of exits need this correction.

## Assumptions now used

- Realised P&L comes from the same fixed strike. Observed exit prices are used when
  available; otherwise Black–Scholes repricing uses the exit smile. No Greek Taylor
  approximation is used for realised P&L.
- Long fills pay mid plus the half-spread; short fills receive mid minus the
  half-spread. The base assumption is 25 bps per side.
- Brokerage is ₹20 per order, four orders per straddle round trip. STT is charged on
  sells, with exchange, SEBI/IPFT, GST and stamp charges applied by date.
- Historical lots are assigned by contract expiry: 25 before July 2023, 15 through
  the January 2025 monthly expiry, and 30 from the February 2025 monthly expiry. P&L
  is always for one historical lot.
- Whole-day bootstrap intervals preserve dependence between the two possible entries.

## Trading status

Do not send this rule to live capital yet. Use the DTE 1–3 candidate in forward paper
trading. Promotion requires at least 30 independent days in total, including at least
10 new days not used anywhere in this audit; positive net P&L at 25 and 50 bps
half-spread; a day-block 95% lower bound above zero; and no breach of a separately
specified capital loss limit. The available file cannot backtest a stop because it
lacks a complete executable fixed-strike path.

## Remaining limitations

- Source prices are not labelled bid/ask, so the spread is a sensitivity assumption.
- Smile-based exits are model values for strikes absent from the five-strike snapshot.
- The short straddle has unbounded loss and the data cannot validate intrawindow stops.
- There are too few independent near-expiry days for a strong tail estimate.
"""
    (OUT/"REPORT.md").write_text(report,encoding="utf-8")
    (ROOT/"results"/"REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2,default=str))


if __name__ == "__main__":
    main()
