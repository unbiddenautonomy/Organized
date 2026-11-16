"""Corrected five-strike study for the requested DTE 4-20 range."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from frozen_near_expiry_strategy import (
    BASE_HALF_SPREAD_BPS, DEVELOPMENT_END, bootstrap_total,
    collapse_exit_chains, price,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"data/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
OUT = ROOT/"results/dte4_20"
CACHE = OUT/"exit_chains.parquet"
HORIZON_MIN = 120
ENTRY_TIMES = ("09:30","11:45")


def build_cache() -> None:
    if CACHE.exists(): return
    OUT.mkdir(parents=True,exist_ok=True)
    src=str(SOURCE).replace("'","''"); dst=str(CACHE).replace("'","''")
    q=f"""COPY (
      WITH base AS MATERIALIZED (
        SELECT datetime::TIMESTAMP datetime,expiry::DATE expiry,strike::DOUBLE strike,
          spot_price::DOUBLE spot_price,days_to_expiry::DOUBLE dte,
          call_price::DOUBLE call_mid,put_price::DOUBLE put_mid,
          iv_call::DOUBLE iv_call,iv_put::DOUBLE iv_put
        FROM read_csv_auto('{src}',header=true,sample_size=100000)
      ), ranked AS (
        SELECT *,dense_rank() over(partition by datetime order by expiry) expiry_rank,
          row_number() over(partition by datetime,expiry order by abs(strike-spot_price),strike) atm_rank
        FROM base
      ), entries AS (
        SELECT *,datetime+INTERVAL {HORIZON_MIN} MINUTE exit_datetime FROM ranked
        WHERE expiry_rank=1 AND atm_rank=1 AND dte BETWEEN 4 AND 20.999999
          AND strftime(datetime,'%H:%M') IN {ENTRY_TIMES}
      )
      SELECT e.*,x.strike exit_chain_strike,x.spot_price exit_spot,
        x.call_mid exit_chain_call_mid,x.put_mid exit_chain_put_mid,
        x.iv_call exit_chain_iv_call,x.iv_put exit_chain_iv_put
      FROM entries e JOIN base x ON x.datetime=e.exit_datetime AND x.expiry=e.expiry
      ORDER BY e.datetime,x.strike
    ) TO '{dst}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
    c=duckdb.connect(); c.execute("set threads=8"); c.execute(q); c.close()


def load() -> pd.DataFrame:
    build_cache(); c=duckdb.connect(); x=c.execute("select * from read_parquet(?)",[str(CACHE)]).fetchdf(); c.close()
    x["datetime"]=pd.to_datetime(x.datetime)
    return collapse_exit_chains(x,HORIZON_MIN)


def metrics(z: pd.DataFrame, variant: str, sample: str, spread: float) -> dict:
    daily=z.groupby("date").net_pnl.sum(); curve=daily.cumsum(); cut=z.net_pnl.quantile(.05)
    lo,hi,p=bootstrap_total(z)
    return {"variant":variant,"sample":sample,"half_spread_bps":spread,"trades":len(z),
      "days":z.date.nunique(),"net_pnl":z.net_pnl.sum(),"mean_trade":z.net_pnl.mean(),
      "hit_rate":float((z.net_pnl>0).mean()),"max_drawdown":float((curve.cummax()-curve).max()),
      "cvar5":z.loc[z.net_pnl<=cut,"net_pnl"].mean(),"ci_low":lo,"ci_high":hi,
      "p_net_le_zero":p,"imputed_exit_fraction":z.exit_imputed.mean()}


def main() -> None:
    raw=load(); rows=[]; base=None
    variants={"all_dte_4_20":lambda x:x,"dte_4_7":lambda x:x[x.dte<8],
              "dte_8_14":lambda x:x[(x.dte>=8)&(x.dte<15)],
              "dte_15_20":lambda x:x[x.dte>=15]}
    for spread in (0.,10.,25.,50.,100.):
      priced=price(raw,spread)
      if spread==BASE_HALF_SPREAD_BPS: base=priced
      for name,fn in variants.items():
        eligible=fn(priced)
        rows.append(metrics(eligible[eligible.date<=DEVELOPMENT_END],name,"development_2019_2023",spread))
        rows.append(metrics(eligible[eligible.date>DEVELOPMENT_END],name,"evaluation_2024_2025",spread))
    result=pd.DataFrame(rows); result.to_csv(OUT/"cost_sensitivity.csv",index=False)
    base.to_csv(OUT/"trade_ledger_25bps.csv",index=False)
    chosen=result[(result.variant=="dte_4_7")&(result.half_spread_bps==25)]
    decisions=pd.DataFrame([
      {"dte_band":"4-7","punching_status":"PAPER_ONLY","reason":"positive historical point estimate; needs untouched forward sample"},
      {"dte_band":"8-14","punching_status":"NO_TRADE","reason":"development sample negative after corrected costs"},
      {"dte_band":"15-20","punching_status":"NO_TRADE","reason":"development and evaluation evidence negative"},
    ]); decisions.to_csv(OUT/"punching_policy.csv",index=False)
    payload={"rule":{"entry_times":list(ENTRY_TIMES),"horizon_min":HORIZON_MIN,
      "structure":"ATM short straddle","expiry":"nearest monthly"},
      "dte_4_7":chosen.to_dict("records"),"punching_policy":decisions.to_dict("records")}
    (OUT/"results.json").write_text(json.dumps(payload,indent=2,default=str))
    table=result[(result.half_spread_bps==25)&result.variant.isin(["dte_4_7","dte_8_14","dte_15_20"])][
      ["variant","sample","trades","days","net_pnl","hit_rate","max_drawdown","cvar5","ci_low","ci_high"]]
    report="""# Corrected DTE 4-20 research

Every DTE band was evaluated, but a trade is not forced merely because its DTE is in
range. Entries use the nearest monthly ATM straddle at 09:30 or 11:45 and exit after
120 minutes. Missing fixed-strike exits are retained and repriced from the exit smile.

```text
"""+table.round(2).to_string(index=False)+"""
```

The operational policy is: DTE 4-7 is paper-only; DTE 8-14 and 15-20 are no-trade.
This keeps the puncher capable of handling the whole 4-20 range while declining bands
whose corrected historical evidence is weak or negative. No live-capital promotion is
made because all of this history has already influenced strategy development.
"""
    (OUT/"REPORT.md").write_text(report,encoding="utf-8")
    print(table.to_string(index=False)); print(decisions.to_string(index=False))


if __name__=="__main__":main()
