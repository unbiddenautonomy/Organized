"""Corrected session-end ATM straddle research for DTE 4-20."""

from __future__ import annotations

import duckdb
import pandas as pd

from dte4_20_strategy import OUT, SOURCE, metrics
from frozen_near_expiry_strategy import BASE_HALF_SPREAD_BPS, DEVELOPMENT_END, collapse_exit_chains, price


CACHE=OUT/"session_end_exit_chains.parquet"


def build_cache():
    if CACHE.exists(): return
    OUT.mkdir(parents=True,exist_ok=True); src=str(SOURCE).replace("'","''"); dst=str(CACHE).replace("'","''")
    q=f"""COPY (
    WITH base AS MATERIALIZED (
      SELECT datetime::TIMESTAMP datetime,expiry::DATE expiry,strike::DOUBLE strike,
       spot_price::DOUBLE spot_price,days_to_expiry::DOUBLE dte,call_price::DOUBLE call_mid,
       put_price::DOUBLE put_mid,iv_call::DOUBLE iv_call,iv_put::DOUBLE iv_put
      FROM read_csv_auto('{src}',header=true,sample_size=100000)
    ), ranked AS (
      SELECT *,dense_rank() over(partition by datetime order by expiry) er,
       row_number() over(partition by datetime,expiry order by abs(strike-spot_price),strike) ar
      FROM base
    ), candidate AS (
      SELECT *,row_number() over(partition by date_trunc('day',datetime) order by datetime) day_rank
      FROM ranked WHERE er=1 AND ar=1 AND dte BETWEEN 4 AND 20.999999
       AND strftime(datetime,'%H:%M') BETWEEN '09:45' AND '09:59'
    ), entry AS (SELECT * FROM candidate WHERE day_rank=1),
    exit_times AS (
      SELECT date_trunc('day',datetime) date,expiry,max(datetime) exit_datetime
      FROM base WHERE strftime(datetime,'%H:%M') BETWEEN '15:15' AND '15:30' GROUP BY 1,2
    )
    SELECT e.*,t.exit_datetime,x.strike exit_chain_strike,x.spot_price exit_spot,
      x.call_mid exit_chain_call_mid,x.put_mid exit_chain_put_mid,
      x.iv_call exit_chain_iv_call,x.iv_put exit_chain_iv_put
    FROM entry e JOIN exit_times t ON t.date=date_trunc('day',e.datetime) AND t.expiry=e.expiry
    JOIN base x ON x.datetime=t.exit_datetime AND x.expiry=e.expiry
    ORDER BY e.datetime,x.strike
    ) TO '{dst}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
    c=duckdb.connect();c.execute('set threads=8');c.execute(q);c.close()


def main():
    build_cache();c=duckdb.connect();raw=c.execute('select * from read_parquet(?)',[str(CACHE)]).fetchdf();c.close()
    raw.datetime=pd.to_datetime(raw.datetime); raw=collapse_exit_chains(raw,None)
    rows=[]
    for spread in (10.,25.,50.,100.):
      x=price(raw,spread)
      for name,eligible in [("session_dte_4_7",x[x.dte<8]),
                            ("session_dte_8_14",x[(x.dte>=8)&(x.dte<15)]),
                            ("session_dte_15_20",x[x.dte>=15])]:
        rows.append(metrics(eligible[eligible.date<=DEVELOPMENT_END],name,'development_2019_2023',spread))
        rows.append(metrics(eligible[eligible.date>DEVELOPMENT_END],name,'evaluation_2024_2025',spread))
    out=pd.DataFrame(rows);out.to_csv(OUT/'session_end_cost_sensitivity.csv',index=False)
    print(out[out.half_spread_bps.eq(BASE_HALF_SPREAD_BPS)].to_string(index=False))

if __name__=='__main__':main()
