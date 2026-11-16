"""Corrected structure search for BANKNIFTY DTE 8-20.

Every entry and exit five-strike smile is retained.  A leg that leaves the exit
snapshot is Black-Scholes repriced from the exit parity forward and smile.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from frozen_near_expiry_strategy import bootstrap_total, lot_size
from pricing import bs_price


ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/'data/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv'
OUT=ROOT/'results/dte8_20_structures'; CACHE=OUT/'entry_exit_chains.parquet'; BPS=25.
STRUCTURES={
 'short_atm_straddle':[('call',0,-1),('put',0,-1)],
 'short_1_strangle':[('call',1,-1),('put',-1,-1)],
 'short_2_strangle':[('call',2,-1),('put',-2,-1)],
 'iron_condor_1_2':[('call',1,-1),('put',-1,-1),('call',2,1),('put',-2,1)],
}


def build_cache():
    if CACHE.exists():return
    OUT.mkdir(parents=True,exist_ok=True);src=str(SOURCE).replace("'","''");dst=str(CACHE).replace("'","''")
    q=f"""COPY (
    WITH b AS MATERIALIZED (
      SELECT datetime::TIMESTAMP datetime,expiry::DATE expiry,strike::DOUBLE strike,
       spot_price::DOUBLE spot_price,days_to_expiry::DOUBLE dte,call_price::DOUBLE call_mid,
       put_price::DOUBLE put_mid,iv_call::DOUBLE iv_call,iv_put::DOUBLE iv_put
      FROM read_csv_auto('{src}',header=true,sample_size=100000)
    ), expiry_rank AS (SELECT datetime,expiry,dense_rank() over(partition by datetime order by expiry) er FROM (SELECT DISTINCT datetime,expiry FROM b)),
    ranked AS (
      SELECT b.*,e.er,row_number() over(partition by b.datetime,b.expiry order by b.strike) sr,
       row_number() over(partition by b.datetime,b.expiry order by abs(b.strike-b.spot_price),b.strike) ar
      FROM b JOIN expiry_rank e USING(datetime,expiry)
    ), candidates AS (
      SELECT *,row_number() over(partition by date_trunc('day',datetime) order by datetime) dr
      FROM ranked WHERE er=1 AND ar=1 AND dte BETWEEN 8 AND 20.999999
       AND strftime(datetime,'%H:%M') BETWEEN '09:45' AND '09:59'
    ), keys AS (
      SELECT datetime,expiry,dte,spot_price,strike atm_strike FROM candidates WHERE dr=1
    ), exits AS (
      SELECT date_trunc('day',datetime) date,expiry,max(datetime) exit_datetime FROM b
      WHERE strftime(datetime,'%H:%M') BETWEEN '15:15' AND '15:30' GROUP BY 1,2
    )
    SELECT k.datetime,k.expiry,k.dte,k.spot_price,k.atm_strike,x.exit_datetime,
      'entry' kind,b.strike,b.call_mid,b.put_mid,b.iv_call,b.iv_put,b.spot_price chain_spot
    FROM keys k JOIN exits x ON x.date=date_trunc('day',k.datetime) AND x.expiry=k.expiry
    JOIN b ON b.datetime=k.datetime AND b.expiry=k.expiry
    UNION ALL
    SELECT k.datetime,k.expiry,k.dte,k.spot_price,k.atm_strike,x.exit_datetime,
      'exit' kind,b.strike,b.call_mid,b.put_mid,b.iv_call,b.iv_put,b.spot_price chain_spot
    FROM keys k JOIN exits x ON x.date=date_trunc('day',k.datetime) AND x.expiry=k.expiry
    JOIN b ON b.datetime=x.exit_datetime AND b.expiry=k.expiry
    ORDER BY datetime,kind,strike
    ) TO '{dst}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
    c=duckdb.connect();c.execute('set threads=8');c.execute(q);c.close()


def exit_price(exit_chain, strike, typ, tau):
    exact=exit_chain[np.isclose(exit_chain.strike,strike)]
    if len(exact):return float(exact[f'{typ}_mid'].iloc[0]),False
    z=exit_chain.sort_values('strike'); strikes=z.strike.to_numpy(float); iv=z[f'iv_{typ}'].to_numpy(float)
    valid=np.isfinite(iv)&(iv>0)
    if valid.sum()<2:return np.nan,True
    forward=float(np.nanmedian(strikes+z.call_mid.to_numpy(float)-z.put_mid.to_numpy(float)))
    x=np.log(strikes/max(forward,1e-9)); target=np.log(strike/max(forward,1e-9)); sigma=float(np.interp(target,x[valid],iv[valid]))
    return float(bs_price(forward,strike,tau,sigma,typ=='call')),True


def costs(legs,date,lot,bps):
    total=0.
    for leg in legs:
      q=leg['qty']; spread=bps/10000; entry=leg['entry']*(1-spread if q<0 else 1+spread); exit_=leg['exit']*(1+spread if q<0 else 1-spread)
      sell=(entry if q<0 else exit_)*lot*abs(q);buy=(exit_ if q<0 else entry)*lot*abs(q);turn=sell+buy
      slippage=(leg['entry']+leg['exit'])*spread*lot*abs(q)
      d=pd.Timestamp(date);stt=.000625 if d<pd.Timestamp('2024-10-01') else .001;ex=.0005 if d<pd.Timestamp('2024-04-01') else (.000495 if d<pd.Timestamp('2024-10-01') else .0003503)
      brokerage=40.;total+=slippage+brokerage+round(sell*stt)+turn*ex+turn*.000001+turn*.000000001+buy*.00003+.18*(brokerage+turn*ex+turn*.000001+turn*.000000001)
    return total


def realised():
    build_cache();c=duckdb.connect();raw=c.execute('select * from read_parquet(?)',[str(CACHE)]).fetchdf();c.close();rows=[]
    for dt,g in raw.groupby('datetime',sort=False):
      entry=g[g.kind.eq('entry')].sort_values('strike').copy();exit_=g[g.kind.eq('exit')].sort_values('strike').copy()
      entry['pos']=np.arange(len(entry))-int(np.argmin(abs(entry.strike-entry.atm_strike.iloc[0])))
      tau=max((float(entry.dte.iloc[0])-(pd.Timestamp(entry.exit_datetime.iloc[0])-pd.Timestamp(dt)).total_seconds()/86400)/365,1e-9)
      for name,spec in STRUCTURES.items():
        legs=[];gross_points=0.;imputed=False;valid=True
        for typ,pos,qty in spec:
          pick=entry[entry.pos.eq(pos)]
          if pick.empty:valid=False;break
          e=float(pick[f'{typ}_mid'].iloc[0]);xp,flag=exit_price(exit_,float(pick.strike.iloc[0]),typ,tau)
          if not np.isfinite(xp):valid=False;break
          legs.append({'qty':qty,'entry':e,'exit':xp});gross_points+=qty*(xp-e);imputed|=flag
        if not valid:continue
        expiry=pd.Timestamp(entry.expiry.iloc[0]);lot=lot_size(expiry);date=pd.Timestamp(dt).normalize();cost=costs(legs,date,lot,BPS)
        rows.append({'datetime':dt,'date':date,'expiry':expiry,'dte':float(entry.dte.iloc[0]),'structure':name,
          'gross_pnl':gross_points*lot,'costs':cost,'net_pnl':gross_points*lot-cost,'exit_imputed':imputed,
          'entry_premium':sum(-x['qty']*x['entry'] for x in legs if x['qty']<0)})
    return pd.DataFrame(rows)


def report(x):
    x['dte_band']=pd.cut(x.dte,[8,15,21],right=False,labels=['8-14','15-20']);x['sample']=np.where(x.date<='2023-12-31','development','evaluation')
    rows=[]
    for (structure,band,sample),g in x.groupby(['structure','dte_band','sample'],observed=True):
      lo,hi,p=bootstrap_total(g)
      rows.append({'structure':structure,'dte_band':band,'sample':sample,'trades':len(g),'net_pnl':g.net_pnl.sum(),
        'mean':g.net_pnl.mean(),'hit_rate':(g.net_pnl>0).mean(),'worst':g.net_pnl.min(),'ci_low':lo,'ci_high':hi,
        'p_net_le_zero':p,'imputed_fraction':g.exit_imputed.mean()})
    return pd.DataFrame(rows)


def main():
    x=realised();x.to_csv(OUT/'structure_trade_ledger_25bps.csv',index=False);s=report(x);s.to_csv(OUT/'structure_performance.csv',index=False);print(s.to_string(index=False))

if __name__=='__main__':main()
