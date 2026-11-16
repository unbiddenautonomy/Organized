"""Walk-forward multi-structure research backtest on the five-strike data.

The script intentionally skips cells without matched fixed-strike exits.  It
does not modify any live service or artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from model_router import calibrated_forecast, dm_daily
from pricing import modeled_fill
from selector import CandidateScore, choose
from structures import calendar_structures, retained_greek_fraction, same_expiry_structures


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv'
SPOT=ROOT/'data/raw/icharts_banknifty_spot_1min.csv'
OUT=ROOT/'results/multi_structure'
CACHE=OUT/'matched_five_strike_windows.parquet'
CONFIG=json.loads((ROOT/'research/structure_backtest_config.json').read_text())
HALF_SPREAD=float(CONFIG['execution']['modeled_half_spread_bps'])
HORIZONS=CONFIG['horizons']; TRAIN_DAYS=int(CONFIG['train_days'])


def read_parquet(path):
    c=duckdb.connect(); x=c.execute('select * from read_parquet(?)',[str(path)]).fetchdf(); c.close(); return x


def write_parquet(frame,path):
    c=duckdb.connect(); c.register('x',frame); c.execute('copy x to ? (format parquet,compression zstd)',[str(path)]); c.close()


def build_cache():
    if CACHE.exists(): return
    OUT.mkdir(parents=True,exist_ok=True); src=str(SOURCE).replace("'","''"); dst=str(CACHE).replace("'","''")
    hvals=','.join(f'({h})' for h in HORIZONS)
    query=f"""
    COPY (
      WITH base AS MATERIALIZED (
        SELECT datetime::TIMESTAMP datetime, expiry::DATE expiry, strike::DOUBLE strike,
          spot_price::DOUBLE spot_price, days_to_expiry::DOUBLE days_to_expiry,
          call_price::DOUBLE call_price, put_price::DOUBLE put_price,
          iv_call::DOUBLE iv_call, iv_put::DOUBLE iv_put,
          call_oi::DOUBLE call_oi, put_oi::DOUBLE put_oi
        FROM read_csv_auto('{src}',header=true,sample_size=100000)
        WHERE datetime::TIMESTAMP >= TIMESTAMP '2024-01-01'
      ), expiries AS (
        SELECT datetime,expiry,dense_rank() over(partition by datetime order by expiry) expiry_rank
        FROM (SELECT DISTINCT datetime,expiry FROM base)
      ), entry AS (
        SELECT b.*,e.expiry_rank,h.horizon_min::INTEGER horizon_min,
          b.datetime+h.horizon_min*INTERVAL 1 MINUTE exit_datetime
        FROM base b JOIN expiries e USING(datetime,expiry) CROSS JOIN (VALUES {hvals}) AS h(horizon_min)
        WHERE e.expiry_rank<=2 AND minute(b.datetime)%15=0
          AND (e.expiry_rank=2 OR b.days_to_expiry BETWEEN 0 AND 25)
          AND b.datetime+h.horizon_min*INTERVAL 1 MINUTE <= date_trunc('day',b.datetime)+INTERVAL 15 HOUR+INTERVAL 30 MINUTE
      )
      SELECT e.*,x.spot_price exit_spot,x.call_price exit_call_price,x.put_price exit_put_price,
        x.iv_call exit_iv_call,x.iv_put exit_iv_put
      FROM entry e LEFT JOIN base x
        ON x.datetime=e.exit_datetime AND x.expiry=e.expiry AND x.strike=e.strike
      ORDER BY e.datetime,e.horizon_min,e.expiry_rank,e.strike
    ) TO '{dst}' (FORMAT PARQUET,COMPRESSION ZSTD)
    """
    c=duckdb.connect(); c.execute('set threads=8'); c.execute("set memory_limit='10GB'"); c.execute(query); c.close()


def lot_size(date): return 15 if pd.Timestamp(date)<=pd.Timestamp('2024-12-24') else 30


def leg_costs(legs,date,lot):
    total=0.
    for leg in legs:
        q=abs(float(leg['qty'])); action=1 if leg['qty']>0 else -1; units=lot*q
        buy=float(leg['entry_mid'] if action>0 else leg['exit_mid'])*units
        sell=float(leg['exit_mid'] if action>0 else leg['entry_mid'])*units; turnover=buy+sell
        d=pd.Timestamp(date); stt_rate=.000625 if d<pd.Timestamp('2024-10-01') else .001
        exchange=.0005 if d<pd.Timestamp('2024-04-01') else (.000495 if d<pd.Timestamp('2024-10-01') else .0003503)
        brokerage=40.; stt=round(sell*stt_rate); exch=turnover*exchange; sebi=turnover*.000001; ipft=turnover*.000000001; stamp=buy*.00003
        gst=.18*(brokerage+exch+sebi+ipft); total+=brokerage+stt+exch+sebi+ipft+stamp+gst
    return total


def actual_structure_row(structure,exit_lookup,entry_row,horizon,wing_hit):
    lot=lot_size(entry_row['datetime']); gross=0.; legs=[]
    for leg in structure.legs:
        key=(pd.Timestamp(leg['expiry']),float(leg['strike']))
        if key not in exit_lookup: return None
        out=exit_lookup[key]; typ=leg['option_type']; exit_mid=float(getattr(out,f'exit_{typ}_price'))
        if not np.isfinite(exit_mid) or exit_mid<=0: return None
        qty=float(leg['qty']); ef=modeled_fill(float(leg['entry_mid']),qty,True,HALF_SPREAD)
        xf=modeled_fill(exit_mid,qty,False,HALF_SPREAD)
        gross+=qty*(xf-ef)*lot; legs.append({**leg,'exit_mid':exit_mid})
    costs=leg_costs(legs,entry_row['datetime'],lot)
    g=structure.net_greeks(); credit=structure.net_entry_premium()*lot
    return {'datetime':entry_row['datetime'],'date':entry_row['datetime'].normalize(),'horizon_min':horizon,
      'structure':structure.name,'wing_definition':structure.wing_definition or 'none','wing_k':structure.wing_k or 0.,
      'dte':float(entry_row['days_to_expiry']),'dte_bucket':pd.cut([float(entry_row['days_to_expiry'])],[-1,3,7,10,25],labels=['0-3','4-7','8-10','11-25'])[0],
      'spot_price':float(entry_row['spot_price']),'exit_spot':float(getattr(next(iter(exit_lookup.values())),'exit_spot')),
      'atm_iv':float(np.nanmean([entry_row['iv_call'],entry_row['iv_put']])),'entry_credit':credit,
      'gross_pnl':gross,'costs':costs,'net_pnl':gross-costs,'leg_count':len(legs),'wing_hit':wing_hit,
      'gamma':g['gamma'],'vega':g['vega'],'theta_day':g['theta_day'],'vanna':g['vanna'],'volga':g['volga'],
      'gamma_retained':retained_greek_fraction(structure,'gamma') if structure.name in {'iron_fly','iron_condor'} else np.nan,
      'vega_retained':retained_greek_fraction(structure,'vega') if structure.name in {'iron_fly','iron_condor'} else np.nan}


def causal_move_multiplier(pred):
    out=[]
    for h,g in pred.groupby('horizon_min'):
        g=g.sort_values('datetime').copy(); vals=[]
        for row in g.itertuples():
            prior=g[g.date<row.date]; days=prior.date.drop_duplicates().sort_values().iloc[-60:]
            z=prior[prior.date.isin(days)]; target=f'future_abs_move_{h}'
            ratio=z[target]/(z.spot_price*np.sqrt(z['har']))
            vals.append(float(ratio.median()) if ratio.notna().sum()>100 else .72)
        g['move_multiplier']=vals; out.append(g)
    return pd.concat(out)


def build_realised_structures():
    raw=read_parquet(CACHE); raw['datetime']=pd.to_datetime(raw.datetime); raw['expiry']=pd.to_datetime(raw.expiry)
    mag=causal_move_multiplier(read_parquet(ROOT/'outputs/spot_magnitude_models/walk_forward_predictions.parquet'))
    mag=mag[['datetime','horizon_min','har','move_multiplier']]
    lookup=mag.set_index(['datetime','horizon_min'])
    rows=[]
    for (dt,h),block in raw.groupby(['datetime','horizon_min'],sort=False):
        if (dt,h) not in lookup.index: continue
        if block.exit_spot.notna().sum()==0: continue
        m=lookup.loc[(dt,h)]; near=block[block.expiry_rank.eq(1)].dropna(subset=['call_price','put_price','iv_call','iv_put'])
        if len(near)<3: continue
        expected=float(near.spot_price.iloc[0]*np.sqrt(float(m['har']))*float(m.move_multiplier))
        expiry_move=float(near.spot_price.iloc[0]*np.nanmean([near.iv_call.median(),near.iv_put.median()])*np.sqrt(max(near.days_to_expiry.iloc[0],1/1440)/365))
        exit_lookup={(pd.Timestamp(x.expiry),float(x.strike)):x for x in block.itertuples() if np.isfinite(x.exit_spot)}
        seen=set()
        for definition in CONFIG['structures']['wing_definitions']:
          for k in CONFIG['structures']['wing_k']:
            try: structs=same_expiry_structures(near,expected,expiry_move,definition,float(k))
            except ValueError: continue
            for s in structs:
                identity=(s.name,s.wing_definition,s.wing_k)
                if s.name in {'naked_short_straddle','long_straddle'}: identity=(s.name,'none',0.)
                if identity in seen: continue
                seen.add(identity)
                wing_strikes=[x['strike'] for x in s.legs if x['qty']>0] if s.name in {'iron_fly','iron_condor'} else [x['strike'] for x in s.legs]
                ex=float(block.exit_spot.dropna().iloc[0]); wing_hit=bool(wing_strikes and (ex<=min(wing_strikes) or ex>=max(wing_strikes)))
                row=actual_structure_row(s,exit_lookup,near.iloc[len(near)//2],h,wing_hit)
                if row: rows.append(row)
        far=block[block.expiry_rank.eq(2)].dropna(subset=['call_price','put_price','iv_call','iv_put'])
        if len(far)>=3:
            try:
                for s in calendar_structures(near,far):
                    row=actual_structure_row(s,exit_lookup,near.iloc[len(near)//2],h,False)
                    if row: rows.append(row)
            except ValueError: pass
    return pd.DataFrame(rows)


def model_panel():
    rv=read_parquet(ROOT/'outputs/rv_prediction_models/intraday_walk_forward_predictions.parquet')
    style=read_parquet(ROOT/'outputs/volatility_specific_models/walk_forward_predictions.parquet')[['datetime','horizon_min','Realized_GARCH_style']]
    iv=read_parquet(ROOT/'outputs/atm_iv_change_models/simple_five_feature/walk_forward_predictions.parquet')
    keep=['datetime','horizon_min','quantile_median_5f']+[f'target_d_iv_{h}' for h in HORIZONS]
    out=rv.merge(style,on=['datetime','horizon_min'],how='left').merge(iv[keep],on=['datetime','horizon_min'],how='left')
    return out.drop(columns=['date','spot_price'],errors='ignore')


def spot_clock(through_dates):
    spot=pd.read_csv(SPOT,usecols=['datetime','spot_price'],parse_dates=['datetime']).dropna(); spot=spot[spot.spot_price>0].sort_values('datetime')
    spot['date']=spot.datetime.dt.normalize(); spot['ret']=spot.groupby('date').spot_price.transform(lambda x:np.log(x/x.shift()))
    z=spot[spot.date.isin(through_dates)].copy(); z['minute']=z.datetime.dt.hour*60+z.datetime.dt.minute-555
    prof=z[z.minute.between(1,375)].groupby('minute').ret.apply(lambda x:float(np.nanmean(x*x))).reindex(range(1,376)).interpolate().fillna(0)
    profile=(prof/prof.sum()).to_numpy(); daily=z.groupby('date').agg(intra=('ret',lambda x:float(np.nansum(x*x))),close=('spot_price','last'))
    cc=np.log(daily.close/daily.close.shift()).pow(2); omega=float(daily.intra.mean()/cc.mean()) if cc.mean()>0 else 1.
    return omega,profile


def walk_forward(realised,models):
    data=realised.merge(models,on=['datetime','horizon_min'],how='inner'); data['time_bucket']=(data.datetime.dt.hour*60+data.datetime.dt.minute)//30
    decisions=[]; diagnostics=[]
    for month in data.datetime.dt.to_period('M').drop_duplicates().sort_values():
      test=data[data.datetime.dt.to_period('M').eq(month)]; prior=data[data.date<test.date.min()]; days=prior.date.drop_duplicates().sort_values().iloc[-TRAIN_DAYS:]; train=prior[prior.date.isin(days)]
      if len(train)<500: continue
      omega,profile=spot_clock(days)
      for arm,col in [('extended','Realized_GARCH_extended'),('style','Realized_GARCH_style'),('har','HAR')]:
        for (dt,h),group in test.groupby(['datetime','horizon_min']):
          base=group.iloc[0]; train_h=train[train.horizon_min.eq(h)]; calibration_train=train_h.drop_duplicates(['datetime','horizon_min'])
          if len(train_h)<300 or not np.isfinite(base.get(col,np.nan)): continue
          forecast=calibrated_forecast(base,calibration_train,col)
          target=f'target_d_iv_{h}'; ivtrain=calibration_train.dropna(subset=['quantile_median_5f',target]).copy(); ivtrain['iv_resid']=ivtrain[target]-ivtrain.quantile_median_5f
          bucket=ivtrain[ivtrain.time_bucket.eq(base.time_bucket)]; bucket=bucket if len(bucket)>=100 else ivtrain
          pred_div=float(base.quantile_median_5f+bucket.iv_resid.mean())
          candidates=[]; edges={}
          for row_index,r in group.iterrows():
            pool=train_h[(train_h.structure.eq(r.structure))&(train_h.dte_bucket.eq(r.dte_bucket))&(train_h.wing_definition.eq(r.wing_definition))&(train_h.wing_k.eq(r.wing_k))]
            if len(pool)<30: continue
            scale=max(abs(float(r.entry_credit)),1.)/max(pool.entry_credit.abs().median(),1.); draws=pool.net_pnl.to_numpy(float)*scale
            mean=float(draws.mean()); cutoff=np.quantile(draws,.05); cvar=float(draws[draws<=cutoff].mean()); ploss=float(np.mean(draws<0))
            start=dt.hour*60+dt.minute-555; mass=float(profile[max(0,start+1):min(375,start+1+h)].sum())
            ivvar=float(r.atm_iv*r.atm_iv/252*omega*mass); edge=float(np.log(max(ivvar,1e-12)/max(forecast.mean,1e-12)))
            label=f"{r.structure}|{r.wing_definition}|{r.wing_k}"
            score=CandidateScore(label,mean,cvar,ploss,ivvar,forecast.mean,forecast.q90,pred_div,payload_index=int(row_index))
            candidates.append(score); edges[label]=edge-CONFIG['risk']['edge_z']*forecast.residual_sd*np.sign(edge)
          winner,runners=choose(candidates,float(CONFIG['risk']['lambda_cvar']),edges)
          if winner:
            chosen=group.loc[winner.payload_index]
            decisions.append({**chosen.to_dict(),'router_arm':arm,'rv_mean':forecast.mean,'rv_q90':forecast.q90,'predicted_mean_net':winner.mean_net,'predicted_cvar5':winner.cvar5,'predicted_loss_probability':winner.loss_probability,'runner_ups':'|'.join(x.structure for x in runners)})
      diagnostics.append({'month':str(month),'omega':omega,'training_days':len(days)})
    selected=pd.DataFrame(decisions)
    if selected.empty:return selected,pd.DataFrame(diagnostics)
    # No overlapping position within each independent router arm.
    keep=[]
    for arm,g in selected.sort_values('datetime').groupby('router_arm'):
      free=pd.Timestamp.min; day_count={}
      for i,r in g.iterrows():
        if r.datetime<free or day_count.get(r.date,0)>=CONFIG['risk']['max_trades_per_day']:continue
        keep.append(i); free=r.datetime+pd.Timedelta(minutes=int(r.horizon_min)); day_count[r.date]=day_count.get(r.date,0)+1
    return selected.loc[keep].copy(),pd.DataFrame(diagnostics)


def summary(trades):
    if trades.empty:return pd.DataFrame()
    rows=[]
    keys=['structure','dte_bucket','horizon_min','router_arm']
    for key,g in trades.groupby(keys,dropna=False):
      p=g.net_pnl.to_numpy(float); daily=g.groupby('date').net_pnl.sum(); curve=daily.cumsum(); dd=(curve.cummax()-curve).max(); cut=np.quantile(p,.05)
      rows.append(dict(zip(keys,key))|{'trades':len(g),'total_net':p.sum(),'hit_rate':np.mean(p>0),'avg_win':p[p>0].mean() if np.any(p>0) else np.nan,'avg_loss':p[p<0].mean() if np.any(p<0) else np.nan,'max_drawdown':dd,'cvar5':p[p<=cut].mean(),'sharpe_like':daily.mean()/daily.std(ddof=1)*np.sqrt(252) if len(daily)>1 and daily.std()>0 else np.nan,'turnover':g.leg_count.sum()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True); build_cache(); realised_path=OUT/'realised_structures.parquet'
    if realised_path.exists(): realised=read_parquet(realised_path)
    else: realised=build_realised_structures(); write_parquet(realised,realised_path)
    models=model_panel(); selected_path=OUT/'selected_trades.parquet'
    if selected_path.exists():
        trades=read_parquet(selected_path); perf=pd.read_csv(OUT/'performance.csv'); diag=pd.read_csv(OUT/'clock_diagnostics.csv')
    else:
        trades,diag=walk_forward(realised,models); write_parquet(trades,selected_path)
        perf=summary(trades); perf.to_csv(OUT/'performance.csv',index=False); diag.to_csv(OUT/'clock_diagnostics.csv',index=False)
    wing=realised.groupby(['structure','wing_definition','wing_k'],as_index=False).agg(gamma_retained=('gamma_retained','mean'),vega_retained=('vega_retained','mean'),wing_hit_rate=('wing_hit','mean'),n=('wing_hit','size')); wing.to_csv(OUT/'wing_diagnostics.csv',index=False)
    dm=dm_daily(models.dropna(subset=['Realized_GARCH_extended','Realized_GARCH_style']),'Realized_GARCH_extended','Realized_GARCH_style')
    result={'matched_structure_rows':len(realised),'selected_trades':len(trades),'dm_extended_vs_style':dm,'hedged_mode':'UNTESTABLE: no complete fixed-strike bid/ask/IV minute paths','stop_loss':'UNTESTABLE for same reason','bid_ask':'UNAVAILABLE: 25 bps modeled half-spread','configurations_reported':int(len(perf))}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=str))
    table=('```text\n'+perf.round(3).to_string(index=False)+'\n```') if len(perf) else 'No configuration passed every frozen gate.'
    lines=['# Multi-structure walk-forward report','',f"Matched realised structure rows: {len(realised):,}. Selected non-overlapping trades: {len(trades):,}.",'',f"Extended-versus-Style daily DM p-value: {dm['p_two_sided']:.4f}.",'','## Performance','',table,'','## What does NOT work','','- Executable bid/ask backtesting is unavailable; every result uses the configured 25 bps half-spread proxy.','- Discretely hedged and stop-loss results are skipped because complete fixed-strike executable minute paths are absent.','- Calendar rows are included only when both expiries and every fixed-strike exit are present; missing cells are skipped.','- Positive point estimates with confidence intervals spanning zero remain inconclusive.','', '## Verdict','', 'No live promotion is made by this script. Verdicts are based on after-cost point estimates and must remain inconclusive without an untouched forward sample and executable quotes.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__':main()
