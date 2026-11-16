"""Build fixed-contract paths from the FULL chain, never the moving 5-strike window."""
from pathlib import Path
import json
import os
import duckdb
import numpy as np
import pandas as pd
from weeks43_features import FEATURES, feature_row, history_add, historical_quantity, pnl

ROOT = Path(__file__).resolve().parent

def save_frame(frame, path):
    conn=duckdb.connect(); conn.register('frame',frame)
    conn.execute('COPY frame TO ? (FORMAT PARQUET)',[str(path)]); conn.close()

def read_frame(path):
    with duckdb.connect() as conn:
        return conn.execute('SELECT * FROM read_parquet(?)',[str(path)]).fetchdf()

def prepare():
    cfg = json.loads((ROOT/'config.json').read_text())
    for key, env in (('entry_source', 'VOLTR_43W_ENTRY_SOURCE'),
                     ('full_chain_source', 'VOLTR_43W_FULL_CHAIN_SOURCE')):
        selected = os.environ.get(env, cfg[key])
        path = Path(selected).expanduser()
        cfg[key] = str((path if path.is_absolute() else ROOT/path).resolve())
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(f'{key}: {cfg[key]} (set {env} for an external data file)')
    cache = ROOT/'data'; cache.mkdir(exist_ok=True)
    con = duckdb.connect(); con.execute("SET threads=4")
    if not (cache/'snapshots.parquet').exists():
        print('Reading source snapshots', flush=True)
        raw = con.execute('''SELECT CAST(datetime AS TIMESTAMP) AS nominal,
          CAST(Date AS DATE) AS trade_date, CAST(tick_time AS VARCHAR) AS tick_time,
          CAST(expiry AS DATE) AS expiry, strike, atm_strike, spot_price, future_open,
          call_price, put_price, call_oi, put_oi
          FROM read_csv_auto(?) WHERE dte_flag='M1' AND CAST(is_monthly AS BOOLEAN)
          AND days_to_expiry>=21 AND days_to_expiry<35''', [cfg['entry_source']]).fetchdf()
        raw = raw.sort_values(['nominal','expiry','strike']).drop_duplicates(['nominal','expiry','strike'])
        # Select earliest eligible expiry at THIS timestamp, never full-day dominant expiry.
        first = raw.groupby('nominal')['expiry'].transform('min')
        raw = raw.loc[raw.expiry.eq(first)].copy()
        base = raw.loc[raw.strike.eq(raw.atm_strike)].copy()
        base['call_strike'] = base.atm_strike + 100
        base['put_strike'] = base.atm_strike - 100
        q = raw[['nominal','expiry','strike','call_price','put_price','call_oi','put_oi']]
        base = base.merge(q[['nominal','expiry','strike','call_price','call_oi']].rename(columns={'strike':'call_strike','call_price':'call','call_oi':'otm_call_oi'}),on=['nominal','expiry','call_strike'],validate='one_to_one')
        base = base.merge(q[['nominal','expiry','strike','put_price','put_oi']].rename(columns={'strike':'put_strike','put_price':'put','put_oi':'otm_put_oi'}),on=['nominal','expiry','put_strike'],validate='one_to_one')
        rows=[]
        for r in base.itertuples(index=False):
            observed = pd.Timestamp(str(r.trade_date.date())+' '+r.tick_time)
            if observed < r.nominal or observed > r.nominal + pd.Timedelta(seconds=60): continue
            dte = (r.expiry+pd.Timedelta(hours=15,minutes=30)-r.nominal).total_seconds()/86400
            rows.append(dict(signal_time=r.nominal.isoformat(sep=' '), available_at=observed.isoformat(sep=' '),
                expiry=str(r.expiry.date()), days_to_expiry=dte, spot=float(r.spot_price),future=float(r.future_open),
                atm=float(r.atm_strike),step=100.,atm_call=float(r.call_price),atm_put=float(r.put_price),
                atm_call_oi=float(r.call_oi),atm_put_oi=float(r.put_oi),call=float(r.call),put=float(r.put),
                call_oi=float(r.otm_call_oi),put_oi=float(r.otm_put_oi),call_strike=float(r.call_strike),put_strike=float(r.put_strike)))
        save_frame(pd.DataFrame(rows),cache/'snapshots.parquet')
    snapshots=read_frame(cache/'snapshots.parquet')
    if not (cache/'chain.parquet').exists():
        print('Extracting fixed-strike minute paths from 16GB full chain (one pass)',flush=True)
        wanted=pd.concat([snapshots[['signal_time','expiry',c]].rename(columns={c:'strike'}) for c in ['call_strike','put_strike']])
        wanted['trade_date']=pd.to_datetime(wanted.signal_time).dt.normalize()
        wanted['expiry']=pd.to_datetime(wanted.expiry)
        wanted=wanted[['trade_date','expiry','strike']].drop_duplicates()
        con.register('wanted',wanted)
        chain=con.execute('''WITH c AS (SELECT CAST(datetime AS TIMESTAMP) AS nominal,
            CAST(Date AS DATE) AS trade_date, CAST(expiry AS TIMESTAMP) AS expiry,
            CAST(tick_time AS VARCHAR) tick_time, CAST(strike AS DOUBLE) strike,
            CAST(call_price AS DOUBLE) AS "call", CAST(put_price AS DOUBLE) AS "put"
            FROM read_csv_auto(?) WHERE CAST(is_monthly AS BOOLEAN))
          SELECT c.* FROM c SEMI JOIN wanted w ON c.trade_date=w.trade_date AND c.expiry=w.expiry AND c.strike=w.strike''',[cfg['full_chain_source']]).fetchdf()
        chain['time']=pd.to_datetime(chain.trade_date.dt.strftime('%Y-%m-%d')+' '+chain.tick_time)
        chain=chain.loc[(chain.time>=chain.nominal)&(chain.time<=chain.nominal+pd.Timedelta(seconds=60))]
        chain=chain.sort_values('time').drop_duplicates(['nominal','expiry','strike'])
        save_frame(chain,cache/'chain.parquet')
    print('Building causal features and OTM stop/target labels',flush=True)
    history={}; feats=[]
    for r in snapshots.to_dict('records'):
        history_add(history,r); f=feature_row(r,history)
        if f is not None: feats.append(dict(r,**f))
    frame=pd.DataFrame(feats)
    chain=read_frame(cache/'chain.parquet')
    candidates=[]; paths=[]; missing=0
    for day, group in frame.groupby(frame.signal_time.str[:10],sort=True):
        cq=chain.loc[chain.trade_date.eq(pd.Timestamp(day))]
        curves={}
        for (expiry,strike),q in cq.groupby(['expiry','strike']):
            curves[(str(expiry.date()),strike)]=q.set_index('nominal').sort_index()
        for r in group.to_dict('records'):
            t=pd.Timestamp(r['signal_time']); available=pd.Timestamp(r['available_at'])
            execution=available+pd.Timedelta(seconds=60)
            end=execution+pd.Timedelta(minutes=120)
            if t.time()<pd.Timestamp('10:15').time() or end>pd.Timestamp(day+' '+cfg['square_off']): continue
            c=curves.get((r['expiry'],r['call_strike'])); p=curves.get((r['expiry'],r['put_strike']))
            if c is None or p is None: missing+=1; continue
            # Exact minute joins, no ffill across missing quotes and no changing strikes.
            nominal=pd.date_range(t+pd.Timedelta(minutes=1),periods=121,freq='min')
            cp=c.reindex(nominal); pp=p.reindex(nominal)
            valid=(cp['call'].gt(0)&pp['put'].gt(0)&cp.time.notna()&pp.time.notna())
            # No fabricated fills. Sparse paths may label only observed quotes;
            # require exact entry/end and at least 80% of intervening snapshots.
            if not (valid.iloc[0] and valid.iloc[-1] and valid.mean()>=.80): missing+=1; continue
            observed=pd.concat([cp.time,pp.time],axis=1).max(axis=1)
            if observed.iloc[0] < execution or (observed-observed.index).dt.total_seconds().max()>60:
                missing+=1; continue
            premium=cp['call'].to_numpy()+pp['put'].to_numpy()
            qty=historical_quantity(r['expiry'])
            net=np.array([pnl(premium[0],v,qty,cfg,day)['net_pnl'] if np.isfinite(v) else np.nan for v in premium])
            hits=np.flatnonzero((net[1:]>=cfg['profit_target_inr'])|(net[1:]<=cfg['stop_loss_inr']))
            exit_i=int(hits[0]+1) if len(hits) else 120
            r.update(row_id=len(candidates),execution_time=observed.iloc[0].isoformat(sep=' '),
                exit_time=observed.iloc[exit_i].isoformat(sep=' '),label_end=end.isoformat(sep=' '),quantity=qty,
                entry_premium=premium[0],exit_premium=premium[exit_i],net_points=net[exit_i]/qty,
                horizon_net_points=net[-1]/qty,adverse_points=max(0.,-np.nanmin(net)/qty),path_coverage=float(valid.mean()),
                exit_reason='TARGET' if net[exit_i]>=1500 else 'STOP' if net[exit_i]<=-750 else 'HORIZON')
            candidates.append(r)
            paths.append(np.c_[cp['call'].to_numpy(),pp['put'].to_numpy()])
    out=pd.DataFrame(candidates)
    save_frame(out,cache/'training_rows.parquet')
    np.save(cache/'fixed_paths.npy',np.array(paths,dtype=np.float32))
    audit={'source':cfg['entry_source'],'full_chain':cfg['full_chain_source'],'snapshot_rows':len(snapshots),
        'complete_feature_rows':len(frame),'labeled_rows':len(out),'missing_path_rows_excluded':missing,
        'feature_names':FEATURES,'label_horizon':120,'holding_strikes':'ATM_t+100 CE / ATM_t-100 PE, same expiry',
        'feature_clock':'completed minute nominal :59; source tick_time controls availability',
        'fill_delay_seconds':60,'path_resolution':'one-minute snapshots; intra-minute stop hits are unobservable',
        'missing_path_policy':'excluded only from supervised training; chronological replay must retain missing-quote positions',
        'by_year':out.groupby(out.signal_time.str[:4]).size().to_dict()}
    (cache/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__': prepare()
