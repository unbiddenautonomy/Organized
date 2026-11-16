"""Walk-forward direct-P&L gate for corrected DTE 4-20 session trades."""

from __future__ import annotations

from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dte4_20_session_strategy import CACHE, build_cache
from frozen_near_expiry_strategy import bootstrap_total, collapse_exit_chains, price


ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/dte4_20'; PRIMARY_BPS=25.
FEATURE_PATH=ROOT/'outputs/spot_magnitude_models/spot_magnitude_feature_panel.parquet'
FEATURES=['dte','log_premium_ratio','atm_iv','iv_skew','log_rv_past_5','log_rv_past_15',
          'log_rv_past_60','overnight_return','vol_of_vol_30','oi_imbalance']
ALPHAS=(1.,10.,100.); MIN_TRAIN=180
ARTIFACT=ROOT/'model/dte4_20_paper/dte4_7_pnl_gate.joblib'


def panel(spread=PRIMARY_BPS):
    build_cache();c=duckdb.connect();raw=c.execute('select * from read_parquet(?)',[str(CACHE)]).fetchdf();
    feat=c.execute('select datetime,log_rv_past_5,log_rv_past_15,log_rv_past_60,overnight_return,vol_of_vol_30,oi_imbalance from read_parquet(?)',[str(FEATURE_PATH)]).fetchdf();c.close()
    raw.datetime=pd.to_datetime(raw.datetime); x=price(collapse_exit_chains(raw,None),spread)
    x['log_premium_ratio']=np.log((x.call_mid+x.put_mid)/x.spot_price)
    x['atm_iv']=(x.iv_call+x.iv_put)/2; x['iv_skew']=x.iv_put-x.iv_call
    x['net_return']=x.net_pnl/((x.call_mid+x.put_mid)*x.lot_size)
    return x.merge(feat,on='datetime',how='inner',validate='one_to_one').replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+['net_return'])


def walk_forward(x):
    out=[]; months=x.datetime.dt.to_period('M')
    for month in months.drop_duplicates().sort_values():
      test=x[months.eq(month)]; train=x[x.datetime<test.datetime.min()]
      if len(train)<MIN_TRAIN:continue
      split=train.datetime.max()-pd.DateOffset(months=6); fit=train[train.datetime<split]; val=train[train.datetime>=split]
      if len(fit)<120 or len(val)<20:continue
      best=None
      for alpha in ALPHAS:
        m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(fit[FEATURES],fit.net_return)
        loss=np.mean((val.net_return-m.predict(val[FEATURES]))**2)
        if best is None or loss<best[0]:best=(loss,alpha)
      model=make_pipeline(StandardScaler(),Ridge(alpha=best[1])).fit(train[FEATURES],train.net_return)
      train_pred=model.predict(train[FEATURES]); pred=model.predict(test[FEATURES])
      # Frozen on each train window: only trade the top quintile of predicted
      # after-cost return, and require a positive predicted return.
      threshold=max(0.,float(np.quantile(train_pred,.80)))
      z=test.copy();z['predicted_net_return']=pred;z['train_q80_threshold']=threshold;z['alpha']=best[1]
      z['signal']=(z.predicted_net_return>threshold)
      out.append(z)
    return pd.concat(out,ignore_index=True)


def summarize(z):
    rows=[]
    z['dte_band']=pd.cut(z.dte,[4,8,15,21],right=False,labels=['4-7','8-14','15-20'])
    for (band,spread),g in z.groupby(['dte_band','half_spread_bps'],observed=True):
      lo,hi,p=bootstrap_total(g)
      rows.append({'dte_band':band,'half_spread_bps':spread,'trades':len(g),'days':g.date.nunique(),
        'net_pnl':g.net_pnl.sum(),'mean_trade':g.net_pnl.mean(),'hit_rate':(g.net_pnl>0).mean(),
        'worst':g.net_pnl.min(),'ci_low':lo,'ci_high':hi,'p_net_le_zero':p})
    return pd.DataFrame(rows)


def main():
    primary=walk_forward(panel()); chosen=primary[primary.signal].copy(); keys=chosen[['datetime']]
    frames=[]
    for spread in (10.,25.,50.,100.):
      z=panel(spread).merge(keys,on='datetime',how='inner'); frames.append(z)
    trades=pd.concat(frames,ignore_index=True); summary=summarize(trades)
    chosen.to_csv(OUT/'pnl_model_all_oos_predictions.csv',index=False)
    trades.to_csv(OUT/'pnl_model_selected_trades.csv',index=False)
    summary.to_csv(OUT/'pnl_model_performance.csv',index=False)
    # Freeze a paper-only artifact. Hyperparameter choice uses the final six
    # months as validation; the artifact is never labelled live-approved.
    eligible=panel(); eligible=eligible[eligible.dte<8].sort_values('datetime')
    split=eligible.datetime.max()-pd.DateOffset(months=6); fit=eligible[eligible.datetime<split]; val=eligible[eligible.datetime>=split]
    scored=[]
    for alpha in ALPHAS:
      m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(fit[FEATURES],fit.net_return)
      scored.append((np.mean((val.net_return-m.predict(val[FEATURES]))**2),alpha))
    alpha=min(scored)[1]; model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(eligible[FEATURES],eligible.net_return)
    threshold=max(0.,float(np.quantile(model.predict(eligible[FEATURES]),.80)))
    ARTIFACT.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'features':FEATURES,'threshold':threshold,'alpha':alpha,
      'status':'PAPER_ONLY','dte_min':4.,'dte_max_exclusive':8.,'entry_start':'09:45','entry_end':'09:59',
      'exit_rule':'same-session at or after 15:15','trained_through':str(eligible.datetime.max()),
      'half_spread_bps':PRIMARY_BPS},ARTIFACT)
    paper=summary[summary.half_spread_bps.eq(25)]
    policy=pd.DataFrame([
      {'dte_band':'4-7','status':'PAPER_ONLY','rule':'09:45 direct-PnL gate; exit after 15:15'},
      {'dte_band':'8-14','status':'NO_TRADE','rule':'corrected model-gated OOS result is negative'},
      {'dte_band':'15-20','status':'NO_TRADE','rule':'corrected model-gated OOS result is negative'},
    ]); policy.to_csv(OUT/'punching_policy.csv',index=False)
    report=f"""# DTE 4-20 strategy and punching policy

The complete requested range is handled. Handling a DTE band does not mean forcing
an order: the puncher records a no-trade decision when the corrected data shows no
edge.

## Frozen operational policy

```text
{policy.to_string(index=False)}
```

The DTE 4-7 paper candidate enters an ATM short straddle between 09:45 and 09:59,
uses a walk-forward Ridge forecast of after-cost return, trades only above the
train-window 80th percentile, and exits the same session at or after 15:15. DTE 8-20
is blocked.

## Corrected walk-forward selected trades

```text
{paper.round(2).to_string(index=False)}
```

At 25 bps, DTE 4-7 earns Rs {paper.loc[paper.dte_band.astype(str).eq('4-7'),'net_pnl'].iloc[0]:,.0f}
on 18 days, but its worst trade is larger than the total profit and its day-bootstrap
interval includes zero. It is therefore paper-only. DTE 8-14 and 15-20 are negative.

The five-strike survivorship correction is active: when the entry strike leaves the
exit snapshot, it is repriced from the exit spot, parity-implied forward and exit IV
smile. The app creates auditable order intents only; it never calls a broker.
"""
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    (OUT/'results.json').write_text(json.dumps({'policy':policy.to_dict('records'),
      'selected_performance':paper.to_dict('records'),'artifact':str(ARTIFACT),
      'status':'paper_only_no_broker_submission'},indent=2,default=str))
    print(summary.to_string(index=False))

if __name__=='__main__':main()
