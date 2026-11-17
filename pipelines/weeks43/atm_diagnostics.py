"""Archived ATM models are visible for comparison, NEVER OTM entry vetoes."""
from collections import deque
from datetime import timedelta
import json
import math
import joblib
import pandas as pd
import numpy as np
from .engine import ROOT, parse_time

def atm_features(rows):
    r=rows[-1]; t=parse_time(r['datetime'])+timedelta(seconds=119)
    f={k:float(r[k]) for k in ['days_to_expiry','delta_call','delta_put','vega_call','vega_put','gamma_call','gamma_put','theta_call','theta_put','call_oi','put_oi']}
    f['days_to_expiry']-=59/86400
    f.update(iv_call=float(r['entry_iv_call']),iv_put=float(r['entry_iv_put']),future_oi=float(r['Future_oi']),
        signal_straddle=r['call_price']+r['put_price'],basis=r['future_open']/r['spot_price']-1,
        oi_ratio=r['put_oi']/r['call_oi'],moneyness=(r['spot_price']-r['entry_strike'])/100,
        time_sin=math.sin(2*math.pi*(t.hour*60+t.minute-555)/375),time_cos=math.cos(2*math.pi*(t.hour*60+t.minute-555)/375),
        day_of_week=t.weekday(),four_week=float(f['days_to_expiry']>=28),month_sin=math.sin(2*math.pi*(t.month-1)/12),month_cos=math.cos(2*math.pi*(t.month-1)/12))
    f.update(premium_balance=(r['call_price']-r['put_price'])/f['signal_straddle'],
        iv_mean=(f['iv_call']+f['iv_put'])/2,iv_skew=f['iv_put']-f['iv_call'],
        oi_imbalance=(r['put_oi']-r['call_oi'])/(r['put_oi']+r['call_oi']))
    for kind in ['theta','vega','gamma']: f[kind+'_total']=f[kind+'_call']+f[kind+'_put']
    f.update(theta_per_premium=f['theta_total']/f['signal_straddle'],gamma_per_premium=f['gamma_total']/f['signal_straddle'],premium_per_dte=f['signal_straddle']/f['days_to_expiry'])
    bytime={x['datetime']:x for x in rows if x['expiry']==r['expiry'] and x['datetime'][:10]==r['datetime'][:10]}
    for lag in [5,15,30,60]:
        p=bytime.get((parse_time(r['datetime'])-timedelta(minutes=lag)).isoformat(sep=' '))
        if p is None: return None
        f[f'future_return_{lag}']=math.log(r['future_open']/p['future_open'])
        f[f'abs_future_return_{lag}']=abs(f[f'future_return_{lag}'])
        f[f'spot_return_{lag}']=math.log(r['spot_price']/p['spot_price'])
        f[f'straddle_return_{lag}']=f['signal_straddle']/(p['call_price']+p['put_price'])-1
        f[f'change_{lag}_signal_straddle']=f['signal_straddle']-p['call_price']-p['put_price']
        f[f'change_{lag}_oi_ratio']=f['oi_ratio']-p['put_oi']/p['call_oi']
        for side in ['call','put']: f[f'change_{lag}_iv_{side}']=f['iv_'+side]-p['entry_iv_'+side]
        f[f'iv_impulse_{lag}']=(f[f'change_{lag}_iv_call']+f[f'change_{lag}_iv_put'])/2
        f[f'premium_velocity_{lag}']=f[f'straddle_return_{lag}']/lag
        f[f'return_velocity_{lag}']=f[f'abs_future_return_{lag}']/math.sqrt(lag)
    for name in ['return','premium','iv']:
        src='iv_impulse' if name=='iv' else name+'_velocity'
        f[name+'_acceleration_5_60']=f[src+'_5']-f[src+'_60']
    f.update(trend_5_minus_30=f['future_return_5']-f['future_return_30'],
        iv_to_realized_proxy=f['iv_mean']/(abs(f['return_velocity_60'])+1e-5),gamma_iv_interaction=f['gamma_per_premium']*f['iv_mean'])
    return f if all(math.isfinite(v) for v in f.values()) else None

class AtmDiagnostics:
    def __init__(self,root=ROOT):
        self.root=root; self.rows=deque(maxlen=70); self.models={}; self.histories={}
        for p in (root/'atm_reference_models').glob('*/model.joblib'):
            self.models[p.parent.name]=joblib.load(p)
        for p in (root/'atm_reference_models/tail_veto').glob('tail_veto_*m.joblib'):
            self.models[p.stem]=joblib.load(p)
        for name,p in self.models.items():
            for value in p.values():
                if hasattr(value,'get_params'):
                    for key in list(value.get_params()):
                        if key.endswith('n_jobs'): value.set_params(**{key:1})
            if 'rolling_score_history' in p: self.histories[name]=deque(p['rolling_score_history'],maxlen=p.get('rolling_score_window',1000))
        saved=root/'runtime/atm_rolling_scores.json'
        if saved.exists():
            data=json.loads(saved.read_text())
            for name,values in data.items():
                if name in self.histories: self.histories[name].extend(values)
    def score(self,row):
        self.rows.append(row)
        try: f=atm_features(list(self.rows))
        except (ValueError,TypeError,KeyError,ZeroDivisionError): f=None
        output={}
        for name,p in self.models.items():
            record={'controls_otm_entries':False,'horizon_minutes':p.get('horizon_minutes'),'status':'DIAGNOSTIC_ONLY'}
            if f is None: record['status']='WAITING_FOR_VALID_ATM_FEATURES'; output[name]=record; continue
            try:
                features=p.get('features',p.get('entry_features'))
                X=pd.DataFrame([[f[c] for c in features]],columns=features)
                if 'entry_model' in p:
                    record['win_probability']=float(p['entry_model'].predict_proba(X)[0,1])
                    risk=float(p['tail_veto_model'].predict_proba(pd.DataFrame([[f[c] for c in p['veto_features']]],columns=p['veto_features']))[0,1])
                    record['tail_risk_score']=risk
                elif 'classifier' in p:
                    record['win_probability']=float(p['classifier'].predict_proba(X)[0,1])
                    record['expected_net_points']=float(p['regressor'].predict(X)[0])
                else:
                    risk=float(p['model'].predict_proba(X)[0,1])
                    record['tail_risk_score' if name.startswith('tail_veto_') else 'win_probability']=risk
                if name in self.histories and 'tail_risk_score' in record:
                    h=self.histories[name]
                    record['rolling_threshold']=float(np.quantile(h,.8)) if h else p.get('veto_threshold')
                    record['hypothetical_veto']=record['tail_risk_score']>=record['rolling_threshold'] if record['rolling_threshold'] is not None else None
                    h.append(record['tail_risk_score'])
            except Exception as exc: record.update(status='ERROR',error=str(exc))
            output[name]=record
        (self.root/'runtime/atm_rolling_scores.json').write_text(json.dumps({k:list(v) for k,v in self.histories.items()}))
        return output
