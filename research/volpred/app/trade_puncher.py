"""Create auditable paper order intents for the DTE 4-20 policy.

This module never calls a broker.  A live adapter may consume PAPER_ORDER_READY
only after a separate promotion decision and executable quote validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
NEAR_MODEL=ROOT/'model/dte4_20_paper/dte4_7_pnl_gate.joblib'
LONGER_MODEL=ROOT/'model/dte4_20_paper/dte8_20_huber_gate.joblib'
MULTI_MODEL=ROOT/'model/dte4_20_paper/multi_structure_selector.joblib'
DEFAULT_OUTPUT=ROOT/'runtime/trade_intents.jsonl'


def latest_chain(path: Path) -> pd.DataFrame:
    frame=pd.read_csv(path); required={'datetime','expiry','strike','spot_price','days_to_expiry','call_price','put_price'}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f"chain missing columns: {sorted(missing)}")
    frame['datetime']=pd.to_datetime(frame.datetime);frame['expiry']=pd.to_datetime(frame.expiry)
    timestamp=frame.datetime.max(); return frame[frame.datetime.eq(timestamp)].copy()


def build_intent(chain: pd.DataFrame, live_record: dict, artifact: dict) -> dict:
    now=pd.Timestamp(chain.datetime.max()); near=chain[chain.expiry.eq(chain.expiry.min())].copy()
    atm=near.loc[(near.strike-near.spot_price).abs().idxmin()]; dte=float(atm.days_to_expiry)
    base={'timestamp':now.isoformat(),'expiry':pd.Timestamp(atm.expiry).date().isoformat(),
          'dte':dte,'policy':'dte4_20','model_status':artifact['status']}
    if not 4<=dte<=20.999999:return base|{'status':'NO_TRADE','reason':'outside DTE 4-20'}
    if dte>=8 and 'models' not in artifact:
        return base|{'status':'NO_TRADE','reason':'DTE 8-20 failed corrected gate because its model artifact was not supplied'}
    multi_structure='structure_legs' in artifact
    if 'models' in artifact:
        band='8-14' if dte<15 else '15-20'
        if multi_structure:
            band_models=artifact['models'][band]
            artifact={**artifact,'dte_band':band}
        else:
            band_model=artifact['models'][band]
            artifact={**artifact,**band_model,'dte_band':band}
    hhmm=now.strftime('%H:%M')
    if not artifact['entry_start']<=hhmm<=artifact['entry_end']:
        return base|{'status':'NO_TRADE','reason':'outside 09:45-09:59 entry window'}
    f=live_record.get('features',{}); needed=['log_rv_past_5','log_rv_past_15','log_rv_past_60','overnight_return','vol_of_vol_30']
    feature_values={name:f.get(name,atm.get(name,np.nan)) for name in needed}
    missing=[name for name,value in feature_values.items() if not np.isfinite(value)]
    if missing:return base|{'status':'BLOCKED','reason':f"missing live features: {missing}"}
    oi_total=float(atm.get('call_oi',0))+float(atm.get('put_oi',0)); oi_imbalance=(float(atm.get('put_oi',0))-float(atm.get('call_oi',0)))/max(oi_total,1.)
    row=pd.DataFrame([{'dte':dte,'log_premium_ratio':np.log((atm.call_price+atm.put_price)/atm.spot_price),
      'atm_iv':np.nanmean([atm.get('iv_call',np.nan),atm.get('iv_put',np.nan)]),
      'iv_skew':float(atm.get('iv_put',np.nan))-float(atm.get('iv_call',np.nan)),
      'oi_imbalance':oi_imbalance,**{name:float(feature_values[name]) for name in needed}}])
    if row[artifact['features']].isna().any(axis=None):return base|{'status':'BLOCKED','reason':'invalid IV or model feature'}
    if multi_structure:
        scores={}
        for name,item in band_models.items():
            if not item['positive_training_pnl']:continue
            mean=float(item['model'].predict(row[artifact['features']])[0])
            scores[name]={'predicted_mean_points':mean,'residual_q25':float(item['residual_q25']),
                          'conservative_score':mean+float(item['residual_q25'])}
        if not scores:return base|{'status':'NO_TRADE','reason':'no structure has positive aggregate training P&L'}
        structure=max(scores,key=lambda name:scores[name]['conservative_score']);predicted=scores[structure]['conservative_score']
        base|={'structure':structure,'candidate_scores':scores,'predicted_net_points_conservative':predicted,
               'threshold':0.,'strike':float(atm.strike),'dte_band':artifact['dte_band']}
        if predicted<=0:return base|{'status':'NO_TRADE','reason':'every structure has non-positive conservative after-cost score'}
    else:
        predicted=float(artifact['model'].predict(row[artifact['features']])[0])
        base|={'structure':'short_atm_straddle','predicted_net_return':predicted,
               'threshold':float(artifact['threshold']),'strike':float(atm.strike),
               'dte_band':artifact.get('dte_band','4-7')}
        if predicted<=artifact['threshold']:return base|{'status':'NO_TRADE','reason':'predicted edge below train-only threshold'}
    has_quotes={'call_bid','call_ask','put_bid','put_ask'}.issubset(chain.columns)
    if multi_structure:
        ordered=near.sort_values('strike').copy();atm_index=int(np.argmin(abs(ordered.strike-ordered.spot_price)))
        ordered['position']=np.arange(len(ordered))-atm_index;orders=[]
        for option_type,position,qty in artifact['structure_legs'][structure]:
            leg=ordered[ordered.position.eq(position)]
            if leg.empty:return base|{'status':'BLOCKED','reason':f'missing strike position {position} for {structure}'}
            orders.append({'side':'BUY' if qty>0 else 'SELL','option_type':option_type.upper(),
                           'strike':float(leg.strike.iloc[0]),'quantity_lots':abs(int(qty))})
    else:
        orders=[{'side':'SELL','option_type':'CALL','strike':float(atm.strike),'quantity_lots':1},
                {'side':'SELL','option_type':'PUT','strike':float(atm.strike),'quantity_lots':1}]
    return base|{'status':'PAPER_ORDER_READY','executable_quotes':has_quotes,
      'orders':orders,
      'exit_rule':artifact['exit_rule'],
      'research_status':artifact.get('research_status','paper_only'),
      'note':'paper intent only; no broker call performed'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chain',type=Path,required=True);p.add_argument('--prediction',type=Path,required=True);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args()
    lines=[x for x in a.prediction.read_text(encoding='utf-8').splitlines() if x.strip()]; live=json.loads(lines[-1])
    chain=latest_chain(a.chain)
    near=chain[chain.expiry.eq(chain.expiry.min())]
    dte=float(near.loc[(near.strike-near.spot_price).abs().idxmin(),'days_to_expiry'])
    model_path=NEAR_MODEL if dte<8 else (LONGER_MODEL if dte<15 else MULTI_MODEL)
    intent=build_intent(chain,live,joblib.load(model_path));a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('a',encoding='utf-8') as f:f.write(json.dumps(intent,separators=(',',':'))+'\n')
    print(json.dumps(intent,indent=2))

if __name__=='__main__':main()
