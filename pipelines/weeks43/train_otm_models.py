"""Freeze OTM models using pre-2024 data only. Test is never used for selection."""
from pathlib import Path
import json
import hashlib
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss, mean_absolute_error
from .prepare_otm_data import read_frame
from .features import FEATURES, pnl

ROOT=Path(__file__).resolve().parent

def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    return value

def summarize(trades, eligible_days=0):
    t=pd.DataFrame(trades)
    if t.empty: return dict(trades=0,win_rate=0,net_pnl=0,profit_factor=0,maximum_drawdown=0,
        signal_days=0,green_days=0,red_days=0,flat_days=0,eligible_days=eligible_days,coverage=0)
    net=t.net_pnl.to_numpy(float); daily=t.groupby('date').net_pnl.sum(); equity=np.cumsum(net)
    loss=-net[net<0].sum()
    return dict(trades=len(t),win_rate=float((net>0).mean()),net_pnl=float(net.sum()),
        gross_pnl=float(t.gross_pnl.sum()),total_costs=float(t.total_costs.sum()),
        profit_factor=float(net[net>0].sum()/loss) if loss else (None if (net>0).any() else 0),
        maximum_drawdown=float((equity-np.maximum.accumulate(np.r_[0,equity])[1:]).min()),
        signal_days=len(daily),green_days=int((daily>0).sum()),red_days=int((daily<0).sum()),
        flat_days=int((daily==0).sum()),eligible_days=eligible_days,
        no_trade_days=max(0,eligible_days-len(daily)),coverage=len(daily)/max(1,eligible_days))

def select_trades(frame, probabilities, edge, tail, thresholds, cfg):
    out=[]; open_until=[]
    for i,r in enumerate(frame.itertuples(index=False)):
        t=r.execution_time
        open_until=[x for x in open_until if x>t]
        b='4_week' if r.days_to_expiry>=28 else '3_week'
        if (len(open_until)>=cfg['max_open_positions'] or probabilities[i]<thresholds[b]
            or edge[i]*r.quantity<cfg['expected_net_buffer_inr'] or tail[i]*r.quantity>cfg['tail_loss_limit_inr']): continue
        result=pnl(r.entry_premium,r.exit_premium,r.quantity,cfg,r.signal_time[:10])
        out.append(dict(date=r.signal_time[:10],signal_time=r.signal_time,entry_time=t,exit_time=r.exit_time,
            regime=b,call_strike=r.call_strike,put_strike=r.put_strike,quantity=r.quantity,
            gross_pnl=result['gross_pnl'],total_costs=result['costs']['total'],net_pnl=result['net_pnl'],
            exit_reason=r.exit_reason,probability=float(probabilities[i]),expected_net_inr=float(edge[i]*r.quantity),
            tail_loss_inr=float(tail[i]*r.quantity)))
        open_until.append(r.exit_time)
    return out

def calibrated(raw, calibrator):
    x=np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))).reshape(-1,1)
    return calibrator.predict_proba(x)[:,1]

def train():
    cfg=json.loads((ROOT/'config.json').read_text())
    out=ROOT/'otm_models'; out.mkdir(exist_ok=True)
    f=read_frame(ROOT/'data/training_rows.parquet').sort_values('signal_time').reset_index(drop=True)
    # Purge by label END, not merely by feature timestamp; splits are full dates.
    tr=f[f.label_end<'2023-01-01'].copy()
    ca=f[(f.signal_time>='2023-01-01')&(f.label_end<'2023-07-01')].copy()
    va=f[(f.signal_time>='2023-07-01')&(f.label_end<'2024-01-01')].copy()
    te=f[f.signal_time>='2024-01-01'].copy()
    assert tr.label_end.max()<ca.signal_time.min() and ca.label_end.max()<va.signal_time.min()
    assert va.label_end.max()<te.signal_time.min()
    print('Split rows:',{k:len(v) for k,v in [('train',tr),('calibration',ca),('validation',va),('test',te)]},flush=True)
    X=tr[FEATURES]; y=(tr.net_points>0).astype(int)
    def pipeline(model,scale=False):
        steps=[('imputer',SimpleImputer(strategy='median'))]
        if scale: steps.append(('scaler',StandardScaler()))
        return Pipeline(steps+[('estimator',model)])
    classifiers={
        'logistic':pipeline(LogisticRegression(C=.1,max_iter=2000,random_state=43),True),
        'extra_trees':pipeline(ExtraTreesClassifier(n_estimators=160,max_depth=5,min_samples_leaf=20,n_jobs=2,random_state=43)),
        'lightgbm':pipeline(lgb.LGBMClassifier(n_estimators=120,num_leaves=7,max_depth=3,min_child_samples=30,learning_rate=.03,n_jobs=2,verbosity=-1,random_state=43))}
    regressors={
        'ridge':pipeline(Ridge(alpha=100),True),
        'lightgbm_huber':pipeline(lgb.LGBMRegressor(objective='huber',n_estimators=120,num_leaves=7,max_depth=3,min_child_samples=30,learning_rate=.03,n_jobs=2,verbosity=-1,random_state=43))}
    tail=pipeline(lgb.LGBMRegressor(objective='quantile',alpha=.90,n_estimators=120,num_leaves=7,max_depth=3,min_child_samples=30,learning_rate=.03,n_jobs=2,verbosity=-1,random_state=43))
    weights=1/tr.groupby(tr.signal_time.str[:10]).signal_time.transform('size').to_numpy()
    weights*=len(weights)/weights.sum()
    models={}; cal={}; scores={k:{} for k in ['calibration','validation','test']}; evals={}
    for name,model in classifiers.items():
        print('Training probability model',name,flush=True)
        model.fit(X,y,estimator__sample_weight=weights)
        raw=model.predict_proba(ca[FEATURES])[:,1]
        calibrator=LogisticRegression(C=.1,max_iter=1000)
        logits=np.log(np.clip(raw,1e-6,1-1e-6)/(1-np.clip(raw,1e-6,1-1e-6))).reshape(-1,1)
        calibrator.fit(logits,(ca.net_points>0).astype(int))
        cal[name]=calibrator; models[name]=model
        for key,split in [('calibration',ca),('validation',va),('test',te)]:
            scores[key][name]=calibrated(model.predict_proba(split[FEATURES])[:,1],calibrator)
        evals[name]={'task':'win_probability','validation_brier':brier_score_loss(va.net_points>0,scores['validation'][name])}
    for name,model in regressors.items():
        print('Training net decay model',name,flush=True)
        model.fit(X,tr.net_points,estimator__sample_weight=weights); models[name]=model
        for key,split in [('calibration',ca),('validation',va),('test',te)]: scores[key][name]=model.predict(split[FEATURES])
        evals[name]={'task':'exit_adjusted_net_points','validation_mae':mean_absolute_error(va.net_points,scores['validation'][name])}
    print('Training OTM path-loss quantile model',flush=True)
    tail.fit(X,tr.adverse_points,estimator__sample_weight=weights); models['otm_loss_q90']=tail
    # Add a calibration residual quantile for empirical 90% coverage (chronological calibration only).
    correction=max(0.,float(np.quantile(ca.adverse_points-tail.predict(ca[FEATURES]),.90)))
    for key,split in [('calibration',ca),('validation',va),('test',te)]:
        scores[key]['otm_loss_q90']=np.maximum(0,tail.predict(split[FEATURES])+correction)
    evals['otm_loss_q90']={'task':'maximum_adverse_net_loss_points_q90','calibration_additive_correction':correction,
        'validation_coverage':float((va.adverse_points<=scores['validation']['otm_loss_q90']).mean())}
    eligible_days=int(va.signal_time.str[:10].nunique())
    rankings=[]
    for classifier in classifiers:
        for regressor in regressors:
            for t4 in [.50,.55,.60,.65,.70,.75]:
                for t3 in [t4,min(.80,t4+.05),min(.85,t4+.10)]:
                    thresholds={'3_week':round(t3,2),'4_week':round(t4,2)}
                    trades=select_trades(va,scores['validation'][classifier],scores['validation'][regressor],scores['validation']['otm_loss_q90'],thresholds,cfg)
                    stats=summarize(trades,eligible_days)
                    # Sparse 4-week sample: global models; thresholds shrink to same/nearby values.
                    feasible=stats['trades']>=20 and stats['coverage']>=.5 and stats['net_pnl']>0 and (stats['profit_factor'] is None or stats['profit_factor']>=1.5)
                    rankings.append(dict(classifier=classifier,regressor=regressor,thresholds=thresholds,meets_constraints=feasible,**stats))
    ranked=sorted(rankings,key=lambda s:(s['meets_constraints'],s['trades']>=10,s['win_rate'],s['net_pnl'],s['coverage']),reverse=True)
    best=ranked[0]
    # Selected BEFORE looking at any test performance; test is descriptive, not an optimizer input.
    selection={k:best[k] for k in ['classifier','regressor','thresholds','meets_constraints']}
    selection.update(horizon_minutes=120,tail_model='otm_loss_q90',tail_correction=correction,
        expected_net_buffer_inr=cfg['expected_net_buffer_inr'],tail_loss_limit_inr=cfg['tail_loss_limit_inr'],
        max_open_positions=3,profit_target_inr=1500,stop_loss_inr=-750,
        frozen_before_test='2024-01-01',selection_period='2023-07-01 through 2023-12-31',
        status='paper_research_candidate' if best['meets_constraints'] else 'experimental_validation_constraints_not_met')
    (out/'selection.json').write_text(json.dumps(clean(selection),indent=2))
    for name,model in models.items():
        d=out/name; d.mkdir(exist_ok=True)
        package=dict(model=model,features=FEATURES,name=name,train_end_exclusive='2023-01-01',
            calibrator=cal.get(name),tail_correction=correction if name=='otm_loss_q90' else 0)
        joblib.dump(package,d/'model.joblib',compress=3)
        joblib.dump(model,d/'pipeline.joblib',compress=3)
        joblib.dump(model[-1],d/'estimator.joblib',compress=3)
        joblib.dump(model[:-1],d/'preprocessor.joblib',compress=3)
        joblib.dump(model.named_steps.get('scaler'),d/'scaler.joblib',compress=3)
        if name in cal: joblib.dump(cal[name],d/'calibrator.joblib',compress=3)
        (d/'feature_manifest.json').write_text(json.dumps({'feature_order':FEATURES,'scaler':'StandardScaler' if 'scaler' in model.named_steps else 'not required (saved None)',**evals[name]},indent=2))
    test_trades=select_trades(te,scores['test'][selection['classifier']],scores['test'][selection['regressor']],scores['test']['otm_loss_q90'],selection['thresholds'],cfg)
    pd.DataFrame(test_trades).to_csv(out/'complete_path_test_trades.csv',index=False)
    pd.DataFrame(ranked).to_json(out/'validation_candidates.json',orient='records',indent=2)
    report={'selection':selection,'validation_selected':best,'complete_path_test':summarize(test_trades,int(te.signal_time.str[:10].nunique())),
        'test_by_bucket':{b:summarize([t for t in test_trades if t['regime']==b],int(te.loc[(te.days_to_expiry>=28) if b=='4_week' else (te.days_to_expiry<28)].signal_time.str[:10].nunique())) for b in ['3_week','4_week']},
        'model_metrics':evals,'splits':{k:{'rows':len(v),'first':v.signal_time.min(),'last':v.signal_time.max(),'days':int(v.signal_time.str[:10].nunique())} for k,v in [('train',tr),('calibration',ca),('validation',va),('test',te)]},
        'limitations':['2024+ benchmark was examined in previous research; not a new untouched confirmation.',
            'Selection uses complete historical quote paths; chronological replay separately reports missing paths and unresolved positions.',
            'Minute data cannot establish intra-minute stop fills; gaps can exceed the stop.',
            'Source-report historical cost and lot schedules are assumptions, live uses current configured charges and contract lot size.',
            'No model refit on calibration, validation or test data; four-week thresholds share global models.',
            'Supervised labels require exact start/end quotes and at least 80 percent intermediate minute coverage; missing peaks remain unknown.',
            'Fitting observations are sparse and overlapping; see split row/day counts rather than treating rows as independent trials.',
            'Historical lot sizes differ from the live contract; probabilities are estimates, not guarantees at a new lot size.',
            'Validation constraints do not impose a measured drawdown ceiling; drawdown is reported, not promised.'],
        'versions':{'sklearn':sklearn.__version__,'lightgbm':lgb.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'feature_hash':hashlib.sha256('\n'.join(FEATURES).encode()).hexdigest()}
    (out/'training_report.json').write_text(json.dumps(clean(report),indent=2))
    # Score exports permit inspection of EVERY trained candidate; never feed labels to a predictor.
    for key,split in [('validation',va),('test',te)]:
        exported=split[['signal_time','expiry','call_strike','put_strike','quantity']].copy()
        for name,values in scores[key].items(): exported[name]=values
        exported.to_csv(out/f'{key}_all_model_outputs.csv',index=False)
    print(json.dumps(clean(report),indent=2),flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    train()
