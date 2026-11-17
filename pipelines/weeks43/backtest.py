"""Chronological replay using the live engine; no selection on future quote availability."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import os
import traceback
import numpy as np
import pandas as pd
from .prepare_otm_data import read_frame
from .features import feature_row, history_add, historical_quantity
from .engine import ROOT, Ledger, ModelRunner, PaperEngine, second, json_safe
from .train_otm_models import summarize
from .settings import read_settings, validate_settings, write_snapshot, atomic_write_json

def key(expiry,strike,side): return f'{str(expiry)[:10]}:{float(strike):g}:{side}'

def replay_summary(trades,eligible_days):
    result=summarize(trades,eligible_days)
    if not trades:
        result.update(win_rate=None,gross_pnl=0.,total_costs=0.,profit_factor=None,no_trade_days=eligible_days)
    return result

def replay(run,settings_file=None):
    cfg=json.loads((ROOT/'config.json').read_text()); root=ROOT/'backtests'/run
    entry_settings=validate_settings(json.loads(Path(settings_file).read_text(encoding='utf-8'))) if settings_file else read_settings(ROOT)
    if root.exists(): raise ValueError('Run already exists: '+run)
    root.mkdir(parents=True); runtime=ROOT/'runtime'; runtime.mkdir(exist_ok=True)
    write_snapshot(root/'entry_settings.json',entry_settings)
    runner=ModelRunner(); ledger=Ledger(root,durable=False); engine=PaperEngine(cfg,ledger,runner,entry_settings=entry_settings)
    snapshots=read_frame(ROOT/'data/snapshots.parquet')
    snapshots=snapshots.loc[snapshots.signal_time>=cfg['test_start']].sort_values('signal_time').copy()
    history={}; features=[]; indices=[]
    for index,r in enumerate(snapshots.to_dict('records')):
        history_add(history,r); f=feature_row(r,history)
        if f is not None: features.append(f); indices.append(index)
    batch=runner.batch(pd.DataFrame(features))
    scored={index:{name:float(values[i]) for name,values in batch.items()} for i,index in enumerate(indices)}
    chain=read_frame(ROOT/'data/chain.parquet'); chain=chain.loc[chain.time>=pd.Timestamp(cfg['test_start'])]
    snapshots['row_index']=range(len(snapshots)); snapshots['date']=snapshots.signal_time.str[:10]
    eligible={b:set() for b in ['combined','3_week','4_week']}
    for r in snapshots.itertuples(index=False):
        b='4_week' if r.days_to_expiry>=28 else '3_week'
        eligible[b].add(r.date); eligible['combined'].add(r.date)
    days=sorted(eligible['combined'])
    for count,day in enumerate(days,1):
        print('Replaying',day,f'{count}/{len(days)}',flush=True)
        cg=chain[chain.trade_date.eq(pd.Timestamp(day))]
        quote_events={}
        for time_,group in cg.groupby('time',sort=True):
            ts=second(time_); quotes=[]
            for r in group.itertuples(index=False):
                if np.isfinite(r.call) and r.call>0: quotes.append((key(r.expiry,r.strike,'CE'),float(r.call),ts))
                if np.isfinite(r.put) and r.put>0: quotes.append((key(r.expiry,r.strike,'PE'),float(r.put),ts))
            quote_events[ts]=quotes
        signal_events={}
        for r in snapshots.loc[snapshots.date.eq(day)].to_dict('records'):
            index=r.pop('row_index'); r.pop('date')
            r.update(call_key=key(r['expiry'],r['call_strike'],'CE'),put_key=key(r['expiry'],r['put_strike'],'PE'),quantity=historical_quantity(r['expiry'])*cfg['lots_per_position'])
            signal_events.setdefault(second(r['available_at']),[]).append((index,r))
        # Include minute clocks for missing-path records and timed exits with stale data.
        clocks=set(range(second(day+' 09:15:00'),second(day+' 15:31:00'),60))
        for ts in sorted(set(quote_events)|set(signal_events)|clocks):
            engine.on_quotes(quote_events.get(ts,[]),ts)
            for index,r in signal_events.get(ts,[]):
                engine.on_snapshot(r,ts,precomputed=scored.get(index,{}))
        progress={'state':'RUNNING','run':run,'completed_days':count,'total_days':len(days),'last_day':day}
        atomic_write_json(runtime/'backtest_status.json',progress)
    all_positions=ledger.all('positions'); closed=[p for p in all_positions if p['status']=='CLOSED']
    trades=[dict(date=p['entry_time'][:10],regime=p['regime'],gross_pnl=p['gross_pnl'],net_pnl=p['net_pnl'],total_costs=p['costs']['total']) for p in sorted(closed,key=lambda x:x['exit_time'])]
    signals=ledger.all('signals')
    reasons=pd.Series([x['reason'] for x in signals]).value_counts().to_dict()
    result={'state':'COMPLETE','period_start':cfg['test_start'],'period_end':snapshots.signal_time.max()[:10],
        'strategy':'one-strike OTM; 120 minutes; up to 3 positions; net TP1500 / SL-750',
        'selection':runner.selection,'entry_settings':entry_settings,'summary':replay_summary(trades,len(eligible['combined'])),
        'by_bucket':{b:replay_summary([t for t in trades if t['regime']==b],len(eligible[b])) for b in ['3_week','4_week']},
        'signals_logged':len(signals),'model_output_rows':sum(bool(s['outputs']) for s in signals),
        'decision_reasons':reasons,'unresolved_positions':list(engine.positions.values()),'pending_entries':list(engine.pending.values()),
        'execution':'Next available pair after a 60-second signal delay; original strikes; quote age <=60 seconds. Gaps do not fabricate fills.',
        'limitations':['Historical data is minute snapshots, not tick/order-book replay; stop losses can overshoot.',
            'Cost rates and historical lots follow the provided source report; live configuration has separate current rates.',
            'The previously examined 2024+ period is a retrospective benchmark, not fresh validation.',
            'Reported realized P&L excludes unresolved positions; these are listed and retain capacity.',
            'Daily color is the sum of closed-trade net P&L assigned to entry date.'],
        'cost_config':{k:v for k,v in cfg.items() if 'rate' in k or 'slippage' in k or 'brokerage' in k or 'cost_mode' in k}}
    trade_columns=['id','signal_time','entry_time','exit_time','regime','expiry','atm_at_t','call_strike','put_strike',
        'quantity','entry_premium','exit_premium','exit_reason','gross_pnl','costs','net_pnl','win_probability',
        'signal_threshold','entry_threshold','signal_settings_revision','entry_settings_revision']
    (pd.DataFrame(closed) if closed else pd.DataFrame(columns=trade_columns)).to_csv(root/'trades.csv',index=False)
    pd.DataFrame([dict(time=s['signal_time'],regime=s['regime'],decision=s['decision'],reason=s['reason'],
        win_probability=s['win_probability'],threshold=s['threshold'],probability_passed=s['probability_passed'],
        settings_revision=s['settings_revision'],comparison=s['comparison'],**s['outputs']) for s in signals]).to_csv(root/'all_model_outputs.csv',index=False)
    daily=pd.DataFrame(trades)
    if daily.empty:
        daily=pd.DataFrame(0,index=pd.Index(days,name='date'),columns=['trades','gross_pnl','costs','net_pnl'])
    else:
        daily=daily.groupby('date').agg(trades=('net_pnl','size'),gross_pnl=('gross_pnl','sum'),costs=('total_costs','sum'),net_pnl=('net_pnl','sum')).reindex(days,fill_value=0)
    daily['day_color']=np.where(daily.net_pnl>0,'GREEN',np.where(daily.net_pnl<0,'RED',np.where(daily.trades>0,'FLAT','NO_TRADE')))
    daily.to_csv(root/'daily_pnl.csv')
    engine.close()
    # Publish completion only after every export and the ledger are saved.
    atomic_write_json(root/'summary.json',json_safe(result))
    atomic_write_json(runtime/'backtest_status.json',{'state':'COMPLETE','run':run})
    print(json.dumps(json_safe(result),indent=2),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',default='replay_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--settings-file',help='Immutable policy snapshot captured when this replay was launched'); args=parser.parse_args()
    import re
    if not re.fullmatch(r'(replay_\d{8}_\d{6}|initial_2024_test)',args.run): parser.error('invalid run name')
    try: replay(args.run,args.settings_file)
    except Exception as exc:
        (ROOT/'runtime').mkdir(exist_ok=True)
        atomic_write_json(ROOT/'runtime/backtest_status.json',{'state':'ERROR','run':args.run,'error':str(exc)})
        traceback.print_exc(); raise
