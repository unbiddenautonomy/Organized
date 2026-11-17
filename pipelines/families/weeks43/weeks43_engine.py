"""Independent paper-only ledger. No broker imports, credentials or order API."""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import json
import math
import sqlite3
import time
import numpy as np
import pandas as pd
import joblib
from weeks43_features import FEATURES, feature_row, history_add, pnl, parse_time
from weeks43_settings import MAX_POSITIONS, defaults, validate_settings
from voltr.core.clock import NSE_EPOCH as EPOCH

ROOT=Path(__file__).resolve().parent

def second(t): return int((parse_time(t)-EPOCH).total_seconds())
def stamp(t): return (EPOCH+timedelta(seconds=float(t))).isoformat(sep=' ')
def json_safe(v):
    if isinstance(v,dict): return {str(k):json_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,deque)): return [json_safe(x) for x in v]
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    if isinstance(v,(np.floating,float)): return float(v) if math.isfinite(v) else None
    if isinstance(v,(datetime,Path)): return str(v)
    return v

def summarize_positions(positions,eligible_days):
    closed=sorted((p for p in positions if p['status']=='CLOSED'),key=lambda p:p['exit_time'])
    daily={}; net=0.; peak=0.; drawdown=0.; gains=0.; losses=0.
    for p in closed:
        value=p['net_pnl'];net+=value;peak=max(peak,net);drawdown=min(drawdown,net-peak)
        gains+=max(0,value);losses+=max(0,-value)
        day=p['entry_time'][:10];daily[day]=daily.get(day,0)+value
    return dict(trades=len(closed),win_rate=sum(p['net_pnl']>0 for p in closed)/len(closed) if closed else None,
        net_pnl=net,gross_pnl=sum(p['gross_pnl'] for p in closed),total_costs=sum(p['costs']['total'] for p in closed),
        signal_days=len(daily),eligible_days=len(eligible_days),coverage=len(daily)/len(eligible_days) if eligible_days else 0,
        no_trade_days=max(0,len(eligible_days)-len(daily)),green_days=sum(v>0 for v in daily.values()),
        red_days=sum(v<0 for v in daily.values()),flat_days=sum(v==0 for v in daily.values()),
        profit_factor=gains/losses if losses else None,maximum_drawdown=drawdown)

class Ledger:
    def __init__(self,root,durable=True):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'ledger.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL' if durable else 'PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS records(kind TEXT,id TEXT,time TEXT,payload TEXT,PRIMARY KEY(kind,id))')
        self.db.execute('CREATE INDEX IF NOT EXISTS record_time ON records(kind,time)')
        self.db.commit(); self.handles={}
    def put(self,kind,id,value,mirror=True):
        value=json_safe(value); t=str(value.get('time',value.get('signal_time','')))
        body=json.dumps(value,allow_nan=False,separators=(',',':'))
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',(kind,str(id),t,body))
        if mirror:
            day=t[:10] if len(t)>=10 else datetime.now().date().isoformat()
            folder=self.root/day; folder.mkdir(exist_ok=True)
            key=(day,kind)
            if key not in self.handles: self.handles[key]=(folder/(kind+'.jsonl')).open('a',encoding='utf-8',buffering=1)
            self.handles[key].write(body+'\n')
    def get(self,kind,id):
        r=self.db.execute('SELECT payload FROM records WHERE kind=? AND id=?',(kind,str(id))).fetchone()
        return json.loads(r[0]) if r else None
    def all(self,kind):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM records WHERE kind=? ORDER BY time',(kind,))]
    def recent(self,kind,limit):
        rows=self.db.execute('SELECT payload FROM records WHERE kind=? ORDER BY time DESC LIMIT ?',(kind,limit)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]
    def close(self):
        for h in self.handles.values(): h.close()
        self.db.close()

class ModelRunner:
    def __init__(self,root=ROOT):
        self.root=Path(root)
        self.selection=json.loads((self.root/'otm_models/selection.json').read_text())
        self.models={p.parent.name:joblib.load(p) for p in (self.root/'otm_models').glob('*/model.joblib')}
        # Constrain CPU usage in the separate worker, including archived tree ensembles.
        for pack in self.models.values():
            model=pack['model']
            for key in list(model.get_params()):
                if key.endswith('n_jobs'): model.set_params(**{key:1})
    def batch(self,frame):
        result={}
        X=frame[FEATURES]
        for name,p in self.models.items():
            if p.get('calibrator') is not None:
                raw=np.clip(p['model'].predict_proba(X)[:,1],1e-6,1-1e-6)
                result[name]=p['calibrator'].predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]
            else:
                values=p['model'].predict(X)
                result[name]=np.maximum(0,values+p.get('tail_correction',0)) if name=='otm_loss_q90' else values
        return result
    def score(self,features):
        return {name:float(values[0]) for name,values in self.batch(pd.DataFrame([features])).items()}

class PaperEngine:
    def __init__(self,config,ledger,runner,live=False,entry_settings=None):
        self.cfg=dict(config, max_open_positions=MAX_POSITIONS); self.ledger=ledger; self.runner=runner; self.live=live
        self.selection=runner.selection
        self.entry_settings=validate_settings(entry_settings or defaults(self.selection))
        self.settings_error=None
        self.history={}; self.quotes={}; self.pending={}; self.now=0
        self.last_quote_second=None
        self.eligible_days={'3_week':set(),'4_week':set()}
        self.positions={p['id']:p for p in ledger.all('positions') if p['status']=='OPEN'}
        # Reconcile mirrors after a crash between authoritative state and log writes.
        for p in ledger.all('positions'):
            kind='trades' if p['status']=='CLOSED' else 'entries'
            if ledger.get(kind,p['id']) is None: ledger.put(kind,p['id'],p)
        for s in ledger.all('signals'):
            if s.get('eligible_regime'): self.eligible_days[s['regime']].add(s['signal_time'][:10])
            if s.get('decision')=='PENDING' and ledger.get('positions',s['id']) is None:
                ledger.put('events',s['id']+':restart_cancel',dict(time=s['time'],event='ENTRY_CANCELLED_ON_RESTART',signal_id=s['id']))
        self.tracks={p['id']:p for p in ledger.all('tracks') if p.get('active')}
        self.last_path_minute=None; self.last_status=0; self.gaps=0; self.error=None
        # Restore only feature snapshots, never old quotes or pending fills.
        for r in ledger.recent('snapshots',65): history_add(self.history,r['row'])
        self.last_signal=None

    def apply_settings(self,settings,now):
        settings=validate_settings(settings)
        changed=settings!=self.entry_settings or self.settings_error is not None
        self.entry_settings=settings; self.settings_error=None
        if changed:
            self.ledger.put('events','settings:'+settings['revision'],dict(time=stamp(now),event='ENTRY_SETTINGS_APPLIED',entry_settings=settings))
            self._recheck_pending(now)
            self.persist_status(now)

    def _recheck_pending(self,now):
        for id,p in list(self.pending.items()):
            cutoff=self.entry_settings['thresholds'][p['regime']]
            probability=p['win_probability']
            reason='entry_settings_unavailable' if self.settings_error else 'feed_or_worker_error' if self.error else None
            if reason is None and not (math.isfinite(probability) and 0<=probability<=1 and probability>=cutoff):
                reason='win_probability_below_updated_threshold'
            if reason:
                self.ledger.put('events',id+':policy_cancel',dict(time=stamp(now),event='ENTRY_CANCELLED',signal_id=id,
                    reason=reason,win_probability=probability,threshold=cutoff,settings_revision=self.entry_settings['revision']))
                del self.pending[id]

    def fresh(self,call_key,put_key,now,min_time=None):
        q=[self.quotes.get(str(k)) for k in (call_key,put_key)]
        if any(v is None for v in q): return None
        if any(v[1]>now or now-v[1]>self.cfg['max_quote_age_seconds'] for v in q): return None
        if min_time is not None and any(v[1]<min_time for v in q): return None
        return q[0][0],q[1][0],q[0][1],q[1][1]

    def _save_position(self,p):
        self.ledger.put('positions',p['id'],p,mirror=False)

    def on_quotes(self,quotes,now):
        self.now=max(self.now,int(now))
        for key,price,ts in quotes:
            old=self.quotes.get(str(key))
            if math.isfinite(float(price)) and price>0 and ts<=now and (old is None or ts>=old[1]):
                self.quotes[str(key)]=(float(price),int(ts))
                self.last_quote_second=max(self.last_quote_second or 0,int(ts))
        # Exits release capacity before pending entries. All fills use only observed fresh prices.
        for id,p in list(self.positions.items()):
            q=self.fresh(p['call_key'],p['put_key'],now)
            if q is None:
                p['mark_status']='STALE_OR_MISSING'; continue
            result=pnl(p['entry_premium'],q[0]+q[1],p['quantity'],self.cfg,p['date'] if not self.live else None)
            p.update(result,call_mark=q[0],put_mark=q[1],mark_time=stamp(now),mark_status='FRESH')
            reason=None
            if result['net_pnl']>=self.cfg['profit_target_inr']: reason='TARGET'
            elif result['net_pnl']<=self.cfg['stop_loss_inr']: reason='STOP'
            elif now>=p['exit_due']: reason='HORIZON'
            elif stamp(now)[11:19]>=self.cfg['square_off']: reason='SQUARE_OFF'
            if reason:
                p.update(status='CLOSED',time=stamp(now),exit_time=stamp(now),exit_premium=q[0]+q[1],
                    exit_call=q[0],exit_put=q[1],exit_reason=reason,exit_call_quote_time=stamp(q[2]),exit_put_quote_time=stamp(q[3]))
                # Closed position is authoritative even if interrupted before the JSONL mirror.
                self._save_position(p); self.ledger.put('trades',id,p)
                self.ledger.put('events',id+':exit',dict(time=stamp(now),event='EXIT',trade_id=id,reason=reason,net_pnl=p['net_pnl']))
                del self.positions[id]
                from position_alert import notify_position_action
                notify_position_action()
        self._recheck_pending(now)
        for id,p in list(self.pending.items()):
            if now>p['expires']:
                self.ledger.put('events',id+':expired',dict(time=stamp(now),event='ENTRY_EXPIRED',signal_id=id,reason='no_fresh_post_decision_quotes'))
                del self.pending[id]; continue
            q=self.fresh(p['call_key'],p['put_key'],now,p['not_before'])
            if q is None or now<p['not_before']: continue
            if len(self.positions)>=self.cfg['max_open_positions']: continue
            if stamp(now)[11:19]>=self.cfg['square_off']: continue
            p.update(status='OPEN',time=stamp(now),entry_time=stamp(now),date=stamp(now)[:10],
                entry_threshold=self.entry_settings['thresholds'][p['regime']],entry_settings_revision=self.entry_settings['revision'],
                entry_call=q[0],entry_put=q[1],entry_premium=q[0]+q[1],entry_call_quote_time=stamp(q[2]),entry_put_quote_time=stamp(q[3]),
                exit_due=min(now+self.cfg['horizon_minutes']*60,second(stamp(now)[:10]+' '+self.cfg['square_off'])),
                mark_status='FRESH',cost_config={k:v for k,v in self.cfg.items() if 'rate' in k or 'slippage' in k or 'brokerage' in k})
            p.update(pnl(p['entry_premium'],p['entry_premium'],p['quantity'],self.cfg,p['date'] if not self.live else None))
            self.positions[id]=p; self._save_position(p)
            if id in self.tracks:
                self.tracks[id]['end']=now+self.cfg['horizon_minutes']*60
                self.ledger.put('tracks',id,self.tracks[id],mirror=False)
            self.ledger.put('entries',id,p)
            self.ledger.put('events',id+':entry',dict(time=stamp(now),event='ENTRY',trade_id=id,call_strike=p['call_strike'],put_strike=p['put_strike']))
            from position_alert import notify_position_action
            notify_position_action()
            del self.pending[id]
        self.mark_paths(now)
        if now-self.last_status>=1: self.persist_status(now)

    def mark_paths(self,now,force=False):
        minute=now//60
        if self.last_path_minute==minute and not force: return
        self.last_path_minute=minute
        for id,p in list(self.tracks.items()):
            if now<p['start']: continue
            observation_time=min(now,p['end'])
            q=self.fresh(p['call_key'],p['put_key'],observation_time)
            record=dict(time=stamp(observation_time),signal_id=id,signal_time=p['signal_time'],expiry=p['expiry'],
                atm_at_t=p['atm_at_t'],call_strike_at_t=p['call_strike'],put_strike_at_t=p['put_strike'],
                horizon_minutes=120,elapsed_minutes=(observation_time-p['start'])/60,
                call=None if q is None else q[0],put=None if q is None else q[1],
                call_quote_time=None if q is None else stamp(q[2]),put_quote_time=None if q is None else stamp(q[3]),
                quote_status='STALE_OR_MISSING' if q is None else 'FRESH',final=now>=p['end'])
            self.ledger.put('strike_paths',id+':'+str(observation_time//60),record)
            if now>=p['end']:
                p['active']=False; self.ledger.put('tracks',id,p,mirror=False); del self.tracks[id]

    def on_snapshot(self,row,now,precomputed=None,diagnostics=None):
        self.now=max(self.now,int(now)); t=second(row['signal_time'])
        id=row['expiry']+':'+row['signal_time']
        if self.ledger.get('signals',id) is not None: return
        # Keep all input snapshots, including warmup and rejected entries.
        history_add(self.history,row)
        self.ledger.put('snapshots',id,dict(time=stamp(now),row=row))
        f=feature_row(row,self.history)
        output=precomputed if precomputed is not None else self.runner.score(f) if f is not None else {}
        bucket='4_week' if row['days_to_expiry']>=28 else '3_week'
        eligible_regime=21<=row['days_to_expiry']<35
        if eligible_regime: self.eligible_days[bucket].add(row['signal_time'][:10])
        event=dict(id=id,time=stamp(now),signal_time=row['signal_time'],expiry=row['expiry'],regime=bucket,
            eligible_regime=eligible_regime,days_to_expiry=row['days_to_expiry'],
            atm_at_t=row['atm'],call_strike=row['call_strike'],put_strike=row['put_strike'],
            quantity=row['quantity'],horizon_minutes=120,outputs=output,features=f,atm_diagnostics=diagnostics or {},
            selected_classifier=self.selection['classifier'],selected_regressor=self.selection['regressor'],
            threshold=self.entry_settings['thresholds'][bucket],comparison='>=',settings_revision=self.entry_settings['revision'],
            settings_source=self.entry_settings.get('source'),max_open_positions=MAX_POSITIONS,
            open_positions_at_signal=len(self.positions),pending_entries_at_signal=len(self.pending),decision='REJECTED',reason='')
        probability=output.get(self.selection['classifier'])
        edge=output.get(self.selection['regressor']); tail=output.get('otm_loss_q90')
        valid_probability=not isinstance(probability,(bool,np.bool_)) and isinstance(probability,(int,float,np.number)) and math.isfinite(probability) and 0<=probability<=1
        event.update(win_probability=probability,probability_passed=bool(probability>=event['threshold']) if valid_probability else None,
            expected_net_inr=edge*row['quantity'] if edge is not None else None,
            predicted_loss_q90_inr=tail*row['quantity'] if tail is not None else None)
        reason=None
        if f is None: reason='need_5_15_30_60_minute_lags_max_age_1m_and_13_observations'
        elif not (21<=row['days_to_expiry']<35): reason='outside_3_or_4_week_regime'
        elif row['signal_time'][11:19]<self.cfg['entry_start']: reason='before_entry_window'
        elif stamp(now+120*60)[11:19]>self.cfg['square_off'] or stamp(now)[:10]!=row['signal_time'][:10]: reason='insufficient_session_time'
        elif now-second(row.get('available_at',row['signal_time']))>self.cfg['max_signal_age_seconds']: reason='stale_signal'
        elif self.error: reason='feed_or_worker_error'
        elif self.settings_error: reason='entry_settings_unavailable'
        else:
            if not valid_probability: reason='invalid_win_probability'
            elif edge is None or tail is None or not math.isfinite(edge) or not math.isfinite(tail): reason='invalid_risk_prediction'
            elif not event['probability_passed']: reason='win_probability_below_threshold'
            elif event['expected_net_inr']<self.cfg['expected_net_buffer_inr']: reason='expected_net_below_buffer'
            elif event['predicted_loss_q90_inr']>self.cfg['tail_loss_limit_inr']: reason='otm_loss_limit_exceeded'
            elif len(self.positions)+len(self.pending)>=self.cfg['max_open_positions']: reason='three_position_limit'
        if f is not None and 21<=row['days_to_expiry']<35:
            track=dict(id=id,signal_time=row['signal_time'],expiry=row['expiry'],atm_at_t=row['atm'],
                call_strike=row['call_strike'],put_strike=row['put_strike'],call_key=str(row['call_key']),put_key=str(row['put_key']),
                start=t,end=second(row.get('available_at',row['signal_time']))+121*60,active=True)
            self.tracks[id]=track; self.ledger.put('tracks',id,track,mirror=False)
        if reason is None:
            not_before=max(now,second(row.get('available_at',row['signal_time']))+60)
            if self.live:
                from zoneinfo import ZoneInfo
                wall=datetime.now(ZoneInfo('Asia/Kolkata')).replace(tzinfo=None)
                not_before=max(not_before,second(wall)+1)
            p=dict(id=id,signal_time=row['signal_time'],expiry=row['expiry'],atm_at_t=row['atm'],regime=bucket,
                call_strike=row['call_strike'],put_strike=row['put_strike'],call_key=str(row['call_key']),put_key=str(row['put_key']),
                quantity=row['quantity'],not_before=not_before,expires=not_before+30,model_outputs=output,
                win_probability=probability,signal_threshold=event['threshold'],signal_settings_revision=event['settings_revision'],
                probability_passed=True,comparison='>=',max_open_positions=MAX_POSITIONS)
            self.pending[id]=p
            event.update(decision='PENDING',reason='awaiting_fresh_post_decision_quotes')
        else: event['reason']=reason
        self.ledger.put('signals',id,event); self.last_signal=event
        if event['decision']=='PENDING':
            from position_alert import notify_signal
            notify_signal()
        self.persist_status(now)

    def persist_status(self,now):
        self.last_status=now
        for p in self.positions.values(): self._save_position(p)
        closed=[p for p in self.ledger.all('positions') if p['status']=='CLOSED']
        today=stamp(now)[:10]
        today_closed=[p for p in closed if p.get('exit_time','')[:10]==today]
        summary=summarize_positions(closed,self.eligible_days['3_week']|self.eligible_days['4_week'])
        by_bucket={b:summarize_positions([p for p in closed if p['regime']==b],days) for b,days in self.eligible_days.items()}
        self.ledger.put('status','latest',dict(time=stamp(now),mode='LIVE_PAPER' if self.live else 'BACKTEST',
            last_quote_time=stamp(self.last_quote_second) if self.last_quote_second is not None else None,
            summary=summary,by_bucket=by_bucket,
            entry_settings=self.entry_settings,settings_error=self.settings_error,
            capacity=dict(max_positions=MAX_POSITIONS,open=len(self.positions),pending=len(self.pending),
                available=max(0,MAX_POSITIONS-len(self.positions)-len(self.pending))),
            selection=self.selection,config={k:self.cfg[k] for k in ['horizon_minutes','max_open_positions','profit_target_inr','stop_loss_inr','square_off','lots_per_position','round_trip_slippage_points']},
            open_positions=list(self.positions.values()),pending_entries=list(self.pending.values()),
            open_count=len(self.positions),tracking_signals=len(self.tracks),closed_trades=len(closed),
            realized_net=sum(p['net_pnl'] for p in closed),today_realized_net=sum(p['net_pnl'] for p in today_closed),
            unrealized_net=sum(p.get('net_pnl',0) for p in self.positions.values()),
            last_signal=self.last_signal,error=self.error),mirror=False)

    def close(self):
        for id in self.pending:
            self.ledger.put('events',id+':shutdown_cancel',dict(time=stamp(self.now),event='ENTRY_CANCELLED_ON_SHUTDOWN',signal_id=id))
        self.pending.clear()
        if self.now: self.persist_status(self.now)
        self.ledger.close()
