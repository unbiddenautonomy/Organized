"""Small stdlib-only adapter for the existing UDP app; all ML runs in a child process.

Queues never wait in the original feed thread. Any overflow stops NEW OTM entries
and is visible in the new dashboard. The original paper engine is never called.
"""
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import math
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from weeks43_settings import read_settings, save_settings, write_snapshot, atomic_write_json, SettingsConflict
from voltr.core.clock import NSE_EPOCH as EPOCH

ROOT=Path(__file__).resolve().parent

def python_executable(cfg):
    configured = cfg.get('python_executable', 'auto')
    if configured in ('', 'auto', None): return sys.executable
    candidate = Path(configured).expanduser()
    return str((candidate if candidate.is_absolute() else ROOT/candidate).resolve())
LOG=logging.getLogger('weeks43_bridge')

def read_records(kind,limit=100,run=None,offset=0,signal_id=None):
    folder=ROOT/'logs' if run is None else ROOT/'backtests'/run
    db=folder/'ledger.sqlite3'
    if not db.exists(): return []
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True,timeout=.5)
    try:
        if kind=='scored':
            rows=con.execute("SELECT payload FROM records WHERE kind='signals' AND json_extract(payload,'$.outputs') != '{}' ORDER BY time DESC LIMIT ? OFFSET ?",(limit,offset))
        elif kind=='strike_paths' and signal_id:
            rows=con.execute('SELECT payload FROM records WHERE kind=? AND id LIKE ? ORDER BY time LIMIT ? OFFSET ?',
                (kind,signal_id+':%',limit,offset))
        else:
            rows=con.execute('SELECT payload FROM records WHERE kind=? ORDER BY time DESC LIMIT ? OFFSET ?',
                (kind,limit,offset))
        result=[json.loads(r[0]) for r in rows]
        # The original t snapshot is already persisted in snapshots.jsonl/SQLite.
        # Include it in the path viewer before subsequent minute observations.
        if kind=='strike_paths' and signal_id and offset==0:
            saved=con.execute("SELECT payload FROM records WHERE kind='snapshots' AND id=?",(signal_id,)).fetchone()
            if saved:
                snapshot=json.loads(saved[0]); r=snapshot['row']; available=r.get('available_at',r['signal_time'])
                initial=dict(time=available,recorded_at=snapshot['time'],signal_id=signal_id,signal_time=r['signal_time'],
                    expiry=r['expiry'],atm_at_t=r['atm'],call_strike_at_t=r['call_strike'],put_strike_at_t=r['put_strike'],
                    elapsed_minutes=(datetime.fromisoformat(available)-datetime.fromisoformat(r['signal_time'])).total_seconds()/60,
                    call=r['call'],put=r['put'],call_quote_time=available,put_quote_time=available,
                    quote_status='SIGNAL_SNAPSHOT',final=False)
                result.insert(0,initial)
        return result
    finally: con.close()

def status():
    records=read_records('status',1)
    result=records[0] if records else {'mode':'WAITING','open_positions':[],'pending_entries':[],'open_count':0}
    p=ROOT/'runtime/bridge_status.json'
    if p.exists():
        try: result['worker']=json.loads(p.read_text())
        except (OSError,ValueError): pass
    model=ROOT/'otm_models/selection.json'
    if model.exists(): result['selection']=json.loads(model.read_text())
    result['server_time']=datetime.now().isoformat(sep=' ')
    try: result['requested_entry_settings']=read_settings(ROOT)
    except (OSError,ValueError) as exc: result['settings_error']=str(exc)
    return result

def settings_status():
    records=read_records('status',1)
    saved=records[0] if records else {}
    return dict(requested=read_settings(ROOT),applied=saved.get('entry_settings'),
                applied_at=saved.get('time'),error=saved.get('settings_error'))

def backtests():
    out=[]
    for path in sorted((ROOT/'backtests').glob('*/summary.json'),reverse=True):
        try: out.append(dict(run=path.parent.name,**json.loads(path.read_text())))
        except (OSError,ValueError): pass
    progress=ROOT/'runtime/backtest_status.json'
    return {'runs':out,'job':json.loads(progress.read_text()) if progress.exists() else None}

BACKTEST_LOCK=threading.Lock()
BACKTEST_PROCESS=None
def launch_backtest():
    global BACKTEST_PROCESS
    with BACKTEST_LOCK:
        if BACKTEST_PROCESS is not None and BACKTEST_PROCESS.poll() is None:
            return {'started':False,'reason':'already_running'}
        cfg=json.loads((ROOT/'config.json').read_text())
        name='replay_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        runtime=ROOT/'runtime'; runtime.mkdir(exist_ok=True)
        settings_file=runtime/(name+'.entry_settings.json')
        write_snapshot(settings_file,read_settings(ROOT))
        atomic_write_json(runtime/'backtest_status.json',{'state':'RUNNING','run':name})
        log=(runtime/(name+'.log')).open('ab',buffering=0)
        try:
            BACKTEST_PROCESS=subprocess.Popen([python_executable(cfg),'-u',str(ROOT/'backtest_otm.py'),'--run',name,'--settings-file',str(settings_file)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        finally: log.close()
        return {'started':True,'run':name}

class Bridge:
    def __init__(self,engine):
        self.engine=engine; self.queue=queue.Queue(maxsize=50000); self.dropped=0; self.failed=False; self.closed=False
        self.runtime=ROOT/'runtime'; self.runtime.mkdir(exist_ok=True)
        cfg=json.loads((ROOT/'config.json').read_text())
        self.config=cfg; self.oi_history={}
        log=(self.runtime/'worker.log').open('ab',buffering=0)
        try:
            self.process=subprocess.Popen([python_executable(cfg),'-u',str(ROOT/'weeks43_worker.py')],
                cwd=ROOT,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        finally: log.close()
        self.writer=threading.Thread(target=self._write,name='weeks43-feed-writer',daemon=True); self.writer.start()
        self._status('STARTING')
        self.submit({'kind':'metadata','lot_size':int(engine.lot_size),'expiry':str(engine.expiry)})

    def submit(self,event):
        if self.closed: return
        try: self.queue.put_nowait(event)
        except queue.Full: self.dropped+=1; self.failed=True

    def tick(self,token,price,exchange_second):
        self.submit({'kind':'quote','key':str(token),'price':float(price),'second':int(exchange_second)})

    def oi(self,token,value,exchange_second):
        history=self.oi_history.setdefault(token,deque(maxlen=1000))
        if history and exchange_second==history[-1][1]: history[-1]=(value,exchange_second)
        elif not history or exchange_second>history[-1][1]: history.append((value,exchange_second))

    def snapshot(self,minute):
        e=self.engine; spot_bar=e.spot_bars.get(minute); future_bar=e.future_bars.get(minute)
        if spot_bar is None or future_bar is None: return
        nearest=min(range(len(e.strikes)),key=lambda i:abs(e.strikes[i]-spot_bar.close))
        if nearest==0 or nearest>=len(e.strikes)-1: return
        atm=e.strikes[nearest]; cs=e.strikes[nearest+1]; ps=e.strikes[nearest-1]
        if cs-atm!=100 or atm-ps!=100:
            self.submit({'kind':'error','message':'unexpected_strike_spacing'}); return
        cp=e.pairs[cs]['CE']; pp=e.pairs[ps]['PE']; ac=e.pairs[atm]['CE']; ap=e.pairs[atm]['PE']
        bars=[e.option_bars.get(token,{}).get(minute) for token in [cp,pp,ac,ap]]
        if any(b is None for b in bars):
            self.submit({'kind':'notice','second':int(e.latest_exchange_second),'reason':'missing_otm_minute_quote','minute':minute}); return
        ts=EPOCH+timedelta(minutes=minute,seconds=59)
        expiry=datetime.combine(e.expiry,datetime.min.time())+timedelta(hours=15,minutes=30)
        # Filter OI by information availability at this bar's end. Never use later OI for older bars.
        def oi(token):
            value=next((v for v in reversed(self.oi_history.get(token,())) if v[1]<=minute*60+59),None)
            end=minute*60+59
            return float(value[0]) if value and 0<=end-value[1]<=900 else math.nan
        row=dict(signal_time=ts.isoformat(sep=' '),available_at=(ts+timedelta(seconds=1)).isoformat(sep=' '),
            expiry=str(e.expiry),days_to_expiry=(expiry-ts).total_seconds()/86400,spot=float(spot_bar.close),future=float(future_bar.open),
            atm=float(atm),step=100.,call_strike=float(cs),put_strike=float(ps),call=float(bars[0].close),put=float(bars[1].close),
            atm_call=float(bars[2].close),atm_put=float(bars[3].close),call_oi=oi(cp),put_oi=oi(pp),atm_call_oi=oi(ac),atm_put_oi=oi(ap),
            call_key=str(cp),put_key=str(pp),quantity=int(e.lot_size)*int(self.config['lots_per_position']))
        # Latest quote snapshot initializes marks; inference still requires post-decision fills.
        quotes=[[str(token),float(v[0]),int(v[1])] for token,v in e.latest_option_ticks.items()]
        self.submit({'kind':'snapshot','row':row,'second':int(e.latest_exchange_second),'quotes':quotes})

    def atm_row(self,row,now):
        self.submit({'kind':'atm_row','row':row,'second':int(now)})

    def _status(self,state,error=None):
        value={'state':state,'error':error,'pid':self.process.pid,'queue_depth':self.queue.qsize(),
            'dropped_events':self.dropped,'updated_at':datetime.now().isoformat(sep=' ')}
        p=self.runtime/'bridge_status.json'; temp=p.with_suffix('.tmp')
        try: temp.write_text(json.dumps(value),encoding='utf-8'); os.replace(temp,p)
        except OSError: pass

    def _write(self):
        last=0; last_status=0; reported_gap=False
        try:
            while True:
                try: event=self.queue.get(timeout=.5)
                except queue.Empty:
                    if self.process.poll() is not None: raise RuntimeError('worker exited '+str(self.process.returncode))
                    self._status('FEED_GAP' if self.failed else 'RUNNING'); continue
                if event is None: break
                if self.failed and not reported_gap:
                    self.process.stdin.write(json.dumps({'kind':'error','message':'input_queue_overflow_new_entries_disabled'})+'\n'); reported_gap=True
                self.process.stdin.write(json.dumps(event,separators=(',',':'))+'\n')
                if time.monotonic()-last>=.1:
                    self.process.stdin.flush(); last=time.monotonic()
                if event['kind'] in ('snapshot','metadata','atm_row'): self.process.stdin.flush()
                if time.monotonic()-last_status>=1:
                    self._status('FEED_GAP' if self.failed else 'RUNNING'); last_status=time.monotonic()
        except Exception as exc:
            self.failed=True; LOG.exception('Isolated 43Weeks worker stopped'); self._status('ERROR',str(exc))
        finally:
            try: self.process.stdin.close()
            except OSError: pass

    def close(self):
        self.closed=True
        try: self.queue.put_nowait(None)
        except queue.Full: pass
        self.writer.join(timeout=3)
        try: self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # Only our own subprocess; never terminate/restart the existing trader.
            self.process.terminate(); self.process.wait(timeout=3)
        self._status('STOPPED')

def attach(engine):
    cfg=json.loads((ROOT/'config.json').read_text())
    return Bridge(engine) if cfg.get('enabled') else None

def handle_get(handler,parsed):
    from urllib.parse import parse_qs
    import re
    query=parse_qs(parsed.query)
    if parsed.path in ('/firstTwoWeeks','/firstTwoWeeks/'):
        handler._send((ROOT/'firstTwoWeeks.html').read_bytes(),'text/html; charset=utf-8'); return
    suffix=parsed.path.removeprefix('/firstTwoWeeks/api/')
    if suffix=='status': handler._json(status()); return
    if suffix=='backtests': handler._json(backtests()); return
    if suffix=='settings':
        try: handler._json(settings_status())
        except (OSError,ValueError) as exc: handler._json({'error':str(exc)},503)
        return
    if suffix=='models':
        p=ROOT/'otm_models/training_report.json'
        handler._json(json.loads(p.read_text()) if p.exists() else {'state':'not_trained'}); return
    kinds={'signals':'signals','scored':'scored','trades':'positions','events':'events','paths':'strike_paths','atm':'atm_outputs'}
    if suffix not in kinds: handler.send_error(404); return
    limit=min(1000,max(1,int(query.get('limit',['100'])[0])))
    offset=max(0,int(query.get('offset',['0'])[0]))
    run=query.get('run',[None])[0]
    if run and not re.fullmatch(r'(replay_\d{8}_\d{6}|initial_2024_test)',run):
        handler._json({'error':'invalid_run'},400); return
    handler._json(read_records(kinds[suffix],limit,run,offset,query.get('signal_id',[None])[0]))

def handle_post(handler,parsed):
    from urllib.parse import urlparse
    import ipaddress
    if parsed.path not in ('/firstTwoWeeks/api/backtest','/firstTwoWeeks/api/settings'): handler.send_error(404); return
    if not ipaddress.ip_address(handler.client_address[0]).is_loopback:
        handler._json({'error':'changes may only be made from this computer'},403); return
    origin=handler.headers.get('Origin')
    if origin and urlparse(origin).netloc!=handler.headers.get('Host'):
        handler._json({'error':'cross-origin requests are not allowed'},403); return
    if parsed.path.endswith('/settings'):
        try:
            length=int(handler.headers.get('Content-Length','0'))
            if not 0<length<=4096: raise ValueError('Provide a JSON settings object of at most 4096 bytes')
            if handler.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
                raise ValueError('Content-Type must be application/json')
            body=json.loads(handler.rfile.read(length))
            if not isinstance(body,dict) or set(body)!={'thresholds','expected_revision'}:
                raise ValueError('Provide thresholds and expected_revision only; position cap is fixed at 3')
            saved=save_settings(body['thresholds'],body['expected_revision'],ROOT)
            handler._json({'saved':saved,'message':'Saved. The running worker reloads within about one second; open trades keep their exit rules.'})
        except SettingsConflict as exc: handler._json({'error':str(exc)},409)
        except (ValueError,UnicodeError) as exc: handler._json({'error':str(exc)},400)
        except OSError as exc: handler._json({'error':'Settings could not be saved: '+str(exc)},503)
        return
    try: handler._json(launch_backtest(),202)
    except (OSError,ValueError) as exc: handler._json({'error':str(exc)},503)
