"""Private worker started/stopped with the existing daily session."""
import json
import sys
import queue
import threading
import traceback
import time
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from weeks43_engine import ROOT, Ledger, ModelRunner, PaperEngine, second, stamp
from weeks43_settings import read_settings

def refresh_entry_settings(engine,now):
    try:
        engine.apply_settings(read_settings(ROOT),now)
    except (OSError,ValueError) as exc:
        message=str(exc)
        if engine.settings_error!=message:
            engine.ledger.put('events','settings_error:'+str(time.time_ns()),
                dict(time=stamp(now),event='ENTRY_SETTINGS_ERROR',reason=message))
        engine.settings_error=message
        engine._recheck_pending(now)
        engine.persist_status(now)

def main():
    import msvcrt
    runtime=ROOT/'runtime'; runtime.mkdir(exist_ok=True)
    lock=(runtime/'worker.lock').open('a+b'); lock.seek(0)
    if lock.read(1)==b'': lock.write(b'0'); lock.flush()
    lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        raise RuntimeError('Another 43Weeks worker owns the paper ledger')
    cfg=json.loads((ROOT/'config.json').read_text())
    runner=ModelRunner(); engine=PaperEngine(cfg,Ledger(ROOT/'logs'),runner,live=True)
    try:
        from weeks43_atm_diagnostics import AtmDiagnostics
        diagnostics=AtmDiagnostics(ROOT) if cfg.get('show_atm_diagnostics') else None
    except Exception as exc:
        print('ATM diagnostics unavailable:',exc,flush=True); diagnostics=None
    incoming=queue.Queue(maxsize=100000)
    def reader():
        for line in sys.stdin:
            try: incoming.put(json.loads(line))
            except ValueError: continue
        incoming.put(None)
    thread=threading.Thread(target=reader,daemon=True); thread.start()
    print('43Weeks paper worker ready',flush=True)
    last_settings=0
    try:
        while True:
            wall=second(datetime.now(ZoneInfo('Asia/Kolkata')).replace(tzinfo=None))
            if time.monotonic()-last_settings>=1:
                refresh_entry_settings(engine,wall); last_settings=time.monotonic()
            try: event=incoming.get(timeout=.5)
            except queue.Empty:
                now=second(datetime.now(ZoneInfo('Asia/Kolkata')).replace(tzinfo=None))
                engine.on_quotes([],now)
                continue
            if event is None: break
            kind=event['kind']
            try:
                if kind=='quote': engine.on_quotes([(event['key'],event['price'],event['second'])],event['second'])
                elif kind=='snapshot':
                    now=event['second']
                    wall=second(datetime.now(ZoneInfo('Asia/Kolkata')).replace(tzinfo=None))
                    refresh_entry_settings(engine,wall)
                    if wall-now>cfg['max_signal_age_seconds']:
                        engine.error='feed_behind_wall_clock_new_entries_disabled'
                    engine.on_quotes(event['quotes'],now)
                    engine.on_snapshot(event['row'],now)
                elif kind=='atm_row' and diagnostics:
                    result=diagnostics.score(event['row'])
                    engine.ledger.put('atm_outputs',event['row']['datetime'],dict(time=stamp(event['second']),outputs=result,controls_entries=False))
                elif kind=='error':
                    engine.error=event['message']; engine.pending.clear()
                    engine.ledger.put('events','error:'+str(time.time_ns()),dict(time=datetime.now().isoformat(sep=' '),event='ERROR',reason=engine.error))
                elif kind=='notice':
                    engine.ledger.put('events','notice:'+str(event.get('minute')),dict(time=stamp(event['second']),event='NO_SIGNAL',reason=event['reason']))
            except Exception as exc:
                engine.error=str(exc); engine.pending.clear(); traceback.print_exc()
                engine.ledger.put('events','exception:'+str(time.time_ns()),dict(time=datetime.now().isoformat(sep=' '),event='ERROR',reason=str(exc)))
    finally:
        engine.close(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()

if __name__=='__main__': main()
