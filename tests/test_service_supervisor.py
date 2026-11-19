from datetime import datetime,date
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import io
import json
import queue
import sys
import threading
import time
import pytest
from voltr.service.fanout import FeedChannel,Fanout,HEADER,read_frame


def test_feed_workers_are_not_started_below_normal_priority(monkeypatch,tmp_path):
    captured={}
    class Process:
        stdin=None
    def popen(*args,**kwargs):
        captured.update(kwargs)
        raise RuntimeError('stop after capturing process options')
    monkeypatch.setattr('voltr.service.fanout.subprocess.Popen',popen)
    monkeypatch.setattr('voltr.service.fanout.subprocess.CREATE_NO_WINDOW',0x08000000,raising=False)
    monkeypatch.setattr('voltr.service.fanout.subprocess.ABOVE_NORMAL_PRIORITY_CLASS',0x00008000,raising=False)
    monkeypatch.setattr('voltr.service.fanout.subprocess.BELOW_NORMAL_PRIORITY_CLASS',0x00004000,raising=False)
    try:
        FeedChannel('priority',['worker'],tmp_path,tmp_path/'worker.log')
    except RuntimeError as exc:
        assert str(exc)=='stop after capturing process options'
    flags=captured['creationflags']
    assert flags & 0x00008000
    assert not flags & 0x00004000
from voltr.service.supervisor import daily_window,prepare_contracts,Supervisor
from voltr.service import registry

# The shipped pipeline registry (config/service.json), as the live service sees it.
SERVICE=registry.service_settings(registry.REPO/'config'/'live_config.json')

def test_frame_handles_partial_reads_and_rejects_truncation():
    payload=b'udp-packet';raw=HEADER.pack(b'F',19,123.,1,len(payload))+payload
    class Short(io.BytesIO):
        def read(self,n=-1):return super().read(min(n,3))
    assert read_frame(Short(raw))==(b'F',19,123.,True,payload)
    with pytest.raises(EOFError):read_frame(io.BytesIO(raw[:-2]))

def test_memory_feed_reports_overflow_once_then_resumes_clean_frames():
    class Pipe(io.BytesIO):
        def close(self):pass

    channel=FeedChannel.__new__(FeedChannel)
    channel.queue=queue.Queue();channel.journal=None;channel.closing=False
    channel.fault_lock=threading.Lock();channel.fault='FEED_QUEUE_OVERFLOW';channel.fault_generation=1
    channel.transport_error=None;channel.delivered=0;channel.last_sequence=0
    channel.process=SimpleNamespace(stdin=Pipe())
    channel.queue.put((b'F',1,100.,b'first'))
    channel.queue.put((b'F',2,101.,b'second'))
    channel.queue.put(None)
    channel._write()
    frames=io.BytesIO(channel.process.stdin.getvalue())
    assert read_frame(frames)==(b'F',1,100.,True,b'first')
    assert read_frame(frames)==(b'F',2,101.,False,b'second')
    assert channel.fault is None

def test_daily_9am_includes_weekends_and_rolls_after_close():
    tz=ZoneInfo('Asia/Kolkata')
    a,b=daily_window(datetime(2026,10,31,8,55,tzinfo=tz))
    assert a.isoformat()=='2026-10-31T09:00:00+05:30' and b.hour==15 and b.minute==40
    a,_=daily_window(datetime(2026,10,31,15,40,tzinfo=tz))
    assert a.date()==date(2026,11,1)

def test_daily_contract_prep_forces_download_and_excludes_secrets(tmp_path,monkeypatch):
    import voltr.instruments.download as bootstrap
    contract=tmp_path/'master.csv.gz';contract.write_bytes(b'validated-source-fixture')
    calls=[]
    def obtain(config,asof):
        calls.append((config,asof))
        return SimpleNamespace(downloaded=True,contract_file=contract,file_date=date(2026,10,28),manifest_file=tmp_path/'tokens.json')
    monkeypatch.setattr(bootstrap,'ensure_latest_contract_file',obtain)
    now=datetime(2026,10,29,9,tzinfo=ZoneInfo('Asia/Kolkata'))
    cfg=dict(contract_force_download=False,contract_download_enabled=False,unrelated_secret='never-serialize')
    result=prepare_contracts(cfg,tmp_path/'config.json',now,tmp_path)
    assert calls[0][0]['contract_force_download'] and calls[0][0]['contract_download_enabled']
    assert calls[0][1]==date(2026,10,29) and result['downloaded']
    assert 'never-serialize' not in (tmp_path/'session_context.json').read_text()
    assert cfg['contract_force_download'] is False

def test_real_subprocess_backpressure_and_crash_do_not_block_healthy_workers(tmp_path):
    root=Path(__file__).resolve().parents[1]  # repository root, so child scripts can import voltr
    script=tmp_path/'consumer.py'
    script.write_text('''import sys,time,json,os
from pathlib import Path
sys.path.insert(0,sys.argv[3])
from voltr.service.fanout import read_frame
from voltr.core.io import atomic_write_json as atomic_json
mode=sys.argv[1];path=Path(sys.argv[2]);count=0;sequences=[]
while True:
    item=read_frame(sys.stdin.buffer)
    if item is None:break
    if item[0]==b'H':continue
    if mode=='fail':os._exit(17)
    if mode=='slow':time.sleep(30)
    count+=1;sequences.append(item[1])
    if count%25==0:atomic_json(path,dict(count=count,sequences=sequences))
''')
    channels={}
    try:
        for name in ('slow','healthy1','healthy2'):
            channels[name]=FeedChannel(name,[sys.executable,'-u',str(script),'slow' if name=='slow' else 'healthy',
                str(tmp_path/(name+'.json')),str(root)],root,tmp_path/(name+'.log'),capacity=8 if name=='slow' else 2048)
        f=Fanout();f.channels=channels
        started=time.monotonic()
        for i in range(600):f.publish(b'F',b'x'*8192)
        publish_seconds=time.monotonic()-started
        assert publish_seconds<2.,publish_seconds
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            values=[]
            for name in ('healthy1','healthy2'):
                path=tmp_path/(name+'.json')
                try:values.append(json.loads(path.read_text()) if path.exists() else {})
                except (OSError,ValueError):values.append({}) # Windows atomic replace can briefly deny readers.
            if all(v.get('count')==600 for v in values):break
            time.sleep(.1)
        assert all(v.get('sequences')==list(range(1,601)) for v in values)
        assert channels['slow'].dropped>0 and channels['healthy1'].dropped==channels['healthy2'].dropped==0
        channels['slow'].process.terminate();channels['slow'].process.wait(timeout=3)
        for i in range(25):f.publish(b'C',b'next')
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if all(json.loads((tmp_path/(n+'.json')).read_text()).get('count')==625 for n in ('healthy1','healthy2')):break
            time.sleep(.1)
        assert all(json.loads((tmp_path/(n+'.json')).read_text())['count']==625 for n in ('healthy1','healthy2'))
    finally:
        for c in channels.values():c.close()

def test_supervisor_starts_other_pipelines_when_one_startup_fails(tmp_path,monkeypatch):
    from voltr.service import supervisor as runtime
    monkeypatch.setattr(runtime,'RUNTIME',tmp_path)
    monkeypatch.setattr(runtime,'atomic_json',lambda *args:None)
    launched=[]
    class Channel:
        def __init__(self,name,*args,**kwargs):
            if name=='ivrv':raise RuntimeError('Deliberate IV/RV model startup failure')
            launched.append(name);self.process=SimpleNamespace(pid=123,poll=lambda:None)
        def status(self):return {'pid':123}
    monkeypatch.setattr(runtime,'FeedChannel',Channel)
    start=datetime(2026,10,29,9,tzinfo=ZoneInfo('Asia/Kolkata'))
    everything=dict(SERVICE,enabled_pipelines=list(SERVICE['pipelines']))
    supervisor=Supervisor(everything,tmp_path/'config.json',start,start.replace(hour=15,minute=40),Fanout())
    supervisor.context={'session_id':'test','contract_file':'fixture'}
    class Stop:
        count=0
        def wait(self,_):self.count+=1;return self.count>1
        def is_set(self):return False
    supervisor.stop_event=Stop();supervisor._monitor()
    assert launched==['pcr','43weeks']
    assert supervisor.entries['ivrv']['state']=='RETRY_WAIT'
    assert set(supervisor.fanout.channels)=={'pcr','43weeks'}

def test_log_archive_keeps_unresolved_positions(tmp_path,monkeypatch):
    from pipelines.ivrv import live as app
    import os
    from pipelines.ivrv import paths as ivrv_paths
    monkeypatch.setattr(ivrv_paths,'BASE_DIR',tmp_path)
    path=tmp_path/'logs/paper_position_state.json';path.parent.mkdir()
    path.write_text(json.dumps({'open_positions':[{'position_id':'retained'}]}))
    os.utime(path,(0,0))
    assert app.archive_previous_day_logs({},date(2026,10,29))==[] and path.exists()

def test_otm_bars_do_not_need_ivrv_model_or_feature_state(monkeypatch):
    from pipelines.weeks43.pipeline import OtmFeatureStream
    from pipelines.ivrv import live as legacy
    def cannot_load(*args):raise AssertionError('OTM must not load the IV/RV pipeline')
    monkeypatch.setattr(legacy,'FixedStrikeInferencePipeline',cannot_load)
    contracts={}
    for i,k in enumerate((49900.,50000.,50100.)):
        for side,offset in [('CE',1),('PE',2)]:
            token=i*2+offset;contracts[token]=SimpleNamespace(token=token,strike_price=k,option_type=side)
    source=dict(decode_banknifty_7207=lambda *args:None,implied_volatility=lambda *args:None,
        _greeks=lambda *args:None,decode_oi_message=lambda *args:[],OiDecodeError=ValueError)
    called=[];bridge=SimpleNamespace(snapshot=called.append,diagnostics=None)
    a=OtmFeatureStream({},contracts,30,date(2026,11,24),99,source,bridge)
    b=OtmFeatureStream({},contracts,30,date(2026,11,24),99,source,bridge)
    a.spot_bars[100]=SimpleNamespace(close=50000)
    a._finalize_minute(100)
    assert called==[100] and b.spot_bars=={} and not hasattr(a,'pipeline') and not hasattr(a,'paper_engine')

def test_service_sidecar_changes_only_runtime_options():
    from pipelines.ivrv.live import load_live_config
    path=Path(__file__).resolve().parents[1]/'config/live_config.json'
    cfg=load_live_config(path)
    assert cfg['isolate_pipelines'] and cfg['contract_force_download']
    assert cfg['enabled_pipelines']==['ivrv']
    assert cfg['session_start']=='09:00' and cfg['schedule_prepare_seconds']==300
    assert cfg['paper_max_open_positions']==3 and cfg['paper_entry_convergence_probability_threshold']==.70


def test_loading_worker_does_not_queue_or_fault_before_subscription(tmp_path):
    root=Path(__file__).resolve().parents[1]  # repository root, so child scripts can import voltr
    code='import sys; sys.stdin.buffer.read()'
    ch=FeedChannel('loading',[sys.executable,'-u','-c',code],root,tmp_path/'loading.log',capacity=4,start_paused=True)
    try:
        f=Fanout();f.channels={'loading':ch}
        for _ in range(10000):f.publish(b'F',b'x'*512)
        assert ch.pre_ready_packets==10000 and ch.accepted==ch.dropped==ch.queue.qsize()==0
        assert ch.fault is None and not ch.subscribed
        ch.activate();f.publish(b'F',b'first fresh frame')
        assert ch.subscribed and ch.accepted==1 and ch.dropped==0 and ch.fault is None
    finally:ch.close()


def test_old_dashboard_redirect_preserves_path_and_query():
    import urllib.request,urllib.error
    from voltr.service.redirect import DashboardRedirect
    redirect=DashboardRedirect(0,8780);redirect.start()
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args):return None
    client=urllib.request.build_opener(NoRedirect)
    try:
        for path in ('/','/firstTwoWeeks','/api/predictions?limit=10'):
            try:client.open(f'http://127.0.0.1:{redirect.server.server_port}'+path,timeout=3)
            except urllib.error.HTTPError as response:
                assert response.code==307
                assert response.headers['Location']=='http://127.0.0.1:8780'+path
                assert response.headers['Cache-Control']=='no-store'
            else:raise AssertionError('Old port must redirect, not serve a preview')
    finally:redirect.close()


def test_dashboard_does_not_present_previous_worker_as_current(tmp_path,monkeypatch):
    from voltr.service import supervisor as runtime
    from pipelines.ivrv import live as legacy
    from voltr.core.io import atomic_write_json as atomic_json
    monkeypatch.setattr(runtime,'RUNTIME',tmp_path)
    monkeypatch.setattr(legacy,'LogDashboardView',lambda config:SimpleNamespace(status=lambda:dict(state='READY',predictions={'old':1},paper_trading={'open_positions':[]})))
    atomic_json(tmp_path/'ivrv_status.json',dict(pid=123,session_id='old',dashboard=dict(state='READY',predictions={'old':1})))
    atomic_json(tmp_path/'ivrv_recent.json',[{'old':1}])
    supervisor=SimpleNamespace(folder=tmp_path,context={'session_id':'new'},fanout=SimpleNamespace(channels={'ivrv':SimpleNamespace(process=SimpleNamespace(pid=456))}))
    view=runtime.DashboardView(dict(SERVICE),dict(supervisor=supervisor))
    assert view.status()['state']=='WAITING' and 'predictions' not in view.status()
    assert view.recent(20)==[]


def test_transient_missing_status_does_not_restart_running_worker(tmp_path,monkeypatch):
    from voltr.service import supervisor as runtime
    monkeypatch.setattr(runtime,'RUNTIME',tmp_path)
    monkeypatch.setattr(runtime,'atomic_json',lambda *args:None)
    start=datetime(2026,10,29,9,tzinfo=ZoneInfo('Asia/Kolkata'))
    fanout=Fanout();s=Supervisor(dict(SERVICE),tmp_path/'config.json',start,start.replace(hour=15),fanout)
    s.context={'session_id':'current'}
    class Channel:
        process=SimpleNamespace(pid=123,poll=lambda:None)
        started=time.time()-200
        transport_error=None
        subscribed=True
        def status(self):return dict(pid=123)
        def close(self):raise AssertionError('A fresh last-known heartbeat must survive a transient read failure')
    fanout.channels={'ivrv':Channel()}
    s.observed_workers['ivrv']=dict(pid=123,session_id='current',heartbeat=time.time(),state='RUNNING')
    s.entries['pcr']['state']=s.entries['43weeks']['state']='FAILED'
    monkeypatch.setattr(runtime,'read_json',lambda *args:{})
    class Stop:
        count=0
        def wait(self,_):self.count+=1;return self.count>1
        def is_set(self):return False
    s.stop_event=Stop();s._monitor()
    assert s.entries['ivrv']['state']=='RUNNING' and s.entries['ivrv']['restarts']==0
    assert s.status()['pipelines']['ivrv']['worker']['heartbeat']>time.time()-5
