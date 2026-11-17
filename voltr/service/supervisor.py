"""Daily supervisor and raw UDP receiver for independent paper pipelines (listed in config/service.json)."""
from datetime import datetime,timedelta,time as day_time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
import selectors
import sys
import threading
import time
import uuid
from voltr.core.io import atomic_write_json as atomic_json,file_lock as single_writer,read_json

from .fanout import Fanout,FeedChannel
from voltr.feed.journal import FeedJournal
from voltr.core.clock import MARKET_TIMEZONE as TZ

from . import registry

REPO=registry.REPO
RUNTIME=REPO/'runtime'/'service'  # replaced by service.json "runtime_dir" when the service starts
LOG=logging.getLogger('isolated_service')

def daily_window(now,start='09:00',end='15:40'):
    if now.tzinfo is None:raise ValueError('Timezone-aware time required')
    date=now.date();a=datetime.combine(date,day_time.fromisoformat(start),tzinfo=TZ)
    b=datetime.combine(date,day_time.fromisoformat(end),tzinfo=TZ)
    if b<=a:raise ValueError('Session end must follow start')
    if now>=b:a+=timedelta(days=1);b+=timedelta(days=1)
    return a,b

def prepare_contracts(config,config_path,start,folder):
    from voltr.instruments.download import ensure_latest_contract_file
    cfg=dict(config,contract_force_download=True,contract_download_enabled=True)
    result=ensure_latest_contract_file(cfg,asof=start.date())
    if not result.downloaded:raise RuntimeError('A newly downloaded contract file is required')
    context=dict(config_path=str(config_path),session_id=start.date().isoformat()+'_'+uuid.uuid4().hex[:12],
        session_date=start.date().isoformat(),contract_file=str(result.contract_file),
        contract_file_date=result.file_date.isoformat(),downloaded=True,
        downloaded_at=datetime.now(TZ).isoformat(),contract_sha256=hashlib.sha256(result.contract_file.read_bytes()).hexdigest(),
        manifest_file=str(result.manifest_file),start_at=start.isoformat())
    atomic_json(folder/'session_context.json',context)
    return context

class Supervisor:
    def __init__(self,config,config_path,start,stop,fanout):
        self.config=config;self.config_path=config_path;self.start=start;self.end=stop;self.fanout=fanout
        names=tuple(config.get('pipelines',{}))
        self.enabled_pipelines=tuple(config.get('enabled_pipelines',names))
        self.folder=RUNTIME/'sessions'/start.date().isoformat();self.folder.mkdir(parents=True,exist_ok=True)
        self.stop_event=threading.Event();self.context=None;self.contract_error=None
        self.entries={n:dict(restarts=0,next_attempt=0,state='WAITING_FOR_CONTRACTS' if n in self.enabled_pipelines else 'DISABLED') for n in names}
        self.observed_workers={}
        self.cached_status=dict(pipelines={n:dict(v) for n,v in self.entries.items()},contracts=None,contract_error=None)
        self.last_receive=None;self.last_state_minute=None
        self.thread=threading.Thread(target=self._monitor,daemon=True,name='pipeline-supervisor')
        self.preparer=threading.Thread(target=self._prepare,daemon=True,name='contract-download')
    def begin(self):self.preparer.start();self.thread.start()
    def _prepare(self):
        load_live_config=registry.host(self.config).load_live_config
        while not self.stop_event.is_set():
            try:
                self.context=prepare_contracts(load_live_config(self.config_path),self.config_path,self.start,self.folder)
                self.contract_error=None;return
            except Exception as exc:
                self.contract_error=str(exc);LOG.error('Contract preparation: %s',exc)
                self.stop_event.wait(float(self.config.get('contract_retry_seconds',60)))
    def _event(self,name,event,**details):
        with (self.folder/'supervisor_events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(time=datetime.now(TZ).isoformat(),pipeline=name,event=event,**details))+'\n')
    def _launch(self,name,entry):
        if self.stop_event.is_set():return
        context=dict(self.context)
        if (entry['restarts'] and self.fanout.journal is None or self.fanout.journal and self.fanout.journal.recovered) and time.time()>=self.start.timestamp():
            context['entry_fault']='RESTART_AFTER_FEED_INTERRUPTION'
        path=self.folder/(name+'_context.json');atomic_json(path,context)
        executable=str(Path(sys.executable).with_name('python.exe')) if os.name=='nt' else sys.executable
        command=[executable,'-u','-m','voltr.service.worker','--name',name,'--context',str(path),'--status',str(self.folder/(name+'_status.json'))]
        channel=FeedChannel(name,command,str(REPO),self.folder/(name+'.log'),
            int(self.config.get('pipeline_queue_capacity',4096)),bool(context.get('entry_fault')),start_paused=True,
            journal=self.fanout.journal)
        if self.stop_event.is_set():channel.close();return
        self.fanout.channels={**self.fanout.channels,name:channel}
        self.observed_workers.pop(name,None)
        entry.update(state='STARTING',last_error=None,pid=channel.process.pid)
        self._event(name,'WORKER_STARTED',pid=channel.process.pid,restarts=entry['restarts'])
    def _monitor(self):
        while not self.stop_event.wait(1):
            for name,entry in self.entries.items():
                if self.stop_event.is_set():break
                if name not in self.enabled_pipelines:continue
                try:
                    channel=self.fanout.channels.get(name)
                    if channel:
                        state=read_json(self.folder/(name+'_status.json'),{})
                        own=state.get('pid')==channel.process.pid and state.get('session_id')==self.context['session_id']
                        if own:self.observed_workers[name]=state
                        # A transient Windows file-sharing/read failure is not
                        # evidence that an already-running worker never started.
                        observed=self.observed_workers.get(name,{})
                        if observed.get('pid')!=channel.process.pid:observed={}
                        heartbeat=observed.get('heartbeat',channel.started)
                        timeout=float(self.config.get('pipeline_stall_seconds',60)) if observed else float(self.config.get('pipeline_startup_seconds',180))
                        if channel.process.poll() is not None or channel.transport_error or time.time()-heartbeat>timeout:
                            reason=observed.get('error')
                            reason=reason or channel.transport_error or ('WORKER_STALLED' if channel.process.poll() is None else 'WORKER_EXITED')
                            self._event(name,'WORKER_FAILED',reason=reason,transport=channel.status())
                            self.fanout.channels={k:v for k,v in self.fanout.channels.items() if k!=name}
                            channel.close();entry['restarts']+=1
                            entry.update(state='RETRY_WAIT' if entry['restarts']<=int(self.config.get('pipeline_restart_limit',3)) else 'FAILED',
                                last_error=reason,next_attempt=time.time()+min(15*entry['restarts'],60))
                        else:
                            entry['state']=observed.get('state','STARTING')
                            if observed.get('state') in ('READY','RUNNING','DEGRADED') and not channel.subscribed:
                                channel.activate();self._event(name,'FEED_SUBSCRIBED',pid=channel.process.pid)
                    elif self.context and entry['state']!='FAILED' and time.time()>=entry['next_attempt']:
                        self._launch(name,entry)
                except Exception as exc:
                    entry['restarts']+=1;entry.update(last_error=str(exc),state='RETRY_WAIT' if entry['restarts']<=int(self.config.get('pipeline_restart_limit',3)) else 'FAILED',next_attempt=time.time()+30)
                    LOG.exception('Only %s startup/monitor failed',name)
            try:
                snapshot=self.status();self.cached_status=snapshot;atomic_json(self.folder/'supervisor_status.json',snapshot)
                minute=int(time.time()//60)
                if minute!=self.last_state_minute:
                    self.last_state_minute=minute
                    history=self.folder/'state_history';history.mkdir(exist_ok=True)
                    for pipeline in self.enabled_pipelines:
                        state=read_json(self.folder/(pipeline+'_status.json'),{})
                        if self.context and state.get('session_id')==self.context.get('session_id'):
                            atomic_json(history/(f'{pipeline}_{minute:012d}.json'),state)
                    for old in history.glob('*.json'):
                        try:
                            if int(old.stem.rsplit('_',1)[-1])<minute-60:old.unlink()
                        except (ValueError,OSError):pass
                for name,path in [(n,registry.repo_path(p['status_file'])) for n,p in self.config.get('pipelines',{}).items() if p.get('status_file')]:
                    entry=snapshot['pipelines'][name]
                    atomic_json(path,dict(state=entry['state'],error=entry.get('last_error') or entry.get('fault'),pid=entry.get('pid'),
                        queue_depth=entry.get('queue_depth',0),dropped_events=entry.get('dropped',0),updated_at=datetime.now(TZ).isoformat(),
                        transport='independent_raw_udp_queue'))
            except Exception:LOG.exception('Status publication failed; feed receiver continues')
    def status(self):
        states={}
        for name,entry in self.entries.items():
            channel=self.fanout.channels.get(name);states[name]=dict(entry)
            if channel:states[name].update(channel.status())
            detail=read_json(self.folder/(name+'_status.json'),{})
            if channel and detail.get('pid')!=channel.process.pid:detail=self.observed_workers.get(name,{})
            if channel and detail.get('pid')==channel.process.pid:states[name]['worker']=detail
        return dict(session_start=self.start.isoformat(),session_end=self.end.isoformat(),contracts=self.context,
            contract_error=self.contract_error,pipelines=states)
    def close(self):
        self.stop_event.set();self.thread.join(timeout=10)
        channels=list(self.fanout.channels.values());self.fanout.channels={}
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda c:c.close(),channels))
        for value in self.entries.values():value['state']='STOPPED'

class DashboardView:
    def __init__(self,config,service):self.config=config;self.service=service
    def pipeline_status(self):return read_json(RUNTIME/'status.json',dict(state='STARTING'))
    def status(self):
        LogDashboardView=registry.host(self.config).LogDashboardView;host=self.config.get('dashboard_pipeline','ivrv')
        sup=self.service.get('supervisor')
        value=read_json(sup.folder/(host+'_status.json'),{}) if sup else {}
        channel=sup.fanout.channels.get(host) if sup else None
        current=bool(channel and sup.context and value.get('pid')==channel.process.pid and value.get('session_id')==sup.context['session_id'])
        if current:dashboard=value.get('dashboard',{})
        else:
            previous=LogDashboardView(self.config).status()
            dashboard=dict(state='WAITING',reason='waiting_for_current_worker',paper_trading=previous.get('paper_trading',{}),
                runtime=dict(mode='standby_dashboard',paper_only=True,required_feature_minutes=61))
        dashboard=dict(dashboard);dashboard['pipeline_health']=self.pipeline_status()
        return dashboard
    def recent(self,limit):
        sup=self.service.get('supervisor');host=self.config.get('dashboard_pipeline','ivrv')
        channel=sup.fanout.channels.get(host) if sup else None
        state=read_json(sup.folder/(host+'_status.json'),{}) if sup else {}
        if not channel or not sup.context or state.get('pid')!=channel.process.pid or state.get('session_id')!=sup.context['session_id']:return []
        return read_json(sup.folder/(host+'_recent.json'),[])[-limit:]

def run_service(config,config_path,run_now=False):
    global RUNTIME
    host=registry.host(config)
    DashboardServer,archive_previous_day_logs,configure_logging=host.DashboardServer,host.archive_previous_day_logs,host.configure_logging
    RUNTIME=registry.repo_path(config.get('runtime_dir','runtime/service'))
    from voltr.feed.udp import NseFoUdpReceiver
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with single_writer(RUNTIME/'service.lock'):
        configure_logging(RUNTIME/'service.log',str(config.get('log_level','INFO')))
        publisher_stop=threading.Event();publisher=threading.Thread(target=_publish_status,args=(publisher_stop,config_path),name='service-status-writer',daemon=True);publisher.start()
        service={};view=DashboardView(config,service)
        dashboard=DashboardServer(str(config['dashboard_host']),int(config['dashboard_port']),host.DASHBOARD_HTML,view)
        dashboard.start();supervisor=None;selector=None;receivers={};fanout=Fanout();socket_errors={};last_publish=0
        from .redirect import DashboardRedirect
        aliases=[]
        for port in config.get('dashboard_alias_ports',[8782]):
            try:
                alias=DashboardRedirect(port,int(config['dashboard_port']));alias.start();aliases.append(alias)
            except (OSError,ValueError):LOG.exception('Dashboard bookmark port %s unavailable; live service continues',port)
        try:
            while True:
                now=datetime.now(TZ);start,end=daily_window(now,config.get('session_start','09:00'),config.get('session_end','15:40'))
                if run_now and supervisor is None:start=now;end=now+timedelta(hours=7)
                if supervisor and now>=supervisor.end:
                    if selector:selector.close();selector=None
                    for receiver in receivers.values():receiver.close()
                    receivers={};supervisor.close();supervisor=None;service.clear()
                    if fanout.journal:fanout.journal.close()
                    if run_now:break
                if supervisor is None and now>=start-timedelta(seconds=int(config.get('schedule_prepare_seconds',300))):
                    archive_previous_day_logs(config,start.date())
                    # A service restart begins a new live session. Reusing the
                    # day's old journal would replay millions of stale packets
                    # before this worker can reach the current market feed.
                    journal=FeedJournal(RUNTIME/'sessions'/start.date().isoformat()/'feed_journal_runs'/uuid.uuid4().hex,
                        config.get('feed_journal_retention_minutes',65))
                    fanout=Fanout(journal);supervisor=Supervisor(config,config_path,start,end,fanout);service['supervisor']=supervisor;supervisor.begin()
                if supervisor and now>=supervisor.start:
                    if selector is None:selector=selectors.DefaultSelector()
                    for kind in (b'F',b'C'):
                        if kind in receivers:continue
                        previous=socket_errors.get(kind,{})
                        if time.time()<previous.get('retry_at',0):continue
                        cfg=dict(config)
                        if kind==b'C':cfg.update(multicast_ip=config['cm_multicast_ip'],port=int(config['cm_port']),interface_ip=config.get('cm_interface_ip',config['interface_ip']))
                        receiver=NseFoUdpReceiver(cfg,None)
                        try:
                            receiver.open();receiver.socket.setblocking(False);selector.register(receiver.socket,selectors.EVENT_READ,kind)
                            receivers[kind]=receiver;socket_errors.pop(kind,None)
                        except OSError as exc:
                            receiver.close();socket_errors[kind]=dict(error=str(exc),retry_at=time.time()+15)
                if selector and receivers:
                    for key,_ in selector.select(timeout=.05):
                        try:
                            data,_=key.fileobj.recvfrom(int(config['max_datagram_bytes']))
                            fanout.publish(key.data,data)
                        except BlockingIOError:pass
                        except OSError as exc:
                            kind=key.data;selector.unregister(key.fileobj);receivers.pop(kind).close()
                            socket_errors[kind]=dict(error=str(exc),retry_at=time.time()+15)
                else:time.sleep(.1)
                if time.monotonic()-last_publish>=1:
                    state=dict(supervisor.cached_status) if supervisor else dict(
                        session_start=start.isoformat(),session_end=end.isoformat(),
                        pipelines={n:dict(state='WAITING_FOR_SESSION' if n in config.get('enabled_pipelines',tuple(config.get('pipelines',{}))) else 'DISABLED') for n in config.get('pipelines',{})})
                    state.update(state='SESSION' if supervisor and now>=supervisor.start else 'PREPARING' if supervisor else 'WAITING_FOR_SESSION',
                        updated_at=now.isoformat(),service_pid=os.getpid(),timezone='Asia/Kolkata',daily_including_weekends=True,
                        preparation_seconds=int(config.get('schedule_prepare_seconds',300)),feed=dict(datagrams=fanout.datagrams,bytes=fanout.bytes,
                            last_receive=fanout.last_receive,open_streams=[k.decode() for k in receivers],errors={k.decode():v for k,v in socket_errors.items()}))
                    # Disk work is bounded and off the high-rate dispatch path; status
                    # publication itself runs on a separate daemon below.
                    pending_status[0]=state;last_publish=time.monotonic()
        except KeyboardInterrupt:LOG.info('Service stopped by user')
        finally:
            if selector:selector.close()
            for receiver in receivers.values():receiver.close()
            if supervisor:supervisor.close()
            if fanout.journal:fanout.journal.close()
            dashboard.close()
            for alias in aliases:alias.close()
            publisher_stop.set();publisher.join(timeout=2)
            atomic_json(RUNTIME/'status.json',dict(state='STOPPED',updated_at=datetime.now(TZ).isoformat(),service_pid=os.getpid()))
    return 0

# A size-one latest-value mailbox prevents dashboard writes from blocking UDP.
pending_status=[None]
def _publish_status(stop,config_path):
    readiness={};checked=0
    while not stop.is_set():
        state=pending_status[0]
        if state is not None:
            try:
                if time.monotonic()-checked>=30:
                    load_live_config=registry.host(registry.service_settings(config_path)).load_live_config
                    from voltr.instruments.download import _auth_token
                    try:
                        _auth_token(load_live_config(config_path))
                        readiness=dict(contract_auth_ready=True)
                    except Exception as exc:readiness=dict(contract_auth_ready=False,contract_auth_error=str(exc))
                    checked=time.monotonic()
                value=dict(state,readiness=readiness)
                if not readiness.get('contract_auth_ready'):value['contract_error']=value.get('contract_error') or readiness.get('contract_auth_error')
                atomic_json(RUNTIME/'status.json',value)
            except Exception:LOG.exception('Service status write failed')
        stop.wait(1)
