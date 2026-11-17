"""Real artifacts and real worker processes; isolated files, no market sockets."""
from pathlib import Path
import csv
import gzip
import io
import sys
import time
from voltr.service.fanout import FeedChannel,Fanout
from voltr.core.io import atomic_write_json as atomic_json,read_json
from voltr.instruments.testing import contract_gzip

def test_all_real_workers_start_and_survivors_receive_after_one_crashes(tmp_path):
    repo=Path(__file__).resolve().parents[1];package=repo/'pipelines';root=package/'ivrv'
    config_path=repo/'config/live_config.json'
    # Build a no-feed contract fixture without depending on an archived master.
    fixture=tmp_path/'NSE_FO_contract_fixture.csv.gz'
    rows=list(csv.reader(io.StringIO(gzip.decompress(contract_gzip()).decode('utf-8'))))
    for i,strike in enumerate((49800,49900,50100,50200)):
        for side in ('CE','PE'):
            rows.append([400+i*2+(side=='PE'),26009,'OPTIDX','BANKNIFTY','2026-10-29',
                         strike*100,side,30,'N'])
    data=io.StringIO();csv.writer(data,lineterminator='\n').writerows(rows)
    fixture.write_bytes(gzip.compress(data.getvalue().encode('utf-8')))
    for sub in ('pcr','weeks43'):(tmp_path/sub).mkdir()
    for sub in ('pcr','weeks43'):(tmp_path/sub/'config.json').write_bytes((package/sub/'config.json').read_bytes())
    (tmp_path/'weeks43/entry_settings.json').write_bytes((package/'weeks43/entry_settings.json').read_bytes())
    bootstrap=tmp_path/'isolated_worker_fixture.py'
    bootstrap.write_text(f'''
import sys
from pathlib import Path
sys.path.insert(0,{str(repo)!r})
from pipelines.ivrv import live as legacy
from voltr.service import worker
from voltr.core.io import file_lock as real_lock
audit=Path({str(tmp_path)!r});production=Path({str(package)!r})
load=legacy.load_live_config
def isolated_config(path):
    cfg=load(path)
    for key,filename in [('prediction_log','predictions.jsonl'),('fixed_strike_path_log','paths.jsonl'),('latest_prediction','latest.json'),('paper_trade_events_log','trades.jsonl'),('paper_position_state','positions.json')]:
        cfg[key]=str(audit/'ivrv'/filename)
    return cfg
legacy.load_live_config=isolated_config
import pipelines.weeks43.pipeline as otm_pipeline
otm_pipeline.ROOT=audit/'weeks43'
worker_name=sys.argv[sys.argv.index('--name')+1]
worker.single_writer=lambda path:real_lock(audit/(worker_name+'-worker.lock'))
import pipelines.pcr.bridge as pcr_bridge
pcr_bridge.ROOT=audit/'pcr'
import pipelines.weeks43.engine as otm
RealRunner=otm.ModelRunner
otm.ModelRunner=lambda ignored:RealRunner(production/'weeks43')
import pipelines.weeks43.atm_diagnostics as atm
RealAtm=atm.AtmDiagnostics
class IsolatedAtm(RealAtm):
    def __init__(self,folder):
        super().__init__(production/'weeks43');self.root=folder
atm.AtmDiagnostics=IsolatedAtm
(audit/'weeks43/runtime').mkdir(exist_ok=True)
worker.main()
''')
    context=dict(config_path=str(config_path),contract_file=str(fixture),session_date='2026-10-28',session_id='isolated-artifact-test',fixture_only=True)
    atomic_json(tmp_path/'context.json',context)
    channels={};fanout=Fanout()
    def status(name):return read_json(tmp_path/(name+'.json'),{})
    try:
        for name in ('ivrv','pcr','43weeks'):
            channels[name]=FeedChannel(name,[sys.executable,'-u',str(bootstrap),'--context',str(tmp_path/'context.json'),
                '--name',name,'--status',str(tmp_path/(name+'.json'))],repo,tmp_path/(name+'.log'),capacity=1024)
        fanout.channels=channels
        deadline=time.monotonic()+55
        while time.monotonic()<deadline:
            for name,ch in channels.items():assert ch.process.poll() is None,(name,(tmp_path/(name+'.log')).read_text()[-4000:])
            if all(status(n).get('state') in ('READY','RUNNING') for n in channels):break
            time.sleep(.2)
        assert all(status(n).get('state') in ('READY','RUNNING') for n in channels),{n:status(n) for n in channels}
        for _ in range(200):fanout.publish(b'F',b'\0\0\0\0') # Valid empty sequential packet, no fabricated prices.
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and not all(status(n).get('processed')==200 for n in channels):time.sleep(.2)
        assert all(status(n).get('processed')==200 for n in channels)
        channels['ivrv'].process.terminate();channels['ivrv'].process.wait(timeout=3)
        for _ in range(100):fanout.publish(b'C',b'\0\0\0\0')
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and not all(status(n).get('processed')==300 for n in ('pcr','43weeks')):time.sleep(.2)
        assert all(status(n).get('processed')==300 for n in ('pcr','43weeks'))
        assert all(channels[n].process.poll() is None for n in ('pcr','43weeks'))
        atomic_json(tmp_path/'verification.json',dict(real_workers_started=3,received_before_failure=200,
            failed_pipeline='ivrv',survivor_received=300,network_sockets_opened=False,fixture_only=True,
            statuses={n:status(n) for n in channels}))
    finally:
        for ch in channels.values():ch.close()
