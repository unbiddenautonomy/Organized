"""Completion checks using isolated ledgers; never start the original live trader."""
from datetime import datetime,timedelta,date
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time
import numpy as np
import pandas as pd
from pipelines.weeks43.tests.test_weeks43 import row,config,Runner
from pipelines.weeks43.engine import ROOT
from pipelines.weeks43.settings import read_settings,save_settings

def test_no_trade_replay_still_exports_every_eligible_day(tmp_path,config,monkeypatch):
    from pipelines.weeks43 import backtest as replay
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'otm_models').mkdir()
    selection=dict(Runner.selection,thresholds={'3_week':1.,'4_week':1.})
    (tmp_path/'otm_models/selection.json').write_text(json.dumps(selection))
    snapshots=pd.DataFrame([row(datetime(2026,10,day,10,15,59)-timedelta(minutes=i))
        for day in (29,30) for i in range(60,-1,-1)])
    chain=pd.DataFrame({'time':pd.to_datetime([]),'trade_date':pd.to_datetime([])})
    class Models:
        def __init__(self): self.selection=selection
        def batch(self,frame):
            return {name:np.full(len(frame),value) for name,value in dict(logistic=.9,extra_trees=.8,
                lightgbm=.7,ridge=20.,lightgbm_huber=19.,otm_loss_q90=5.).items()}
    monkeypatch.setattr(replay,'ROOT',tmp_path);monkeypatch.setattr(replay,'ModelRunner',Models)
    monkeypatch.setattr(replay,'read_frame',lambda path:snapshots.copy() if path.name=='snapshots.parquet' else chain.copy())
    report=replay.replay('replay_20261028_000000')
    out=tmp_path/'backtests/replay_20261028_000000'
    assert report['summary']['trades']==0 and report['summary']['no_trade_days']==2
    assert report['summary']['gross_pnl']==report['summary']['total_costs']==report['summary']['net_pnl']==0
    assert report['summary']['win_rate'] is None
    daily=pd.read_csv(out/'daily_pnl.csv');trades=pd.read_csv(out/'trades.csv')
    assert len(daily)==2 and daily.day_color.tolist()==['NO_TRADE','NO_TRADE']
    assert daily.net_pnl.sum()==0 and trades.empty and 'entry_threshold' in trades
    signals=pd.read_csv(out/'all_model_outputs.csv')
    assert len(signals)==122 and set(signals.threshold)=={1.}
    assert report['entry_settings']['thresholds']==selection['thresholds']
    assert json.loads((tmp_path/'runtime/backtest_status.json').read_text())['state']=='COMPLETE'

def test_real_worker_start_reload_stop_and_restart_are_isolated(tmp_path,config,monkeypatch):
    from pipelines.weeks43 import bridge
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'otm_models').mkdir()
    (tmp_path/'otm_models/selection.json').write_bytes((ROOT/'otm_models/selection.json').read_bytes())
    (tmp_path/'entry_settings.json').write_bytes((ROOT/'entry_settings.json').read_bytes())
    # Use the production bridge and worker, redirecting only their writable root.
    # Models load from the installed package; no synthetic events enter its live ledger.
    bootstrap=f'''
import sys
from pathlib import Path
sys.path.insert(0,{str(ROOT.parents[1])!r})
from pipelines.weeks43 import worker
from pipelines.weeks43 import atm_diagnostics as diagnostics
class IsolatedDiagnostics(diagnostics.AtmDiagnostics):
    def __init__(self,ignored):
        super().__init__(Path({str(ROOT)!r}))
        self.root=Path({str(tmp_path)!r})
diagnostics.AtmDiagnostics=IsolatedDiagnostics
worker.ROOT=Path({str(tmp_path)!r})
worker.main()
'''
    (tmp_path/'isolated_worker.py').write_text(bootstrap)
    monkeypatch.setattr(bridge,'worker_command',lambda cfg:[sys.executable,'-u',str(tmp_path/'isolated_worker.py')])
    monkeypatch.setattr(bridge,'ROOT',tmp_path)
    fake=SimpleNamespace(lot_size=30,expiry=date(2026,11,24))
    def wait_for_policy(process,revision,after_time=''):
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            assert process.poll() is None, (tmp_path/'runtime/worker.log').read_text()
            status=bridge.read_records('status',1)
            if status and status[0].get('entry_settings',{}).get('revision')==revision and status[0]['time']>after_time:
                assert status[0]['config']['max_open_positions']==3
                assert status[0]['open_count']==0 and not status[0]['pending_entries']
                return
            time.sleep(.15)
        raise AssertionError('Worker did not apply policy: '+(tmp_path/'runtime/worker.log').read_text())
    first=bridge.attach(fake)
    try:
        initial=read_settings(tmp_path);wait_for_policy(first.process,initial['revision'])
        changed=save_settings({'3_week':.71,'4_week':.76},initial['revision'],tmp_path)
        wait_for_policy(first.process,changed['revision'])
    finally: first.close()
    assert first.process.returncode==0 and not first.writer.is_alive()
    previous_status_time=bridge.read_records('status',1)[0]['time']
    second=bridge.attach(fake)
    try: wait_for_policy(second.process,changed['revision'],previous_status_time)
    finally: second.close()
    assert second.process.returncode==0 and not second.writer.is_alive()
    assert not bridge.read_records('positions') and not bridge.read_records('signals')
