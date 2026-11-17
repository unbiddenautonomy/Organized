"""Behavior tests for the operator cutoff and the shared three-position book."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
import io
import json
import pytest
from pipelines.weeks43.tests.test_weeks43 import config, warmed, row, enter, Runner
from pipelines.weeks43.engine import Ledger, PaperEngine, second
from pipelines.weeks43.settings import defaults, read_settings, save_settings, SettingsConflict, write_snapshot

def policy(cutoff3=.8,cutoff4=.85,revision='test-policy'):
    return dict(defaults(Runner.selection),thresholds={'3_week':cutoff3,'4_week':cutoff4},revision=revision,source='test')

def settings_root(tmp_path):
    (tmp_path/'otm_models').mkdir()
    (tmp_path/'otm_models/selection.json').write_text(json.dumps(Runner.selection))
    return tmp_path

@pytest.mark.parametrize('configured_cap',[1,3,7])
def test_three_open_positions_hard_cap_and_reuse_after_exit(tmp_path,config,configured_cap):
    config['max_open_positions']=configured_cap
    e,t=warmed(tmp_path,config)
    try:
        for i in range(4): enter(e,t+timedelta(minutes=i))
        assert len(e.positions)==3 and not e.pending
        assert e.ledger.all('signals')[-1]['reason']=='three_position_limit'
        now=second(t)+5*60
        e.on_quotes([('fixed_call',420,now),('fixed_put',400,now)],now)
        assert not e.positions
        enter(e,t+timedelta(minutes=6))
        assert len(e.positions)==1
        assert e.ledger.get('status','latest')['capacity']==dict(max_positions=3,open=1,pending=0,available=2)
    finally: e.close()

def test_pending_reservations_prevent_fourth_signal(tmp_path,config):
    e,t=warmed(tmp_path,config)
    try:
        enter(e,t);enter(e,t+timedelta(minutes=1))
        for i in (2,3):
            r=row(t+timedelta(minutes=i));e.on_snapshot(r,second(r['available_at']))
        assert len(e.pending)==1 and len(e.positions)==2
        assert e.last_signal['reason']=='three_position_limit'
        now=second(t)+181;e.on_quotes([('fixed_call',460,now),('fixed_put',440,now)],now)
        assert len(e.positions)==3 and not e.pending
    finally: e.close()

@pytest.mark.parametrize('probability,expected',[(.799,False),(.8,True),(.801,True)])
def test_cutoff_below_equal_above_and_audit(tmp_path,config,probability,expected):
    e,t=warmed(tmp_path,config)
    try:
        e.apply_settings(policy(),second(t))
        e.on_snapshot(row(t),second(t)+1,precomputed=dict(logistic=probability,ridge=20.,otm_loss_q90=5.))
        assert bool(e.pending)==expected
        signal=e.last_signal
        assert signal['probability_passed']==expected and signal['threshold']==.8
        assert signal['win_probability']==probability and signal['settings_revision']=='test-policy'
        if expected:
            now=second(t)+61;e.on_quotes([('fixed_call',460,now),('fixed_put',440,now)],now)
            p=next(iter(e.positions.values()))
            assert p['signal_threshold']==p['entry_threshold']==.8
            assert p['signal_settings_revision']==p['entry_settings_revision']=='test-policy'
            assert p['win_probability']==probability
        else: assert signal['reason']=='win_probability_below_threshold'
    finally: e.close()

@pytest.mark.parametrize('probability',[float('nan'),float('inf'),-.01,1.01,None])
def test_invalid_probability_never_enters(tmp_path,config,probability):
    e,t=warmed(tmp_path,config)
    try:
        e.on_snapshot(row(t),second(t)+1,precomputed=dict(logistic=probability,ridge=20.,otm_loss_q90=5.))
        assert not e.pending and e.last_signal['reason']=='invalid_win_probability'
    finally: e.close()

def test_bucket_cutoffs_and_risk_checks_remain_independent(tmp_path,config):
    e,t=warmed(tmp_path,config)
    try:
        e.apply_settings(policy(.8,.95),second(t))
        r=row(t);r['days_to_expiry']=30
        e.on_snapshot(r,second(t)+1)
        assert e.last_signal['threshold']==.95 and not e.pending
        t+=timedelta(minutes=1)
        e.on_snapshot(row(t),second(t)+1,precomputed=dict(logistic=.9,ridge=0.,otm_loss_q90=5.))
        assert e.last_signal['probability_passed'] and e.last_signal['reason']=='expected_net_below_buffer'
        t+=timedelta(minutes=1)
        e.on_snapshot(row(t),second(t)+1,precomputed=dict(logistic=.9,ridge=20.,otm_loss_q90=30.))
        assert e.last_signal['probability_passed'] and e.last_signal['reason']=='otm_loss_limit_exceeded'
    finally: e.close()

def test_raised_cutoff_cancels_pending_but_preserves_open_trade(tmp_path,config):
    e,t=warmed(tmp_path,config)
    try:
        enter(e,t); original=dict(next(iter(e.positions.values())))
        t+=timedelta(minutes=1);e.on_snapshot(row(t),second(t)+1)
        assert len(e.pending)==1
        e.apply_settings(policy(.95,.95),second(t)+2)
        assert not e.pending and next(iter(e.positions.values()))==original
        assert any(x.get('reason')=='win_probability_below_updated_threshold' for x in e.ledger.all('events'))
        now=second(t)+61;e.on_quotes([('fixed_call',420,now),('fixed_put',400,now)],now)
        trade=e.ledger.all('trades')[0]
        assert trade['exit_reason']=='TARGET' and trade['entry_threshold']==.5
    finally: e.close()

def test_pending_fill_logs_signal_and_entry_policies(tmp_path,config):
    e,t=warmed(tmp_path,config)
    try:
        e.on_snapshot(row(t),second(t)+1)
        e.apply_settings(policy(.8,.85),second(t)+2)
        now=second(t)+61;e.on_quotes([('fixed_call',460,now),('fixed_put',440,now)],now)
        p=next(iter(e.positions.values()))
        assert p['signal_threshold']==.5 and p['entry_threshold']==.8
        assert p['signal_settings_revision']=='frozen-selection' and p['entry_settings_revision']=='test-policy'
    finally: e.close()

def test_policy_persists_history_and_rejects_lost_update(tmp_path):
    root=settings_root(tmp_path); before=read_settings(root)
    saved=save_settings({'3_week':.7,'4_week':.75},before['revision'],root)
    assert read_settings(root)==saved
    audit=json.loads(next((root/'settings_history').glob('*.json')).read_text())
    assert audit['before']==before and audit['after']==saved
    assert json.loads(next((root/'logs').glob('*/settings_changes.jsonl')).read_text())==audit
    with pytest.raises(SettingsConflict): save_settings(before['thresholds'],before['revision'],root)
    assert read_settings(root)==saved
    assert save_settings(saved['thresholds'],saved['revision'],root)==saved
    assert len(list((root/'settings_history').glob('*.json')))==1

@pytest.mark.parametrize('bad',[{'3_week':.6}, {'3_week':.6,'4_week':float('nan')},
    {'3_week':True,'4_week':.6}, {'3_week':60,'4_week':.6}, {'3_week':'.6','4_week':.6}])
def test_invalid_settings_never_publish(tmp_path,bad):
    root=settings_root(tmp_path);before=read_settings(root)
    with pytest.raises(ValueError): save_settings(bad,before['revision'],root)
    assert read_settings(root)==before and not (root/'entry_settings.json').exists()

def test_worker_corrupt_settings_blocks_entries_but_exits_work(tmp_path,config,monkeypatch):
    from pipelines.weeks43 import worker
    settings_root(tmp_path);monkeypatch.setattr(worker,'ROOT',tmp_path)
    e,t=warmed(tmp_path/'ledger',config)
    try:
        enter(e,t);t+=timedelta(minutes=1);e.on_snapshot(row(t),second(t)+1)
        (tmp_path/'entry_settings.json').write_text('{broken')
        worker.refresh_entry_settings(e,second(t)+2)
        assert e.settings_error and not e.pending and len(e.positions)==1
        now=second(t)+61;e.on_quotes([('fixed_call',420,now),('fixed_put',400,now)],now)
        assert not e.positions and e.ledger.all('trades')[0]['exit_reason']=='TARGET'
        write_snapshot(tmp_path/'entry_settings.json',policy())
        worker.refresh_entry_settings(e,now+1)
        assert e.settings_error is None and e.entry_settings['thresholds']['3_week']==.8
    finally: e.close()

class Handler:
    client_address=('127.0.0.1',1000)
    def __init__(self,body=None,origin='http://localhost:8780'):
        data=json.dumps(body).encode() if body is not None else b''
        self.rfile=io.BytesIO(data)
        self.headers={'Content-Type':'application/json','Content-Length':str(len(data)), 'Origin':origin,'Host':'localhost:8780'}
    def _json(self,value,status=200): self.result=value;self.status=status
    def send_error(self,status): self.status=status

def test_settings_api_valid_invalid_conflict_and_origin(tmp_path,monkeypatch):
    from pipelines.weeks43 import bridge
    root=settings_root(tmp_path);monkeypatch.setattr(bridge,'ROOT',root)
    url=urlparse('/firstTwoWeeks/api/settings')
    h=Handler();bridge.handle_get(h,url);before=h.result['requested'];assert h.status==200
    body=dict(thresholds={'3_week':.7,'4_week':.8},expected_revision=before['revision'])
    h=Handler(body);bridge.handle_post(h,url);assert h.status==200
    h=Handler(body);bridge.handle_post(h,url);assert h.status==409
    h=Handler(dict(body,thresholds={'3_week':90,'4_week':.8}));bridge.handle_post(h,url);assert h.status==400
    h=Handler(body,origin='https://elsewhere.example');bridge.handle_post(h,url);assert h.status==403
    h=Handler(dict(body,max_open_positions=4));bridge.handle_post(h,url);assert h.status==400
    assert read_settings(root)['thresholds']==body['thresholds']

def test_backtest_launch_freezes_cutoffs(tmp_path,monkeypatch):
    from pipelines.weeks43 import bridge
    root=settings_root(tmp_path);(root/'config.json').write_text(json.dumps(dict(python_executable='python')))
    monkeypatch.setattr(bridge,'ROOT',root);monkeypatch.setattr(bridge,'BACKTEST_PROCESS',None)
    captured=[]
    def launch(args,**kwargs): captured.append(args);return SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(bridge.subprocess,'Popen',launch)
    before=read_settings(root);result=bridge.launch_backtest()
    assert result['started'] and '--settings-file' in captured[0]
    snapshot=Path(captured[0][-1]);assert json.loads(snapshot.read_text())==before
    save_settings({'3_week':.9,'4_week':.9},before['revision'],root)
    assert json.loads(snapshot.read_text())==before
    assert bridge.launch_backtest()=={'started':False,'reason':'already_running'}

def test_atomic_write_retries_windows_reader_lock(tmp_path,monkeypatch):
    from pipelines.weeks43 import settings
    target=tmp_path/'status.json';target.write_text('{"old":true}')
    original=settings.os.replace;attempts=[]
    def intermittent_lock(source,destination):
        attempts.append(1)
        if len(attempts)<=2:
            assert json.loads(target.read_text())=={'old':True}
            raise PermissionError('Simulated Windows reader lock')
        return original(source,destination)
    monkeypatch.setattr(settings.os,'replace',intermittent_lock)
    settings.atomic_write_json(target,{'complete':True})
    assert json.loads(target.read_text())=={'complete':True} and len(attempts)==3
    assert not list(tmp_path.glob('*.tmp'))

def test_failed_publish_preserves_previous_settings(tmp_path,monkeypatch):
    from pipelines.weeks43 import settings
    target=tmp_path/'entry_settings.json';settings.write_snapshot(target,policy())
    def denied(*args): raise PermissionError('File remains locked')
    monkeypatch.setattr(settings.os,'replace',denied);monkeypatch.setattr(settings.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError): settings.write_snapshot(target,policy(.9,.9,'new'))
    assert json.loads(target.read_text())==policy() and not list(tmp_path.glob('*.tmp'))
