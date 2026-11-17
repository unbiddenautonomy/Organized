from pathlib import Path
from datetime import datetime,timedelta,date
from types import SimpleNamespace
import importlib.util
import json
import sys
import urllib.request
import pytest
from pipelines.weeks43.engine import ROOT,Ledger,PaperEngine,second,stamp,ModelRunner
from pipelines.weeks43.features import costs,pnl,feature_row,history_add,FEATURES

class Runner:
    selection={'classifier':'logistic','regressor':'ridge','thresholds':{'3_week':.5,'4_week':.5}}
    def score(self,f): return {'logistic':.9,'ridge':20.,'otm_loss_q90':5.}

@pytest.fixture
def config(): return json.loads((ROOT/'config.json').read_text())

def row(ts,atm=50000):
    return dict(signal_time=ts.isoformat(sep=' '),available_at=(ts+timedelta(seconds=1)).isoformat(sep=' '),
        expiry='2026-11-24',days_to_expiry=25.,spot=atm+10.,future=atm+25.,atm=atm,step=100.,
        call_strike=atm+100.,put_strike=atm-100.,atm_call=510.,atm_put=490.,call=460.,put=440.,
        call_oi=100.,put_oi=110.,atm_call_oi=120.,atm_put_oi=130.,call_key='fixed_call',put_key='fixed_put',quantity=30)

def warmed(tmp_path,config):
    engine=PaperEngine(config,Ledger(tmp_path),Runner()); t=datetime(2026,10,30,10,15,59)
    for i in range(61,0,-1): history_add(engine.history,row(t-timedelta(minutes=i)))
    return engine,t

def enter(engine,t,atm=50000):
    r=row(t,atm); engine.on_snapshot(r,second(r['available_at']))
    engine.on_quotes([('fixed_call',460,second(t)+61),('fixed_put',440,second(t)+61)],second(t)+61)

def test_maximum_three_counts_pending_and_open(tmp_path,config):
    e,t=warmed(tmp_path,config)
    for i in range(4): enter(e,t+timedelta(minutes=i))
    assert len(e.positions)==3
    assert len(e.pending)==0
    assert e.ledger.all('signals')[-1]['reason']=='three_position_limit'
    e.close()

def test_net_costs_control_profit_target_and_stop(tmp_path,config):
    e,t=warmed(tmp_path,config);enter(e,t)
    now=second(t)+120
    # Gross +1500 is not a NET target; full round-trip costs keep the trade open.
    e.on_quotes([('fixed_call',435,now),('fixed_put',415,now)],now)
    assert len(e.positions)==1
    now+=1;e.on_quotes([('fixed_call',420,now),('fixed_put',400,now)],now)
    assert not e.positions
    p=e.ledger.all('positions')[0]
    assert p['exit_reason']=='TARGET' and p['net_pnl']>=1500 and p['costs']['total']>0
    assert p['gross_pnl']-p['costs']['total']==p['net_pnl']
    e.close()

def test_signal_open_and_close_have_distinct_alerts(tmp_path,config,monkeypatch):
    from voltr.web import alerts as position_alert
    alerts=[]
    monkeypatch.setattr(position_alert,'notify_signal',lambda:alerts.append('signal'))
    monkeypatch.setattr(position_alert,'notify_position_action',lambda:alerts.append('action'))
    e,t=warmed(tmp_path,config)
    enter(e,t)
    now=second(t)+120
    e.on_quotes([('fixed_call',420,now),('fixed_put',400,now)],now)
    assert alerts==['signal','action','action']
    e.close()

def test_stop_can_overshoot_and_keeps_original_contracts(tmp_path,config):
    e,t=warmed(tmp_path,config);enter(e,t)
    now=second(t)+120
    e.on_quotes([('new_atm_call',1,now),('new_atm_put',1,now)],now)
    assert len(e.positions)==1
    e.on_quotes([('fixed_call',490,now),('fixed_put',460,now)],now)
    p=e.ledger.all('positions')[0]
    assert p['exit_reason']=='STOP' and p['net_pnl']<-750
    assert p['call_strike']==50100 and p['put_strike']==49900
    assert p['call_key']=='fixed_call' and p['put_key']=='fixed_put'
    e.close()

def test_no_fill_before_delay_or_using_future_quote(tmp_path,config):
    e,t=warmed(tmp_path,config);r=row(t);now=second(t)+1
    e.on_quotes([('fixed_call',1,now+60),('fixed_put',1,now+60)],now)
    assert not e.quotes
    e.on_snapshot(r,now)
    e.on_quotes([('fixed_call',460,now),('fixed_put',440,now)],now)
    assert not e.positions and len(e.pending)==1
    e.on_quotes([],now+60)
    assert not e.positions  # quote is pre-decision even if still at age limit
    e.close()

def test_stale_exit_is_unresolved_not_fabricated(tmp_path,config):
    e,t=warmed(tmp_path,config);enter(e,t)
    e.on_quotes([],second(t)+130*60)
    assert len(e.positions)==1 and next(iter(e.positions.values()))['mark_status']=='STALE_OR_MISSING'
    e.close()

def test_restart_deduplicates_signal_and_restores_fixed_position(tmp_path,config):
    e,t=warmed(tmp_path,config);enter(e,t);e.close()
    e=PaperEngine(config,Ledger(tmp_path),Runner())
    assert len(e.positions)==1 and not e.quotes
    e.on_snapshot(row(t),second(t)+1)
    assert not e.pending and len(e.ledger.all('signals'))==1
    p=next(iter(e.positions.values()));assert (p['call_strike'],p['put_strike'])==(50100,49900)
    e.close()

def test_path_continues_after_early_exit(tmp_path,config):
    e,t=warmed(tmp_path,config);enter(e,t)
    now=second(t)+120;e.on_quotes([('fixed_call',420,now),('fixed_put',400,now)],now)
    assert not e.positions and len(e.tracks)==1
    now=second(t)+121*60+1;e.on_quotes([('fixed_call',300,now),('fixed_put',250,now)],now)
    paths=e.ledger.all('strike_paths')
    assert paths[-1]['final'] and paths[-1]['call_strike_at_t']==50100
    assert not e.tracks
    e.close()

def test_path_end_never_uses_a_quote_after_the_horizon(tmp_path,config):
    e,t=warmed(tmp_path,config);enter(e,t)
    end=next(iter(e.tracks.values()))['end']
    e.on_quotes([('fixed_call',300,end+120),('fixed_put',250,end+120)],end+120)
    final=e.ledger.all('strike_paths')[-1]
    assert final['time']==stamp(end) and final['final']
    assert final['call'] is None and final['quote_status']=='STALE_OR_MISSING'
    e.close()

def test_features_cannot_see_future_and_lag_age_is_bounded():
    t=datetime(2026,10,30,10,15,59);h={}
    for i in range(65,-1,-1): history_add(h,row(t-timedelta(minutes=i)))
    r=row(t);before=feature_row(r,h)
    h[(t+timedelta(minutes=1)).isoformat(sep=' ')]=dict(row(t+timedelta(minutes=1)),future=999999)
    assert feature_row(r,h)==before and list(before)==FEATURES
    del h[(t-timedelta(minutes=5)).isoformat(sep=' ')]
    assert feature_row(r,h) is not None
    del h[(t-timedelta(minutes=6)).isoformat(sep=' ')]
    assert feature_row(r,h) is None

def test_saved_models_use_feature_allowlist_and_pretest_splits():
    runner=ModelRunner(); report=json.loads((ROOT/'otm_models/training_report.json').read_text())
    assert report['splits']['train']['last']<'2023-01-01'
    assert report['splits']['validation']['last']<'2024-01-01'
    assert not any('exit' in f or 'target' in f or 'label' in f or 'net_points' in f for f in FEATURES)
    assert len(runner.models)==6

def test_oi_adapter_rejects_later_information():
    from pipelines.weeks43.bridge import Bridge
    from collections import deque
    b=Bridge.__new__(Bridge);b.oi_history={};b.config={'lots_per_position':1};sent=[];b.submit=sent.append
    minute=second(datetime(2026,10,30,10,15))//60
    bar=SimpleNamespace(close=50000.,open=50025.)
    b.engine=SimpleNamespace(spot_bars={minute:bar},future_bars={minute:bar},strikes=[49900.,50000.,50100.],
        pairs={49900.:{'CE':1,'PE':2},50000.:{'CE':3,'PE':4},50100.:{'CE':5,'PE':6}},
        option_bars={k:{minute:SimpleNamespace(close=450.)} for k in range(1,7)},
        expiry=date(2026,11,24),latest_option_ticks={},latest_exchange_second=minute*60+120,lot_size=30)
    for k in range(1,7): b.oi(k,100,minute*60+50);b.oi(k,999,minute*60+70)
    b.snapshot(minute)
    assert sent[-1]['row']['call_oi']==100 and sent[-1]['row']['atm_put_oi']==100

def test_existing_routes_and_new_route_are_isolated(monkeypatch):
    app=ROOT.parent/'ivrv'
    from pipelines.ivrv.live import DashboardServer
    from pipelines.ivrv import live as app_module
    from pipelines.weeks43 import bridge as weeks43_bridge
    monkeypatch.setattr(weeks43_bridge,'launch_backtest',lambda:{'started':True,'run':'test_dispatch_only'})
    calls=[]
    class OtherExtension:
        @staticmethod
        def handle_get(handler,parsed): return False
        @staticmethod
        def handle_post(handler,parsed):
            calls.append(parsed.path)
            if parsed.path=='/other-extension': handler._json({'other':True}); return True
            return False
    monkeypatch.setattr(app_module,'PCR_WEB',OtherExtension)
    view=SimpleNamespace(status=lambda:{'original':True},recent=lambda limit:[])
    server=DashboardServer('127.0.0.1',0,app/'dashboard/live_dashboard.html',view);server.start()
    try:
        base='http://127.0.0.1:'+str(server.server.server_port)
        assert json.load(urllib.request.urlopen(base+'/api/status'))=={'original':True}
        assert '3 &amp; 4 Week OTM' in urllib.request.urlopen(base+'/firstTwoWeeks').read().decode()
        assert urllib.request.urlopen(base+'/').read()==(app/'dashboard/live_dashboard.html').read_bytes()
        assert json.load(urllib.request.urlopen(base+'/firstTwoWeeks/api/models'))['selection']['horizon_minutes']==120
        request=urllib.request.Request(base+'/firstTwoWeeks/api/backtest',method='POST')
        assert json.load(urllib.request.urlopen(request))['run']=='test_dispatch_only'
        assert calls==[]
        request=urllib.request.Request(base+'/other-extension',method='POST')
        assert json.load(urllib.request.urlopen(request))=={'other':True}
        assert calls==['/other-extension']
    finally: server.close()
