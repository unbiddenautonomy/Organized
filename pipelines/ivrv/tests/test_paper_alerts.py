from datetime import date, datetime

from pipelines.ivrv import live as live_inference
from voltr.web import alerts as position_alert


def test_ivrv_open_and_close_each_alert_once(tmp_path, monkeypatch):
    monkeypatch.setattr(live_inference, 'BASE_DIR', tmp_path)
    alerts=[]
    monkeypatch.setattr(position_alert, 'notify_position_action', lambda: alerts.append('action'))
    engine=live_inference.PaperTradingEngine({
        'paper_trade_events_log':'events.jsonl', 'paper_position_state':'state.json',
        'paper_profit_target_inr':1500., 'paper_stop_loss_inr':-750.,
    },30)
    at=lambda stamp:int((datetime.fromisoformat(stamp)-live_inference.NSE_EPOCH).total_seconds())
    entry=at('2026-10-22 10:00:00')
    assert engine.try_enter(exchange_second=entry,signal_minute=entry//60,
        expiry=date(2026,10,29),strike=50000.,call_token=1,put_token=2,
        call_price=200.,put_price=200.,triggered_rules=['test'],convergence_probability=.95)
    position_id=next(iter(engine.positions))
    assert engine.mark(position_id,at('2026-10-22 10:05:00'),170.,170.) is not None
    assert alerts==['action','action']
    engine.close()
