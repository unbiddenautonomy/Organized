from live import position_alert
from live import live_inference
from datetime import date, datetime


def test_signal_and_action_launch_distinct_copied_sounds(monkeypatch):
    calls = []
    monkeypatch.delenv('POSITION_ALERT_DISABLED', raising=False)
    monkeypatch.setattr(position_alert.subprocess, 'Popen', lambda *args, **kwargs: calls.append((args, kwargs)))
    position_alert.notify_signal()
    position_alert.notify_position_action()
    assert len(calls) == 2
    assert calls[0][0][0][-1] == str(position_alert.SIGNAL_SOUND)
    assert calls[1][0][0][-1] == str(position_alert.ACTION_SOUND)
    assert all(options['cwd'] == position_alert.SOUND.parents[2] for _, options in calls)
    assert position_alert.SIGNAL_SOUND.is_file() and position_alert.ACTION_SOUND.is_file()


def test_audio_launch_failure_does_not_interrupt_entry(monkeypatch):
    monkeypatch.delenv('POSITION_ALERT_DISABLED', raising=False)
    def failed(*args, **kwargs):
        raise OSError('audio device unavailable')
    monkeypatch.setattr(position_alert.subprocess, 'Popen', failed)
    position_alert.notify_signal()
    position_alert.notify_position_action()


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
