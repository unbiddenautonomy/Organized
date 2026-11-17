"""Signal and position sounds play in a separate process and never interrupt trading."""
from voltr.web import alerts as position_alert


def test_signal_and_action_launch_distinct_copied_sounds(monkeypatch):
    calls = []
    monkeypatch.delenv('POSITION_ALERT_DISABLED', raising=False)
    monkeypatch.setattr(position_alert.subprocess, 'Popen', lambda *args, **kwargs: calls.append((args, kwargs)))
    position_alert.notify_signal()
    position_alert.notify_position_action()
    assert len(calls) == 2
    assert calls[0][0][0][-1] == str(position_alert.SIGNAL_SOUND)
    assert calls[1][0][0][-1] == str(position_alert.ACTION_SOUND)
    assert all(options['cwd'] == position_alert.SOUND.parents[3] for _, options in calls)  # repository root
    assert position_alert.SIGNAL_SOUND.is_file() and position_alert.ACTION_SOUND.is_file()


def test_audio_launch_failure_does_not_interrupt_entry(monkeypatch):
    monkeypatch.delenv('POSITION_ALERT_DISABLED', raising=False)
    def failed(*args, **kwargs):
        raise OSError('audio device unavailable')
    monkeypatch.setattr(position_alert.subprocess, 'Popen', failed)
    position_alert.notify_signal()
    position_alert.notify_position_action()
