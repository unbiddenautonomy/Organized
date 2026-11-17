"""Play distinct sounds for qualified signals and paper position actions.

Playback is a separate process so audio devices and codecs cannot block a
market-data worker. The failure is logged but never changes a trade decision.
"""
from pathlib import Path
import ctypes
import logging
import os
import subprocess
import sys
from datetime import datetime

SIGNAL_SOUND = Path(__file__).resolve().parent / 'assets' / 'peshaan-ravi-kishan.mp3'
ACTION_SOUND = Path(__file__).resolve().parent / 'assets' / 'koteswara-ravi-kishan_U5lot3o.mp3'
# Kept for callers that imported the original position alert asset.
SOUND = ACTION_SOUND
ERROR_LOG = Path(__file__).resolve().parents[2] / 'runtime' / 'logs' / 'position_alert_errors.log'
LOGGER = logging.getLogger(__name__)


def _notify(sound):
    if os.environ.get('POSITION_ALERT_DISABLED') == '1':
        return
    try:
        if not sound.is_file():
            raise FileNotFoundError(sound)
        subprocess.Popen(
            [sys.executable, '-m', 'voltr.web.alerts', '--play', str(sound)],
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            close_fds=True,
        )
    except (OSError, ValueError) as exc:
        LOGGER.warning('Position alert could not start: %s', exc)


def notify_signal():
    _notify(SIGNAL_SOUND)


def notify_position_action():
    _notify(ACTION_SOUND)


def notify_position_opened():
    notify_position_action()


def _command(command):
    result = ctypes.windll.winmm.mciSendStringW(command, None, 0, None)
    if result:
        message = ctypes.create_unicode_buffer(256)
        ctypes.windll.winmm.mciGetErrorStringW(result, message, len(message))
        raise OSError(f'MCI error {result}: {message.value}')


def play(sound=ACTION_SOUND, probe=False):
    if os.name != 'nt':
        raise OSError('MP3 playback requires Windows')
    sound = Path(sound).resolve()
    if sound not in (SIGNAL_SOUND, ACTION_SOUND):
        raise ValueError('Unknown position alert sound')
    alias = f'position_alert_{os.getpid()}'
    # The explicit MPEGVideo MCI device accepts MP3 files on Windows.
    _command(f'open "{sound}" type mpegvideo alias {alias}')
    try:
        if not probe:
            _command(f'play {alias} wait')
    finally:
        _command(f'close {alias}')


if __name__ == '__main__':
    try:
        sound = next((Path(arg) for arg in sys.argv[1:] if arg.lower().endswith('.mp3')), ACTION_SOUND)
        play(sound, probe='--probe' in sys.argv)
    except Exception as exc:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open('a', encoding='utf-8') as handle:
            handle.write(f'{datetime.now().isoformat()} {type(exc).__name__}: {exc}\n')
        raise
