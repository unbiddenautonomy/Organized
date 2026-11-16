"""Persistent operator entry policy; fitted models and their selection stay frozen."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math
import os
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
MAX_POSITIONS = 3
BUCKETS = ('3_week', '4_week')
_LOCK = threading.Lock()

class SettingsConflict(ValueError):
    pass

def validate_thresholds(values):
    if not isinstance(values, dict) or set(values) != set(BUCKETS):
        raise ValueError('Provide thresholds for exactly 3_week and 4_week')
    result = {}
    for bucket in BUCKETS:
        value = values[bucket]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(bucket + ' probability cutoff must be a finite number from 0 to 1')
        result[bucket] = float(value)
    return result

def validate_settings(value):
    if not isinstance(value, dict):
        raise ValueError('Entry settings must be an object')
    result = dict(value)
    result['thresholds'] = validate_thresholds(value.get('thresholds'))
    if value.get('max_open_positions') != MAX_POSITIONS or value.get('comparison') != '>=':
        raise ValueError('The book requires a maximum of 3 positions and probability >= cutoff')
    if not isinstance(value.get('revision'), str) or not value['revision']:
        raise ValueError('Entry settings need a revision')
    return result

def defaults(selection):
    return dict(thresholds=validate_thresholds(selection['thresholds']),
                max_open_positions=MAX_POSITIONS, comparison='>=', revision='frozen-selection',
                source='pre_2024_validation', updated_at=None)

def read_settings(root=ROOT):
    root = Path(root)
    path = root / 'entry_settings.json'
    if path.exists():
        return validate_settings(json.loads(path.read_text(encoding='utf-8')))
    return defaults(json.loads((root / 'otm_models/selection.json').read_text(encoding='utf-8')))

def write_snapshot(path, settings):
    atomic_write_json(path, validate_settings(settings))

def atomic_write_json(path, value):
    """Publish a whole JSON document, tolerating short Windows reader locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        for attempt in range(8):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt==7: raise
                time.sleep(min(.01*2**attempt,.2))
    finally:
        if temp.exists(): temp.unlink()

@contextmanager
def _settings_lock(root):
    # Different dashboard processes share the same lock, as well as threads.
    with _LOCK:
        runtime = root / 'runtime'; runtime.mkdir(parents=True, exist_ok=True)
        with (runtime / 'entry_settings.lock').open('a+b') as f:
            f.seek(0)
            if not f.read(1): f.write(b'0'); f.flush()
            f.seek(0)
            if os.name == 'nt':
                import msvcrt
                try: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc: raise SettingsConflict('Another settings update is in progress; refresh and retry') from exc
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX)
            try: yield
            finally:
                f.seek(0)
                if os.name == 'nt': msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else: fcntl.flock(f, fcntl.LOCK_UN)

def save_settings(thresholds, expected_revision, root=ROOT):
    root = Path(root); thresholds = validate_thresholds(thresholds)
    with _settings_lock(root):
        previous = read_settings(root)
        if expected_revision != previous['revision']:
            raise SettingsConflict('Settings changed in another window; refresh before saving')
        if thresholds == previous['thresholds'] and (root / 'entry_settings.json').exists():
            return previous
        now = datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')
        initial = not (root / 'entry_settings.json').exists() and thresholds == previous['thresholds']
        current = dict(thresholds=thresholds, max_open_positions=MAX_POSITIONS, comparison='>=',
                       revision=uuid.uuid4().hex, source='pre_2024_validation' if initial else 'operator_override', updated_at=now)
        # Preserve the before/after policy before publishing the new current version.
        history = root / 'settings_history'; history.mkdir(exist_ok=True)
        record = dict(time=now, event='ENTRY_SETTINGS_SAVED', before=previous, after=current)
        with (history / (current['revision'] + '.json')).open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
        daily = root / 'logs' / now[:10]; daily.mkdir(parents=True, exist_ok=True)
        with (daily / 'settings_changes.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
        write_snapshot(root / 'entry_settings.json', current)
        return current
