"""Persistent operator entry policy; fitted models and their selection stay frozen."""
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import math
import os
import threading
import time
import uuid

# Until the pipelines move under the repository root (refactor step 9), make
# the shared `voltr` package importable when this folder is the entry point.
import sys as _sys
from pathlib import Path as _Path
_REPO = next(p for p in _Path(__file__).resolve().parents if (p / "voltr" / "__init__.py").is_file())
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from voltr.core.io import atomic_write_json as _atomic_write_json, file_lock

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
    """Publish a whole JSON document (newline-terminated, fsynced), tolerating short Windows reader locks."""
    _atomic_write_json(path, value, trailing_newline=True, fsync=True)

@contextmanager
def _settings_lock(root):
    # Different dashboard processes share the same lock, as well as threads.
    # Windows reports a concurrent update as a conflict; POSIX waits for it.
    with _LOCK, ExitStack() as stack:
        try:
            stack.enter_context(file_lock(root / 'runtime' / 'entry_settings.lock', blocking=os.name != 'nt'))
        except OSError as exc:
            raise SettingsConflict('Another settings update is in progress; refresh and retry') from exc
        yield

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
