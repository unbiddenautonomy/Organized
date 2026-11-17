"""Tests for voltr.core: the bytes each writer produces and the lock semantics callers rely on."""
import json
import math
from datetime import date, datetime

import numpy as np
import pytest

from voltr.core.clock import MARKET_TIMEZONE, NSE_EPOCH
from voltr.core.io import (append_jsonl, atomic_write_json, file_lock, json_clean, read_json, read_jsonl,
                           read_text)
from voltr.core.machine import relative_to_config, resolve_interface


def test_atomic_write_json_bytes_match_the_previous_writers(tmp_path):
    # The previous writers used text mode, so line endings follow the platform (CRLF on Windows).
    value = {"b": 1.5, "a": [1, 2]}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf-8")
    plain = tmp_path / "plain.json"
    atomic_write_json(plain, value)
    assert plain.read_bytes() == reference.read_bytes()
    reference.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    newline = tmp_path / "nested" / "newline.json"
    atomic_write_json(newline, value, trailing_newline=True, fsync=True)
    assert newline.read_bytes() == reference.read_bytes()
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_write_json_refuses_nan_and_keeps_the_old_file(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"bad": math.nan})
    assert read_json(path) == {"ok": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_prepare_cleans_numpy_dates_and_non_finite_values(tmp_path):
    path = tmp_path / "clean.json"
    atomic_write_json(path, {"x": np.float64(2.0), "d": date(2026, 11, 2), "n": math.inf, 3: (1,)},
                      prepare=json_clean)
    assert read_json(path) == {"x": 2.0, "d": "2026-11-02", "n": None, "3": [1]}


def test_jsonl_round_trip_skips_a_torn_last_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    append_jsonl(path, {"i": 1})
    append_jsonl(path, {"i": np.int64(2)}, prepare=json_clean)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"i": 3')
    assert read_jsonl(path) == [{"i": 1}, {"i": 2}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []
    assert read_text(path).startswith('{"i": 1}')


def test_read_json_default_for_missing_or_corrupt(tmp_path):
    assert read_json(tmp_path / "missing.json", {"d": 1}) == {"d": 1}
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert read_json(tmp_path / "bad.json") is None


def test_file_lock_is_exclusive_and_released(tmp_path):
    lock = tmp_path / "locks" / "worker.lock"
    with file_lock(lock):
        with pytest.raises(OSError):
            with file_lock(lock):
                pass
    with file_lock(lock):
        pass


def test_clock_and_machine_helpers(tmp_path, monkeypatch):
    assert NSE_EPOCH == datetime(1980, 1, 1)
    assert str(MARKET_TIMEZONE) == "Asia/Kolkata"
    config = tmp_path / "cfg" / "c.json"
    assert relative_to_config("models/m.joblib", config) == (tmp_path / "cfg" / "models" / "m.joblib").resolve()
    assert relative_to_config(str(tmp_path / "abs"), config) == (tmp_path / "abs").resolve()
    monkeypatch.setenv("VOLTR_TEST_INTERFACE", "10.1.2.3")
    assert resolve_interface("auto", environment="VOLTR_TEST_INTERFACE") == "10.1.2.3"
    monkeypatch.delenv("VOLTR_TEST_INTERFACE")
    assert resolve_interface("192.168.0.9", environment="VOLTR_TEST_INTERFACE") == "192.168.0.9"
