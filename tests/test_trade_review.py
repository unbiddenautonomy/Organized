"""The trade-review tool (tools/trade_review) reads IVRV logs; it never touches the trading engine."""
import importlib
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "trade_review"
DAY = "2026-11-17"


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def review(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL))
    source = importlib.import_module("serve_dashboard")
    stop = importlib.import_module("replay_stop")
    daily = importlib.import_module("generate_daily")
    source.market_history.cache_clear()
    source.execution_history.cache_clear()
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(source, "PREDICTIONS", logs / "live_predictions.jsonl")
    monkeypatch.setattr(stop, "EXTRACTOR", tmp_path / "missing_extractor.exe")

    _jsonl(logs / "live_predictions.jsonl", [
        {"event_type": "fixed_strike_inference", "exchange_minute": f"{DAY} 10:{m:02d}:00",
         "fixed_entry_strike": 50000.0,
         "market": {"spot": 50010.0, "combined_iv": 0.15, "straddle_price": 400.0},
         "strategy_signal": {"current_rv": 0.12}, "predictions": {"pred_rv_120m_xgboost": 0.11},
         "paper_trading": {"open_positions": []}}
        for m in range(3)
    ])
    position = {"side": "SHORT_STRADDLE", "position_id": f"{DAY} 10:00:00|2026-11-23|50000",
                "entry_datetime": f"{DAY} 10:00:00", "fixed_strike": 50000.0, "expiry": "2026-11-23",
                "call_token": 1, "put_token": 2, "quantity": 30, "entry_straddle_price": 400.0,
                "target_inr": 1500.0, "stop_loss_inr": -750.0, "maximum_holding_minutes": 120,
                "square_off_time": "15:14:30", "peak_unrealized_pnl_inr": 120.0,
                "max_drawdown_inr": -880.0}
    exit_premium = 400.0 + 760.0 / 30
    _jsonl(logs / "paper_trade_events.jsonl", [
        {**position, "event_type": "paper_trade_entry"},
        {**position, "event_type": "paper_trade_exit", "exit_datetime": f"{DAY} 10:10:00",
         "exit_reason": "stop_loss", "realized_pnl_inr": -760.0, "exit_straddle_price": exit_premium},
    ])
    # Minute closes after the stop: the premium falls far enough to reach the 1,500 target.
    _jsonl(logs / "fixed_strike_120m_paths.jsonl", [
        {"expiry": "2026-11-23", "entry_strike": 50000.0, "observation_datetime": f"{DAY} 10:{m}:00",
         "observation_straddle_price": price, "status": "partial"}
        for m, price in ((11, 420.0), (20, 340.0))
    ])
    (logs / "live_predictions_demo_positions.json").write_text(json.dumps({"positions": [
        {"position_id": "demo-1", "side": "SHORT_STRADDLE", "signal_time": f"{DAY} 10:01:00",
         "entry_time": f"{DAY} 10:01:05", "expiry": "2026-11-23", "strike": 50000.0, "quantity": 30,
         "entry_premium": 400.0, "horizon_pnl_inr": None, "status": "TRACKING", "probability": 0.8,
         "paper_skip_reason": "maximum_open_positions_reached", "target_hit": False, "stop_hit": False},
    ]}), encoding="utf-8")
    return source, daily, logs


def test_stopped_trade_gets_a_1500_loss_limit_continuation(review):
    source, _, logs = review
    files = [source.PREDICTIONS, logs / "paper_trade_events.jsonl"]
    result = json.loads(source.execution_history(
        tuple((f, f.stat().st_mtime_ns, f.stat().st_size) for f in files), DAY))
    assert result["date"] == DAY and len(result["market"]) == 3
    (trade,) = result["trades"]
    assert trade["actual_reason"] == "stop_loss" and trade["actual_pnl"] == -760.0
    assert trade["reason"] == "profit_target" and trade["pnl"] == 1800.0
    assert trade["basis"] == "Minute-close estimate"  # no tick extractor available
    assert trade["peak"] == 1800.0 and trade["drawdown"] == -880.0


def test_daily_review_fills_every_placeholder(review):
    _, daily, _ = review
    page = daily.render(DAY)
    assert "@@" not in page
    assert "Tuesday, 17 November 2026" in page
    assert f"{DAY} 10:01:05" in page and "maximum_open_positions_reached" in page


def test_server_routes_read_the_saved_logs(review):
    source, _, _ = review
    server = ThreadingHTTPServer(("127.0.0.1", 0), source.Dashboard)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/positions", timeout=10) as response:
            assert [row["position_id"] for row in json.load(response)["positions"]] == ["demo-1"]
        with urlopen(base + "/cutoff-marks", timeout=10) as response:
            assert json.load(response) == []
        with pytest.raises(HTTPError) as missing:
            urlopen(base + "/no-such-route", timeout=10)
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
