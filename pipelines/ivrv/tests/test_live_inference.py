from pathlib import Path

from datetime import date, datetime
import io
import json
import os
import struct
from zoneinfo import ZoneInfo

import pandas as pd
from pipelines.ivrv import paths as ivrv_paths
from pipelines.ivrv import live as live_inference
from pipelines.ivrv.live import (
    LAGS,
    REQUIRED_FEATURE_MINUTES,
    archive_previous_day_logs,
    convergence_entry_eligible,
    current_costs,
    historical_replay_config,
    install_source_imports,
    load_live_config,
    next_session_window,
    paper_entry_route,
    PaperTradingEngine,
    required_live_input_features,
    run_historical_inference,
)
from voltr.feed.udp import NseFoUdpReceiver


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CONFIG = REPO / "config" / "live_config.json"


def test_live_config_resolves_local_paths_and_adapter_override(monkeypatch):
    monkeypatch.setenv("VOLTR_FO_INTERFACE", "10.0.0.12")
    monkeypatch.setenv("VOLTR_CM_INTERFACE", "10.0.0.13")
    config = load_live_config(CONFIG)
    assert config["multicast_ip"] == "233.1.2.5"
    assert config["port"] == 34330
    assert config["cm_port"] == 8222
    assert config["interface_ip"] == "10.0.0.12"
    assert config["cm_interface_ip"] == "10.0.0.13"
    assert Path(config["lzo_dll"]).resolve() == (REPO / "voltr" / "feed" / "native" / "lzo2.dll").resolve()
    assert config["fixed_strike_path_horizon_minutes"] == 120
    assert config["paper_profit_target_inr"] == 1500.0
    assert config["paper_stop_loss_inr"] == -750.0
    assert config["paper_entry_convergence_probability_threshold"] == 0.70
    assert config["paper_entry_iv_rv_near_gap_decimal"] == 0.005
    assert config["paper_entry_rv_decline_event_score_threshold"] == 0.0
    assert config["paper_max_open_positions"] == 3


def test_live_runtime_uses_shared_udp_receiver(monkeypatch):
    monkeypatch.setenv("VOLTR_FO_INTERFACE", "10.0.0.12")
    monkeypatch.setenv("VOLTR_CM_INTERFACE", "10.0.0.13")
    config = load_live_config(CONFIG)
    source = install_source_imports(config)
    assert source["NseFoUdpReceiver"] is NseFoUdpReceiver
    assert source["NseFoUdpReceiver"].__module__ == "voltr.feed.udp"


def test_local_udp_receiver_unpacks_uncompressed_sequential_packet():
    class FakeLzo:
        def decompress(self, compressed: bytes, max_output: int) -> bytes:
            raise AssertionError("uncompressed packet must not call LZO")

    message = bytearray(48)
    struct.pack_into(">H", message, 46, 40)
    datagram = bytearray(4)
    struct.pack_into(">h", datagram, 2, 1)
    datagram.extend(struct.pack(">H", 0))
    datagram.extend(message)
    receiver = NseFoUdpReceiver({"max_decompressed_bytes": 65535}, FakeLzo())
    assert list(receiver.unpack_datagram(bytes(datagram))) == [bytes(message[8:])]


def test_current_costs_have_no_brokerage(monkeypatch):
    monkeypatch.setenv("VOLTR_FO_INTERFACE", "10.0.0.12")
    monkeypatch.setenv("VOLTR_CM_INTERFACE", "10.0.0.13")
    config = load_live_config(CONFIG)
    costs = current_costs(600.0, 550.0, 30, config)
    assert costs["brokerage"] == 0.0
    assert costs["estimated_total"] > 0
    assert abs(costs["stt"] - 600.0 * 30 * 0.0015) < 1e-12


def test_dashboard_exposes_every_prediction_and_spot_bands():
    html = (ROOT / "dashboard" / "live_dashboard.html").read_text(encoding="utf-8")
    assert "Every prediction" in html
    assert "Predicted SPOT boundaries" in html
    assert "one_sigma_low" in html
    assert "approx_90pct_high" in html
    assert "Regime-routed paper short straddles" in html
    assert "paper.open_positions" in html
    assert "Aggregate unrealized P&amp;L" in html
    assert "All prediction signals in the active session" in html
    assert "/api/predictions?limit=500" in html
    assert "All values" in html


def test_convergence_probability_helper_uses_a_strict_operator():
    field = "pred_convergence_probability_classifier"
    assert convergence_entry_eligible({field: 0.900001}, 0.90) is True
    assert convergence_entry_eligible({field: 0.90}, 0.90) is False
    assert convergence_entry_eligible({field: 0.899999}, 0.90) is False
    assert convergence_entry_eligible({}, 0.90) is False


def test_paper_entry_routes_each_iv_rv_regime_to_its_assigned_model():
    common = {
        "pred_convergence_probability_classifier": 0.750001,
        "pred_strategy_rule_iv_above_rv_iv_declines_to_rv": True,
        "pred_no120_event_score": 0.0,
    }
    above = paper_entry_route(
        {**common, "pred_no120_gate_iv_minus_rv": 0.006}, 0.75, 0.005, 0.0
    )
    assert above["eligible"] is True
    assert above["route"] == "iv_above_rv_convergence"
    at_threshold = paper_entry_route(
        {
            **common,
            "pred_no120_gate_iv_minus_rv": 0.006,
            "pred_convergence_probability_classifier": 0.75,
        },
        0.75,
        0.005,
        0.0,
    )
    assert at_threshold["eligible"] is False

    near = paper_entry_route(
        {
            **common,
            "pred_no120_gate_iv_minus_rv": 0.005,
            "pred_convergence_probability_classifier": 0.750001,
        },
        0.75,
        0.005,
        0.0,
    )
    assert near["eligible"] is True
    assert near["route"] == "iv_near_rv_sharp_rv_decline"

    below = paper_entry_route(
        {
            **common,
            "pred_no120_gate_iv_minus_rv": -0.006,
            "pred_convergence_probability_classifier": 0.750001,
        },
        0.75,
        0.005,
        0.0,
    )
    assert below["eligible"] is True
    assert below["route"] == "iv_below_rv_sharp_rv_decline"
    low_confidence = paper_entry_route(
        {**common, "pred_no120_gate_iv_minus_rv": 0.005,
         "pred_convergence_probability_classifier": 0.70},
        0.70, 0.005, 0.0,
    )
    assert low_confidence["eligible"] is False
    assert low_confidence["reason"] == "convergence_probability_below_entry_threshold"
    failed_event = paper_entry_route(
        {**common, "pred_no120_gate_iv_minus_rv": -0.006, "pred_no120_event_score": -1e-9},
        0.75,
        0.005,
        0.0,
    )
    assert failed_event["eligible"] is False


def test_live_warmup_is_one_hour():
    assert LAGS == (5, 15, 30, 60)
    assert REQUIRED_FEATURE_MINUTES == 61
    html = (ROOT / "dashboard" / "live_dashboard.html").read_text(encoding="utf-8")
    assert "0 / 61" in html
    assert "61-minute warmup" in html
    assert "0 / 121" not in html


def test_scheduler_uses_weekday_0900_to_1540_kolkata_window():
    timezone = ZoneInfo("Asia/Kolkata")
    start, end = next_session_window(datetime(2026, 10, 22, 8, 0, tzinfo=timezone))
    assert (start.hour, start.minute) == (9, 0)
    assert (end.hour, end.minute) == (15, 40)

    monday_start, _ = next_session_window(datetime(2026, 10, 23, 16, 0, tzinfo=timezone))
    assert monday_start.date().isoformat() == "2026-10-26"


def test_scheduled_session_date_overrides_cached_contract_asof(monkeypatch):
    expected = date(2026, 10, 28)
    config = {"contract_asof_date": "2026-10-27"}

    def fake_bootstrap(received_config, *, asof):
        assert received_config is config
        assert asof == expected
        raise RuntimeError("bootstrap date verified")

    monkeypatch.setattr(live_inference, "ensure_latest_contract_file", fake_bootstrap)

    try:
        live_inference.run_live_session(config, session_date=expected)
    except RuntimeError as error:
        assert str(error) == "bootstrap date verified"
    else:
        raise AssertionError("run_live_session did not invoke contract bootstrap")


def test_previous_day_logs_are_archived_and_current_day_logs_are_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(ivrv_paths, "BASE_DIR", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    stale = logs / "live_runtime.log"
    current = logs / "live_predictions.jsonl"
    stale.write_text("yesterday", encoding="utf-8")
    current.write_text("today", encoding="utf-8")
    timezone = ZoneInfo("Asia/Kolkata")
    old_stamp = datetime(2026, 10, 21, 12, 0, tzinfo=timezone).timestamp()
    current_stamp = datetime(2026, 10, 22, 8, 0, tzinfo=timezone).timestamp()
    os.utime(stale, (old_stamp, old_stamp))
    os.utime(current, (current_stamp, current_stamp))

    archived = archive_previous_day_logs({}, datetime(2026, 10, 22).date())

    assert archived == [logs / "prevDayLogs" / "2026-10-21" / "live_runtime.log"]
    assert archived[0].read_text(encoding="utf-8") == "yesterday"
    assert not stale.exists()
    assert current.read_text(encoding="utf-8") == "today"


def test_historical_mode_filters_date_and_writes_predictions(tmp_path, monkeypatch):
    source = tmp_path / "features.csv"
    pd.DataFrame({
        "datetime": ["2026-10-21 15:30:00", "2026-10-22 15:30:00"],
        "feature": [1.0, 2.0],
    }).to_csv(source, index=False)

    class FakePipeline:
        def __init__(self, _config):
            pass

        def predict(self, combined, legs, rv):
            assert legs is None and rv is None
            assert combined["feature"].tolist() == [2.0]
            return pd.DataFrame({"prediction": [42.0]})

    monkeypatch.setattr(live_inference, "FixedStrikeInferencePipeline", FakePipeline)
    destination = tmp_path / "results" / "predictions.csv"
    result = run_historical_inference(
        {"inference_config": str(tmp_path / "config.yaml")},
        source,
        output_path=destination,
        trading_date=datetime(2026, 10, 22).date(),
    )

    assert result == destination
    assert pd.read_csv(destination)["prediction"].tolist() == [42.0]


def test_historical_udp_replay_overrides_ports_and_isolates_logs():
    timezone = ZoneInfo("Asia/Kolkata")
    replay = historical_replay_config(
        {"port": 34331, "cm_port": 8223},
        35001,
        cm_port=35002,
        started_at=datetime(2026, 10, 22, 16, 15, 30, tzinfo=timezone),
    )

    assert replay["port"] == 35001
    assert replay["cm_port"] == 35002
    assert replay["runtime_mode"] == "historical_udp_replay"
    assert replay["runtime_log"] == "logs/historicalReplay/2026-10-22/live_runtime_161530.log"
    assert replay["prediction_log"] == "logs/historicalReplay/2026-10-22/live_predictions_161530.jsonl"
    assert replay["fixed_strike_path_log"] == "logs/historicalReplay/2026-10-22/fixed_strike_120m_paths_161530.jsonl"


def test_fixed_strike_path_saves_the_entry_atm_without_reselecting_at_exit():
    engine = live_inference.LiveFeatureInference.__new__(live_inference.LiveFeatureInference)
    engine.config = {
        "fixed_strike_path_horizon_minutes": 120,
        "paper_profit_target_inr": 1500.0,
        "paper_stop_loss_inr": -750.0,
        "option_oi_max_age_minutes": 15,
    }
    engine.expiry = date(2026, 10, 29)
    engine.lot_size = 30
    engine.rate = 0.065
    engine.rows_by_minute = {
        100: {
            "entry_strike": 50000.0,
            "spot_price": 50020.0,
            "future_open": 50040.0,
            "call_price": 200.0,
            "put_price": 210.0,
            "entry_straddle_price": 410.0,
            "entry_iv_call": 0.20,
            "entry_iv_put": 0.21,
            "entry_atm_iv_combined": 0.205,
            "quality_entry_ok": True,
            "entry_no120_event_score": 0.01,
            "entry_no120_sharp_rv_decline": True,
            "entry_no120_iv_rv_gate_open": True,
            "entry_short_straddle_signal": True,
        }
    }
    engine.pairs = {50000.0: {"CE": 1, "PE": 2}}
    engine.option_bars = {
        1: {220: live_inference.MinuteBar(170.0, 170.0, 170.0, 170.0)},
        2: {220: live_inference.MinuteBar(180.0, 180.0, 180.0, 180.0)},
    }
    engine.oi_by_token = {1: (1000, 220 * 60, 7202), 2: (1100, 220 * 60, 7202)}
    engine.implied_volatility = lambda *_args: 0.19
    engine.fixed_strike_output = io.StringIO()

    engine._save_fixed_strike_path_observations(220, 50100.0, 50120.0)
    records = [json.loads(line) for line in engine.fixed_strike_output.getvalue().splitlines()]

    assert len(records) == 1
    record = records[0]
    assert record["entry_strike"] == 50000.0
    assert record["entry_short_straddle_signal"] is True
    assert record["holding_minutes"] == 120
    assert record["is_horizon_exit"] is True
    assert record["observation_straddle_price"] == 350.0
    assert record["short_pnl_per_lot"] == 1800.0
    assert record["profit_target_hit_at_observation"] is True


def test_paper_engine_allows_overlaps_and_enforces_all_exit_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(ivrv_paths, "BASE_DIR", tmp_path)

    def exchange_second(value: str) -> int:
        return int((datetime.fromisoformat(value) - live_inference.NSE_EPOCH).total_seconds())

    scenarios = [
        ("profit_target", "2026-10-22 10:00:00", "2026-10-22 10:05:00", 170.0, 170.0),
        ("stop_loss", "2026-10-22 11:00:00", "2026-10-22 11:05:00", 215.0, 215.0),
        ("time_exit_120m", "2026-10-22 12:00:00", "2026-10-22 14:00:00", 200.0, 200.0),
        ("session_square_off_151430", "2026-10-22 15:00:00", "2026-10-22 15:14:30", 195.0, 195.0),
    ]
    for index, (reason, entry_time, exit_time, exit_call, exit_put) in enumerate(scenarios):
        engine = PaperTradingEngine({
            "paper_trade_events_log": f"events_{index}.jsonl",
            "paper_position_state": f"state_{index}.json",
            "paper_square_off_time": "15:14:30",
            "fixed_strike_path_horizon_minutes": 120,
            "paper_profit_target_inr": 1500.0,
            "paper_stop_loss_inr": -750.0,
        }, 30)
        entered = engine.try_enter(
            exchange_second=exchange_second(entry_time),
            signal_minute=exchange_second(entry_time) // 60,
            expiry=date(2026, 10, 29),
            strike=50000.0,
            call_token=1,
            put_token=2,
            call_price=200.0,
            put_price=200.0,
            triggered_rules=["test_rule"],
            convergence_probability=0.95,
        )
        assert entered is True
        position_id = next(iter(engine.positions))
        event = engine.mark(position_id, exchange_second(exit_time), exit_call, exit_put)
        assert event is not None
        assert event["exit_reason"] == reason
        assert not engine.positions
        engine.close()

    overlap = PaperTradingEngine({
        "paper_trade_events_log": "overlap_events.jsonl",
        "paper_position_state": "overlap_state.json",
        "paper_square_off_time": "15:14:30",
        "fixed_strike_path_horizon_minutes": 120,
        "paper_profit_target_inr": 1500.0,
        "paper_stop_loss_inr": -750.0,
        "paper_max_open_positions": 3,
        "paper_entry_convergence_probability_threshold": 0.90,
    }, 30)
    threshold_second = exchange_second("2026-10-22 09:59:00")
    assert overlap.try_enter(
        exchange_second=threshold_second,
        signal_minute=threshold_second // 60,
        expiry=date(2026, 10, 29),
        strike=49900.0,
        call_token=99,
        put_token=100,
        call_price=200.0,
        put_price=200.0,
        triggered_rules=["threshold_must_be_strict"],
        convergence_probability=0.90,
    ) is False
    for minute_offset, strike in ((0, 50000.0), (1, 50100.0), (2, 50200.0)):
        entry_second = exchange_second("2026-10-22 10:00:00") + minute_offset * 60
        assert overlap.try_enter(
            exchange_second=entry_second,
            signal_minute=entry_second // 60,
            expiry=date(2026, 10, 29),
            strike=strike,
            call_token=1 + minute_offset * 2,
            put_token=2 + minute_offset * 2,
            call_price=200.0,
            put_price=200.0,
            triggered_rules=["overlap"],
            convergence_probability=0.95,
        ) is True
    rejected_second = exchange_second("2026-10-22 10:03:00")
    assert overlap.try_enter(
        exchange_second=rejected_second,
        signal_minute=rejected_second // 60,
        expiry=date(2026, 10, 29),
        strike=50300.0,
        call_token=7,
        put_token=8,
        call_price=200.0,
        put_price=200.0,
        triggered_rules=["overlap"],
        convergence_probability=0.96,
    ) is False
    snapshot = overlap.snapshot()
    assert snapshot["overlapping_positions_allowed"] is True
    assert snapshot["maximum_open_positions"] == 3
    assert snapshot["available_position_slots"] == 0
    assert snapshot["open_position_count"] == 3
    assert len(snapshot["open_positions"]) == 3
    overlap.close()

    routed = PaperTradingEngine({
        "paper_trade_events_log": "routed_events.jsonl",
        "paper_position_state": "routed_state.json",
        "paper_square_off_time": "15:14:30",
        "fixed_strike_path_horizon_minutes": 120,
        "paper_profit_target_inr": 1500.0,
        "paper_stop_loss_inr": -750.0,
        "paper_entry_convergence_probability_threshold": 0.75,
    }, 30)
    routed_second = exchange_second("2026-10-22 10:30:00")
    assert routed.try_enter(
        exchange_second=routed_second,
        signal_minute=routed_second // 60,
        expiry=date(2026, 10, 29),
        strike=50000.0,
        call_token=1,
        put_token=2,
        call_price=200.0,
        put_price=200.0,
        triggered_rules=["iv_near_rv_sharp_rv_decline"],
        convergence_probability=0.10,
        entry_route="iv_near_rv_sharp_rv_decline",
        entry_model_score=0.0,
    ) is True
    routed.close()


def test_long_paper_book_uses_opposite_pnl_and_separate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ivrv_paths, "BASE_DIR", tmp_path)
    config = {"paper_trade_events_log": "replay/trades.jsonl",
              "paper_position_state": "replay/state.json"}
    short = PaperTradingEngine(config, 30)
    long = PaperTradingEngine(config, 30, "LONG_STRADDLE")
    second = int((datetime(2026, 10, 22, 10, 0) - live_inference.NSE_EPOCH).total_seconds())
    for engine in (short, long):
        assert engine.try_enter(
            exchange_second=second, signal_minute=second // 60,
            expiry=date(2026, 10, 29), strike=50000.0,
            call_token=1, put_token=2, call_price=200.0, put_price=200.0,
            triggered_rules=["iv_above_rv_rising_rv_convergence"],
            convergence_probability=0.9,
            entry_route="iv_above_rv_rising_rv_convergence" if engine is long else "iv_above_rv_convergence",
        )
    later = second + 60
    short_exit = short.mark(next(iter(short.positions)), later, 230.0, 230.0)
    long_exit = long.mark(next(iter(long.positions)), later, 230.0, 230.0)
    assert short_exit["realized_pnl_inr"] == -1800.0
    assert long_exit["realized_pnl_inr"] == 1800.0
    assert long_exit["side"] == "LONG_STRADDLE"
    assert short.state_path != long.state_path
    assert short.events_path != long.events_path
    short.close()
    long.close()


def test_rising_rv_signal_opens_long_paper_position(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(ivrv_paths, "BASE_DIR", tmp_path)
    engine = live_inference.LiveFeatureInference.__new__(live_inference.LiveFeatureInference)
    engine.long_paper_engine = PaperTradingEngine({"paper_trade_events_log": "trades.jsonl",
                                                  "paper_position_state": "state.json"}, 30, "LONG_STRADDLE")
    engine.pipeline = SimpleNamespace(config=SimpleNamespace(rising_rv_probability_threshold=0.5))
    engine.expiry = date(2026, 10, 29)
    engine.pairs = {50000.0: {"CE": 1, "PE": 2}}
    engine.latest_exchange_second = int((datetime(2026, 10, 22, 10, 1) - live_inference.NSE_EPOCH).total_seconds())
    engine._fresh_paper_prices = lambda *args, **kwargs: (200.0, 200.0)
    minute = engine.latest_exchange_second // 60 - 1
    row = {"entry_strike": 50000.0, "quality_entry_ok": True}
    prediction = {"pred_rising_rv_convergence_probability": 0.8,
                  "pred_rising_rv_convergence_signal": True}
    # Disabled by default: a qualifying signal is logged but opens nothing.
    engine.config = {}
    decision = engine._try_open_long_paper_position(minute, row, prediction)
    assert decision["entered"] is False
    assert decision["reason"] == "long_entries_disabled"
    assert not engine.long_paper_engine.positions
    engine.config = {"paper_long_entries_enabled": True}
    decision = engine._try_open_long_paper_position(minute, row, prediction)
    assert decision["entered"] is True
    assert len(engine.long_paper_engine.positions) == 1
    assert next(iter(engine.long_paper_engine.positions.values()))["side"] == "LONG_STRADDLE"
    engine.long_paper_engine.close()


def test_machine_local_accepts_only_machine_settings(tmp_path):
    import pytest
    from voltr.core.machine import machine_local_overrides
    config = tmp_path / "live_config.json"
    assert machine_local_overrides(config) == {}
    machine = tmp_path / "machine.local.json"
    machine.write_text(json.dumps({"interface_ip": "10.0.0.5", "dashboard_port": 8791}), encoding="utf-8")
    assert machine_local_overrides(config) == {"interface_ip": "10.0.0.5", "dashboard_port": 8791}
    for text in ('{"paper_entry_convergence_probability_threshold": 0.5}', '["interface_ip"]'):
        machine.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="machine.local.json"):
            machine_local_overrides(config)


def test_shipped_live_config_disables_long_entries(monkeypatch):
    monkeypatch.setenv("VOLTR_FO_INTERFACE", "10.0.0.12")
    monkeypatch.setenv("VOLTR_CM_INTERFACE", "10.0.0.13")
    config = load_live_config(CONFIG)
    assert config["paper_long_entries_enabled"] is False


def test_live_validation_excludes_features_created_inside_pipeline():
    artifacts = {
        "rv_xgboost": {"features": ["spot_return_60m", "entry_iv_weighted"]},
        "combined_direct_lightgbm": {
            "features": [
                "entry_atm_iv_combined",
                "rv_pred_120m",
                "entry_gap_to_rv_pred_combined",
            ]
        },
        "rv_iv_sharp_decline_event": {
            "features": [
                "entry_atm_iv_combined",
                "rv_trailing_60m",
                "engineered_iv_minus_rv_60m",
                "engineered_log_entry_iv",
            ]
        },
    }

    assert required_live_input_features(artifacts) == [
        "entry_atm_iv_combined",
        "entry_iv_weighted",
        "rv_trailing_60m",
        "spot_return_60m",
    ]
