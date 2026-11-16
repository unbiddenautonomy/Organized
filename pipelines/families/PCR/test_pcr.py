from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import threading
import subprocess
import time
from types import SimpleNamespace
import urllib.request

import numpy as np
import pandas as pd
import pytest

from PCR import features, web
from PCR.runtime import Book, Models, PCRRuntime, costs, load_config, ROOT
from PCR.storage import read_lines


def context():
    return {"expiry": "2026-11-24", "lot_size": 30, "future_token": 99,
            "risk_free_rate": .065, "pairs": {"50000.0": {"CE": 1, "PE": 2}}, "mode": "recorded_replay",
            "cost_rates": {"stt_sell_premium_rate": .0015, "nse_options_transaction_rate": .0003553,
                           "sebi_turnover_rate": .000001, "stamp_buy_premium_rate": .00003, "gst_rate": .18}}


def sig(key):
    return {"id": key, "outputs": {"win_probability_60m": .9}, "thresholds": {},
            "feature_timestamp": "2026-10-28T10:14:00", "expiry_group": "3_4"}


def test_three_concurrent_positions_slot_released_only_after_exit_and_restart(tmp_path):
    config, ctx = load_config(), context()
    b = Book(tmp_path, config, ctx)
    t = datetime(2026, 10, 28, 10, 16)
    for i in range(3):
        assert b.enter(t + timedelta(minutes=i), 50000 + i*100, {"CE": i*2+1, "PE": i*2+2}, [500, 500], 120, sig(f"s{i}"))
    assert not b.enter(t + timedelta(minutes=3), 51000, {"CE": 9, "PE": 10}, [500, 500], 120, sig("s3"))
    b = Book(tmp_path, config, ctx)
    assert len(b.positions) == 3
    b.mark(t + timedelta(minutes=4), lambda tokens, now: [530, 530] if tokens[0] == 1 else [500, 500])
    assert len(b.positions) == 2 and b.closed[0]["exit_reason"] == "STOP"
    assert b.closed[0]["net_rupees"] < -1500
    assert b.enter(t + timedelta(minutes=5), 51000, {"CE": 9, "PE": 10}, [500, 500], 120, sig("s3"))
    assert len(b.positions) == 3
    assert b.closed[0]["exit_strike"] == b.closed[0]["entry_strike"] == 50000
    assert b.closed[0]["brokerage_rupees"] == 0
    assert len(Book(tmp_path, config, ctx).positions) == 3


def test_target_uses_net_after_costs_and_slippage(tmp_path):
    b = Book(tmp_path, load_config(), context())
    t = datetime(2026, 10, 28, 10, 16)
    assert b.enter(t, 50000, {"CE": 1, "PE": 2}, [500, 500], 120, sig("s1"))
    b.mark(t + timedelta(minutes=1), lambda *_: [450, 450])  # Gross exactly 3000, net below target.
    assert len(b.positions) == 1
    b.mark(t + timedelta(minutes=2), lambda *_: [440, 440])
    assert len(b.closed) == 1 and b.closed[0]["exit_reason"] == "TARGET"
    p = b.closed[0]
    assert p["net_rupees"] == pytest.approx(p["gross_rupees"] - p["total_cost_rupees"])
    assert p["slippage_rupees"] == 60


def test_open_and_close_each_alert_once(tmp_path,monkeypatch):
    from PCR import position_alert
    alerts=[]
    monkeypatch.setattr(position_alert,'notify_position_action',lambda:alerts.append('action'))
    book=Book(tmp_path,load_config(),context())
    now=datetime(2026,10,28,10,16)
    assert book.enter(now,50000,{"CE":1,"PE":2},[500,500],120,sig('alert-test'))
    book.mark(now+timedelta(minutes=2),lambda *_:[440,440])
    assert alerts==['action','action']


def test_cutoff_missing_quote_keeps_position_and_marks_pending(tmp_path):
    b = Book(tmp_path, load_config(), context())
    assert b.enter(datetime(2026,10,28,13), 50000, {"CE":1,"PE":2}, [500,500], 120, sig("cut"))
    b.mark(datetime(2026,10,28,15,14,30), lambda *_: None)
    assert b.positions["cut"]["pending_exit"] == "CUTOFF"
    b.mark(datetime(2026,10,28,15,14,31), lambda *_: [500,500])
    assert b.closed[0]["exit_reason"] == "CUTOFF"
    assert b.closed[0]["exit_datetime"].endswith("15:14:31")
    assert not b.enter(datetime(2026,10,28,13,0,1), 50000,{"CE":1,"PE":2}, [500,500],60,sig("late"))


def raw_rows():
    rows = []
    for i in range(100):
        stamp = pd.Timestamp("2026-10-28 09:15") + pd.Timedelta(minutes=i)
        rows.append({"datetime": stamp, "expiry": pd.Timestamp("2026-11-24"), "strike": 50000., "atm_strike":50000.,
                     "days_to_expiry":27.-i/1440, "future_open":50020.+i, "spot_price":50000.+i,
                     "call_price":500.+i*.1,"put_price":510.-i*.3,"call_oi":1000.+i*2,"put_oi":1200.+i,
                     "Future_oi":9000.,"iv_call":.2,"iv_put":.21,"theta_call":-22.,"theta_put":-20.,
                     "gamma_call":.0002,"gamma_put":.0003,"vega_call":40.,"vega_put":41.,
                     "delta_call":.5,"delta_put":-.5})
    return rows


def test_feature_parity_with_training_and_future_poison():
    sys.path.insert(0, str(ROOT / "research"))
    import train_win_classifier as training
    rows = raw_rows()
    names = list(training.FEATURES) + [f"expiry_week_{i}" for i in range(5)]
    expected = training.build_atm_panel(pd.DataFrame(rows))
    stamp = pd.Timestamp("2026-10-28 10:30")
    actual, group, when = features.feature_at(rows, stamp, 27., names)
    for f in training.FEATURES:
        assert actual[f] == pytest.approx(expected.loc[stamp-pd.Timedelta(minutes=1), f], nan_ok=True), f
    contaminated = [dict(r) for r in rows]
    for r in contaminated:
        if r["datetime"] >= stamp:
            for k,v in list(r.items()):
                if isinstance(v,(float,int)):
                    r[k] = 1e20
    poisoned, _, _ = features.feature_at(contaminated, stamp, 27., names)
    assert poisoned == actual
    assert group == "3_4" and when == "2026-10-28T10:29:00"


class FakeModels:
    def __init__(self, names):
        self.names = names
        self.thresholds = {g: {"minimum_expected_points":0.,"minimum_win_probability":.5,"maximum_large_loss_probability":.15} for g in ("0","1","2","3_4")}
        self.bundles = {}
    def predict(self, _):
        return {"decay_60_points":20., "win_probability_60m":.9, "holding_60_points":20.,"holding_120_points":30.,"tail_loss_probability":.01}


def test_runtime_full_signals_max3_and_user_win_cutoff(tmp_path):
    names = Models().names
    config = load_config()
    rt = PCRRuntime(context(), tmp_path, models=FakeModels(names), config=config)
    start = datetime(2026,10,28,9,15)
    for i in range(70):
        t = start + timedelta(minutes=i)
        stamp = t.isoformat()
        event = {"timestamp":stamp,"spot":[50000.,stamp],"future":[50020.,stamp],
                 "future_open":{stamp:50020.},"quotes":{"1":[500.,stamp],"2":[500.,stamp]},
                 "oi":{"1":[1000,stamp],"2":[1200,stamp]},"market_update":True}
        rt.on_snapshot(event)
    assert len(rt.book.positions) == 3
    events = read_lines(tmp_path / "2026-10-28" / "signals.jsonl")
    assert sum(e["entered"] for e in events) == 3
    assert any(e["reason"] == "capacity_full" for e in events)
    assert all(e["feature_timestamp"] < e["timestamp"] for e in events if e["outputs"])
    for p in rt.book.positions.values():
        assert p["entry_strike"] == 50000 and p["horizon_minutes"] == 120
    rt.config["win_probability_thresholds"]["3_4"] = .95
    stamp = (start+timedelta(minutes=70)).isoformat()
    event.update({"timestamp":stamp,"spot":[50000.,stamp],"future":[50020.,stamp],
                  "quotes":{"1":[500.,stamp],"2":[500.,stamp]},"oi":{"1":[1000,stamp],"2":[1200,stamp]}})
    rt.on_snapshot(event)
    assert rt.latest_signal["gates"]["win"] is False
    assert rt.latest_signal["reason"] == "model_gate_rejected"


def test_copied_model_hashes_and_prediction_parity():
    manifest = json.loads((ROOT / "model_files.sha256.json").read_text())
    for relative, expected in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    model = Models()
    candidates = pd.read_csv(ROOT / "reference" / "all_test_candidates.csv.gz").iloc[::800]
    mapping = {"decay_60_points":"predicted_decay_60_points", "holding_60_points":"predicted_pnl_60_points",
               "holding_120_points":"predicted_pnl_120_points", "win_probability_60m":"win_probability",
               "tail_loss_probability":"large_loss_probability"}
    for _, row in candidates.iterrows():
        output = model.predict(row[model.names].to_dict())
        for key, column in mapping.items():
            assert output[key] == pytest.approx(row[column], abs=1e-10)


def test_original_dashboard_routes_and_new_pcr_route_coexist(tmp_path):
    app = ROOT.parent / "ivrv" / "engine" / "Inference"
    sys.path.insert(0, str(app))
    from live.live_inference import DashboardServer
    class Dummy:
        def status(self): return {"original_strategy":"unchanged"}
        def recent(self, limit): return []
    html = tmp_path / "old.html"
    html.write_text("original-dashboard")
    server = DashboardServer("127.0.0.1", 0, html, Dummy())
    server.start()
    prefix = f"http://127.0.0.1:{server.server.server_address[1]}"
    try:
        assert urllib.request.urlopen(prefix+"/").read() == b"original-dashboard"
        assert json.load(urllib.request.urlopen(prefix+"/api/status"))["original_strategy"] == "unchanged"
        assert b"MAX 3 OPEN" in urllib.request.urlopen(prefix+"/PCR").read()
        pcr = json.load(urllib.request.urlopen(prefix+"/PCR/api/status"))
        assert pcr["desired_config"]["maximum_open_positions"] == 3
        request = urllib.request.Request(prefix+"/PCR/api/thresholds", data=b'{}', method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request)
        assert err.value.code == 403
    finally:
        server.close()


def test_original_strategy_configuration_unchanged():
    # IVRV's strategy configuration must not change by accident. The baseline
    # was re-taken on 2026-11-16 after deliberate changes; the 2026-10-28
    # install-time hashes remain in integration_backup/hashes_before.json.
    # Paths are relative to pipelines/families. Line endings are normalised
    # so a Git checkout with CRLF conversion hashes the same.
    baseline = json.loads((ROOT / "integration_backup" / "hashes_current.json").read_text())
    for name, expected in baseline.items():
        content = (ROOT.parent / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == expected, name


def test_worker_lock_prevents_duplicate_ledger_owner(tmp_path):
    from PCR.worker_lock import single_writer
    with single_writer(tmp_path):
        with pytest.raises(RuntimeError, match="Another PCR worker"):
            with single_writer(tmp_path):
                pytest.fail("duplicate ledger owner admitted")
    with single_writer(tmp_path):
        pass  # OS lock releases on a normal stop and can be acquired on restart.


def test_recorded_day_restores_unresolved_carry_in(tmp_path):
    names = Models().names
    cfg, ctx = load_config(), dict(context(), mode="live")
    original = PCRRuntime(ctx, tmp_path / "source", models=FakeModels(names), config=cfg)
    assert original.book.enter(datetime(2026,10,27,12),50000,{"CE":1,"PE":2},[500,500],120,sig("carry"))
    event = {"timestamp":"2026-10-28T09:15:00", "market_update":False}
    original.on_snapshot(event)
    saved = json.loads((tmp_path/"source"/"2026-10-28"/"session_context.json").read_text())
    replay = PCRRuntime(dict(saved, mode="recorded_replay", source_mode="live"),tmp_path/"replay",
                        models=FakeModels(names),config=cfg)
    replay.on_snapshot(event)
    assert list(replay.book.positions) == ["carry"]
    assert replay.book.positions["carry"]["pending_exit"] == "CUTOFF"
    assert replay.book.status() == original.book.status()


def test_late_cross_stream_observations_preserve_asof_times():
    from PCR.bridge import Bridge, NSE_EPOCH
    b = Bridge.__new__(Bridge)
    b.enabled, b.process, b.closed = True, object(), False
    b.second, b.batch, b.future_minute = 100, None, None
    b.future_token, b.price_divisor = 99, 100
    b.contracts, b.oi_versions = {1:None,2:None}, {}
    b.source = {"decode_banknifty_7207":lambda msg,divisor:msg,
                "decode_oi_message":lambda msg:msg, "OiDecodeError":ValueError}
    b.index_message((99,50000))
    b.market(SimpleNamespace(log_time=98,token=1,last_traded_price=500))
    b.index_message((98,49900))
    b.market(SimpleNamespace(log_time=97,token=1,last_traded_price=490))
    b.oi_message([SimpleNamespace(log_time=97,token=1,open_interest=1000,transaction_code=17202)])
    b.oi_message([SimpleNamespace(log_time=97,token=1,open_interest=999,transaction_code=7201)])
    assert b.batch["timestamp"] == (NSE_EPOCH+timedelta(seconds=100)).isoformat()
    assert b.batch["spot"][1] == (NSE_EPOCH+timedelta(seconds=99)).isoformat()
    assert b.batch["quotes"]["1"][1] == (NSE_EPOCH+timedelta(seconds=98)).isoformat()
    assert b.batch["quotes"]["1"][0] == 500
    assert b.batch["oi"]["1"] == [1000,(NSE_EPOCH+timedelta(seconds=97)).isoformat()]


def test_replay_preserves_live_latency_rejections(tmp_path):
    names = Models().names
    ctx = dict(context(), mode="live")
    live = PCRRuntime(ctx, tmp_path / "source", models=FakeModels(names), config=load_config())
    for i in range(70):
        stamp = (datetime(2026,10,28,9,15)+timedelta(minutes=i)).isoformat()
        event = {"timestamp":stamp,"spot":[50000.,stamp],"future":[50020.,stamp],
                 "future_open":{stamp:50020.},"quotes":{"1":[500.,stamp],"2":[500.,stamp]},
                 "oi":{"1":[1000,stamp],"2":[1200,stamp]},"market_update":True}
        live.on_snapshot(event, processing_delay=30. if i % 2 else 0.)
    replay = PCRRuntime(dict(ctx, mode="recorded_replay", source_mode="live"), tmp_path / "replay",
                        models=FakeModels(names), config=load_config())
    for event in read_lines(tmp_path / "source" / "2026-10-28" / "inputs.jsonl"):
        replay.on_snapshot(event, processing_delay=1e9)
    original = read_lines(tmp_path / "source" / "2026-10-28" / "signals.jsonl")
    replayed = read_lines(tmp_path / "replay" / "2026-10-28" / "signals.jsonl")
    assert any(s["reason"] == "processing_delay_or_clock_skew" for s in original)
    assert original == replayed
    assert live.book.status() == replay.book.status()


def test_feed_bridge_real_worker_and_recorded_replay(tmp_path, monkeypatch):
    from PCR import bridge
    from PCR.storage import atomic_json
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    # Worker module and real models resolve from this installed package; all
    # resulting ledgers and bridge status stay inside the pytest sandbox.
    atomic_json(tmp_path / "config.json", load_config())
    ctx = context()
    config = dict(ctx["cost_rates"], runtime_mode="historical_udp_replay", price_divisor=100,
                  risk_free_rate=.065, contract_asof_date="2026-10-28")
    contracts = {i:SimpleNamespace(token=i, strike_price=50000., option_type=side)
                 for i,side in ((1,"CE"),(2,"PE"))}
    source = {"decode_banknifty_7207":lambda message, divisor:message,
              "decode_oi_message":lambda message:message, "OiDecodeError":ValueError}
    b = bridge.Bridge(config, contracts,30,datetime(2026,11,24).date(),99,source)
    # Popen otherwise uses the normal installation's parent cwd; keep it valid
    # after isolating the bridge's output root.
    original_popen = subprocess.Popen
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *a, **kw: original_popen(*a, **dict(kw, cwd=str(ROOT.parent))))
    b.start()
    try:
        for i in range(70):
            now = datetime(2026,10,28,9,15)+timedelta(minutes=i)
            sec = int((now-bridge.NSE_EPOCH).total_seconds())
            b.index_message((sec,50000.))
            for token, price in ((99,50020.),(1,500.),(2,500.)):
                b.market(SimpleNamespace(log_time=sec,token=token,last_traded_price=price))
            b.oi_message([SimpleNamespace(log_time=sec,token=t,open_interest=1000+t*100) for t in (1,2,99)])
        b._flush()
        deadline = time.monotonic()+30
        state = {}
        while time.monotonic() < deadline:
            status_file = b.directory / "status.json"
            if status_file.exists():
                state = web.read_json(status_file, {})
                if state.get("updated_at") == now.isoformat():
                    break
            if b.process.poll() is not None:
                pytest.fail((b.directory / "worker.log").read_text())
            time.sleep(.1)
        assert state.get("updated_at") == now.isoformat()
        assert state["latest_signal"]["outputs"] is not None
        assert state["fault"] is None
    finally:
        b.close()
    day = b.directory / "2026-10-28"
    replay = tmp_path / "recorded_replay"
    result = subprocess.run([sys.executable,"-m","PCR.worker","--context",str(day/"session_context.json"),
                             "--inputs",str(day/"inputs.jsonl"),"--output",str(replay)],
                            cwd=ROOT.parent,capture_output=True,text=True,timeout=45)
    assert result.returncode == 0, result.stderr[-4000:]
    for kind in ("signals", "trades", "marks", "minute_rows"):
        assert read_lines(day/f"{kind}.jsonl") == read_lines(replay/"2026-10-28"/f"{kind}.jsonl"), kind
