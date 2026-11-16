"""Capacity replacement rules use real paper ledger entries and fresh quotes."""
from datetime import date, datetime
import json

import pytest

from live import live_inference as live


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setattr(live, 'BASE_DIR', tmp_path)
    monkeypatch.setenv('POSITION_ALERT_DISABLED', '1')
    engine = live.LiveFeatureInference.__new__(live.LiveFeatureInference)
    engine.paper_engine = live.PaperTradingEngine({
        'paper_trade_events_log': 'events.jsonl',
        'paper_position_state': 'positions.json',
        'paper_entry_convergence_probability_threshold': .70,
        'paper_replacement_min_confidence_improvement': .10,
        'paper_replacement_cooldown_seconds': 600,
        'paper_max_replacements_per_day': 3,
        'paper_max_open_positions': 3,
        'paper_profit_target_inr': 1_000_000,
        'paper_stop_loss_inr': -1_000_000,
    }, 30)
    engine.config = {'paper_price_max_age_seconds': 60}
    engine.expiry = date(2026, 11, 23)
    now = int((datetime(2026, 11, 3, 11, 0) - live.NSE_EPOCH).total_seconds())
    engine.latest_exchange_second = now
    engine.pairs = {float(50000 + i * 100): {'CE': i * 2 + 1, 'PE': i * 2 + 2} for i in range(5)}
    engine.latest_option_ticks = {token: (100., now) for token in range(1, 11)}
    yield engine, now, tmp_path
    engine.paper_engine.close()


def add_position(engine, now, index, confidence):
    strike = float(50000 + index * 100)
    pair = engine.pairs[strike]
    assert engine.paper_engine.try_enter(
        exchange_second=now - (4 - index) * 60,
        signal_minute=now // 60 - (4 - index),
        expiry=engine.expiry, strike=strike,
        call_token=pair['CE'], put_token=pair['PE'],
        call_price=100., put_price=100., triggered_rules=['test'],
        convergence_probability=confidence,
        entry_route='iv_above_rv_convergence', entry_model_score=confidence,
    )


def signal(engine, now, confidence):
    return live.LiveFeatureInference._try_open_paper_position(
        engine, now // 60, {'entry_strike': 50400.}, {
            'pred_convergence_probability_classifier': confidence,
            'pred_no120_gate_iv_minus_rv': .006,
            'pred_strategy_rule_iv_above_rv_iv_declines_to_rv': True,
        })


@pytest.mark.parametrize('holdings,incoming,replaced', [
    ((.71, .78, .82), .82, .71),
    ((.72, .75, .80), .85, .72),
    ((.75, .80, .87), .95, .75),
])
def test_stronger_signal_replaces_lowest_eligible_confidence(stack, holdings, incoming, replaced):
    engine, now, folder = stack
    for index, confidence in enumerate(holdings):
        add_position(engine, now, index, confidence)
    decision = signal(engine, now, incoming)
    assert decision['entered'] and decision['replaced_position_confidence'] == replaced
    assert len(engine.paper_engine.positions) == 3
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    exits = [event for event in events if event['event_type'] == 'paper_trade_exit']
    assert len(exits) == 1
    assert exits[0]['exit_reason'] == 'higher_confidence_signal'
    assert exits[0]['replacement_signal_confidence'] == incoming


@pytest.mark.parametrize('holdings,incoming', [
    ((.71, .75, .80), .80),
    ((.81, .82, .90), .85),
    ((.71, .72, .79), .70),
])
def test_non_matching_signal_keeps_open_positions(stack, holdings, incoming):
    engine, now, folder = stack
    for index, confidence in enumerate(holdings):
        add_position(engine, now, index, confidence)
    decision = signal(engine, now, incoming)
    assert not decision['entered']
    assert len(engine.paper_engine.positions) == 3
    assert not any(json.loads(line)['event_type'] == 'paper_trade_exit'
                   for line in (folder / 'events.jsonl').read_text().splitlines())


def test_replacement_requires_fresh_exit_quotes(stack):
    engine, now, _ = stack
    for index, confidence in enumerate((.71, .75, .80)):
        add_position(engine, now, index, confidence)
    engine.latest_option_ticks[1] = (100., now - 61)
    decision = signal(engine, now, .85)
    assert not decision['entered']
    assert decision['reason'] == 'replacement_exit_prices_stale_or_missing'
    assert len(engine.paper_engine.positions) == 3


def test_available_slot_opens_without_forced_exit(stack):
    engine, now, folder = stack
    add_position(engine, now, 0, .72)
    decision = signal(engine, now, .75)
    assert decision['entered'] and 'replaced_position_id' not in decision
    assert len(engine.paper_engine.positions) == 2
    assert not any(json.loads(line)['event_type'] == 'paper_trade_exit'
                   for line in (folder / 'events.jsonl').read_text().splitlines())


def test_replacement_cooldown_and_daily_limit_survive_restart(stack):
    engine, now, folder = stack
    for index, confidence in enumerate((.71, .73, .75)):
        add_position(engine, now, index, confidence)
    assert signal(engine, now, .85)['entered']
    assert engine.paper_engine.replacement_history_seconds == [now]
    engine.latest_option_ticks = {token: (100., now + 60) for token in range(1, 11)}
    engine.latest_exchange_second = now + 60
    blocked = signal(engine, now + 60, .95)
    assert blocked['reason'] == 'replacement_cooldown_active'

    config = {
        'paper_trade_events_log': 'events.jsonl',
        'paper_position_state': 'positions.json',
        'paper_entry_convergence_probability_threshold': .70,
    }
    restored = live.PaperTradingEngine(config, 30)
    try:
        assert restored.replacement_history_seconds == [now]
        restored.replacement_history_seconds = [now - 1800, now - 1200, now]
        candidate, reason = restored.replacement_candidate(.95, 'iv_above_rv_convergence', now + 600)
        assert candidate is None and reason == 'daily_replacement_limit_reached'
        candidate, reason = restored.replacement_candidate(.95, 'iv_above_rv_convergence', now + 86400)
        assert candidate is not None and reason == 'replacement_allowed'
    finally:
        restored.close()


def test_replacement_requires_same_route_and_ten_point_improvement(stack):
    engine, now, _ = stack
    for index, confidence in enumerate((.71, .73, .75)):
        add_position(engine, now, index, confidence)
    candidate, reason = engine.paper_engine.replacement_candidate(.85, 'iv_near_rv_sharp_rv_decline', now)
    assert candidate is None and reason == 'no_same_route_position_with_sufficient_improvement'
    candidate, reason = engine.paper_engine.replacement_candidate(.80, 'iv_above_rv_convergence', now)
    assert candidate is None and reason == 'no_same_route_position_with_sufficient_improvement'
    candidate, reason = engine.paper_engine.replacement_candidate(.81, 'iv_above_rv_convergence', now)
    assert candidate['entry_convergence_probability'] == .71 and reason == 'replacement_allowed'
