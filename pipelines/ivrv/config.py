"""Session configuration: live_config.json merged with feed.json, config.yaml, machine.local.json and service.json; trading-day schedule."""
from __future__ import annotations

from datetime import datetime, time, timedelta
import json
from pathlib import Path
import yaml
from voltr.core.machine import machine_local_overrides, relative_to_config, resolve_interface


DEFAULT_SESSION_START = time(9, 0)


DEFAULT_SESSION_END = time(15, 40)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_live_config(path: Path) -> dict:
    path = path.resolve()
    local = _load_json(path)
    local.update(machine_local_overrides(path))
    source_config = relative_to_config(local["source_config"], path)
    source = _load_json(source_config)
    for key in ("contract_file", "spot_model_dir", "iv_model_dir", "pnl_model_dir", "hmm_model_dir",
                "expanded_hybrid_model_file", "rv120_model_file"):
        if source.get(key):
            source[key] = str(relative_to_config(source[key], source_config))
    inference_config = relative_to_config(local["inference_config"], path)
    inference_yaml = yaml.safe_load(inference_config.read_text(encoding="utf-8")) or {}
    udp = inference_yaml.get("live_udp", {})
    if udp:
        required = {
            "fo_multicast_ip", "fo_port", "cm_multicast_ip", "cm_port",
            "interface_ip", "cm_interface_ip",
        }
        missing = required.difference(udp)
        if missing:
            raise ValueError(f"config.yaml live_udp missing keys: {', '.join(sorted(missing))}")
        source.update({
            "multicast_ip": str(udp["fo_multicast_ip"]),
            "port": int(udp["fo_port"]),
            "cm_multicast_ip": str(udp["cm_multicast_ip"]),
            "cm_port": int(udp["cm_port"]),
            "interface_ip": str(udp["interface_ip"]),
            "cm_interface_ip": str(udp["cm_interface_ip"]),
        })
    signal = inference_yaml.get("rv_iv_sharp_decline_signal", {})
    if signal:
        source.update({
            "fixed_strike_path_horizon_minutes": int(signal["horizon_minutes"]),
            "paper_profit_target_inr": float(signal["profit_target_inr"]),
            "paper_stop_loss_inr": float(signal["stop_loss_inr"]),
        })
    source.update(local)
    source["source_config"] = str(source_config)
    for key in ("inference_config", "contract_file", "contract_download_directory",
                "lzo_dll", "contract_auth_token_file"):
        if source.get(key):
            source[key] = str(relative_to_config(source[key], path))
    source["interface_ip"] = resolve_interface(source.get("interface_ip"), environment="VOLTR_FO_INTERFACE")
    source["cm_interface_ip"] = resolve_interface(source.get("cm_interface_ip"), environment="VOLTR_CM_INTERFACE")
    service_path=path.with_name('service.json')
    if service_path.is_file():
        service=_load_json(service_path)
        allowed={'isolate_pipelines','session_start','session_end','schedule_prepare_seconds','pipeline_queue_capacity',
            'pipeline_reader_queue_capacity','pipeline_stall_seconds','pipeline_startup_seconds','pipeline_restart_limit',
            'pipeline_max_backlog_seconds','pipeline_feed_silence_seconds','feed_journal_retention_minutes','contract_force_download','contract_retry_seconds','contract_auth_token_file','dashboard_alias_ports','enabled_pipelines',
            'host','dashboard_pipeline','runtime_dir','pipelines'}
        if set(service)-allowed:raise ValueError('Unknown service configuration keys: '+str(sorted(set(service)-allowed)))
        for key in ('pipeline_queue_capacity','pipeline_reader_queue_capacity'):
            if key in service and not 1<=int(service[key])<=100000:raise ValueError(key+' must be bounded from 1 to 100000')
        if 'enabled_pipelines' in service:
            names=service['enabled_pipelines']
            if not isinstance(names,list) or not names or len(names)!=len(set(names)) or any(name not in service.get('pipelines',('ivrv','pcr','43weeks')) for name in names):
                raise ValueError('enabled_pipelines must be a nonempty list of unique pipeline names')
        source.update(service)
    return source


def parse_schedule_time(value: str) -> time:
    """Parse a 24-hour HH:MM schedule value."""
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise ValueError(f"invalid schedule time {value!r}; expected HH:MM") from error


def next_session_window(
    now: datetime,
    start: time = DEFAULT_SESSION_START,
    end: time = DEFAULT_SESSION_END,
) -> tuple[datetime, datetime]:
    """Return the current or next weekday session in the supplied timezone."""
    if now.tzinfo is None:
        raise ValueError("scheduler requires a timezone-aware datetime")
    if start >= end:
        raise ValueError("session start must be earlier than session end")

    candidate = now.date()
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    session_start = datetime.combine(candidate, start, tzinfo=now.tzinfo)
    session_end = datetime.combine(candidate, end, tzinfo=now.tzinfo)
    if now >= session_end:
        candidate += timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        session_start = datetime.combine(candidate, start, tzinfo=now.tzinfo)
        session_end = datetime.combine(candidate, end, tzinfo=now.tzinfo)
    return session_start, session_end
