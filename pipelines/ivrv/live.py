"""IVRV session entry point: `python -m pipelines.ivrv.live --config config/live_config.json`.

With `isolate_pipelines` (the default in config/service.json) this starts the
shared voltr.service supervisor. Otherwise it runs the older single-process
session, or an offline historical replay. This module also re-exports the IVRV
pieces other code imports by name (config loader, paper book, feature engine,
dashboard). Paper only: nothing here places orders.

    config.py     session config and schedule        strategy.py  entry rules and costs
    book.py       paper straddle book                engine.py    features, scoring, publishing
    dashboard.py  HTTP dashboard and standby view    model.py     the saved model stack
    pipeline.py   adapter for the isolated service   paths.py     where IVRV files live
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import selectors
import shutil
import sys
import time as clock
from typing import Any

import pandas as pd

from voltr.core.clock import MARKET_TIMEZONE, NSE_EPOCH  # noqa: F401  (NSE_EPOCH re-exported)
from voltr.feed.lzo import Lzo1zDecoder, LzoError
from voltr.feed.messages.indices import decode_banknifty_7207
from voltr.feed.messages.market_by_price import DecodeError, decode_7208
from voltr.feed.messages.open_interest import SUPPORTED_OI_CODES, OiDecodeError, decode_oi_message
from voltr.feed.udp import NseFoUdpReceiver, PacketError
from voltr.instruments.download import ensure_latest_contract_file
from voltr.instruments.master import load_front_future, load_monthly_contracts
from voltr.market.atm_features import (ANNUAL_MINUTES, LAGS, REQUIRED_FEATURE_MINUTES,  # noqa: F401
                                       AtmFeatureStream, quadratic_surface as _quadratic_surface)
from voltr.market.bars import MinuteBar  # noqa: F401  (re-exported for the replay tools)
from voltr.market.options import greeks, implied_volatility

from . import extensions, paths
from .book import PaperTradingEngine  # noqa: F401
from .config import (DEFAULT_SESSION_END, DEFAULT_SESSION_START, _load_json, load_live_config,  # noqa: F401
                     next_session_window, parse_schedule_time)
from .dashboard import DashboardServer, LogDashboardView  # noqa: F401
from .engine import (PIPELINE_GENERATED_FEATURES, LiveFeatureInference, _json_value,  # noqa: F401
                     _weeks43_notify, required_live_input_features)
from .model import FixedStrikeInferencePipeline
from .paths import DASHBOARD_HTML, DEFAULT_CONFIG, LIVE_DIR, PROJECT_DIR, REPO_DIR  # noqa: F401
from .strategy import convergence_entry_eligible, current_costs, paper_entry_route  # noqa: F401

LOGGER = logging.getLogger("fixed_strike_live")


def archive_previous_day_logs(config: dict, session_date: date) -> list[Path]:
    """Move stale active log files into prevDayLogs/<file-date>/.

    Files already modified on the session date are retained so restarting the
    runtime intraday appends to that day's logs instead of archiving them.
    """
    configured = {
        (paths.BASE_DIR / str(config.get("runtime_log", "logs/live_runtime.log"))).resolve(),
        (paths.BASE_DIR / str(config.get("prediction_log", "logs/live_predictions.jsonl"))).resolve(),
        (paths.BASE_DIR / str(config.get("fixed_strike_path_log", "logs/fixed_strike_120m_paths.jsonl"))).resolve(),
        (paths.BASE_DIR / str(config.get("paper_trade_events_log", "logs/paper_trade_events.jsonl"))).resolve(),
        (paths.BASE_DIR / str(config.get("paper_position_state", "logs/paper_position_state.json"))).resolve(),
        (paths.BASE_DIR / str(config.get("latest_prediction", "logs/latest_prediction.json"))).resolve(),
    }
    archived: list[Path] = []
    archive_root = paths.BASE_DIR / "logs" / "prevDayLogs"
    log_root = paths.BASE_DIR / "logs"
    if log_root.is_dir():
        configured.update(path.resolve() for path in log_root.iterdir() if path.is_file())
    for source in sorted(configured):
        if not source.is_file():
            continue
        state_path=(paths.BASE_DIR / str(config.get('paper_position_state','logs/paper_position_state.json'))).resolve()
        if source==state_path:
            try:
                saved=_load_json(source)
                if saved.get('open_positions') or saved.get('status')=='OPEN':
                    LOGGER.warning('Retaining unresolved paper positions during daily log archive')
                    continue
            except (OSError,ValueError):
                LOGGER.warning('Retaining unreadable paper state for inspection')
                continue
        modified = datetime.fromtimestamp(source.stat().st_mtime, MARKET_TIMEZONE).date()
        if modified >= session_date:
            continue
        destination_dir = archive_root / modified.isoformat()
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists():
            destination = destination_dir / f"{source.stem}_{int(source.stat().st_mtime)}{source.suffix}"
        shutil.move(str(source), str(destination))
        archived.append(destination)
    return archived


def filter_historical_date(frame: pd.DataFrame, trading_date: date) -> pd.DataFrame:
    """Select one trading date from a saved feature frame."""
    if "datetime" in frame.columns:
        values = pd.to_datetime(frame["datetime"], errors="raise").dt.date
    elif "Date" in frame.columns:
        values = pd.to_datetime(frame["Date"], errors="raise").dt.date
    else:
        raise ValueError("--historical-date requires a 'datetime' or 'Date' column")
    selected = frame.loc[values == trading_date].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"no historical feature rows found for {trading_date.isoformat()}")
    return selected


def run_historical_inference(
    config: dict,
    combined_path: Path,
    *,
    leg_path: Path | None = None,
    rv_path: Path | None = None,
    output_path: Path | None = None,
    trading_date: date | None = None,
) -> Path:
    """Run the saved inference stack against feature CSVs without live I/O."""
    combined = pd.read_csv(combined_path.resolve(), low_memory=False)
    legs = pd.read_csv(leg_path.resolve(), low_memory=False) if leg_path else None
    rv = pd.read_csv(rv_path.resolve(), low_memory=False) if rv_path else None
    if trading_date is not None:
        combined = filter_historical_date(combined, trading_date)
        if legs is not None:
            legs = filter_historical_date(legs, trading_date)
        if rv is not None:
            rv = filter_historical_date(rv, trading_date)

    pipeline = FixedStrikeInferencePipeline(Path(config["inference_config"]).resolve())
    predictions = pipeline.predict(combined, legs, rv)
    if output_path is None:
        now = datetime.now(MARKET_TIMEZONE)
        result_date = trading_date or now.date()
        output_path = (
            paths.BASE_DIR / "historical_predictions" / result_date.isoformat()
            / f"predictions_{now:%H%M%S}.csv"
        )
    destination = output_path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    print(f"Wrote {len(predictions):,} historical inference rows to {destination}", flush=True)
    return destination


def historical_replay_config(
    config: dict,
    fo_port: int,
    *,
    cm_port: int | None = None,
    started_at: datetime | None = None,
) -> dict:
    """Build an isolated live configuration for an after-hours UDP replay."""
    ports = [fo_port] + ([cm_port] if cm_port is not None else [])
    if any(port is None or not 1 <= port <= 65535 for port in ports):
        raise ValueError("historical replay ports must be between 1 and 65535")
    stamp = started_at or datetime.now(MARKET_TIMEZONE)
    replay = dict(config)
    replay["port"] = int(fo_port)
    if cm_port is not None:
        replay["cm_port"] = int(cm_port)
    directory = f"logs/historicalReplay/{stamp:%Y-%m-%d}"
    suffix = f"{stamp:%H%M%S}"
    replay["runtime_log"] = f"{directory}/live_runtime_{suffix}.log"
    replay["prediction_log"] = f"{directory}/live_predictions_{suffix}.jsonl"
    replay["fixed_strike_path_log"] = f"{directory}/fixed_strike_120m_paths_{suffix}.jsonl"
    replay["paper_trade_events_log"] = f"{directory}/paper_trade_events_{suffix}.jsonl"
    replay["paper_position_state"] = f"{directory}/paper_position_state_{suffix}.json"
    replay["latest_prediction"] = f"{directory}/latest_prediction_{suffix}.json"
    replay["runtime_mode"] = "historical_udp_replay"
    return replay


def install_source_imports(config: dict) -> dict[str, Any]:
    """The feed, contract and option-maths functions each worker uses, by name."""
    return {
        "load_monthly_contracts": load_monthly_contracts,
        "load_front_future": load_front_future,
        "DecodeError": DecodeError,
        "decode_7208": decode_7208,
        "_greeks": greeks,
        "implied_volatility": implied_volatility,
        "OiDecodeError": OiDecodeError,
        "SUPPORTED_OI_CODES": SUPPORTED_OI_CODES,
        "decode_oi_message": decode_oi_message,
        "decode_banknifty_7207": decode_banknifty_7207,
        "Lzo1zDecoder": Lzo1zDecoder,
        "LzoError": LzoError,
        "NseFoUdpReceiver": NseFoUdpReceiver,
        "PacketError": PacketError,
    }


def configure_logging(path: Path, level: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler, console],
        force=True,
    )


def wait_until(moment: datetime) -> None:
    """Sleep until a timezone-aware wall-clock moment."""
    while True:
        remaining = (moment - datetime.now(moment.tzinfo)).total_seconds()
        if remaining <= 0:
            return
        clock.sleep(min(remaining, 30.0))


def run_live_session(
    config: dict,
    *,
    start_at: datetime | None = None,
    stop_at: datetime | None = None,
    session_date: date | None = None,
    preflight: bool = False,
) -> None:
    """Prepare and run one live session, closing every resource on exit."""
    bootstrap_asof = (
        session_date
        or (
            date.fromisoformat(config["contract_asof_date"])
            if config.get("contract_asof_date")
            else datetime.now(MARKET_TIMEZONE).date()
        )
    )
    contract_state = ensure_latest_contract_file(config, asof=bootstrap_asof)
    LOGGER.info(
        "CONTRACT_READY file=%s file_date=%s downloaded=%s nearest_expiry=%s nearest_tokens=%d "
        "pipeline_monthly_expiry=%s pipeline_tokens=%d token_manifest=%s",
        contract_state.contract_file, contract_state.file_date, contract_state.downloaded,
        contract_state.nearest_expiry, contract_state.nearest_option_tokens,
        contract_state.pipeline_monthly_expiry, contract_state.pipeline_option_tokens,
        contract_state.manifest_file,
    )
    source = install_source_imports(config)

    contract_file = Path(config["contract_file"]).resolve()
    asof = (
        date.fromisoformat(config["contract_asof_date"])
        if config.get("contract_asof_date")
        else datetime.now(MARKET_TIMEZONE).date()
    )
    contracts, lot_size, expiry = source["load_monthly_contracts"](
        contract_file, asof, float(config.get("price_divisor", 100.0))
    )
    future_token, future_expiry = source["load_front_future"](config)
    pipeline = FixedStrikeInferencePipeline(Path(config["inference_config"]).resolve())
    engine = LiveFeatureInference(config, contracts, lot_size, expiry, future_token, pipeline, source)
    lzo_path = Path(config["lzo_dll"]).resolve()
    source["Lzo1zDecoder"](lzo_path)
    receiver_module = Path(sys.modules[source["NseFoUdpReceiver"].__module__].__file__).resolve()

    LOGGER.info(
        "PREFLIGHT_OK paper_only=true expiry=%s option_tokens=%d future_token=%d future_expiry=%s "
        "lot_size=%d model_artifacts=%d feature_contract=%d udp_receiver=%s lzo_dll=%s",
        expiry, len(contracts), future_token, future_expiry, lot_size,
        len(pipeline.artifacts), len(engine.required_feature_names),
        receiver_module, lzo_path,
    )
    if preflight:
        engine.close()
        return

    pcr = (
        extensions.PCR_BRIDGE.create_bridge(config, contracts, lot_size, expiry, future_token, source)
        if extensions.PCR_BRIDGE is not None else None
    )

    dashboard = DashboardServer(
        str(config["dashboard_host"]), int(config["dashboard_port"]),
        DASHBOARD_HTML, engine,
    )
    fo_config = dict(config, receiver_name="NSE FO")
    cm_config = dict(
        config,
        receiver_name="NSE CM indices",
        multicast_ip=config["cm_multicast_ip"],
        port=int(config["cm_port"]),
        interface_ip=config.get("cm_interface_ip", config["interface_ip"]),
    )
    fo_receiver = source["NseFoUdpReceiver"](fo_config, source["Lzo1zDecoder"](lzo_path))
    cm_receiver = source["NseFoUdpReceiver"](cm_config, source["Lzo1zDecoder"](lzo_path))
    tracked = set(contracts) | {future_token}

    def handle(message: bytes, receiver: Any) -> None:
        if len(message) < 12:
            return
        code = int.from_bytes(message[10:12], "big", signed=True)
        if code == 7207:
            engine.on_index_message(message)
            if extensions.PCR_BRIDGE is not None:
                extensions.PCR_BRIDGE.safe_call(pcr, "index_message", message)
        elif receiver is not cm_receiver and code in source["SUPPORTED_OI_CODES"]:
            engine.on_oi_message(message)
            if extensions.PCR_BRIDGE is not None:
                extensions.PCR_BRIDGE.safe_call(pcr, "oi_message", message)
        elif receiver is not cm_receiver and code in (7208, 17208):
            try:
                for update in source["decode_7208"](message, float(config["price_divisor"]), tracked):
                    engine.on_market_update(update)
                    if extensions.PCR_BRIDGE is not None:
                        extensions.PCR_BRIDGE.safe_call(pcr, "market", update)
            except source["DecodeError"]:
                return

    selector: selectors.BaseSelector | None = None
    dashboard_started = False
    try:
        if start_at is not None:
            wait_until(start_at)
        if stop_at is not None and datetime.now(stop_at.tzinfo) >= stop_at:
            LOGGER.warning("SESSION_SKIPPED stop time already passed: %s", stop_at.isoformat())
            return

        try:
            extension = extensions.weeks43_module()
            engine.weeks43 = extension.attach(engine) if extension is not None else None
        except Exception:
            LOGGER.exception("43Weeks startup failed; existing paper trader continues")
            engine.weeks43 = None

        if extensions.PCR_BRIDGE is not None:
            extensions.PCR_BRIDGE.safe_call(pcr, "start")
        dashboard.start()
        dashboard_started = True
        LOGGER.info("DASHBOARD_READY http://%s:%s", config["dashboard_host"], config["dashboard_port"])
        fo_receiver.open()
        cm_receiver.open()
        selector = selectors.DefaultSelector()
        selector.register(fo_receiver.socket, selectors.EVENT_READ, fo_receiver)
        selector.register(cm_receiver.socket, selectors.EVENT_READ, cm_receiver)
        LOGGER.info(
            "SESSION_STARTED start=%s scheduled_stop=%s timezone=%s",
            datetime.now(MARKET_TIMEZONE).isoformat(),
            stop_at.isoformat() if stop_at else "manual",
            MARKET_TIMEZONE.key,
        )
        while stop_at is None or datetime.now(stop_at.tzinfo) < stop_at:
            timeout = 1.0
            if stop_at is not None:
                timeout = max(0.0, min(timeout, (stop_at - datetime.now(stop_at.tzinfo)).total_seconds()))
            for key, _ in selector.select(timeout=timeout):
                receiver = key.data
                datagram, _sender = receiver.socket.recvfrom(int(receiver.config["max_datagram_bytes"]))
                try:
                    for message in receiver.unpack_datagram(datagram):
                        handle(message, receiver)
                except (source["PacketError"], source["LzoError"]) as error:
                    LOGGER.debug("discarded malformed packet: %s", error)
            if extensions.PCR_BRIDGE is not None:
                extensions.PCR_BRIDGE.safe_call(pcr, "pulse")
    finally:
        LOGGER.info("SESSION_STOPPED time=%s", datetime.now(MARKET_TIMEZONE).isoformat())
        if selector is not None:
            selector.close()
        fo_receiver.close()
        cm_receiver.close()
        if extensions.PCR_BRIDGE is not None:
            extensions.PCR_BRIDGE.safe_call(pcr, "close")
        if dashboard_started:
            dashboard.close()
        else:
            dashboard.server.server_close()
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live UDP inference for the fixed-strike IV/RV stack")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="validate imports, contracts, and models without opening sockets")
    mode.add_argument("--run-now", action="store_true", help="bypass the daily scheduler and run until interrupted")
    mode.add_argument(
        "--historical-data", type=Path, metavar="CSV",
        help="run after-hours inference on a combined/wide historical feature CSV",
    )
    mode.add_argument(
        "--historical-replay-port", type=int, metavar="PORT",
        help="run after-hours from the historical UDP feed on this FO port",
    )
    parser.add_argument(
        "--historical-replay-cm-port", type=int, metavar="PORT",
        help="optional historical UDP port for the CM/index feed",
    )
    parser.add_argument("--historical-leg-data", type=Path, metavar="CSV", help="optional call/put feature CSV")
    parser.add_argument("--historical-rv-data", type=Path, metavar="CSV", help="optional RV feature CSV")
    parser.add_argument(
        "--historical-date", metavar="YYYY-MM-DD",
        help="CSV trading-date filter or UDP replay contract as-of date",
    )
    parser.add_argument("--historical-output", type=Path, metavar="CSV", help="historical prediction destination")
    args = parser.parse_args()

    if args.historical_data:
        try:
            trading_date = date.fromisoformat(args.historical_date) if args.historical_date else None
        except ValueError:
            parser.error("--historical-date must use YYYY-MM-DD")
        historical_local = _load_json(args.config.resolve())
        inference_config = Path(historical_local["inference_config"])
        if not inference_config.is_absolute():
            inference_config = args.config.resolve().parent / inference_config
        run_historical_inference(
            {"inference_config": str(inference_config.resolve())},
            args.historical_data,
            leg_path=args.historical_leg_data,
            rv_path=args.historical_rv_data,
            output_path=args.historical_output,
            trading_date=trading_date,
        )
        return 0
    if args.historical_replay_port is not None:
        if args.historical_leg_data or args.historical_rv_data or args.historical_output:
            parser.error("CSV historical options cannot be combined with UDP replay mode")
        config = load_live_config(args.config)
        try:
            replay_config = historical_replay_config(
                config,
                args.historical_replay_port,
                cm_port=args.historical_replay_cm_port,
            )
            if args.historical_date:
                replay_config["contract_asof_date"] = date.fromisoformat(args.historical_date).isoformat()
        except ValueError as error:
            parser.error(str(error))
        configure_logging(
            paths.BASE_DIR / replay_config["runtime_log"],
            str(replay_config.get("log_level", "INFO")),
        )
        LOGGER.info(
            "HISTORICAL_REPLAY_MODE fo=%s:%d cm=%s:%d replay_date=%s",
            replay_config["multicast_ip"], replay_config["port"],
            replay_config["cm_multicast_ip"], replay_config["cm_port"],
            replay_config.get("contract_asof_date", "today"),
        )
        try:
            run_live_session(replay_config)
        except KeyboardInterrupt:
            LOGGER.info("historical UDP replay stopped by user")
        return 0
    if args.historical_replay_cm_port is not None:
        parser.error("--historical-replay-cm-port requires --historical-replay-port")
    if args.historical_leg_data or args.historical_rv_data or args.historical_date or args.historical_output:
        parser.error("historical options require --historical-data")

    config = load_live_config(args.config)
    if args.preflight:
        # Validate an already available current master without forcing a new
        # network download; scheduled sessions retain their download policy.
        config["contract_force_download"] = False
    runtime_log = paths.BASE_DIR / config["runtime_log"]
    if config.get('isolate_pipelines',False) and not args.preflight:
        from voltr.service.supervisor import run_service
        return run_service(config,args.config.resolve(),run_now=args.run_now)
    if args.preflight or args.run_now:
        configure_logging(runtime_log, str(config.get("log_level", "INFO")))
        try:
            run_live_session(config, preflight=args.preflight)
        except KeyboardInterrupt:
            LOGGER.info("live inference stopped by user")
        return 0

    session_start = parse_schedule_time(str(config.get("session_start", "09:00")))
    session_end = parse_schedule_time(str(config.get("session_end", "15:40")))
    prepare_seconds = max(0, int(config.get("schedule_prepare_seconds", 300)))

    try:
        while True:
            now = datetime.now(MARKET_TIMEZONE)
            start_at, stop_at = next_session_window(now, session_start, session_end)
            preparation_at = start_at - timedelta(seconds=prepare_seconds)
            if now < preparation_at:
                print(
                    f"Scheduler waiting: next session {start_at:%Y-%m-%d %H:%M}–{stop_at:%H:%M} "
                    f"{MARKET_TIMEZONE.key}",
                    flush=True,
                )
                if bool(config.get("standby_dashboard_enabled", False)):
                    standby = DashboardServer(
                        str(config["dashboard_host"]),
                        int(config["dashboard_port"]),
                        DASHBOARD_HTML,
                        LogDashboardView(config),
                    )
                    try:
                        standby.start()
                        print(
                            f"Standby dashboard ready: http://{config['dashboard_host']}:"
                            f"{config['dashboard_port']}",
                            flush=True,
                        )
                        wait_until(preparation_at)
                    finally:
                        standby.close()
                else:
                    wait_until(preparation_at)

            logging.shutdown()
            archived = archive_previous_day_logs(config, start_at.date())
            configure_logging(runtime_log, str(config.get("log_level", "INFO")))
            for path in archived:
                LOGGER.info("LOG_ARCHIVED destination=%s", path)
            LOGGER.info(
                "SESSION_SCHEDULED start=%s stop=%s preparation_seconds=%d",
                start_at.isoformat(), stop_at.isoformat(), prepare_seconds,
            )
            # ``ensure_latest_contract_file`` records its as-of date in config.
            # Override that cached value for every scheduled session so a
            # long-running scheduler cannot reuse yesterday's expired tokens.
            run_live_session(
                config,
                start_at=start_at,
                stop_at=stop_at,
                session_date=start_at.date(),
            )
            logging.shutdown()
    except KeyboardInterrupt:
        LOGGER.info("scheduler stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
