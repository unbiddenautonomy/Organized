"""Start the NSE FO UDP receiver and print decoded 7208 updates as JSON lines."""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import selectors
from datetime import datetime, timedelta

from voltr.feed.lzo import Lzo1zDecoder, LzoError
from voltr.feed.messages.market_by_price import DecodeError, decode_7208
from voltr.feed.udp import NseFoUdpReceiver, PacketError
from .iv_prediction import IvPredictionPipeline, SUPPORTED_OI_CODES
from .contract_repository import load_nearest_expiry_options
from voltr.feed.order_book import BankNiftyOrderBook
from voltr.instruments.master import contract_source_summary
from .paper_trading import DashboardServer, PaperTradingEngine
from .spot_inference import SpotPredictionPipeline
from .pnl_live_inference import LivePnlInferenceAdapter
from .config import load_source_config
from .live_retraining import LiveRetrainingManager


BASE_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    config = load_source_config(path)
    required = {"multicast_ip", "port", "message_code"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"missing config keys: {', '.join(sorted(missing))}")
    if int(config["message_code"]) != 7208:
        raise ValueError("this receiver currently supports message_code 7208 only")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive and decode NSE FO UDP message 7208")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--lzo-dll", type=Path, help="default: the shared voltr/feed/native/lzo2.dll")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    log_path = Path(config.get("log_file", "logs/feed.log"))
    if not log_path.is_absolute():
        log_path = BASE_DIR / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        log_path,
        mode="a",
        maxBytes=int(config.get("log_max_bytes", 10 * 1024 * 1024)),
        backupCount=int(config.get("log_backup_count", 3)),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, str(config.get("log_level", "INFO")).upper(), logging.INFO),
        handlers=[file_handler, console_handler],
    )
    logger = logging.getLogger("nse_7208")
    logger.info("Starting receiver; per-packet data-feed logging is disabled")

    contracts = load_nearest_expiry_options(config)
    contract_summary = contract_source_summary(config, contracts)
    expiry = next(iter(contracts.values())).expiry
    ce_count = sum(item.option_type == "CE" for item in contracts.values())
    pe_count = sum(item.option_type == "PE" for item in contracts.values())
    orderbook_path = Path(config.get("orderbook_log_file", "logs/banknifty_orderbook.jsonl"))
    if not orderbook_path.is_absolute():
        orderbook_path = BASE_DIR / orderbook_path
    persist_orderbook = bool(config.get("persist_orderbook_updates", False))
    order_book = BankNiftyOrderBook(contracts, orderbook_path, persist_orderbook)
    spot_pipeline = SpotPredictionPipeline(config, contracts)
    iv_pipeline = IvPredictionPipeline(config, contracts, spot_pipeline.future_token)
    pnl_adapter = LivePnlInferenceAdapter(config)
    paper_engine = PaperTradingEngine(config, contracts, order_book, iv_pipeline.oi_by_token, pnl_adapter)
    retraining = LiveRetrainingManager(config, BASE_DIR)
    def dashboard_state() -> dict:
        state=paper_engine.snapshot(); spot_ready=spot_pipeline.readiness_snapshot(); iv_ready=iv_pipeline.readiness_snapshot()
        state["contract_master"] = contract_summary
        state["front_future"] = {"token": spot_pipeline.future_token, "expiry": str(spot_pipeline.future_expiry)}
        state["runtime"] = {"status": "running", "paper_only": True,
            "fo_multicast": f"{config['multicast_ip']}:{config['port']}",
            "cm_multicast": f"{config.get('cm_multicast_ip')}:{config.get('cm_port')}",
            "interface_ip": config.get("interface_ip")}
        state["intraday_retraining"] = retraining.snapshot()
        state["prediction_readiness"]={}
        latest_events=[event for event in (paper_engine.latest_spot_prediction,paper_engine.latest_iv_prediction) if event]
        for horizon in sorted(set(spot_ready)|set(iv_ready),key=int):
            s=spot_ready.get(horizon,{}); v=iv_ready.get(horizon,{})
            remaining=max(int(s.get("bars_remaining",999)),int(v.get("bars_remaining",999)))
            feed_advancing=bool(s.get("latest_spot_minute") is not None and s.get("latest_future_minute") is not None and v.get("option_tokens_with_bars",0)>0)
            horizon_events=[event for event in latest_events if horizon in event.get("predictions",{})]
            latest_prediction=min((datetime.fromisoformat(event["exchange_minute"]) for event in horizon_events),default=None)
            if latest_prediction:
                next_prediction=latest_prediction+timedelta(minutes=1)
            elif feed_advancing:
                feed_minute=min(int(s["latest_spot_minute"]),int(s["latest_future_minute"]),int(v["latest_spot_minute"]),int(v["latest_future_minute"]))
                next_prediction=datetime(1980,1,1)+timedelta(minutes=feed_minute-2+remaining)
            else:
                next_prediction=None
            next_update_at=next_prediction+timedelta(minutes=2) if next_prediction else None
            state["prediction_readiness"][horizon]={"spot":s,"iv":v,"bars_remaining":remaining,
                "estimated_seconds_remaining":0 if remaining == 0 else (max(0,remaining*60-datetime.now().second) if feed_advancing else None),
                "latest_predicted_minute":latest_prediction.isoformat(sep=" ") if latest_prediction else None,
                "next_predicted_minute":next_prediction.isoformat(sep=" ") if next_prediction else None,
                "next_update_seconds":max(0,int((next_update_at-datetime.now()).total_seconds())) if next_update_at else None,
                "feed_advancing":feed_advancing,
                "ready":bool(s.get("ready")) and bool(v.get("ready"))}
        return state
    dashboard = DashboardServer(
        str(config.get("paper_dashboard_host", "127.0.0.1")),
        int(config.get("paper_dashboard_port", 8765)),
        dashboard_state,
    )
    spot_pipeline.add_prediction_listener(paper_engine.on_spot_prediction)
    iv_pipeline.add_prediction_listener(paper_engine.on_iv_prediction)
    spot_pipeline.add_prediction_listener(retraining.on_spot_prediction)
    iv_pipeline.add_prediction_listener(retraining.on_iv_prediction)
    dashboard.start()
    tracked_tokens = set(contracts)
    tracked_tokens.update(item.underlying_token for item in contracts.values())
    tracked_tokens.add(spot_pipeline.future_token)
    logger.info(
        "ORDER_BOOK_READY symbol=%s expiry=%s contracts=%d ce=%d pe=%d underlying_tokens=%d output=%s",
        config.get("contract_symbol", "BANKNIFTY"), expiry, len(order_book), ce_count, pe_count,
        order_book.underlying_count, orderbook_path if persist_orderbook else "disabled (memory only)",
    )
    logger.info("PAPER_TRADING_READY mode=PAPER_ONLY dashboard=http://%s:%d trades=%s decisions=%s",
                config.get("paper_dashboard_host", "127.0.0.1"),
                int(config.get("paper_dashboard_port", 8765)), paper_engine.output_path,
                paper_engine.decision_output_path)

    fo_config = dict(config)
    fo_config["receiver_name"] = "NSE FO"
    cm_config = dict(config)
    cm_config.update({
        "receiver_name": "NSE CM indices",
        "multicast_ip": config.get("cm_multicast_ip", "233.1.2.5"),
        "port": int(config.get("cm_port", 8222)),
        "interface_ip": config.get("cm_interface_ip", config.get("interface_ip", "0.0.0.0")),
    })
    fo_receiver = NseFoUdpReceiver(fo_config, Lzo1zDecoder(args.lzo_dll))
    cm_receiver = NseFoUdpReceiver(cm_config, Lzo1zDecoder(args.lzo_dll))
    fo_receiver.open()
    cm_receiver.open()
    socket_selector = selectors.DefaultSelector()
    socket_selector.register(fo_receiver.socket, selectors.EVENT_READ, fo_receiver)
    socket_selector.register(cm_receiver.socket, selectors.EVENT_READ, cm_receiver)

    def handle_message(message: bytes, source: NseFoUdpReceiver) -> None:
        if len(message) < 12:
            return
        transaction_code = int.from_bytes(message[10:12], "big", signed=True)
        if source is cm_receiver:
            if transaction_code == 7207:
                paper_engine.on_index_message(message)
                spot_pipeline.on_broadcast_message(message)
                iv_pipeline.on_broadcast_message(message)
            elif transaction_code == 7216:
                paper_engine.on_index_message(message)
            return
        if transaction_code in SUPPORTED_OI_CODES:
            iv_pipeline.on_oi_message(message)
            return
        # Accept 7207 on FO too if the configured channel happens to carry it.
        if transaction_code == 7207:
            paper_engine.on_index_message(message)
            spot_pipeline.on_broadcast_message(message)
            iv_pipeline.on_broadcast_message(message)
            return
        if transaction_code == 7216:
            paper_engine.on_index_message(message)
            return
        if transaction_code not in (7208, 17208):
            return
        try:
            for update in decode_7208(message, float(config["price_divisor"]), tracked_tokens):
                spot_pipeline.on_7208(update)
                iv_pipeline.on_7208(update)
                order_book.update(update)
                paper_engine.on_market_update(update)
        except DecodeError:
            return

    try:
        while True:
            for key, _ in socket_selector.select(timeout=1.0):
                source: NseFoUdpReceiver = key.data
                assert source.socket is not None
                datagram, _sender = source.socket.recvfrom(int(source.config["max_datagram_bytes"]))
                try:
                    for message in source.unpack_datagram(datagram):
                        handle_message(message, source)
                except (PacketError, LzoError) as error:
                    # Keep malformed feed packets isolated from the other channel.
                    logging.getLogger("market_receiver").debug("Discarded packet: %s", error)
            retraining.tick()
    except KeyboardInterrupt:
        logging.info("Stopping receiver")
    finally:
        socket_selector.close()
        fo_receiver.close()
        cm_receiver.close()
        order_book.close()
        spot_pipeline.close()
        iv_pipeline.close()
        paper_engine.close()
        retraining.close()
        dashboard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
