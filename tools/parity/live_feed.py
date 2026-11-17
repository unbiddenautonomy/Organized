"""Drive the real IVRV live path with a synthetic feed and save what it writes.

Datagrams from feedgen go through the production packet unpacker, LZO decoder,
message decoders, LiveFeatureInference (minute bars, IV surface, features,
models) and both paper books. Nothing opens a socket or the dashboard. Every
log the engine writes lands in the output folder for parity comparison.

`load_runtime()` is the only place that knows where the code lives; refactor
steps update its imports and nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POSITION_ALERT_DISABLED", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import feedgen  # noqa: E402


def load_runtime():
    """Return the production pieces the harness drives (location-specific)."""
    sys.path.insert(0, str(REPO))
    inference = REPO / "pipelines" / "ivrv"
    from pipelines.ivrv import live as li
    from voltr.feed.lzo import Lzo1zDecoder as FastLzo
    from voltr.feed.udp import NseFoUdpReceiver
    config_path = REPO / "config" / "live_config.json"
    config = li.load_live_config(config_path)
    source = li.install_source_imports(config)
    from voltr.instruments.master import OptionContract
    return dict(
        config=config,
        source=source,
        engine_type=li.LiveFeatureInference,
        pipeline_type=li.FixedStrikeInferencePipeline,
        contract_type=OptionContract,
        receiver_type=NseFoUdpReceiver,
        lzo_types={"ivrv": FastLzo},  # step 3 removed the slower standalone copy
        lzo_dll=(REPO / "voltr" / "feed" / "native" / "lzo2.dll"),
        inference_config=inference / "config.yaml",
    )


def run(day_folder: Path, out: Path) -> dict:
    rt = load_runtime()
    out.mkdir(parents=True, exist_ok=True)
    day = feedgen.load_day(day_folder)
    config = dict(rt["config"])
    logs = out / "logs"
    for key, name in (("prediction_log", "live_predictions.jsonl"),
                      ("fixed_strike_path_log", "fixed_strike_120m_paths.jsonl"),
                      ("paper_trade_events_log", "paper_trade_events.jsonl"),
                      ("paper_position_state", "paper_position_state.json"),
                      ("latest_prediction", "latest_prediction.json")):
        config[key] = str(logs / name)
    config["contract_asof_date"] = day.session.isoformat()
    contracts = feedgen.option_contracts(day, rt["contract_type"])
    pipeline = rt["pipeline_type"](rt["inference_config"])
    engine = rt["engine_type"](config, contracts, 30, day.expiry, day.future_token, pipeline, rt["source"])
    engine.weeks43 = None
    source = rt["source"]
    tracked = set(contracts) | {day.future_token}
    receivers = {s: rt["receiver_type"](dict(config, receiver_name=s), rt["lzo_types"]["ivrv"](rt["lzo_dll"]))
                 for s in ("fo", "cm")}
    fast_lzo = rt["lzo_types"]["ivrv"](rt["lzo_dll"])
    decoded = hashlib.sha256()
    counts = dict(datagrams=0, messages=0, compressed=0, lzo_mismatch=0, mbp_updates=0)
    for stream, datagram, compressed in feedgen.datagrams(day, feedgen.Compressor(rt["lzo_dll"])):
        counts["datagrams"] += 1
        for packed, payload in compressed:
            counts["compressed"] += 1
            a = fast_lzo.decompress(packed, int(config["max_decompressed_bytes"]))
            if a != payload:
                counts["lzo_mismatch"] += 1
        for message in receivers[stream].unpack_datagram(datagram):
            counts["messages"] += 1
            decoded.update(message)
            code = int.from_bytes(message[10:12], "big", signed=True)
            if code == 7207:
                engine.on_index_message(message)
            elif stream == "fo" and code in source["SUPPORTED_OI_CODES"]:
                engine.on_oi_message(message)
            elif stream == "fo" and code in (7208, 17208):
                for update in source["decode_7208"](message, float(config["price_divisor"]), tracked):
                    counts["mbp_updates"] += 1
                    decoded.update(repr(update).encode())
                    engine.on_market_update(update)
    engine.close()
    counts["decoded_sha256"] = decoded.hexdigest()
    (out / "feed_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
