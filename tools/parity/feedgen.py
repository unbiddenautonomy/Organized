"""Build a deterministic NSE-format feed from a prepared one-minute BANKNIFTY day.

The prepared replay folders hold one observed quote per contract per minute.
This module turns those quotes into the binary messages the live receiver
sees (7207 index, 17208 market-by-price, 7202 open interest), wraps them in
sequential packets, and LZO1Z-compresses most of them with the same DLL the
receiver uses. The output is only a parity fixture: identical input bytes for
the code before and after each refactor step.
"""
from __future__ import annotations

import csv
import ctypes
import random
import struct
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

NSE_EPOCH = datetime(1980, 1, 1)
PREFIX = b"\x00" * 8
ENHANCED_RECORD_SIZE = 258


@dataclass(frozen=True)
class Day:
    session: date
    expiry: date
    future_token: int
    contracts: list[dict]          # token, strike, option_type, symbol
    events: list[tuple]            # (exchange_second, order, kind, payload)


def _second(text: str) -> int:
    stamp = datetime.fromisoformat(text)
    return int((stamp - NSE_EPOCH).total_seconds())


def load_day(folder: Path) -> Day:
    futures = list(csv.DictReader((folder / "banknifty_futures_1min.csv").open(newline="", encoding="utf-8")))
    near = [r for r in futures if r["is_near_expiry"] == "True"]
    future_token = int(near[0]["token"])
    expiry = date.fromisoformat(near[0]["expiry_master"])
    session = date.fromisoformat(near[0]["minute"][:10])
    events: list[tuple] = []
    seen: set[tuple[int, str]] = set()
    for r in near:
        if not r["last_traded_price"] or not r["source_tick_time"]:
            continue
        sec = _second(r["source_tick_time"])
        key = (future_token, r["source_tick_time"])
        if key not in seen:
            seen.add(key)
            price = float(r["last_traded_price"])
            events.append((sec, 1, "mbp", (future_token, price, int(r["volume_traded_today"] or 0))))
            # The index is the near future's price, published on the CM stream.
            events.append((sec, 0, "index", price))
            if r["open_interest"]:
                events.append((sec, 2, "oi", (future_token, int(r["open_interest"]))))
        # A second index print late in the minute closes the spot bar.
        minute_sec = _second(r["minute"])
        events.append((minute_sec + 58, 0, "index", float(r["last_traded_price"])))
    contracts: dict[int, dict] = {}
    for r in csv.DictReader((folder / "banknifty_option_chain_1min.csv").open(newline="", encoding="utf-8")):
        token = int(r["option_token"])
        contracts.setdefault(token, dict(token=token, strike=r["strike"], option_type=r["option_type"],
                                         symbol=r["option_symbol"]))
        for prefix in ("", "target_"):
            tick = r[f"{prefix}source_tick_time"]
            ltp = r[f"{prefix}last_traded_price"]
            if not tick or not ltp or (token, tick) in seen:
                continue
            seen.add((token, tick))
            sec = _second(tick)
            events.append((sec, 1, "mbp", (token, float(ltp), int(r[f"{prefix}volume_traded_today"] or 0))))
            oi = r[f"{prefix}open_interest"]
            if oi:
                events.append((sec, 2, "oi", (token, int(oi))))
    events.sort(key=lambda e: (e[0], e[1], repr(e[3])))
    return Day(session, expiry, future_token, sorted(contracts.values(), key=lambda c: c["token"]), events)


def message_index(log_time: int, value: float) -> bytes:
    message = bytearray(42 + 72)
    struct.pack_into(">i", message, 4, log_time)
    struct.pack_into(">h", message, 10, 7207)
    struct.pack_into(">h", message, 38, len(message))
    struct.pack_into(">H", message, 40, 1)
    message[42:42 + 21] = b"Nifty Bank".ljust(21, b" ")
    struct.pack_into(">i", message, 42 + 22, int(round(value * 100)))
    return bytes(message)


def message_mbp(log_time: int, sequence: int, token: int, price: float, volume: int) -> bytes:
    message = bytearray(42 + ENHANCED_RECORD_SIZE)
    struct.pack_into(">i", message, 4, log_time)
    struct.pack_into(">h", message, 10, 17208)
    struct.pack_into(">i", message, 14, sequence)
    struct.pack_into(">h", message, 38, len(message))
    struct.pack_into(">h", message, 40, 1)
    base = 42
    raw = int(round(price * 100))
    struct.pack_into(">IHHqi", message, base, token, 1, 2, volume, raw)
    message[base + 20] = ord("+")
    struct.pack_into(">iiii", message, base + 22, 0, 15, log_time, raw)
    for level in range(10):
        side = 1 if level < 5 else 2
        level_price = raw - 5 * (level + 1) if side == 1 else raw + 5 * (level - 4)
        struct.pack_into(">qiHH", message, base + 60 + level * 16, 15 * (level + 1), level_price, 1, side)
    struct.pack_into(">HHqq", message, base + 220, 0, 0, 1500, 1500)
    struct.pack_into(">Hiiii", message, base + 240, 0, raw, raw, raw, raw)
    return bytes(message)


def message_oi(log_time: int, token: int, open_interest: int) -> bytes:
    message = bytearray(42 + 26)
    struct.pack_into(">i", message, 4, log_time)
    struct.pack_into(">h", message, 10, 7202)
    struct.pack_into(">h", message, 38, len(message))
    struct.pack_into(">H", message, 40, 1)
    struct.pack_into(">I", message, 42, token)
    struct.pack_into(">III", message, 42 + 14, open_interest, open_interest, open_interest)
    return bytes(message)


class Compressor:
    def __init__(self, dll: Path) -> None:
        self.lib = ctypes.CDLL(str(dll.resolve()))
        self.lib.lzo1z_999_compress.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p,
                                                ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p)
        self.lib.lzo1z_999_compress.restype = ctypes.c_int
        self.work = (ctypes.c_ubyte * (16384 * 8 * 4))()

    def compress(self, data: bytes) -> bytes:
        out = (ctypes.c_ubyte * (len(data) + len(data) // 16 + 64 + 3))()
        size = ctypes.c_size_t(len(out))
        rc = self.lib.lzo1z_999_compress(data, ctypes.c_size_t(len(data)), out, ctypes.byref(size), self.work)
        if rc != 0:
            raise RuntimeError(f"lzo1z_999_compress failed: {rc}")
        return bytes(out[: size.value])


def datagrams(day: Day, compressor: Compressor, seed: int = 20261116):
    """Yield (stream, datagram, compressed_payloads) in exchange-time order.

    stream is "cm" for index messages and "fo" for the rest. Packets carry 1 to
    3 messages; about one message in eight is sent uncompressed.
    """
    rng = random.Random(seed)
    sequence = 0
    pending = {"fo": [], "cm": []}

    def flush(stream):
        parts, compressed = [], []
        for raw in pending[stream]:
            payload = PREFIX + raw
            if rng.random() < 0.125:
                parts.append(struct.pack(">H", 0) + payload)
            else:
                packed = compressor.compress(payload)
                compressed.append((packed, payload))
                parts.append(struct.pack(">H", len(packed)) + packed)
        header = struct.pack(">hh", 0, len(parts))
        pending[stream] = []
        return stream, header + b"".join(parts), compressed

    for second, _, kind, payload in day.events:
        if kind == "index":
            pending["cm"].append(message_index(second, payload))
            yield flush("cm")
            continue
        if kind == "mbp":
            sequence += 1
            pending["fo"].append(message_mbp(second, sequence, *payload))
        else:
            pending["fo"].append(message_oi(second, *payload))
        if len(pending["fo"]) >= rng.randint(1, 3):
            yield flush("fo")
    if pending["fo"]:
        yield flush("fo")


def option_contracts(day: Day, contract_type):
    """Contract objects in the shape the live code expects."""
    result = {}
    for c in day.contracts:
        result[c["token"]] = contract_type(
            token=c["token"], underlying_token=0, instrument_name="OPTIDX", symbol="BANKNIFTY",
            expiry=day.expiry, strike_price=Decimal(c["strike"]), option_type=c["option_type"])
    return result
