"""Decoders for NSE FO contract-level open-interest broadcasts."""

from __future__ import annotations

import struct
from dataclasses import dataclass


SUPPORTED_OI_CODES = frozenset((7201, 7202, 17201, 17202))


class OiDecodeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OiUpdate:
    transaction_code: int
    log_time: int
    token: int
    open_interest: int
    day_high_oi: int | None = None
    day_low_oi: int | None = None


def decode_oi_message(message: bytes) -> list[OiUpdate]:
    if len(message) < 42:
        raise OiDecodeError(f"OI message too short: {len(message)}")
    code = struct.unpack_from(">h", message, 10)[0]
    if code not in SUPPORTED_OI_CODES:
        raise OiDecodeError(f"unsupported OI transaction code: {code}")
    log_time = struct.unpack_from(">i", message, 4)[0]
    count = struct.unpack_from(">H", message, 40)[0]
    record_size = {7201: 86, 17201: 90, 7202: 26, 17202: 38}[code]
    maximum = {7201: 5, 17201: 5, 7202: 17, 17202: 12}[code]
    if count > maximum:
        raise OiDecodeError(f"invalid {code} record count: {count}")
    required = 42 + count * record_size
    if len(message) < required:
        raise OiDecodeError(f"{code} requires {required} bytes, got {len(message)}")

    updates: list[OiUpdate] = []
    for index in range(count):
        base = 42 + index * record_size
        token = struct.unpack_from(">I", message, base)[0]
        if code == 7201:
            oi = struct.unpack_from(">I", message, base + 82)[0]
            high = low = None
        elif code == 17201:
            oi = struct.unpack_from(">q", message, base + 82)[0]
            high = low = None
        elif code == 7202:
            oi, high, low = struct.unpack_from(">III", message, base + 14)
        else:
            oi, high, low = struct.unpack_from(">qqq", message, base + 14)
        if token and oi >= 0:
            updates.append(OiUpdate(code, log_time, token, oi, high, low))
    return updates
