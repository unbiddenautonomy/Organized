"""Decoders for NSE CM index broadcasts: 7207 regular indices and 7216 India VIX."""

from __future__ import annotations

import struct


def decode_indices_message(message: bytes, price_divisor: float = 100.0) -> tuple[int, dict[str, float]]:
    """Decode CM regular indices (7207) or India VIX (7216)."""
    if len(message) < 42:
        return 0, {}
    transaction_code = struct.unpack_from(">h", message, 10)[0]
    if transaction_code not in (7207, 7216):
        return 0, {}
    log_time = struct.unpack_from(">i", message, 4)[0]
    count = struct.unpack_from(">H", message, 40)[0]
    record_size = 72
    if count > 6 or len(message) < 42 + count * record_size:
        return log_time, {}
    values: dict[str, float] = {}
    for index in range(count):
        offset = 42 + index * record_size
        name = message[offset:offset + 21].decode("ascii", errors="ignore").strip(" \x00").upper()
        raw = struct.unpack_from(">i", message, offset + 22)[0]
        if name and raw > 0:
            # NSE transmits regular indices with 2 decimals, while the India
            # VIX broadcast uses 4 decimal places.
            divisor = 10000.0 if transaction_code == 7216 else price_divisor
            values[name] = raw / divisor
    return log_time, values


def decode_banknifty_7207(message: bytes, price_divisor: float = 100.0) -> tuple[int, float] | None:
    """Return (exchange_log_time, BANKNIFTY spot) from a BCAST_INDICES message."""
    if len(message) < 42 or struct.unpack_from(">h", message, 10)[0] != 7207:
        return None
    log_time = struct.unpack_from(">i", message, 4)[0]
    record_count = struct.unpack_from(">H", message, 40)[0]
    # CM index records are packed on 2-byte boundaries: the 21-byte name is
    # followed by one alignment byte, then the 4-byte index value.
    record_size = 72
    if record_count > 6 or len(message) < 42 + record_count * record_size:
        return None
    accepted_names = {"BANK NIFTY", "BANKNIFTY", "NIFTY BANK"}
    for index in range(record_count):
        offset = 42 + index * record_size
        name = message[offset : offset + 21].decode("ascii", errors="ignore").strip(" \x00").upper()
        if name in accepted_names:
            raw_value = struct.unpack_from(">i", message, offset + 22)[0]
            if raw_value > 0:
                return log_time, raw_value / price_divisor
    return None
