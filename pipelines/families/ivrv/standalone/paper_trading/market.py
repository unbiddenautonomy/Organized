"""Small decoders used only by the paper-trading risk layer."""

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


# Retained for callers/tests written against the original function name.
decode_indices_7207 = decode_indices_message
