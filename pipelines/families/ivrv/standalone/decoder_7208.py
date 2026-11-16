"""Decoder for NSE FO BCAST_ONLY_MBP transaction code 7208."""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass


BCAST_HEADER_SIZE = 40
MESSAGE_SIZE = 470
MIN_MESSAGE_SIZE = 42
LEGACY_RECORD_SIZE = 214
ENHANCED_RECORD_SIZE = 258
MAX_RECORDS = 2
TRANSACTION_CODE = 7208
TRANSACTION_CODES = frozenset((7208, 17208))


class DecodeError(ValueError):
    """Raised when a broadcast message is malformed or not code 7208."""


@dataclass(slots=True)
class DepthLevel:
    quantity: int
    price_raw: int
    price: float
    number_of_orders: int
    buy_sell_flag: int


@dataclass(slots=True)
class MarketUpdate7208:
    transaction_code: int
    broadcast_sequence_number: int
    log_time: int
    message_length: int
    record_number: int
    token: int
    book_type: int
    trading_status: int
    volume_traded_today: int
    last_traded_price_raw: int
    last_traded_price: float
    net_change_indicator: str
    volume_exceeded_indicator: str
    net_price_change_raw: int
    net_price_change: float
    last_trade_quantity: int
    last_trade_time: int
    average_trade_price_raw: int
    average_trade_price: float
    auction_number: int
    auction_status: int
    initiator_type: int
    initiator_price_raw: int
    initiator_price: float
    initiator_quantity: int
    auction_price_raw: int
    auction_price: float
    auction_quantity: int
    depth: list[DepthLevel]
    total_buy_flag: int
    total_sell_flag: int
    total_buy_quantity: float
    total_sell_quantity: float
    indicator: int
    closing_price: float
    open_price: float
    high_price: float
    low_price: float

    def to_dict(self) -> dict:
        return asdict(self)


def _char(value: int) -> str:
    return "" if value == 0 else bytes((value,)).decode("ascii", errors="replace").strip()


def decode_7208(
    message: bytes,
    price_divisor: float = 100.0,
    token_filter: set[int] | frozenset[int] | None = None,
) -> list[MarketUpdate7208]:
    """Decode one MS_BCAST_ONLY_MBP message, optionally retaining selected tokens only."""
    if len(message) < MIN_MESSAGE_SIZE:
        raise DecodeError(f"7208 message needs at least {MIN_MESSAGE_SIZE} bytes, got {len(message)}")

    transaction_code = struct.unpack_from(">h", message, 10)[0]
    if transaction_code not in TRANSACTION_CODES:
        raise DecodeError(f"expected transaction code 7208/17208, got {transaction_code}")

    log_time = struct.unpack_from(">i", message, 4)[0]
    sequence_number = struct.unpack_from(">i", message, 14)[0]
    message_length = struct.unpack_from(">h", message, 38)[0]
    number_of_records = struct.unpack_from(">h", message, 40)[0]
    if not 0 <= number_of_records <= MAX_RECORDS:
        raise DecodeError(f"invalid 7208 record count: {number_of_records}")
    record_size = ENHANCED_RECORD_SIZE if transaction_code == 17208 else LEGACY_RECORD_SIZE
    required_length = MIN_MESSAGE_SIZE + number_of_records * record_size
    if len(message) < required_length:
        raise DecodeError(
            f"7208 message with {number_of_records} record(s) needs "
            f"{required_length} bytes, got {len(message)}"
        )

    updates: list[MarketUpdate7208] = []
    for record_number in range(number_of_records):
        base = 42 + record_number * record_size
        token = struct.unpack_from(">I", message, base)[0]
        if token_filter is not None and token not in token_filter:
            continue
        book_type, trading_status = struct.unpack_from(">HH", message, base + 4)
        if transaction_code == 17208:
            volume = struct.unpack_from(">q", message, base + 8)[0]
            ltp = struct.unpack_from(">i", message, base + 16)[0]
            net_indicator = _char(message[base + 20])
            volume_indicator = _char(message[base + 21])
            net_change, last_qty, last_time, average_price = struct.unpack_from(">iiii", message, base + 22)
            auction_number, auction_status, initiator_type = struct.unpack_from(">HHH", message, base + 38)
            initiator_price, initiator_qty, auction_price, auction_qty = struct.unpack_from(">iiii", message, base + 44)
            depth_base, depth_stride = base + 60, 16
            totals_base, indicator_offset, prices_base = base + 220, base + 240, base + 242
        else:
            volume, ltp = struct.unpack_from(">Ii", message, base + 8)
            net_indicator = _char(message[base + 16])
            volume_indicator = _char(message[base + 17])
            net_change, last_qty, last_time, average_price = struct.unpack_from(">iiii", message, base + 18)
            auction_number, auction_status, initiator_type = struct.unpack_from(">HHH", message, base + 34)
            initiator_price, initiator_qty, auction_price, auction_qty = struct.unpack_from(">iiii", message, base + 40)
            depth_base, depth_stride = base + 56, 12
            totals_base, indicator_offset, prices_base = base + 176, base + 196, base + 198

        depth: list[DepthLevel] = []
        for level in range(10):
            level_base = depth_base + level * depth_stride
            if transaction_code == 17208:
                quantity, price, order_count, buy_sell = struct.unpack_from(">qiHH", message, level_base)
            else:
                quantity, price, order_count, buy_sell = struct.unpack_from(">iiHH", message, level_base)
            depth.append(DepthLevel(quantity, price, price / price_divisor, order_count, buy_sell))

        total_buy_flag, total_sell_flag = struct.unpack_from(">HH", message, totals_base)
        if transaction_code == 17208:
            total_buy_quantity, total_sell_quantity = struct.unpack_from(">qq", message, totals_base + 4)
        else:
            total_buy_quantity, total_sell_quantity = struct.unpack_from(">dd", message, totals_base + 4)
        indicator = struct.unpack_from(">H", message, indicator_offset)[0]
        close_price, open_price, high_price, low_price = struct.unpack_from(">iiii", message, prices_base)

        updates.append(MarketUpdate7208(
            transaction_code=transaction_code,
            broadcast_sequence_number=sequence_number,
            log_time=log_time,
            message_length=message_length,
            record_number=record_number + 1,
            token=token,
            book_type=book_type,
            trading_status=trading_status,
            volume_traded_today=volume,
            last_traded_price_raw=ltp,
            last_traded_price=ltp / price_divisor,
            net_change_indicator=net_indicator,
            volume_exceeded_indicator=volume_indicator,
            net_price_change_raw=net_change,
            net_price_change=net_change / price_divisor,
            last_trade_quantity=last_qty,
            last_trade_time=last_time,
            average_trade_price_raw=average_price,
            average_trade_price=average_price / price_divisor,
            auction_number=auction_number,
            auction_status=auction_status,
            initiator_type=initiator_type,
            initiator_price_raw=initiator_price,
            initiator_price=initiator_price / price_divisor,
            initiator_quantity=initiator_qty,
            auction_price_raw=auction_price,
            auction_price=auction_price / price_divisor,
            auction_quantity=auction_qty,
            depth=depth,
            total_buy_flag=total_buy_flag,
            total_sell_flag=total_sell_flag,
            total_buy_quantity=total_buy_quantity,
            total_sell_quantity=total_sell_quantity,
            indicator=indicator,
            closing_price=close_price / price_divisor,
            open_price=open_price / price_divisor,
            high_price=high_price / price_divisor,
            low_price=low_price / price_divisor,
        ))
    return updates
