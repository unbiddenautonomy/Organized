"""Efficient latest-state BANKNIFTY option order book keyed by feed token."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from voltr.instruments.master import OptionContract
from voltr.feed.messages.market_by_price import DepthLevel, MarketUpdate7208


@dataclass(slots=True)
class OrderBookEntry:
    contract: OptionContract
    broadcast_sequence_number: int = 0
    log_time: int = 0
    trading_status: int = 0
    volume_traded_today: int = 0
    last_traded_price: float = 0.0
    total_buy_quantity: float = 0.0
    total_sell_quantity: float = 0.0
    bids: tuple[DepthLevel, ...] = ()
    asks: tuple[DepthLevel, ...] = ()


@dataclass(slots=True)
class UnderlyingBookEntry:
    token: int
    symbol: str
    broadcast_sequence_number: int = 0
    log_time: int = 0
    trading_status: int = 0
    volume_traded_today: int = 0
    last_traded_price: float = 0.0
    total_buy_quantity: float = 0.0
    total_sell_quantity: float = 0.0
    bids: tuple[DepthLevel, ...] = ()
    asks: tuple[DepthLevel, ...] = ()


class BankNiftyOrderBook:
    """O(1) token lookup/update with append-only JSONL persistence."""

    def __init__(self, contracts: dict[int, OptionContract], log_path: Path, persist_updates: bool = False) -> None:
        self._entries: dict[int, OrderBookEntry] = {
            token: OrderBookEntry(contract) for token, contract in contracts.items()
        }
        self._underlyings: dict[int, UnderlyingBookEntry] = {}
        for contract in contracts.values():
            self._underlyings.setdefault(
                contract.underlying_token,
                UnderlyingBookEntry(contract.underlying_token, contract.symbol),
            )
        self._stream: TextIO | None = None
        if persist_updates:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = log_path.open("a", encoding="utf-8", buffering=1)
        self.update_count = 0

    def update(self, update: MarketUpdate7208) -> bool:
        entry = self._entries.get(update.token)
        if entry is not None:
            self._apply_update(entry, update)
            self.update_count += 1
            if self._stream is not None:
                self._write_option_event(entry)
            return True

        underlying = self._underlyings.get(update.token)
        if underlying is not None:
            self._apply_update(underlying, update)
            self.update_count += 1
            if self._stream is not None:
                self._write_underlying_event(underlying)
            return True
        return False

    @staticmethod
    def _apply_update(entry: OrderBookEntry | UnderlyingBookEntry, update: MarketUpdate7208) -> None:
        # NSE 7208 RecordBuffer is positional: entries 0..4 are the five
        # best buys and entries 5..9 are the five best sells. The protocol
        # states BbBuySellFlag is currently zero, so it is not a side marker.
        entry.broadcast_sequence_number = update.broadcast_sequence_number
        entry.log_time = update.log_time
        entry.trading_status = update.trading_status
        entry.volume_traded_today = update.volume_traded_today
        entry.last_traded_price = update.last_traded_price
        entry.total_buy_quantity = update.total_buy_quantity
        entry.total_sell_quantity = update.total_sell_quantity
        entry.bids = tuple(update.depth[:5])
        entry.asks = tuple(update.depth[5:10])

    @staticmethod
    def _levels(levels: tuple[DepthLevel, ...]) -> list[dict]:
        return [
            {
                "level": level_number,
                "quantity": level.quantity,
                "price_raw": level.price_raw,
                "price": level.price,
                "number_of_orders": level.number_of_orders,
            }
            for level_number, level in enumerate(levels, 1)
        ]

    def _write_option_event(self, entry: OrderBookEntry) -> None:
        contract = entry.contract
        event = {
            "event_type": "option",
            "fininstrmid": contract.token,
            "undrlygfininstrmid": contract.underlying_token,
            "fininstrmnm": contract.instrument_name,
            "tckrsymb": contract.symbol,
            "xprydt": contract.expiry.isoformat(),
            "strkpric": str(contract.strike_price),
            "optntp": contract.option_type,
            "broadcast_sequence_number": entry.broadcast_sequence_number,
            "log_time": entry.log_time,
            "trading_status": entry.trading_status,
            "volume_traded_today": entry.volume_traded_today,
            "last_traded_price": entry.last_traded_price,
            "total_buy_quantity": entry.total_buy_quantity,
            "total_sell_quantity": entry.total_sell_quantity,
            "bids": self._levels(entry.bids),
            "asks": self._levels(entry.asks),
        }
        assert self._stream is not None
        self._stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    def _write_underlying_event(self, entry: UnderlyingBookEntry) -> None:
        event = {
            "event_type": "underlying",
            "fininstrmid": entry.token,
            "tckrsymb": entry.symbol,
            "broadcast_sequence_number": entry.broadcast_sequence_number,
            "log_time": entry.log_time,
            "trading_status": entry.trading_status,
            "volume_traded_today": entry.volume_traded_today,
            "last_traded_price": entry.last_traded_price,
            "total_buy_quantity": entry.total_buy_quantity,
            "total_sell_quantity": entry.total_sell_quantity,
            "bids": self._levels(entry.bids),
            "asks": self._levels(entry.asks),
        }
        assert self._stream is not None
        self._stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    def get(self, token: int) -> OrderBookEntry | None:
        return self._entries.get(token)

    def reset(self) -> None:
        """Discard quotes from a previous replay pass without reallocating contracts."""
        for token, entry in tuple(self._entries.items()):
            self._entries[token] = OrderBookEntry(entry.contract)
        for token, entry in tuple(self._underlyings.items()):
            self._underlyings[token] = UnderlyingBookEntry(token, entry.symbol)
        self.update_count = 0

    def get_underlying(self, token: int) -> UnderlyingBookEntry | None:
        return self._underlyings.get(token)

    @property
    def underlying_count(self) -> int:
        return len(self._underlyings)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()

    def __len__(self) -> int:
        return len(self._entries)
