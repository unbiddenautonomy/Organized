"""The template adapter honours the worker contract without any market data or models."""
from datetime import date
import struct

from pipelines._template.pipeline import TemplateAdapter
from voltr.feed.messages.indices import decode_banknifty_7207


def index_message(value: float) -> bytes:
    message = bytearray(42 + 72)
    struct.pack_into(">h", message, 10, 7207)
    struct.pack_into(">H", message, 40, 1)
    message[42:42 + 21] = b"Nifty Bank".ljust(21, b" ")
    struct.pack_into(">i", message, 64, int(value * 100))
    return bytes(message)


def test_adapter_contract():
    adapter = TemplateAdapter({}, {}, 30, date(2026, 11, 24), 99, {"decode_banknifty_7207": decode_banknifty_7207})
    adapter.index(index_message(55000.0))
    adapter.market(object())
    adapter.oi(b"")
    adapter.pulse()
    adapter.fault("FEED_GAP_NEW_ENTRIES_DISABLED")
    status = adapter.status()
    assert status["received"] == {"index": 1, "market": 1, "oi": 1}
    assert status["fault"] == "FEED_GAP_NEW_ENTRIES_DISABLED" and adapter.last_pulse
    adapter.close()
