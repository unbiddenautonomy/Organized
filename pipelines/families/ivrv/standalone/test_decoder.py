import struct
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from voltr.feed.messages.market_by_price import DecodeError, decode_7208
from voltr.feed.order_book import BankNiftyOrderBook
from voltr.instruments.master import OptionContract
from voltr.feed.messages.indices import decode_banknifty_7207, decode_indices_message
from voltr.feed.udp import NseFoUdpReceiver


class FakeLzo:
    def decompress(self, compressed, _max_output):
        self.compressed = compressed
        return b"12345678" + make_message()


def make_message():
    message = bytearray(470)
    struct.pack_into(">h", message, 10, 7208)
    struct.pack_into(">h", message, 38, 470)
    return bytes(message)


class Decoder7208Test(unittest.TestCase):
    def test_decodes_enhanced_17208_offsets_and_64_bit_quantities(self):
        message = bytearray(558)
        struct.pack_into(">i", message, 4, 123456)
        struct.pack_into(">h", message, 10, 17208)
        struct.pack_into(">i", message, 14, 77)
        struct.pack_into(">h", message, 38, 558)
        struct.pack_into(">h", message, 40, 1)
        base = 42
        struct.pack_into(">IHHqi", message, base, 68390, 1, 2, 5_000_000_000, 5_721_500)
        message[base + 20] = ord("+")
        struct.pack_into(">iiii", message, base + 22, 100, 15, 123450, 5_720_000)
        for level in range(10):
            struct.pack_into(">qiHH", message, base + 60 + level * 16,
                             5_000_000_000 + level, 5_721_000 + level, level + 1, 1 if level < 5 else 2)
        struct.pack_into(">HHqq", message, base + 220, 0, 0, 7_000_000_000, 8_000_000_000)
        struct.pack_into(">Hiiii", message, base + 240, 3, 5_700_000, 5_710_000, 5_730_000, 5_690_000)

        update = decode_7208(bytes(message))[0]
        self.assertEqual(update.token, 68390)
        self.assertEqual(update.volume_traded_today, 5_000_000_000)
        self.assertEqual(update.last_traded_price, 57215.0)
        self.assertEqual(update.depth[0].quantity, 5_000_000_000)
        self.assertEqual(update.depth[0].price, 57210.0)
        self.assertEqual(update.total_buy_quantity, 7_000_000_000)
        self.assertEqual(update.open_price, 57100.0)

    def test_decodes_banknifty_spot_from_7207(self):
        message = bytearray(114)
        struct.pack_into(">i", message, 4, 1472820000)
        struct.pack_into(">h", message, 10, 7207)
        struct.pack_into(">h", message, 38, 114)
        struct.pack_into(">H", message, 40, 1)
        message[42:63] = b"BANK NIFTY".ljust(21, b" ")
        struct.pack_into(">i", message, 64, 5412345)
        self.assertEqual(decode_banknifty_7207(bytes(message)), (1472820000, 54123.45))

    def test_decodes_india_vix_from_7216_with_four_decimals(self):
        message = bytearray(114)
        struct.pack_into(">i", message, 4, 1472820000)
        struct.pack_into(">h", message, 10, 7216)
        struct.pack_into(">h", message, 38, 114)
        struct.pack_into(">H", message, 40, 1)
        message[42:63] = b"India VIX".ljust(21, b" ")
        struct.pack_into(">i", message, 64, 112675)
        self.assertEqual(decode_indices_message(bytes(message)), (1472820000, {"INDIA VIX": 11.2675}))

    def test_decodes_two_records(self):
        message = bytearray(470)
        struct.pack_into(">i", message, 4, 123456)
        struct.pack_into(">h", message, 10, 7208)
        struct.pack_into(">i", message, 14, 987)
        struct.pack_into(">h", message, 38, 470)
        struct.pack_into(">h", message, 40, 2)
        for index, token in enumerate((101, 202)):
            base = 42 + index * 214
            struct.pack_into(">IHHIi", message, base, token, 1, 2, 1000, 12345)
            message[base + 16] = ord("+")
            struct.pack_into(">iiii", message, base + 18, 50, 10, 111, 12222)
            struct.pack_into(">iiHH", message, base + 56, 500, 12300, 7, 1)
            struct.pack_into(">HHddHiiii", message, base + 176, 0, 0, 10000.0, 9000.0, 3, 12000, 12100, 12500, 11900)

        updates = decode_7208(bytes(message))
        self.assertEqual([item.token for item in updates], [101, 202])
        self.assertEqual(updates[0].last_traded_price, 123.45)
        self.assertEqual(updates[0].depth[0].price, 123.0)
        self.assertEqual(updates[0].broadcast_sequence_number, 987)

    def test_rejects_wrong_transaction_code(self):
        message = bytearray(470)
        struct.pack_into(">h", message, 10, 7200)
        with self.assertRaises(DecodeError):
            decode_7208(bytes(message))

    def test_decodes_compact_one_record_message(self):
        message = bytearray(256)
        struct.pack_into(">h", message, 10, 7208)
        struct.pack_into(">h", message, 38, 256)
        struct.pack_into(">h", message, 40, 1)
        struct.pack_into(">IHHIi", message, 42, 999, 1, 2, 1234, 4567)
        updates = decode_7208(bytes(message))
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].token, 999)
        self.assertEqual(updates[0].last_traded_price, 45.67)

    def test_rejects_legacy_sized_17208_message(self):
        message = bytearray(256)
        struct.pack_into(">h", message, 10, 17208)
        struct.pack_into(">h", message, 38, 256)
        struct.pack_into(">h", message, 40, 1)
        struct.pack_into(">IHHIi", message, 42, 999, 1, 2, 1234, 4567)
        with self.assertRaises(DecodeError):
            decode_7208(bytes(message))

    def test_unpacks_compressed_sequential_packet(self):
        fake_lzo = FakeLzo()
        receiver = NseFoUdpReceiver({"max_decompressed_bytes": 65535}, fake_lzo)
        datagram = b"FO" + struct.pack(">hH", 1, 3) + b"abc"
        messages = list(receiver.unpack_datagram(datagram))
        self.assertEqual(messages, [make_message()])
        self.assertEqual(fake_lzo.compressed, b"abc")

    def test_unpacks_uncompressed_sequential_packet(self):
        receiver = NseFoUdpReceiver({"max_decompressed_bytes": 65535}, FakeLzo())
        datagram = b"FO" + struct.pack(">hH", 1, 0) + b"12345678" + make_message()
        self.assertEqual(list(receiver.unpack_datagram(datagram)), [make_message()])

    def test_in_memory_order_book_splits_top_five_sides(self):
        message = bytearray(256)
        struct.pack_into(">h", message, 10, 7208)
        struct.pack_into(">i", message, 14, 77)
        struct.pack_into(">h", message, 38, 256)
        struct.pack_into(">h", message, 40, 1)
        struct.pack_into(">IHHIi", message, 42, 35000, 1, 2, 100, 7260000)
        for level in range(10):
            struct.pack_into(">iiHH", message, 42 + 56 + level * 12, 10 + level, 1000 + level, 1 + level, 0)
        update = decode_7208(bytes(message))[0]
        contract = OptionContract(35000, 26009, "OPTIDX", "BANKNIFTY", date(2026, 9, 29), Decimal("72600"), "CE")
        with tempfile.TemporaryDirectory() as directory:
            book = BankNiftyOrderBook({35000: contract}, Path(directory) / "book.jsonl")
            try:
                self.assertTrue(book.update(update))
                entry = book.get(35000)
                self.assertIsNotNone(entry)
                self.assertEqual(len(entry.bids), 5)
                self.assertEqual(len(entry.asks), 5)
                self.assertEqual([level.price_raw for level in entry.bids], [1000, 1001, 1002, 1003, 1004])
                self.assertEqual([level.price_raw for level in entry.asks], [1005, 1006, 1007, 1008, 1009])
                self.assertEqual(entry.broadcast_sequence_number, 77)

                underlying_message = bytearray(message)
                struct.pack_into(">I", underlying_message, 42, 26009)
                underlying_update = decode_7208(bytes(underlying_message))[0]
                self.assertTrue(book.update(underlying_update))
                underlying = book.get_underlying(26009)
                self.assertIsNotNone(underlying)
                self.assertEqual(underlying.token, 26009)
                self.assertEqual(len(underlying.bids), 5)
                self.assertEqual(len(underlying.asks), 5)
            finally:
                book.close()


if __name__ == "__main__":
    unittest.main()
