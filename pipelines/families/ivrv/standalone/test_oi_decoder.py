import struct
import unittest

from iv_prediction.oi_decoder import OiDecodeError, decode_oi_message


class OiDecoderTest(unittest.TestCase):
    def make_message(self, code: int, size: int, count: int = 1) -> bytearray:
        message = bytearray(42 + size * count)
        struct.pack_into(">i", message, 4, 123456)
        struct.pack_into(">h", message, 10, code)
        struct.pack_into(">H", message, 40, count)
        return message

    def test_7201_contract_oi(self):
        message = self.make_message(7201, 86)
        struct.pack_into(">I", message, 42, 101)
        struct.pack_into(">I", message, 42 + 82, 999)
        update = decode_oi_message(message)[0]
        self.assertEqual((update.token, update.open_interest), (101, 999))

    def test_17201_contract_oi(self):
        message = self.make_message(17201, 90)
        struct.pack_into(">I", message, 42, 102)
        struct.pack_into(">q", message, 42 + 82, 5_000_000_000)
        self.assertEqual(decode_oi_message(message)[0].open_interest, 5_000_000_000)

    def test_7202_contract_oi_range(self):
        message = self.make_message(7202, 26)
        struct.pack_into(">I", message, 42, 103)
        struct.pack_into(">III", message, 42 + 14, 1000, 1200, 800)
        update = decode_oi_message(message)[0]
        self.assertEqual((update.open_interest, update.day_high_oi, update.day_low_oi), (1000, 1200, 800))

    def test_17202_contract_oi_range(self):
        message = self.make_message(17202, 38)
        struct.pack_into(">I", message, 42, 104)
        struct.pack_into(">qqq", message, 42 + 14, 6_000_000_000, 7_000_000_000, 4_000_000_000)
        update = decode_oi_message(message)[0]
        self.assertEqual((update.open_interest, update.day_high_oi), (6_000_000_000, 7_000_000_000))

    def test_rejects_truncated_message(self):
        with self.assertRaises(OiDecodeError):
            decode_oi_message(self.make_message(17202, 37))


if __name__ == "__main__":
    unittest.main()
