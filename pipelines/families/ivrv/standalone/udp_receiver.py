"""NSE FO multicast receiver and sequential packet unpacker."""

from __future__ import annotations

import logging
import socket
import struct
from collections.abc import Iterator

from lzo_codec import Lzo1zDecoder, LzoError


LOGGER = logging.getLogger(__name__)
PREFIX_SIZE = 8
BCAST_HEADER_SIZE = 40


class PacketError(ValueError):
    """Raised for malformed UDP/sequential-packing data."""


class NseFoUdpReceiver:
    def __init__(self, config: dict, lzo: Lzo1zDecoder) -> None:
        self.config = config
        self.lzo = lzo
        self.socket: socket.socket | None = None

    def open(self) -> None:
        multicast_ip = self.config["multicast_ip"]
        interface_ip = self.config.get("interface_ip", "0.0.0.0")
        port = int(self.config["port"])

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(self.config["receive_buffer_bytes"]))
        sock.settimeout(float(self.config["socket_timeout_seconds"]))
        sock.bind(("", port))
        membership = socket.inet_aton(multicast_ip) + socket.inet_aton(interface_ip)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        self.socket = sock
        LOGGER.info("Listening for %s multicast on %s:%d via interface %s",
                    self.config.get("receiver_name", "NSE FO"), multicast_ip, port, interface_ip)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def receive(self) -> Iterator[bytes]:
        if self.socket is None:
            raise RuntimeError("receiver is not open")
        max_datagram = int(self.config["max_datagram_bytes"])
        while True:
            try:
                datagram, sender = self.socket.recvfrom(max_datagram)
            except socket.timeout:
                continue
            if self.config.get("log_raw_datagrams", True):
                LOGGER.info(
                    "RAW_UDP sender=%s:%d bytes=%d hex=%s",
                    sender[0], sender[1], len(datagram), datagram.hex(),
                )
            try:
                yield from self.unpack_datagram(datagram)
            except (PacketError, LzoError):
                continue

    def unpack_datagram(self, datagram: bytes) -> Iterator[bytes]:
        if len(datagram) < 4:
            raise PacketError("datagram is shorter than the 4-byte packet header")
        number_of_packets = struct.unpack_from(">h", datagram, 2)[0]
        if not 0 <= number_of_packets <= 200:
            raise PacketError(f"invalid sequential packet count: {number_of_packets}")

        cursor = 4
        for _ in range(number_of_packets):
            if cursor + 2 > len(datagram):
                raise PacketError("missing compression-length field")
            compressed_length = struct.unpack_from(">H", datagram, cursor)[0]
            cursor += 2

            if compressed_length:
                end = cursor + compressed_length
                if end > len(datagram):
                    raise PacketError("compressed message exceeds datagram boundary")
                payload = self.lzo.decompress(
                    datagram[cursor:end], int(self.config["max_decompressed_bytes"])
                )
                cursor = end
            else:
                # Uncompressed broadcast data has the same 8-byte prefix. Its
                # BCAST_HEADER MessageLength field determines this packet size.
                length_offset = cursor + PREFIX_SIZE + 38
                if length_offset + 2 > len(datagram):
                    raise PacketError("uncompressed message header is truncated")
                message_length = struct.unpack_from(">H", datagram, length_offset)[0]
                end = cursor + PREFIX_SIZE + message_length
                if message_length < BCAST_HEADER_SIZE or end > len(datagram):
                    raise PacketError(f"invalid uncompressed message length: {message_length}")
                payload = datagram[cursor:end]
                cursor = end

            if len(payload) < PREFIX_SIZE + BCAST_HEADER_SIZE:
                raise PacketError("broadcast payload is too short")
            yield payload[PREFIX_SIZE:]
