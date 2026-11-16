"""ctypes binding for NSE's LZO1Z decompression routine."""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path


class LzoError(RuntimeError):
    """Raised when the native LZO library cannot decompress a message."""


_ERRORS = {
    -1: "generic error",
    -4: "input overrun (truncated compressed data)",
    -5: "output overrun (increase max_decompressed_bytes)",
    -6: "look-behind overrun (corrupt compressed data)",
    -7: "EOF marker not found",
    -8: "compressed input was not fully consumed",
}


class Lzo1zDecoder:
    """Load lzo2.dll and expose lzo1z_decompress_safe."""

    def __init__(self, dll_path: str | Path | None = None) -> None:
        base = Path(__file__).resolve().parent
        # add_dll_directory rejects relative folders, so always pass an absolute path.
        configured = Path(dll_path).resolve() if dll_path else base / "native" / "lzo2.dll"
        if not configured.is_file():
            raise FileNotFoundError(f"LZO DLL not found at {configured}")

        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(configured.parent))
        self._lib = ctypes.CDLL(str(configured))
        self._decompress = self._lib.lzo1z_decompress_safe
        self._decompress.argtypes = (
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        )
        self._decompress.restype = ctypes.c_int
        self._buffers = threading.local()

    def decompress(self, compressed: bytes, max_output: int = 65535) -> bytes:
        if not compressed:
            raise LzoError("empty compressed message")
        source = (ctypes.c_ubyte * len(compressed)).from_buffer_copy(compressed)
        if max_output <= 0:
            raise ValueError('max_output must be positive')
        destination = getattr(self._buffers, 'destination', None)
        if destination is None or ctypes.sizeof(destination) < max_output:
            destination = (ctypes.c_ubyte * max_output)()
            self._buffers.destination = destination
        output_length = ctypes.c_size_t(max_output)
        result = self._decompress(
            source,
            len(compressed),
            destination,
            ctypes.byref(output_length),
            None,
        )
        if result != 0:
            detail = _ERRORS.get(result, "unknown LZO error")
            raise LzoError(f"lzo1z_decompress_safe failed ({result}: {detail})")
        if output_length.value > max_output:
            raise LzoError('native decoder returned an out-of-bounds length')
        # Copy directly from native memory. Array slicing first creates a
        # Python integer list for every byte of every multicast message.
        return ctypes.string_at(destination, output_length.value)
