"""Shared append-only UDP journal, split into UTC minute files.

The receiver writes each datagram once. Every worker has an independent read
cursor, so a slow worker cannot fill another worker's queue. A fresh worker
starts at the oldest retained minute and rebuilds its own feature history.
"""
from pathlib import Path
import struct
import threading
import time

RECORD = struct.Struct('!cQdI')
MAX_DATAGRAM = 65535

class FeedJournal:
    def __init__(self, folder, retention_minutes=65):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.retention_minutes = max(61, int(retention_minutes))
        self.lock = threading.Lock()
        self.minute = None
        self.handle = None
        self.written = 0
        self.error = None
        self.last_flush = time.monotonic()
        self.recovered = False
        self.last_sequence = 0
        self._restore_tail()
        self.stop = threading.Event()
        self.flusher = threading.Thread(target=self._flush_loop, name='feed-journal-flush', daemon=True)
        self.flusher.start()

    def _flush_loop(self):
        while not self.stop.wait(.01):
            with self.lock:
                if self.handle:
                    try:
                        self.handle.flush()
                    except OSError as exc:
                        self.error = str(exc)

    def _restore_tail(self):
        files = sorted(self.folder.glob('*.feed'))
        if not files:
            return
        self.recovered = True
        for last in reversed(files):
            with last.open('r+b') as handle:
                good = 0
                while True:
                    raw = handle.read(RECORD.size)
                    if not raw or len(raw) != RECORD.size:
                        break
                    _kind, seq, _received, length = RECORD.unpack(raw)
                    if length > MAX_DATAGRAM or len(handle.read(length)) != length:
                        break
                    good = handle.tell()
                    self.last_sequence = seq
                handle.truncate(good)
            if self.last_sequence:
                break

    def append(self, kind, sequence, received, datagram):
        if len(datagram) > MAX_DATAGRAM:
            raise ValueError('UDP datagram exceeds journal record bound')
        minute = int(received // 60)
        with self.lock:
            try:
                if minute != self.minute:
                    if self.handle:
                        self.handle.flush()
                        self.handle.close()
                    self.minute = minute
                    self.handle = (self.folder / f'{minute:012d}.feed').open('ab', buffering=1024 * 1024)
                    self._prune(minute)
                self.handle.write(RECORD.pack(kind, sequence, received, len(datagram)))
                self.handle.write(datagram)
                self.written += 1
                self.last_sequence = sequence
                # Batch writes to avoid a syscall for every market packet.
                now = time.monotonic()
                if now - self.last_flush >= .01:
                    self.handle.flush()
                    self.last_flush = now
            except OSError as exc:
                self.error = str(exc)
                raise

    def _prune(self, minute):
        for path in self.folder.glob('*.feed'):
            try:
                if int(path.stem) < minute - self.retention_minutes + 1:
                    path.unlink()
            except (ValueError, OSError):
                pass

    def reader(self, stop, start_sequence=0):
        """Yield complete records, waiting for the live tail without skipping it."""
        path = None
        handle = None
        offset = 0
        try:
            while not stop():
                if handle is None:
                    files = sorted(self.folder.glob('*.feed'))
                    if not files:
                        time.sleep(.01)
                        continue
                    path = files[0]
                    handle = path.open('rb', buffering=1024 * 1024)
                    offset = 0
                handle.seek(offset)
                header = handle.read(RECORD.size)
                if len(header) == RECORD.size:
                    kind, seq, received, length = RECORD.unpack(header)
                    if length > MAX_DATAGRAM:
                        raise ValueError(f'Invalid journal record at {path}:{offset}')
                    data = handle.read(length)
                    if len(data) == length:
                        offset += RECORD.size + length
                        if seq > start_sequence:
                            yield kind, seq, received, data
                        continue
                # The directory only changes when a minute rolls over (or a
                # retained file is pruned). Avoid a full glob for every packet.
                files = sorted(self.folder.glob('*.feed'))
                newer = next((p for p in files if p > path), None)
                if newer:
                    # The writer closes and flushes the old minute before it
                    # creates the new one. A reader may have observed a
                    # partial tail just before that flush. Reopen the old
                    # file and retry its last record before advancing.
                    handle.close()
                    handle = path.open('rb', buffering=1024 * 1024)
                    handle.seek(offset)
                    tail = handle.read(RECORD.size)
                    if tail:
                        if len(tail) != RECORD.size:
                            raise RuntimeError('Incomplete journal header at minute rollover')
                        _kind, _seq, _received, length = RECORD.unpack(tail)
                        if length > MAX_DATAGRAM or len(handle.read(length)) != length:
                            raise RuntimeError('Incomplete journal record at minute rollover')
                        continue
                    handle.close()
                    handle = newer.open('rb', buffering=1024 * 1024)
                    path = newer
                    offset = 0
                else:
                    time.sleep(.01)
        finally:
            if handle:
                handle.close()

    def close(self):
        self.stop.set()
        self.flusher.join(timeout=2)
        with self.lock:
            if self.handle:
                self.handle.flush()
                self.handle.close()
                self.handle = None
