import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from voltr.feed.lzo import Lzo1zDecoder,LzoError


def decoder():
    d=Lzo1zDecoder.__new__(Lzo1zDecoder);d._buffers=threading.local()
    def native(source,length,destination,output_length,_):
        size=ctypes.cast(output_length,ctypes.POINTER(ctypes.c_size_t))
        if length>size[0]:return -5
        ctypes.memmove(destination,source,length);size[0]=length
        return 0
    d._decompress=native
    return d


def test_reused_buffer_does_not_leak_previous_message_or_modify_returned_bytes():
    d=decoder();first=d.decompress(bytes(range(256)),512)
    original=d._buffers.destination
    assert d.decompress(b'new',512)==b'new'
    assert d._buffers.destination is original and first==bytes(range(256))
    assert d.decompress(b'bigger'*200,2048)==b'bigger'*200
    with pytest.raises(LzoError):d.decompress(b'oversized',3)
    with pytest.raises(LzoError):d.decompress(b'')
    with pytest.raises(ValueError):d.decompress(b'x',0)


def test_shared_decoder_keeps_buffers_private_to_each_thread():
    d=decoder();barrier=threading.Barrier(4)
    def read(i):
        barrier.wait()
        data=bytes([i])*2000
        for _ in range(100):assert d.decompress(data)==data
        return id(d._buffers.destination)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len(set(pool.map(read,range(4))))==4
