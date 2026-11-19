"""Bounded, nonblocking raw-UDP fanout. No model code runs in the receiver."""
from pathlib import Path
import os
import queue
import struct
import subprocess
import threading
import time

HEADER=struct.Struct('!cQdBI')
MAX_FRAME=1024*1024

def read_exact(stream,size):
    parts=[];remaining=size
    while remaining:
        data=stream.read(remaining)
        if not data:
            if remaining==size: return None
            raise EOFError('Truncated feed frame')
        parts.append(data);remaining-=len(data)
    return b''.join(parts)

def read_frame(stream):
    raw=read_exact(stream,HEADER.size)
    if raw is None:return None
    kind,seq,received,gap,length=HEADER.unpack(raw)
    if length>MAX_FRAME:raise ValueError('Feed frame exceeds bound')
    body=read_exact(stream,length) if length else b''
    if body is None:raise EOFError('Missing feed payload')
    return kind,seq,received,bool(gap),body

class FeedChannel:
    """Only this channel's writer can wait on its worker's pipe."""
    def __init__(self,name,command,cwd,log_path,capacity=4096,initial_gap=False,start_paused=False,journal=None):
        self.name=name;self.queue=queue.Queue(maxsize=capacity)
        self.journal=journal
        self.dropped=0;self.accepted=0;self.delivered=0;self.last_sequence=0
        self.fault_lock=threading.Lock()
        self.fault='RESTART_AFTER_FEED_INTERRUPTION' if initial_gap else None
        self.fault_generation=1 if initial_gap else 0
        self.closing=False;self.started=time.time();self.transport_error=None
        self.subscribed=not start_paused;self.subscribed_at=None;self.pre_ready_packets=0
        self.ready=threading.Event()
        if self.subscribed:self.ready.set()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        Path(log_path).parent.mkdir(parents=True,exist_ok=True)
        # These workers must keep pace with a loss-intolerant market-data feed.
        # BELOW_NORMAL lets desktop applications starve them on small hosts and
        # eventually turns an otherwise healthy stream into a queue overflow.
        worker_priority=getattr(subprocess,'ABOVE_NORMAL_PRIORITY_CLASS',0)
        with open(log_path,'ab',buffering=0) as log:
            self.process=subprocess.Popen(command,cwd=cwd,stdin=subprocess.PIPE,stdout=log,stderr=log,
                bufsize=0,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|worker_priority)
        self.thread=threading.Thread(target=self._write,name='feed-'+name,daemon=True);self.thread.start()

    def offer(self,kind,sequence,received,datagram):
        # No filesystem, prediction, wait, pipe write or process join here.
        if self.journal is not None:
            if not self.subscribed:self.pre_ready_packets+=1
            return True
        if not self.subscribed:
            self.pre_ready_packets+=1;return False
        if self.closing or self.transport_error:
            self.dropped+=1;return False
        try:self.queue.put_nowait((kind,sequence,received,datagram));self.accepted+=1;return True
        except queue.Full:
            self.dropped+=1
            with self.fault_lock:
                self.fault='FEED_QUEUE_OVERFLOW';self.fault_generation+=1
            return False

    def activate(self):
        # Initial subscription starts a fresh history after model loading. It
        # does not erase a gap/fault from a previously running worker.
        if not self.subscribed:self.subscribed_at=time.time();self.subscribed=True;self.ready.set()

    def _write(self):
        try:
            if self.journal is not None:
                while not self.closing and not self.ready.wait(.1):pass
                if not self.closing:
                    for kind,seq,received,data in self.journal.reader(lambda:self.closing):
                        packet=HEADER.pack(kind,seq,received,bool(self.fault),len(data))+data
                        view=memoryview(packet)
                        while view:
                            sent=self.process.stdin.write(view)
                            if not sent:raise BrokenPipeError('Worker stopped reading')
                            view=view[sent:]
                        self.delivered+=1;self.last_sequence=seq
                return
            while not self.closing:
                try:item=self.queue.get(timeout=.5)
                except queue.Empty:item=(b'H',self.last_sequence,time.time(),b'')
                if item is None:break
                kind,seq,received,data=item
                with self.fault_lock:
                    generation=self.fault_generation
                    signal_gap=kind!=b'H' and self.fault is not None
                packet=HEADER.pack(kind,seq,received,signal_gap,len(data))+data
                view=memoryview(packet)
                while view:
                    sent=self.process.stdin.write(view)
                    if not sent:raise BrokenPipeError('Worker stopped reading')
                    view=view[sent:]
                if kind!=b'H':
                    self.delivered+=1;self.last_sequence=seq
                    # One successful frame reports the gap. A new overflow
                    # during the pipe write must remain pending.
                    if signal_gap:
                        with self.fault_lock:
                            if self.fault_generation==generation:self.fault=None
        except (OSError,ValueError,RuntimeError) as exc:self.transport_error=type(exc).__name__;self.fault='WORKER_TRANSPORT_FAILED'
        finally:
            try:self.process.stdin.close()
            except (OSError,ValueError):pass

    def status(self):
        return dict(pid=self.process.pid,exit_code=self.process.poll(),queue_depth=self.queue.qsize() if self.journal is None else max(0,self.journal.last_sequence-self.last_sequence),
            queue_capacity=self.queue.maxsize,accepted=self.accepted,delivered=self.delivered,dropped=self.dropped,
            fault=self.fault,transport_error=self.transport_error,started_at=self.started,
            subscribed=self.subscribed,subscribed_at=self.subscribed_at,pre_ready_packets=self.pre_ready_packets,
            transport='journal' if self.journal else 'memory_queue')

    def close(self):
        self.closing=True
        self.ready.set()
        try:self.queue.put_nowait(None)
        except queue.Full:pass
        # A wedged worker cannot hold up shutdown of other workers/receiver.
        self.thread.join(timeout=.5)
        try:self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
        self.thread.join(timeout=1)

class Fanout:
    def __init__(self,journal=None):self.channels={};self.journal=journal;self.sequence=journal.last_sequence if journal else 0;self.datagrams=0;self.bytes=0;self.last_receive=None
    def publish(self,kind,datagram,received=None):
        self.sequence+=1;self.datagrams+=1;self.bytes+=len(datagram)
        self.last_receive=received if received is not None else time.time()
        if self.journal is not None:self.journal.append(kind,self.sequence,self.last_receive,datagram)
        for channel in tuple(self.channels.values()):channel.offer(kind,self.sequence,self.last_receive,datagram)
