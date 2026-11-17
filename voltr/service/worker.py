"""One pipeline per process: its framed raw feed arrives on a private stdin pipe.

The adapter class and lock file come from config/service.json (see registry.py)."""
from pathlib import Path
import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import date
from voltr.core.io import atomic_write_json as atomic_json,file_lock as single_writer

from . import registry
from .fanout import read_frame

def run(context,name,status_path):
    settings=registry.service_settings(context['config_path']);legacy=registry.host(settings)
    entry=registry.pipelines(settings)[name]
    config=legacy.load_live_config(Path(context['config_path']))
    config.update(contract_file=context['contract_file'],contract_asof_date=context['session_date'],contract_download_enabled=False)
    source=legacy.install_source_imports(config)
    contracts,lot,expiry=source['load_monthly_contracts'](Path(config['contract_file']),date.fromisoformat(context['session_date']),float(config.get('price_divisor',100)))
    future,_=source['load_front_future'](config)
    lock_path=registry.repo_path(entry['lock'])
    incoming=queue.Queue(maxsize=int(config.get('pipeline_reader_queue_capacity',1024)))
    def reader():
        try:
            while True:
                item=read_frame(sys.stdin.buffer)
                if item is None:break
                incoming.put(item)
        finally:incoming.put(None)
    with single_writer(lock_path):
        adapter=registry.import_object(entry['adapter'])(config,contracts,lot,expiry,future,source)
        if context.get('entry_fault'):adapter.fault(context['entry_fault'])
        receivers={b'F':source['NseFoUdpReceiver'](config,source['Lzo1zDecoder'](Path(config['lzo_dll']))),
            b'C':source['NseFoUdpReceiver'](config,source['Lzo1zDecoder'](Path(config['lzo_dll'])))}
        tracked=set(contracts)|{future};last_seq=None;processed=0;stale=0;malformed=0;fault=context.get('entry_fault')
        last_status=0.;last_recent=None;processing_delay=0.;last_exchange=None;last_gap_flag=False
        def checkpoint(state):
            nonlocal last_recent
            value=dict(pipeline=name,pid=os.getpid(),session_id=context['session_id'],state=state,
                heartbeat=time.time(),processed=processed,last_sequence=last_seq,dropped_stale=stale,
                malformed_packets=malformed,fault=fault,processing_delay_seconds=processing_delay,
                contract_file=config['contract_file'],contract_asof_date=config['contract_asof_date'],
                expiry=str(expiry),lot_size=lot)
            value['dashboard']=adapter.status()
            if state=='RUNNING' and (value['dashboard'].get('error') or value['dashboard'].get('fault')):
                value['state']='DEGRADED'
            atomic_json(status_path,value)
            if hasattr(adapter,'recent'):
                recent=adapter.recent();key=(len(recent),recent[-1].get('exchange_minute') if recent else None)
                if key!=last_recent:atomic_json(Path(status_path).with_name(name+'_recent.json'),recent);last_recent=key
        threading.Thread(target=reader,name='private-feed-reader',daemon=True).start()
        checkpoint('READY')
        try:
            while True:
                try:item=incoming.get(timeout=.25)
                except queue.Empty:item=(b'H',last_seq or 0,time.time(),False,b'')
                if item is None:break
                kind,seq,received,gap,payload=item
                if kind!=b'H':
                    processing_delay=time.time()-received
                    sequence_gap=last_seq is not None and seq!=last_seq+1
                    if gap or sequence_gap:
                        if fault in (None,'REPLAYING_FEED_NEW_ENTRIES_DISABLED'):
                            fault='FEED_GAP_NEW_ENTRIES_DISABLED'
                            adapter.fault(fault)
                        if fault=='FEED_GAP_NEW_ENTRIES_DISABLED' and (sequence_gap or not last_gap_flag) and hasattr(adapter,'note_feed_gap'):
                            adapter.note_feed_gap()
                    last_gap_flag=gap
                    last_seq=seq
                    if processing_delay>float(config.get('pipeline_max_backlog_seconds',10)) or processing_delay < -1:
                        stale+=1
                        if not fault:fault='REPLAYING_FEED_NEW_ENTRIES_DISABLED';adapter.fault(fault)
                    elif fault=='REPLAYING_FEED_NEW_ENTRIES_DISABLED':
                        fault=None;adapter.fault(None)
                    # Backlogged frames are still needed to reconstruct the
                    # complete minute history. Never discard them for age.
                    try:
                        for message in receivers[kind].unpack_datagram(payload):
                            if len(message)<12:continue
                            code=int.from_bytes(message[10:12],'big',signed=True)
                            if code==7207:adapter.index(message)
                            elif kind==b'F' and code in source['SUPPORTED_OI_CODES']:adapter.oi(message)
                            elif kind==b'F' and code in (7208,17208):
                                for update in source['decode_7208'](message,float(config.get('price_divisor',100)),tracked):adapter.market(update)
                        processed+=1
                    except (source['PacketError'],source['LzoError'],source['DecodeError']):malformed+=1
                    except Exception as exc:
                        fault='PIPELINE_ERROR:'+type(exc).__name__;adapter.fault(fault);traceback.print_exc()
                if time.monotonic()-last_status>=1:
                    adapter.pulse()
                    if (fault=='FEED_GAP_NEW_ENTRIES_DISABLED' and
                            0<=processing_delay<=float(config.get('pipeline_max_backlog_seconds',10)) and
                            hasattr(adapter,'feed_gap_recovered') and adapter.feed_gap_recovered()):
                        fault=None;adapter.fault(None)
                    checkpoint('DEGRADED' if fault else 'RUNNING');last_status=time.monotonic()
        finally:
            try:checkpoint('STOPPED')
            finally:adapter.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True)
    p.add_argument('--name',required=True,help='a pipeline listed in config/service.json');p.add_argument('--status',type=Path,required=True)
    args=p.parse_args();context=json.loads(args.context.read_text(encoding='utf-8'))
    try:run(context,args.name,args.status)
    except Exception as exc:
        atomic_json(args.status,dict(pipeline=args.name,pid=os.getpid(),session_id=context['session_id'],
            state='FAILED',heartbeat=time.time(),error=str(exc)))
        traceback.print_exc();raise

if __name__=='__main__':main()
