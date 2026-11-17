"""PCR worker adapter: the service's decoded feed drives PCR's bridge and runtime in this process."""
from datetime import datetime
from types import SimpleNamespace

from voltr.core.clock import MARKET_TIMEZONE

from .bridge import Bridge
from .runtime import PCRRuntime


class PCRAdapter:
    def __init__(self,config,contracts,lot,expiry,future,source):
        class InProcessBridge(Bridge):
            def _flush(self):
                if not self.batch:return
                batch=self.batch;self.batch=None
                if self.transport_fault:batch['transport_fault']=self.transport_fault
                now=datetime.now(MARKET_TIMEZONE).replace(tzinfo=None)
                self.runtime.on_snapshot(batch,processing_delay=(now-datetime.fromisoformat(batch['timestamp'])).total_seconds())
            def close(self):
                self._flush();self.closed=True
                self.runtime.checkpoint('FAULT' if self.runtime.fault else 'STOPPED')
        self.bridge=InProcessBridge(config,contracts,lot,expiry,future,source)
        self.bridge.process=SimpleNamespace(poll=lambda:None) # Its strategy already owns this process.
        self.bridge.runtime=PCRRuntime(self.bridge.context,self.bridge.directory)
    def index(self,message):self.bridge.index_message(message)
    def market(self,update):self.bridge.market(update)
    def oi(self,message):self.bridge.oi_message(message)
    def pulse(self):self.bridge.pulse()
    def fault(self,reason):self.bridge.transport_fault=reason;self.bridge.runtime.fault=reason
    def status(self):return dict(state='FAULT' if self.bridge.runtime.fault else 'RUNNING',fault=self.bridge.runtime.fault)
    def close(self):self.bridge.close()
