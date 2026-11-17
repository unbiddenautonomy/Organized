"""43Weeks worker adapter: its own minute bars and OI from the decoded feed, then the OTM paper engine."""
import json
import logging
from pathlib import Path

from voltr.core.clock import wall_nse_second as wall_second
from voltr.market.atm_features import AtmFeatureStream

# Modules, not names: their classes are looked up when the adapter starts, so a test can swap them.
from . import atm_diagnostics, engine
from .bridge import Bridge
from .engine import stamp
from .settings import read_settings

ROOT = Path(__file__).resolve().parent  # config, models and logs; tests point this at a scratch copy
LOGGER = logging.getLogger("fixed_strike_live")  # historic logger name, kept for existing log readers


def _notify(bridge, method, *args):
    if bridge is not None:
        try:
            getattr(bridge, method)(*args)
        except Exception:
            LOGGER.exception("Optional 43Weeks extension failed during %s", method)


class OtmFeatureStream(AtmFeatureStream):
    """Own bars and OI; no dependency on the IV/RV model or its paper engine."""
    def __init__(self,config,contracts,lot,expiry,future,source,bridge):
        super().__init__(config,contracts,lot,expiry,future,source)
        if len(self.strikes)<3:raise ValueError('OTM pipeline requires three paired strikes')
        self.weeks43=bridge
    def _on_option_tick(self,token,price,log_time):_notify(self.weeks43,'tick',token,price,log_time)
    def _on_open_interest(self,token,open_interest,log_time):_notify(self.weeks43,'oi',token,open_interest,log_time)
    def _finalize_minute(self,minute):
        self.weeks43.snapshot(minute)
        if not self.weeks43.diagnostics:return
        # ATM diagnostics use this independent feature stream and never gate OTM.
        try:
            spot=self.spot_bars[minute].close;future=self.future_bars[minute].open
            n=min(range(len(self.strikes)),key=lambda i:abs(self.strikes[i]-spot))
            if n<2 or n+2>=len(self.strikes):return
            selected=self.strikes[n-2:n+3]
            prices={k:{side:self.option_bars[self.pairs[k][side]][minute].close for side in ('CE','PE')} for k in selected}
            row=self._build_current_row(minute,spot,future,selected,prices)
            if row is not None:self.weeks43.atm_row(row,self.latest_exchange_second)
        except KeyError:pass # Missing bars are ordinary warmup.
        except Exception as exc:self.weeks43.diagnostic_error=str(exc)

class OTMAdapter:
    def __init__(self,config,contracts,lot,expiry,future,source):
        folder=ROOT
        self.root=folder;self.read_settings=read_settings;self.stamp=stamp
        self.config=json.loads((folder/'config.json').read_text())
        if not self.config.get('enabled'):raise RuntimeError('43Weeks disabled by configuration')
        self.paper=engine.PaperEngine(self.config,engine.Ledger(folder/'logs'),engine.ModelRunner(folder),live=True,entry_settings=read_settings(folder))
        owner=self
        class DirectBridge(Bridge):
            def __init__(self):
                self.config=owner.config;self.oi_history={};self.diagnostic_error=None
                try:self.diagnostics=atm_diagnostics.AtmDiagnostics(folder) if owner.config.get('show_atm_diagnostics') else None
                except Exception as exc:self.diagnostics=None;self.diagnostic_error=str(exc)
            def submit(self,event):
                kind=event['kind'];paper=owner.paper
                if kind=='quote':paper.on_quotes([(event['key'],event['price'],event['second'])],event['second'])
                elif kind=='snapshot':
                    owner.reload_settings();now=event['second']
                    if abs(wall_second()-now)>owner.config['max_signal_age_seconds']:paper.error='stale_exchange_clock'
                    paper.on_quotes(event['quotes'],now);paper.on_snapshot(event['row'],now)
                elif kind=='atm_row' and self.diagnostics:
                    output=self.diagnostics.score(event['row'])
                    paper.ledger.put('atm_outputs',event['row']['datetime'],dict(time=stamp(event['second']),outputs=output,controls_entries=False))
                elif kind=='error':owner.fault(event['message'])
                elif kind=='notice':paper.ledger.put('events','notice:'+str(event.get('minute')),dict(time=stamp(event['second']),event='NO_SIGNAL',reason=event['reason']))
        self.bridge=DirectBridge();self.stream=OtmFeatureStream(config,contracts,lot,expiry,future,source,self.bridge)
        self.bridge.engine=self.stream
    def reload_settings(self):
        try:self.paper.apply_settings(self.read_settings(self.root),wall_second())
        except (OSError,ValueError) as exc:self.paper.settings_error=str(exc);self.paper._recheck_pending(wall_second())
    def index(self,message):self.stream.on_index_message(message)
    def market(self,update):self.stream.on_market_update(update)
    def oi(self,message):self.stream.on_oi_message(message)
    def pulse(self):self.reload_settings();self.paper.on_quotes([],wall_second())
    def fault(self,reason):self.paper.error=reason;self.paper._recheck_pending(wall_second())
    def status(self):return dict(state='RUNNING',error=self.paper.error,diagnostic_error=self.bridge.diagnostic_error,open_count=len(self.paper.positions))
    def close(self):self.paper.close()
