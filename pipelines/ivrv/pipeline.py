"""IVRV worker adapter: the service feeds it decoded ticks; models run on a separate minute thread."""
from pathlib import Path
import queue
import threading
import time
from . import live as legacy

from voltr.core.clock import wall_nse_second as wall_second

class IsolatedIVRV(legacy.LiveFeatureInference):
    entry_block_reason=None
    def _try_open_paper_position(self,minute,row,prediction):
        with self.risk_lock:
            reason=self.entry_block_reason
            if self.latest_exchange_second is None or abs(wall_second()-self.latest_exchange_second)>10:
                reason=reason or 'stale_exchange_clock'
            if reason:return dict(state='NO_TRADE',reason=reason,entered=False,eligible=False)
            return super()._try_open_paper_position(minute,row,prediction)

    def _publish(self,*args,**kwargs):
        with self.risk_lock:return super()._publish(*args,**kwargs)


class IVRVMinuteSampler:
    """Keep the exact first/last tick selections used by the live feature row.

    The model thread receives one immutable sample per completed exchange minute.
    Tick-time position marking does not wait for model prediction or history work.
    """
    def __init__(self,contracts,future,source,price_divisor,emit,rewind_tolerance=5):
        self.contracts=set(contracts);self.future=future;self.emit=emit
        self.decode_spot=source['decode_banknifty_7207'];self.decode_oi=source['decode_oi_message']
        self.oi_error=source['OiDecodeError'];self.price_divisor=price_divisor
        self.rewind_tolerance=rewind_tolerance;self.generation=0
        self.spot={};self.future_prices={};self.options={};self.oi={}
        self.latest_spot_minute=None;self.latest_future_minute=None;self.last_emitted=None
        self.latest_exchange_second=None

    def _tick(self,bars,minute,price):
        bar=bars.get(minute)
        if bar is None:bars[minute]=legacy.MinuteBar(price,price,price,price)
        else:bar.update(price)

    def index(self,message):
        decoded=self.decode_spot(message,self.price_divisor)
        if decoded is None:return None
        second,price=decoded;minute=second//60
        self._rewind(minute)
        self._tick(self.spot,minute,price)
        self.latest_exchange_second=max(self.latest_exchange_second or second,second)
        self.latest_spot_minute=max(self.latest_spot_minute or minute,minute)
        return second

    def market(self,update):
        if update.last_traded_price<=0:return None
        second=update.log_time;minute=second//60
        self._rewind(minute)
        self.latest_exchange_second=max(self.latest_exchange_second or second,second)
        if update.token==self.future:
            self._tick(self.future_prices,minute,update.last_traded_price)
            self.latest_future_minute=max(self.latest_future_minute or minute,minute)
        elif update.token in self.contracts:
            self._tick(self.options.setdefault(update.token,{}),minute,update.last_traded_price)
        return second

    def oi_message(self,message):
        try:updates=self.decode_oi(message)
        except self.oi_error:return
        priority={7201:1,7202:2,17201:3,17202:4}
        for update in updates:
            if update.token not in self.contracts and update.token!=self.future:continue
            old=self.oi.get(update.token)
            if old and (update.log_time<old[1] or
                update.log_time==old[1] and priority.get(update.transaction_code,0)<priority.get(old[2],0)):
                continue
            self.oi[update.token]=(update.open_interest,update.log_time,update.transaction_code)

    def _rewind(self,minute):
        latest=max((x for x in (self.latest_spot_minute,self.latest_future_minute) if x is not None),default=None)
        if latest is None or minute>=latest-self.rewind_tolerance:return
        self.spot.clear();self.future_prices.clear();self.options.clear();self.oi.clear()
        self.latest_spot_minute=self.latest_future_minute=self.last_emitted=self.latest_exchange_second=None
        self.generation+=1

    def complete(self):
        if self.latest_spot_minute is None or self.latest_future_minute is None:return
        through=min(self.latest_spot_minute,self.latest_future_minute)-2
        start=through if self.last_emitted is None else self.last_emitted+1
        if start>through:return
        for minute in range(start,through+1):
            spot=self.spot.get(minute);future=self.future_prices.get(minute)
            sample=dict(minute=minute,spot_close=spot.close if spot else None,
                future_open=future.open if future else None,
                option_closes={token:bars[minute].close for token,bars in self.options.items() if minute in bars},
                oi=dict(self.oi),exchange_second=self.latest_exchange_second,received_at=time.time(),
                generation=self.generation)
            self.emit(sample)
        self.last_emitted=through
        cutoff=through-3
        for bars in (self.spot,self.future_prices):
            for old in tuple(bars):
                if old<cutoff:del bars[old]
        for token,bars in tuple(self.options.items()):
            for old in tuple(bars):
                if old<cutoff:del bars[old]
            if not bars:del self.options[token]

class IVRVAdapter:
    def __init__(self,config,contracts,lot,expiry,future,source):
        pipeline=legacy.FixedStrikeInferencePipeline(Path(config['inference_config']).resolve())
        self.engine=IsolatedIVRV(config,contracts,lot,expiry,future,pipeline,source)
        self.engine.risk_lock=threading.RLock()
        self.samples=queue.Queue(maxsize=256);self.stopping=threading.Event()
        self.transport_fault=None;self.model_fault=None
        self.max_backlog=float(config.get('pipeline_max_backlog_seconds',10))
        self.sampler=IVRVMinuteSampler(contracts,future,source,float(config.get('price_divisor',100)),self._offer_sample,
            int(config.get('replay_rewind_tolerance_minutes',5)))
        self.risk_generation=0;self.model_generation=0
        self.gap_after_minute=None
        self.model_thread=threading.Thread(target=self._model_loop,name='ivrv-minute-model',daemon=True)
        self.model_thread.start()
    def _offer_sample(self,sample):
        try:self.samples.put_nowait(sample)
        except queue.Full:
            self.model_fault='MODEL_SNAPSHOT_QUEUE_OVERFLOW';self.engine.entry_block_reason=self.model_fault
    def _model_loop(self):
        while not self.stopping.is_set():
            try:sample=self.samples.get(timeout=.25)
            except queue.Empty:continue
            if sample is None:break
            try:
                if sample['generation']!=self.model_generation:
                    self.engine.rows_by_minute.clear();self.engine.spot_by_minute.clear()
                    self.model_generation=sample['generation']
                delay=time.time()-sample['received_at']
                self.model_fault='MODEL_SNAPSHOT_BACKLOG' if delay>self.max_backlog else None
                self.engine.entry_block_reason=self.transport_fault or self.model_fault
                minute=sample['minute']
                spot=sample['spot_close'];future=sample['future_open']
                if spot is None or future is None:
                    self.engine._set_wait(minute,'missing_completed_spot_or_front_future_bar')
                    continue
                self.engine.spot_bars={minute:legacy.MinuteBar(spot,spot,spot,spot)}
                self.engine.future_bars={minute:legacy.MinuteBar(future,future,future,future)}
                self.engine.option_bars={token:{minute:legacy.MinuteBar(price,price,price,price)}
                    for token,price in sample['option_closes'].items()}
                self.engine.oi_by_token=sample['oi']
                with self.engine.risk_lock:
                    self.engine.latest_exchange_second=max(self.engine.latest_exchange_second or sample['exchange_second'],
                        sample['exchange_second'])
                self.engine.last_finalized_minute=minute
                self.engine._finalize_minute(minute)
            except Exception as exc:
                self.model_fault='MODEL_SNAPSHOT_ERROR:'+type(exc).__name__
                self.engine.entry_block_reason=self.model_fault
                import traceback;traceback.print_exc()
    def _sync_rewind(self):
        if self.sampler.generation==self.risk_generation:return
        with self.engine.risk_lock:
            self.engine.latest_option_ticks.clear()
            self.engine.latest_exchange_second=None
            self.engine.last_paper_time_check_second=None
        self.risk_generation=self.sampler.generation
    def index(self,message):
        second=self.sampler.index(message)
        if second is not None:
            self._sync_rewind()
            with self.engine.risk_lock:
                self.engine.latest_exchange_second=max(self.engine.latest_exchange_second or second,second)
                self.engine.latest_spot_minute=self.sampler.latest_spot_minute
                self.engine._mark_paper_position(second,time_only=True)
            self.sampler.complete()
    def market(self,update):
        second=self.sampler.market(update)
        if second is None:return
        self._sync_rewind()
        with self.engine.risk_lock:
            self.engine.latest_exchange_second=max(self.engine.latest_exchange_second or second,second)
            self.engine.latest_future_minute=self.sampler.latest_future_minute
            if update.token in self.engine.contracts:
                self.engine.latest_option_ticks[update.token]=(float(update.last_traded_price),int(second))
                self.engine._mark_paper_position(second,updated_token=update.token)
            else:self.engine._mark_paper_position(second,time_only=True)
        self.sampler.complete()
    def oi(self,message):self.sampler.oi_message(message)
    def pulse(self):
        with self.engine.risk_lock:self.engine._mark_paper_position(wall_second(),time_only=True)
    def fault(self,reason):
        self.transport_fault=reason;self.engine.entry_block_reason=reason or self.model_fault
    def note_feed_gap(self):
        # Exclude the minute being assembled when a packet went missing.
        with self.engine.lock:
            observed=[m for m in (self.engine.latest_spot_minute,self.engine.latest_future_minute) if m is not None]
            self.gap_after_minute=max(observed,default=wall_second()//60)
    def feed_gap_recovered(self):
        with self.engine.lock:
            end=self.engine.last_finalized_minute
            boundary=self.gap_after_minute
            if end is None or boundary is None or end-boundary<legacy.REQUIRED_FEATURE_MINUTES:
                return False
            return all(m in self.engine.rows_by_minute for m in range(end-max(legacy.LAGS),end+1))
    def status(self):
        with self.engine.risk_lock:
            value=self.engine.status();value['model_snapshot_queue_depth']=self.samples.qsize()
            value['model_fault']=self.model_fault
            if self.model_fault:value['fault']=self.model_fault
            return value
    def recent(self):return self.engine.recent(500)
    def close(self):
        self.stopping.set()
        try:self.samples.put_nowait(None)
        except queue.Full:pass
        self.model_thread.join(timeout=5)
        self.engine.close()
