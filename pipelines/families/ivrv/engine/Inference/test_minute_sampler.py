"""The live sampler must preserve the model's one-minute price selections."""
from types import SimpleNamespace
import threading

from live import pipeline_adapters
from live.pipeline_adapters import IVRVMinuteSampler


def test_sampler_emits_one_snapshot_with_last_spot_option_and_first_future():
    samples=[]
    source={
        'decode_banknifty_7207':lambda message,divisor:message,
        'decode_oi_message':lambda message:message,
        'OiDecodeError':ValueError,
    }
    sampler=IVRVMinuteSampler({11:object(),12:object()},99,source,100,samples.append)

    def market(token,second,price):
        sampler.market(SimpleNamespace(token=token,log_time=second,last_traded_price=price))
        sampler.complete()

    sampler.index((60,100.0))
    sampler.complete()
    sampler.index((85,101.0))
    sampler.complete()
    market(99,61,200.0)
    market(99,90,201.0)
    market(11,62,10.0)
    market(11,91,11.0)
    market(12,63,20.0)
    market(12,92,21.0)
    sampler.oi_message([SimpleNamespace(token=11,log_time=90,open_interest=500,transaction_code=7201)])
    assert len(samples)==1 and samples[0]['spot_close'] is None

    for second in (120,180):
        sampler.index((second,102.0))
        sampler.complete()
        market(99,second,202.0)

    assert len(samples)==3  # Startup minutes before both streams arrived are incomplete.
    sample=samples[-1]
    assert sample['minute']==1
    assert sample['spot_close']==101.0
    assert sample['future_open']==200.0
    assert sample['option_closes']=={11:11.0,12:21.0}
    assert sample['oi'][11]==(500,90,7201)


def test_live_paper_marks_continue_while_minute_model_is_busy(monkeypatch):
    entered=threading.Event();release=threading.Event();marked=[]

    class FakeEngine:
        def __init__(self,*args):
            self.contracts={11:object()};self.latest_option_ticks={}
            self.latest_exchange_second=None;self.entry_block_reason=None
        def _finalize_minute(self,minute):
            entered.set();release.wait(5)
        def _mark_paper_position(self,second,**kwargs):marked.append((second,kwargs))
        def status(self):return {}
        def recent(self,limit):return []
        def close(self):pass
        def _set_wait(self,*args):pass

    monkeypatch.setattr(pipeline_adapters.legacy,'FixedStrikeInferencePipeline',lambda path:object())
    monkeypatch.setattr(pipeline_adapters,'IsolatedIVRV',FakeEngine)
    source={
        'decode_banknifty_7207':lambda message,divisor:message,
        'decode_oi_message':lambda message:message,
        'OiDecodeError':ValueError,
    }
    adapter=pipeline_adapters.IVRVAdapter({'inference_config':'unused'}, {11:object()},30,
        None,99,source)
    try:
        for second in (60,120,180):
            adapter.index((second,100.0))
            adapter.market(SimpleNamespace(token=99,log_time=second,last_traded_price=200.0))
        assert entered.wait(2)
        adapter.market(SimpleNamespace(token=11,log_time=61,last_traded_price=10.0))
        assert marked[-1]==(61,{'updated_token':11})
        assert adapter.engine.latest_option_ticks[11]==(10.0,61)
    finally:
        release.set();adapter.close()
