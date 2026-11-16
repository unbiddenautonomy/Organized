import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'app'))
from trade_puncher import build_intent


def chain(dte):
    return pd.DataFrame([{'datetime':pd.Timestamp('2026-10-12 09:46'),'expiry':pd.Timestamp('2026-10-29'),
      'strike':45000.,'spot_price':45010.,'days_to_expiry':dte,'call_price':500.,'put_price':490.,
      'iv_call':.18,'iv_put':.19,'call_oi':100.,'put_oi':120.}])


def artifact():
    return {'status':'PAPER_ONLY','entry_start':'09:45','entry_end':'09:59','threshold':0.01}


def test_dte_8_to_20_is_explicit_no_trade():
    result=build_intent(chain(10),{},artifact())
    assert result['status']=='NO_TRADE'
    assert 'failed corrected' in result['reason']


def test_dte_4_to_7_blocks_when_live_features_are_incomplete():
    result=build_intent(chain(6),{'features':{}},artifact())
    assert result['status']=='BLOCKED'
