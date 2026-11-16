import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_router import calibrated_forecast, route_column
from selector import CandidateScore,choose

def test_router_conditions():
    assert route_column(pd.Timestamp('2025-01-01 09:15'),True,'extended')=='opening_extended'
    assert route_column(pd.Timestamp('2025-01-01 10:00'),True,'extended')=='Realized_GARCH_extended'
    assert route_column(pd.Timestamp('2025-01-01 10:00'),True,'style')=='Realized_GARCH_style'
    assert route_column(pd.Timestamp('2025-01-01 10:00'),False,'extended')=='HAR'

def test_selector_requires_every_gate():
    good=CandidateScore('iron_fly',100,-500,.3,.02,.01,.015,-.01)
    bad=CandidateScore('iron_condor',100,-500,.3,.01,.01,.015,-.01)
    winner,_=choose([bad,good],.1,{'iron_fly':1,'iron_condor':1})
    assert winner.structure=='iron_fly'

def test_no_lookahead_shift_changes_calibration():
    n=300
    train=pd.DataFrame({'time_bucket':[20]*n,'forecast':[1.0]*n,'actual_var':[1.0]*n})
    row=pd.Series({'time_bucket':20,'forecast':1.0})
    before=calibrated_forecast(row,train,'forecast').mean
    shifted=train.copy(); shifted['actual_var']=shifted.actual_var.shift(-1).fillna(4.0)
    after=calibrated_forecast(row,shifted,'forecast').mean
    assert before != after
