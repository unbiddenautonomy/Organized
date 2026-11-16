from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from trade_puncher import build_intent


FEATURES = ["dte", "log_premium_ratio", "atm_iv", "iv_skew", "log_rv_past_5",
            "log_rv_past_15", "log_rv_past_60", "overnight_return", "vol_of_vol_30",
            "oi_imbalance"]


class ConstantModel:
    def __init__(self, value): self.value = value
    def predict(self, frame): return np.full(len(frame), self.value)


def chain(dte):
    return pd.DataFrame([{"datetime":"2025-05-09 09:50:00", "expiry":"2025-05-29",
      "strike":50000, "spot_price":50010, "days_to_expiry":dte, "call_price":500,
      "put_price":490, "iv_call":.18, "iv_put":.19, "call_oi":100, "put_oi":110}])


def live():
    return {"features":{"log_rv_past_5":-12.,"log_rv_past_15":-11.5,
      "log_rv_past_60":-11.,"overnight_return":.001,"vol_of_vol_30":.1}}


def artifact(value=.01):
    models={band:{"model":ConstantModel(value),"threshold":0.005}
            for band in ("8-14","15-20")}
    return {"models":models,"features":FEATURES,"status":"PAPER_ONLY",
      "entry_start":"09:45","entry_end":"09:59","exit_rule":"after 15:15",
      "research_status":"exploratory_candidate_requires_forward_validation"}


def test_dte_8_14_can_create_experimental_paper_intent():
    intent=build_intent(chain(10),live(),artifact())
    assert intent["status"] == "PAPER_ORDER_READY"
    assert intent["dte_band"] == "8-14"
    assert len(intent["orders"]) == 2


def test_dte_15_20_uses_its_band_model():
    intent=build_intent(chain(18),live(),artifact())
    assert intent["status"] == "PAPER_ORDER_READY"
    assert intent["dte_band"] == "15-20"


def test_gate_blocks_nonpositive_predicted_edge():
    intent=build_intent(chain(10),live(),artifact(value=-.01))
    assert intent["status"] == "NO_TRADE"
    assert "below" in intent["reason"]


def test_multi_structure_selector_emits_selected_strangle_legs():
    rows=[]
    for strike in (49800,49900,50000,50100,50200):
        rows.append({"datetime":"2025-05-09 09:50:00","expiry":"2025-05-29",
          "strike":strike,"spot_price":50010,"days_to_expiry":18,"call_price":500,
          "put_price":490,"iv_call":.18,"iv_put":.19,"call_oi":100,"put_oi":110})
    models={"short_1_strangle":{"model":ConstantModel(10.),"residual_q25":-2.,"positive_training_pnl":True},
            "short_2_strangle":{"model":ConstantModel(20.),"residual_q25":-2.,"positive_training_pnl":True}}
    multi={"models":{"15-20":models},"features":FEATURES,"status":"EXPERIMENTAL_PAPER_ONLY",
      "entry_start":"09:45","entry_end":"09:59","exit_rule":"after 15:15",
      "structure_legs":{"short_1_strangle":[("call",1,-1),("put",-1,-1)],
                        "short_2_strangle":[("call",2,-1),("put",-2,-1)]}}
    intent=build_intent(pd.DataFrame(rows),live(),multi)
    assert intent["status"] == "PAPER_ORDER_READY"
    assert intent["structure"] == "short_2_strangle"
    assert {x["strike"] for x in intent["orders"]} == {49800.,50200.}
