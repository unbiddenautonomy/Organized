"""Past-only, downside-aware selector across option structures.

The selector predicts after-cost P&L in option points.  For each structure it
forms a conservative score equal to the Huber conditional-mean forecast plus
the training residual's 25th percentile.  A structure must also have positive
aggregate training P&L.  The highest positive score wins; otherwise the action
is no-trade.  Selection is performed before observing the test month.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import dte4_20_structure_research as structures
from dte4_20_pnl_model import FEATURES, panel
from frozen_near_expiry_strategy import bootstrap_total, lot_size


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dte4_20_multi_structure"
ARTIFACT = ROOT / "model/dte4_20_paper/multi_structure_selector.joblib"
BANDS = {"4-7": (4., 8., 60), "8-14": (8., 15., 100), "15-20": (15., 21., 100)}
RESIDUAL_QUANTILE = .25


def estimator():
    return make_pipeline(RobustScaler(), HuberRegressor(epsilon=1.35, alpha=.01, max_iter=2000))


def feature_panel(spread: float) -> pd.DataFrame:
    structures.BPS = spread
    realised = structures.realised()
    realised["half_spread_bps"] = spread
    realised["lot_size"] = realised.expiry.map(lot_size)
    realised["net_points"] = realised.net_pnl / realised.lot_size
    base = panel(25.)[["datetime"] + FEATURES]
    return realised.drop(columns=["dte"]).merge(base, on="datetime", how="inner").replace(
        [np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["net_points"])


def band_name(dte: pd.Series) -> pd.Series:
    return pd.cut(dte, [4, 8, 15, 21], right=False, labels=list(BANDS)).astype(str)


def walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy(); frame["dte_band"] = band_name(frame.dte)
    candidate_rows, selected_rows = [], []
    for band, (low, high, minimum) in BANDS.items():
        group = frame[(frame.dte >= low) & (frame.dte < high)]
        dates = pd.Series(sorted(group.datetime.unique()))
        for month in dates.dt.to_period("M").drop_duplicates():
            test_dates = dates[dates.dt.to_period("M").eq(month)]
            cutoff = test_dates.min(); candidates = []
            for name, history in group.groupby("structure"):
                train = history[history.datetime < cutoff]
                test = history[history.datetime.isin(test_dates)]
                if len(train) < minimum or test.empty:
                    continue
                model = estimator().fit(train[FEATURES], train.net_points)
                fitted = model.predict(train[FEATURES])
                residual_q25 = float(np.quantile(train.net_points - fitted, RESIDUAL_QUANTILE))
                scored = test.copy()
                scored["predicted_mean_points"] = model.predict(test[FEATURES])
                scored["residual_q25"] = residual_q25
                scored["conservative_score"] = scored.predicted_mean_points + residual_q25
                scored["positive_training_pnl"] = train.net_pnl.sum() > 0
                scored["train_rows"] = len(train)
                candidates.append(scored)
            if not candidates:
                continue
            candidates = pd.concat(candidates, ignore_index=True)
            candidate_rows.append(candidates)
            eligible = candidates[candidates.positive_training_pnl]
            winners = eligible.sort_values(["datetime", "conservative_score"], ascending=[True, False]).groupby("datetime").head(1)
            selected_rows.append(winners[winners.conservative_score > 0])
    return pd.concat(candidate_rows, ignore_index=True), pd.concat(selected_rows, ignore_index=True)


def performance(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (band, spread), group in frame.groupby(["dte_band", "half_spread_bps"]):
        lo, hi, p = bootstrap_total(group)
        rows.append({"dte_band":band,"half_spread_bps":spread,"trades":len(group),
          "net_pnl":group.net_pnl.sum(),"mean_trade":group.net_pnl.mean(),
          "hit_rate":(group.net_pnl>0).mean(),"worst_trade":group.net_pnl.min(),
          "ci_low":lo,"ci_high":hi,"p_net_le_zero":p})
    return pd.DataFrame(rows)


def freeze(frame: pd.DataFrame) -> dict:
    frame=frame.copy();frame["dte_band"]=band_name(frame.dte); models={}
    for band,(low,high,minimum) in BANDS.items():
        models[band]={}
        for name,train in frame[(frame.dte>=low)&(frame.dte<high)].groupby("structure"):
            if len(train)<minimum:continue
            model=estimator().fit(train[FEATURES],train.net_points); fitted=model.predict(train[FEATURES])
            models[band][name]={"model":model,"residual_q25":float(np.quantile(train.net_points-fitted,.25)),
              "positive_training_pnl":bool(train.net_pnl.sum()>0),"training_rows":len(train)}
    artifact={"models":models,"features":FEATURES,"status":"EXPERIMENTAL_PAPER_ONLY",
      "entry_start":"09:45","entry_end":"09:59","exit_rule":"same-session at or after 15:15",
      "score":"Huber conditional mean points + training residual q25; require positive training aggregate",
      "structure_legs":structures.STRUCTURES,"trained_through":str(frame.datetime.max()),
      "deployment_policy":{"4-7":"use existing ATM Ridge gate","8-14":"use existing ATM Huber gate",
                           "15-20":"use multi-structure selector"}}
    ARTIFACT.parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,ARTIFACT);return artifact


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    primary=feature_panel(25.);candidates,selected=walk_forward(primary)
    keys=selected[["datetime","structure","dte_band","predicted_mean_points","residual_q25","conservative_score"]]
    stress=[]
    for spread in (10.,25.,50.,100.):
        priced=feature_panel(spread)
        stress.append(priced.merge(keys,on=["datetime","structure"],how="inner",validate="one_to_one"))
    trades=pd.concat(stress,ignore_index=True);stats=performance(trades);artifact=freeze(primary)
    candidates.to_csv(OUT/'all_candidate_scores.csv',index=False)
    trades.to_csv(OUT/'selected_trades_cost_stress.csv',index=False)
    stats.to_csv(OUT/'performance.csv',index=False)
    pd.DataFrame([{"dte_band":b,"structure":s,"signals":len(g),"net_pnl":g.net_pnl.sum(),
      "hit_rate":(g.net_pnl>0).mean()} for (b,s),g in selected.groupby(['dte_band','structure'])]).to_csv(OUT/'selection_mix.csv',index=False)
    (OUT/'artifact_metadata.json').write_text(json.dumps({k:v for k,v in artifact.items() if k!='models'},indent=2,default=str),encoding='utf-8')
    print(stats.to_string(index=False));print('\nSelection mix\n',pd.read_csv(OUT/'selection_mix.csv').to_string(index=False))


if __name__=='__main__':main()
