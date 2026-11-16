"""Replay minute snapshots through the live IV/RV feature builder and model."""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INF = ROOT.parent / 'families' / 'ivrv' / 'engine' / 'Inference'
sys.path.insert(0, str(INF))
sys.path.insert(0, str(INF / 'live'))
sys.path.insert(0, str(ROOT.parent / 'families' / 'ivrv' / 'standalone'))
from inference_pipeline import FixedStrikeInferencePipeline
from live_inference import LiveFeatureInference, NSE_EPOCH, paper_entry_route
from iv_inference import implied_volatility, _greeks

PROXY_DTE = os.environ.get('IVRV_PROXY_DTE')
PROXY_VEGA = os.environ.get('IVRV_PROXY_VEGA')
DROP_VEGA = os.environ.get('IVRV_DROP_VEGA') == '1'
OUT = ROOT / 'analysis_outputs' / os.environ.get('IVRV_OUTPUT_DIR', 'previous_day_ivrv_2026-10-29_no_vega' if DROP_VEGA else 'previous_day_ivrv_2026-10-29_dte4_vega27' if PROXY_DTE and PROXY_VEGA else 'previous_day_ivrv_2026-10-29_dte4' if PROXY_DTE else 'previous_day_ivrv_2026-10-29')
OUT.mkdir(parents=True, exist_ok=True)
opt = pd.read_csv(ROOT / 'Data' / 'banknifty_atm5_minute.csv')
fut = pd.read_csv(ROOT / 'Data' / 'banknifty_future.csv')
assert fut.minute.is_unique and set(opt.minute) == set(fut.minute)
assert (opt.groupby('minute').size() == 10).all()
assert (opt.future_token == fut.Token.iloc[0]).all()
assert (opt.expiry == fut.expiry.iloc[0]).all()
assert not opt.columns.duplicated().any()
fut = fut.set_index('minute')
config = json.loads((INF / 'live' / 'live_config.json').read_text())
pipeline = FixedStrikeInferencePipeline(INF / 'config.yaml')
engine = LiveFeatureInference.__new__(LiveFeatureInference)
engine.expiry = date.fromisoformat(str(opt.expiry.iloc[0]))
engine.rate = float(config['risk_free_rate'])
engine.minimum_surface_points = int(config['minimum_surface_points'])
engine.config = config
engine.future_token = int(fut.Token.iloc[0])
engine.implied_volatility = implied_volatility
engine.greeks = _greeks
engine.rows_by_minute = {}
engine.spot_by_minute = {}
engine.pairs = {}
for r in opt[['strike','option_type','token']].drop_duplicates().itertuples(index=False):
    engine.pairs.setdefault(float(r.strike), {})[r.option_type] = int(r.token)
engine.strikes = sorted(engine.pairs)
engine.oi_by_token = {}
rows = []
invalid = []
for stamp, group in opt.groupby('minute', sort=True):
    minute = int((pd.Timestamp(stamp).to_pydatetime() - NSE_EPOCH).total_seconds() // 60)
    spot = float(group.future_ltp.iloc[0])  # user-authorized spot proxy
    if group.future_ltp.nunique() != 1:
        invalid.append((stamp, 'inconsistent_future_ltp')); continue
    selected = sorted(group.strike.unique().astype(float))
    nearest = min(engine.strikes, key=lambda strike: abs(strike-spot))
    if len(selected) != 5 or selected[2] != nearest:
        invalid.append((stamp, 'five_strike_window_mismatch')); continue
    prices = {}
    valid = True
    for strike in selected:
        prices[strike] = {}
        for side in ('CE','PE'):
            leg = group.loc[(group.strike == strike) & (group.option_type == side)]
            if len(leg) != 1:
                valid = False; break
            r = leg.iloc[0]
            bid, ask = float(r.BestBuyPrice_1), float(r.BestSellPrice_1)
            if bid <= 0 or ask < bid or r.BestBuyQty_1 <= 0 or r.BestSellQty_1 <= 0:
                valid = False; break
            prices[strike][side] = (bid + ask) / 2
            engine.oi_by_token[int(r.token)] = (int(r.OpenInterest), minute*60, 0)
    if not valid:
        invalid.append((stamp, 'invalid_quote_or_pair')); continue
    engine.oi_by_token[engine.future_token] = (int(fut.loc[stamp,'OpenInterest']), minute*60, 0)
    row = engine._build_current_row(minute, spot, float(fut.loc[stamp,'LastTradedPrice']), selected, prices)
    if row is None:
        invalid.append((stamp, 'invalid_iv_or_surface')); continue
    if PROXY_DTE:
        row['days_to_expiry'] = float(PROXY_DTE)
    if PROXY_VEGA:
        row['entry_total_vega'] = float(PROXY_VEGA)
    if DROP_VEGA:
        row['entry_total_vega'] = math.nan
    engine.rows_by_minute[minute] = row
    engine.spot_by_minute[minute] = spot
    if all(minute-lag in engine.rows_by_minute for lag in (5,15,30,60)):
        wide = engine._add_lags_and_spot_features(minute,row)
        missing = sorted(set(pipeline.required_feature_names) - set(wide)) if hasattr(pipeline,'required_feature_names') else []
        if missing:
            raise ValueError(f'Missing model features: {missing}')
        rows.append(wide)

features = pd.DataFrame(rows)
if features.empty:
    raise RuntimeError(f'No fully warmed feature rows; invalid={invalid[:10]}')
assert not any(c.startswith('t120_') for c in features)
features.to_csv(OUT / 'model_features.csv', index=False)
pred = pipeline.predict(features)
pred['spot_proxy_from_future_ltp'] = features.spot_price.to_numpy()
pred['future_oi'] = features.Future_oi.to_numpy()
pred['quality_entry_ok'] = features.quality_entry_ok.to_numpy()
pred['entry_bid_call'] = np.nan
pred['entry_bid_put'] = np.nan
pred['exit_ask_call_t120'] = np.nan
pred['exit_ask_put_t120'] = np.nan
for i, row in pred.iterrows():
    group = opt.loc[(opt.minute == str(row['datetime'])[:16]) & (opt.strike == row['entry_strike'])]
    for side, suffix in [('CE','call'),('PE','put')]:
        leg = group.loc[group.option_type == side].iloc[0]
        pred.at[i,'entry_bid_'+suffix] = leg.BestBuyPrice_1
        pred.at[i,'exit_ask_'+suffix+'_t120'] = leg.t120_BestSellPrice_1
pred['gross_short_pnl_t120_per_lot'] = (
    pred.entry_bid_call + pred.entry_bid_put - pred.exit_ask_call_t120 - pred.exit_ask_put_t120
) * 30
routes = [paper_entry_route(r._asdict(), .75, .005, 0.0) for r in pred.itertuples(index=False)]
pred['paper_route'] = [r['route'] for r in routes]
pred['paper_signal'] = [bool(r['eligible']) for r in routes]
pred.to_csv(OUT / 'predictions.csv', index=False)
summary = {'source_rows':len(opt),'feature_rows':len(features),'predictions':len(pred),
           'invalid_minutes':invalid,'quality_entry_ok':int(pred.quality_entry_ok.sum()),
           'routed_signals':int(pred.paper_signal.sum()),
           'evaluated_t120':int(pred.gross_short_pnl_t120_per_lot.notna().sum()),
           'spot_source':'option CSV future_ltp, user-authorized proxy',
           'days_to_expiry_model_feature':float(PROXY_DTE) if PROXY_DTE else 'actual expiry interval',
           'entry_total_vega_model_feature':float(PROXY_VEGA) if PROXY_VEGA else 'calculated from current quotes',
           'entry_total_vega_ablation':'missing value' if DROP_VEGA else None,
           'feature_option_price':'best bid/ask midpoint',
           'entry_price':'best bid','t120_exit_price':'best ask',
           'leakage_check':'t120 columns never passed to feature builder or model'}
(OUT/'run_summary.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
