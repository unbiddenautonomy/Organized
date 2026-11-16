import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricing import adjusted_iv_variance, bs_consistent_theta_window, bs_greeks, bs_price
from structures import Structure, calendar_structures, same_expiry_structures


def test_clock_gamma_equals_theta_when_rv_equals_iv():
    profile = np.full(375, 1 / 375)
    ivar = adjusted_iv_variance(0.20, 0.8, profile, 60, 120)
    gamma, spot = 0.0012, 45000.0
    gamma_pnl = 0.5 * gamma * spot * spot * ivar
    theta_pnl = bs_consistent_theta_window(gamma, spot, ivar)
    assert np.isclose(gamma_pnl + theta_pnl, 0.0)


def test_vega_gamma_identity():
    s, k, t, sigma = 45000.0, 45000.0, 12/365, 0.18
    g = bs_greeks(s, k, t, sigma, True)
    assert np.isclose(g["vega"], g["gamma"] * s * s * sigma * t, rtol=1e-10)


def _chain():
    rows=[]
    for strike in [44000,44500,45000,45500,46000]:
        row={"strike":strike,"spot_price":45000.,"expiry":pd.Timestamp('2025-01-30'),"days_to_expiry":10.}
        for typ in ['call','put']:
            row[f'iv_{typ}']=.18; row[f'{typ}_price']=float(bs_price(45000,strike,10/365,.18,typ=='call'))
        rows.append(row)
    return pd.DataFrame(rows)


def test_payoff_correctness_all_same_expiry_structures():
    structures=same_expiry_structures(_chain(),500,1200,'intraday_move',1.0)
    names={x.name for x in structures}
    assert names == {'naked_short_straddle','iron_fly','iron_condor','long_straddle','long_strangle'}
    for structure in structures:
        values=structure.payoff(np.array([0.,45000.,90000.]))
        assert np.isfinite(values).all()
    assert next(x for x in structures if x.name=='iron_fly').max_loss() < np.inf


def test_calendar_neutralities():
    near=_chain(); far=_chain().assign(expiry=pd.Timestamp('2025-02-27'),days_to_expiry=38.)
    for typ in ['call','put']:
        far[f'{typ}_price']=[float(bs_price(45000,k,38/365,.18,typ=='call')) for k in far.strike]
    structures=calendar_structures(near,far)
    by_name={x.name:x for x in structures}
    assert abs(by_name['vega_neutral_calendar'].net_greeks()['vega']) < 1e-8
    assert abs(by_name['gamma_neutral_vega_positive_calendar'].net_greeks()['gamma']) < 1e-10
    assert by_name['vega_neutral_gamma_positive_calendar'].net_greeks()['gamma'] > 0


def test_full_repricing_matches_greeks_for_small_move():
    s,k,t,v=45000.,45000.,10/365,.18
    ds,dv,dt=1.0,1e-5,1e-6
    p0=bs_price(s,k,t,v,True); p1=bs_price(s+ds,k,t-dt,v+dv,True)
    g=bs_greeks(s,k,t,v,True)
    approx=g['delta']*ds+0.5*g['gamma']*ds*ds+g['vega']*dv+g['theta_day']*(dt*365)
    assert np.isclose(p1-p0,approx,atol=2e-4)
