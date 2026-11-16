"""Configurable option-structure construction on the five-strike chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pricing import bs_greeks


@dataclass
class Structure:
    name: str
    legs: list[dict]
    wing_definition: str | None = None
    wing_k: float | None = None

    def net_greeks(self) -> dict[str, float]:
        keys = ["delta", "gamma", "vega", "theta_day", "vanna", "volga"]
        return {key: float(sum(float(leg["qty"]) * float(leg[key]) for leg in self.legs)) for key in keys}

    def net_entry_premium(self) -> float:
        return float(-sum(float(x["qty"]) * float(x["entry_mid"]) for x in self.legs))

    def payoff(self, terminal_spot: np.ndarray) -> np.ndarray:
        s = np.asarray(terminal_spot, float); result = np.zeros_like(s)
        for leg in self.legs:
            intrinsic = np.maximum(s - leg["strike"], 0.0) if leg["option_type"] == "call" else np.maximum(leg["strike"] - s, 0.0)
            result += leg["qty"] * intrinsic
        return result + self.net_entry_premium()

    def max_loss(self) -> float:
        strikes = [x["strike"] for x in self.legs]
        grid = np.linspace(0, max(strikes) * 3.0, 20001)
        return float(max(0.0, -np.min(self.payoff(grid))))


def _leg(row: pd.Series, option_type: str, qty: float) -> dict:
    iv = float(row[f"iv_{option_type}"])
    tau = max(float(row["days_to_expiry"]) / 365.0, 1e-8)
    greeks = bs_greeks(float(row["spot_price"]), float(row["strike"]), tau, iv, option_type == "call")
    return {
        "option_type": option_type, "strike": float(row["strike"]),
        "expiry": pd.Timestamp(row["expiry"]), "qty": float(qty), "iv": iv,
        "entry_mid": float(row[f"{option_type}_price"]),
        **{key: float(np.asarray(value)) for key, value in greeks.items()},
    }


def _nearest(chain: pd.DataFrame, target: float, side: str | None = None) -> pd.Series:
    z = chain
    if side == "below": z = z[z["strike"].lt(target)]
    if side == "above": z = z[z["strike"].gt(target)]
    if z.empty: raise ValueError("No listed strike on requested side")
    return z.loc[(z["strike"] - target).abs().idxmin()]


def same_expiry_structures(chain: pd.DataFrame, expected_move: float, expiry_move: float,
                           wing_definition: str, k: float) -> list[Structure]:
    chain = chain.sort_values("strike").copy()
    atm = _nearest(chain, float(chain["spot_price"].iloc[0]))
    distance = k * (expected_move if wing_definition == "intraday_move" else expiry_move)
    lower = _nearest(chain, float(atm["strike"]) - distance, "below")
    upper = _nearest(chain, float(atm["strike"]) + distance, "above")
    below = chain[chain["strike"].lt(lower["strike"])]
    above = chain[chain["strike"].gt(upper["strike"])]
    lower_wing = below.iloc[-1] if len(below) else chain.iloc[0]
    upper_wing = above.iloc[0] if len(above) else chain.iloc[-1]
    return [
        Structure("naked_short_straddle", [_leg(atm,"call",-1),_leg(atm,"put",-1)]),
        Structure("iron_fly", [_leg(atm,"call",-1),_leg(atm,"put",-1),_leg(lower,"put",1),_leg(upper,"call",1)], wing_definition,k),
        Structure("iron_condor", [_leg(lower,"put",-1),_leg(upper,"call",-1),_leg(lower_wing,"put",1),_leg(upper_wing,"call",1)], wing_definition,k),
        Structure("long_straddle", [_leg(atm,"call",1),_leg(atm,"put",1)]),
        Structure("long_strangle", [_leg(lower,"put",1),_leg(upper,"call",1)], wing_definition,k),
    ]


def calendar_structures(near_chain: pd.DataFrame, far_chain: pd.DataFrame) -> list[Structure]:
    spot = float(near_chain["spot_price"].iloc[0])
    near = _nearest(near_chain, spot); far = _nearest(far_chain, spot)
    near_pair = [_leg(near,"call",1), _leg(near,"put",1)]
    far_pair = [_leg(far,"call",1), _leg(far,"put",1)]
    nv = sum(x["vega"] for x in near_pair); fv = sum(x["vega"] for x in far_pair)
    ng = sum(x["gamma"] for x in near_pair); fg = sum(x["gamma"] for x in far_pair)

    def scaled(pair: Iterable[dict], qty: float) -> list[dict]:
        return [{**x, "qty": qty} for x in pair]

    return [
        Structure("vega_neutral_calendar", scaled(near_pair,-1)+scaled(far_pair,nv/fv)),
        Structure("gamma_neutral_vega_positive_calendar", scaled(near_pair,-1)+scaled(far_pair,ng/fg)),
        Structure("vega_neutral_gamma_positive_calendar", scaled(near_pair,1)+scaled(far_pair,-nv/fv)),
    ]


def retained_greek_fraction(structure: Structure, naked_name: str = "gamma") -> float:
    shorts = sum(-x["qty"] * x[naked_name] for x in structure.legs if x["qty"] < 0)
    net = -structure.net_greeks()[naked_name]
    return float(net / shorts) if shorts else np.nan

