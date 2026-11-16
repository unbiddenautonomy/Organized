from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from five_strike_regime_trade_research import STRUCTURES


def test_all_structures_stay_inside_five_strike_universe():
    assert STRUCTURES
    for legs in STRUCTURES.values():
        assert all(-2 <= strike_position <= 2 for strike_position, _ in legs)
        assert all(side in {"call", "put"} for _, side in legs)


def test_selected_short_two_strangle_has_symmetric_fixed_legs():
    assert STRUCTURES["short_2_strangle"] == {
        (2, "call"): -1,
        (-2, "put"): -1,
    }
