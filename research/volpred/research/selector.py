"""Structure eligibility and risk-adjusted ranking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateScore:
    structure: str
    mean_net: float
    cvar5: float
    loss_probability: float
    iv_variance: float
    rv_mean: float
    rv_q90: float
    predicted_iv_change: float
    valid_quotes: bool = True
    payload_index: int | None = None

    @property
    def direction(self) -> str:
        base=self.structure.split('|',1)[0]
        return "short" if base in {"naked_short_straddle","iron_fly","iron_condor","vega_neutral_calendar"} else "long"

    def passes(self, lambda_cvar: float, edge_band: float) -> bool:
        if not self.valid_quotes or self.mean_net <= 0: return False
        if self.mean_net <= lambda_cvar*abs(self.cvar5): return False
        if self.direction=="short":
            return self.iv_variance>self.rv_q90 and self.predicted_iv_change<=0 and edge_band>0
        return self.rv_mean>self.iv_variance and self.predicted_iv_change>=0 and edge_band<0

    @property
    def score(self) -> float:
        return self.mean_net/max(abs(self.cvar5),1e-9)


def choose(candidates: list[CandidateScore], lambda_cvar: float, edge_bands: dict[str,float]):
    passing=[c for c in candidates if c.passes(lambda_cvar,edge_bands.get(c.structure,0.0))]
    passing.sort(key=lambda x:x.score,reverse=True)
    return (passing[0] if passing else None),passing[1:]

