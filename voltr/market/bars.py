"""One-minute OHLC bars built from ticks."""
from dataclasses import dataclass


@dataclass(slots=True)
class MinuteBar:
    open: float
    high: float
    low: float
    close: float

    def update(self, value: float) -> None:
        self.high = max(self.high, value)
        self.low = min(self.low, value)
        self.close = value
