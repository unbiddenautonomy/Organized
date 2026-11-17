"""Market clock constants shared by every feed decoder and pipeline."""
from datetime import datetime
from zoneinfo import ZoneInfo

# NSE broadcast timestamps count seconds from this naive exchange-local epoch.
NSE_EPOCH = datetime(1980, 1, 1)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")


def wall_nse_second() -> int:
    """The exchange-local wall clock now, in seconds since NSE_EPOCH."""
    return int((datetime.now(MARKET_TIMEZONE).replace(tzinfo=None) - NSE_EPOCH).total_seconds())
