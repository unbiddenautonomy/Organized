"""Market clock constants shared by every feed decoder and pipeline."""
from datetime import datetime
from zoneinfo import ZoneInfo

# NSE broadcast timestamps count seconds from this naive exchange-local epoch.
NSE_EPOCH = datetime(1980, 1, 1)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
