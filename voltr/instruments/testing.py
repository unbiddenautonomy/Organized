"""A tiny synthetic FO contract master for tests (two expiries, CE/PE pairs, one future)."""
import csv
import gzip
import io

FIELDS = [
    "FinInstrmId", "UndrlygFinInstrmId", "FinInstrmNm", "TckrSymb",
    "XpryDt", "StrkPric", "OptnTp", "NewBrdLotQty", "DelFlg",
]
ROWS = [
    [101, 26009, "OPTIDX", "BANKNIFTY", "2026-10-27", 5000000, "CE", 30, "N"],
    [102, 26009, "OPTIDX", "BANKNIFTY", "2026-10-27", 5000000, "PE", 30, "N"],
    [201, 26009, "OPTIDX", "BANKNIFTY", "2026-10-29", 5000000, "CE", 30, "N"],
    [202, 26009, "OPTIDX", "BANKNIFTY", "2026-10-29", 5000000, "PE", 30, "N"],
    [301, 26009, "FUTIDX", "BANKNIFTY", "2026-10-29", 0, "XX", 30, "N"],
]


def contract_gzip() -> bytes:
    """Gzipped CSV bytes in the daily master's column layout."""
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(ROWS)
    return gzip.compress(text.getvalue().encode("utf-8"))
