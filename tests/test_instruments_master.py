"""Contract selection from a generated master file."""

import csv
from datetime import date
import gzip
import io

import pytest

from voltr.instruments.master import (
    contract_source_summary,
    is_monthly_expiry,
    load_front_future,
    load_nearest_expiry_options,
)


FIELDS = ["FinInstrmId", "UndrlygFinInstrmId", "FinInstrmNm", "TckrSymb",
          "XpryDt", "StrkPric", "OptnTp", "NewBrdLotQty", "DelFlg"]


@pytest.fixture
def config(tmp_path):
    rows = [
        [101, 26009, "OPTIDX", "BANKNIFTY", "2026-10-27", 5000000, "CE", 30, "N"],
        [102, 26009, "OPTIDX", "BANKNIFTY", "2026-10-27", 5000000, "PE", 30, "N"],
        [201, 26009, "OPTIDX", "BANKNIFTY", "2026-11-24", 5000000, "CE", 30, "N"],
        [202, 26009, "OPTIDX", "BANKNIFTY", "2026-11-24", 5000000, "PE", 30, "N"],
        [301, 26009, "FUTIDX", "BANKNIFTY", "2026-10-27", 0, "XX", 30, "N"],
    ]
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(FIELDS)
    writer.writerows(rows)
    master = tmp_path / "contract.csv.gz"
    master.write_bytes(gzip.compress(stream.getvalue().encode("utf-8")))
    settings = {"contract_symbol": "BANKNIFTY", "price_divisor": 100.0}
    settings["contract_file"] = str(master)
    settings["contract_asof_date"] = "2026-10-07"
    return settings


def test_nearest_banknifty_expiry_and_tokens(config):
    contracts = load_nearest_expiry_options(config)
    assert {row.expiry for row in contracts.values()} == {date(2026, 10, 27)}
    assert len(contracts) == 2
    assert sum(row.option_type == "CE" for row in contracts.values()) == 1
    assert sum(row.option_type == "PE" for row in contracts.values()) == 1
    assert contracts[101].strike_price == 50000


def test_front_future_and_monthly_flag(config):
    token, expiry = load_front_future(config)
    assert (token, expiry) == (301, date(2026, 10, 27))
    assert is_monthly_expiry(config, expiry)


def test_dashboard_summary(config):
    contracts = load_nearest_expiry_options(config)
    summary = contract_source_summary(config, contracts)
    assert summary["option_tokens"] == 2
    assert summary["expiry"] == "2026-10-27"
