"""PostgreSQL access for the active BANKNIFTY option contract universe."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import psycopg


@dataclass(frozen=True, slots=True)
class OptionContract:
    token: int
    underlying_token: int
    instrument_name: str
    symbol: str
    expiry: date
    strike_price: Decimal
    option_type: str


@dataclass(frozen=True, slots=True)
class ContractMasterRecord:
    token: int
    underlying_token: int
    instrument_name: str
    symbol: str
    expiry: date
    strike_price: Decimal
    option_type: str


NSE_EPOCH = datetime(1980, 1, 1)


def _nse_date(value: str) -> date:
    text = str(value).strip()
    try:
        return (NSE_EPOCH + timedelta(seconds=int(float(text)))).date()
    except ValueError:
        return datetime.fromisoformat(text).date()


@lru_cache(maxsize=4)
def _load_contract_file(path_text: str, price_divisor: float) -> tuple[ContractMasterRecord, ...]:
    path = Path(path_text)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    records: list[ContractMasterRecord] = []
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "FinInstrmId", "UndrlygFinInstrmId", "FinInstrmNm", "TckrSymb",
            "XpryDt", "StrkPric", "OptnTp",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"contract file is missing columns: {', '.join(sorted(missing))}")
        divisor = Decimal(str(price_divisor))
        for row in reader:
            if str(row.get("DelFlg", "N")).strip().upper() == "Y":
                continue
            token = str(row.get("FinInstrmId", "")).strip()
            expiry = str(row.get("XpryDt", "")).strip()
            if not token or not expiry:
                continue
            underlying = str(row.get("UndrlygFinInstrmId", "0")).strip()
            strike = str(row.get("StrkPric", "0")).strip()
            records.append(ContractMasterRecord(
                token=int(float(token)),
                underlying_token=int(float(underlying or 0)),
                instrument_name=str(row.get("FinInstrmNm", "")).strip().upper(),
                symbol=str(row.get("TckrSymb", "")).strip().upper(),
                expiry=_nse_date(expiry),
                strike_price=Decimal(strike or "0") / divisor,
                option_type=str(row.get("OptnTp", "")).strip().upper(),
            ))
    if not records:
        raise RuntimeError(f"no contracts found in {path}")
    return tuple(records)


def _contract_records(config: dict) -> tuple[ContractMasterRecord, ...]:
    path = Path(config["contract_file"]).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_contract_file(str(path), float(config.get("price_divisor", 100.0)))


def _asof_date(config: dict) -> date:
    configured = config.get("contract_asof_date")
    return date.fromisoformat(str(configured)) if configured else date.today()


def load_front_future(config: dict) -> tuple[int, date]:
    """Return the nearest non-expired FUTIDX token from a configured contract file."""
    symbol = str(config.get("contract_symbol", "BANKNIFTY")).strip().upper()
    candidates = [
        row for row in _contract_records(config)
        if row.symbol == symbol and row.instrument_name == "FUTIDX" and row.expiry >= _asof_date(config)
    ]
    if not candidates:
        raise RuntimeError(f"no non-expired {symbol} FUTIDX contract found in contract file")
    selected = min(candidates, key=lambda row: (row.expiry, row.token))
    return selected.token, selected.expiry


def is_monthly_expiry(config: dict, expiry: date) -> bool:
    symbol = str(config.get("contract_symbol", "BANKNIFTY")).strip().upper()
    option_expiries = {
        row.expiry for row in _contract_records(config)
        if row.symbol == symbol and row.option_type in {"CE", "PE"}
        and row.expiry.year == expiry.year and row.expiry.month == expiry.month
    }
    return bool(option_expiries) and expiry == max(option_expiries)


def contract_source_summary(config: dict, contracts: dict[int, OptionContract]) -> dict:
    return {
        "source": str(Path(config["contract_file"]).resolve()) if config.get("contract_file") else "postgresql",
        "symbol": str(config.get("contract_symbol", "BANKNIFTY")).strip().upper(),
        "expiry": str(next(iter(contracts.values())).expiry),
        "option_tokens": len(contracts),
        "call_tokens": sum(row.option_type == "CE" for row in contracts.values()),
        "put_tokens": sum(row.option_type == "PE" for row in contracts.values()),
    }


CONTRACT_SQL = """
WITH nearest_expiry AS (
    SELECT MIN(xprydt::date) AS expiry
    FROM public.nse_fo_contract
    WHERE UPPER(tckrsymb) = %(symbol)s
      AND UPPER(optntp) IN ('CE', 'PE')
      AND xprydt::date >= CURRENT_DATE
)
SELECT
    fininstrmid,
    undrlygfininstrmid,
    fininstrmnm,
    tckrsymb,
    xprydt::date,
    strkpric::numeric,
    UPPER(optntp)
FROM public.nse_fo_contract
CROSS JOIN nearest_expiry
WHERE UPPER(tckrsymb) = %(symbol)s
  AND UPPER(optntp) IN ('CE', 'PE')
  AND xprydt::date = nearest_expiry.expiry
ORDER BY strkpric::numeric, UPPER(optntp), fininstrmid::bigint
"""


def load_nearest_expiry_options(config: dict) -> dict[int, OptionContract]:
    """Return nearest non-expired CE/PE contracts keyed by integer feed token."""
    symbol = str(config.get("contract_symbol", "BANKNIFTY")).strip().upper()
    if config.get("contract_file"):
        eligible = [
            row for row in _contract_records(config)
            if row.symbol == symbol and row.instrument_name == "OPTIDX"
            and row.option_type in {"CE", "PE"} and row.expiry >= _asof_date(config)
        ]
        if not eligible:
            raise RuntimeError(f"no non-expired CE/PE contracts found for {symbol} in contract file")
        nearest_expiry = min(row.expiry for row in eligible)
        contracts = {
            row.token: OptionContract(
                token=row.token,
                underlying_token=row.underlying_token,
                instrument_name=row.instrument_name,
                symbol=row.symbol,
                expiry=row.expiry,
                strike_price=row.strike_price,
                option_type=row.option_type,
            )
            for row in eligible if row.expiry == nearest_expiry
        }
        if {row.option_type for row in contracts.values()} != {"CE", "PE"}:
            raise RuntimeError(f"nearest expiry {nearest_expiry} does not contain both CE and PE contracts")
        return contracts
    connection_options = {
        "host": config["db_host"],
        "port": int(config["db_port"]),
        "user": config["db_user"],
        "password": config["db_password"],
        "dbname": config["db_name"],
        "sslmode": config.get("db_sslmode", "disable"),
        "connect_timeout": 5,
    }
    with psycopg.connect(**connection_options) as connection:
        rows = connection.execute(CONTRACT_SQL, {"symbol": symbol}).fetchall()

    contracts: dict[int, OptionContract] = {}
    for row in rows:
        contract = OptionContract(
            token=int(row[0]),
            underlying_token=int(row[1]),
            instrument_name=str(row[2]).strip(),
            symbol=str(row[3]).strip().upper(),
            expiry=row[4],
            strike_price=Decimal(row[5]),
            option_type=str(row[6]).strip().upper(),
        )
        contracts[contract.token] = contract
    if not contracts:
        raise RuntimeError(f"no non-expired CE/PE contracts found for {symbol}")
    return contracts
