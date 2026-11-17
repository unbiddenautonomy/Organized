"""Read the NSE FO contract master: option chains, front future, expiries and lot size.

All functions read the daily master CSV (optionally gzipped) that
voltr.instruments.download fetches. Prices in the master are scaled by
`price_divisor` (100 for BANKNIFTY).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
import gzip
from pathlib import Path

from voltr.core.clock import NSE_EPOCH

REQUIRED_COLUMNS = frozenset({"FinInstrmId", "UndrlygFinInstrmId", "FinInstrmNm", "TckrSymb",
                              "XpryDt", "StrkPric", "OptnTp"})


@dataclass(frozen=True, slots=True)
class OptionContract:
    """One master row. Futures use the same record with option_type 'XX'."""
    token: int
    underlying_token: int
    instrument_name: str
    symbol: str
    expiry: date
    strike_price: Decimal
    option_type: str


def nse_date(value: str) -> date:
    """Master expiry: seconds since the NSE epoch, or an ISO date/datetime."""
    text = str(value).strip()
    try:
        return (NSE_EPOCH + timedelta(seconds=int(float(text)))).date()
    except ValueError:
        return datetime.fromisoformat(text).date()


def _open(path: Path):
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


@lru_cache(maxsize=4)
def _load_contract_file(path_text: str, price_divisor: float) -> tuple[OptionContract, ...]:
    path = Path(path_text)
    records: list[OptionContract] = []
    with _open(path) as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
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
            records.append(OptionContract(
                token=int(float(token)),
                underlying_token=int(float(underlying or 0)),
                instrument_name=str(row.get("FinInstrmNm", "")).strip().upper(),
                symbol=str(row.get("TckrSymb", "")).strip().upper(),
                expiry=nse_date(expiry),
                strike_price=Decimal(strike or "0") / divisor,
                option_type=str(row.get("OptnTp", "")).strip().upper(),
            ))
    if not records:
        raise RuntimeError(f"no contracts found in {path}")
    return tuple(records)


def _contract_records(config: dict) -> tuple[OptionContract, ...]:
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
    """True when `expiry` is the last option expiry of its calendar month."""
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


def load_nearest_expiry_options(config: dict) -> dict[int, OptionContract]:
    """Nearest non-expired CE/PE contracts (weekly or monthly), keyed by feed token."""
    symbol = str(config.get("contract_symbol", "BANKNIFTY")).strip().upper()
    eligible = [
        row for row in _contract_records(config)
        if row.symbol == symbol and row.instrument_name == "OPTIDX"
        and row.option_type in {"CE", "PE"} and row.expiry >= _asof_date(config)
    ]
    if not eligible:
        raise RuntimeError(f"no non-expired CE/PE contracts found for {symbol} in contract file")
    nearest_expiry = min(row.expiry for row in eligible)
    contracts = {row.token: row for row in eligible if row.expiry == nearest_expiry}
    if {row.option_type for row in contracts.values()} != {"CE", "PE"}:
        raise RuntimeError(f"nearest expiry {nearest_expiry} does not contain both CE and PE contracts")
    return contracts


def load_monthly_contracts(path: Path, asof: date, price_divisor: float) -> tuple[dict[int, OptionContract], int, date]:
    """Load the nearest non-expired monthly BANKNIFTY option chain, its lot size and expiry."""
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, str]] = []
    with _open(path) as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"contract file missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("DelFlg", "N").strip().upper() == "Y":
                continue
            if row.get("TckrSymb", "").strip().upper() != "BANKNIFTY":
                continue
            if row.get("FinInstrmNm", "").strip().upper() != "OPTIDX":
                continue
            if row.get("OptnTp", "").strip().upper() not in {"CE", "PE"}:
                continue
            expiry = nse_date(row["XpryDt"])
            if expiry >= asof:
                row["_expiry"] = expiry  # type: ignore[assignment]
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no non-expired BANKNIFTY options in {path}")
    monthly = {max(r["_expiry"] for r in rows if r["_expiry"].year == year and r["_expiry"].month == month)
               for year, month in {(r["_expiry"].year, r["_expiry"].month) for r in rows}}
    expiry = min(value for value in monthly if value >= asof)
    selected = [row for row in rows if row["_expiry"] == expiry]
    divisor = Decimal(str(price_divisor))
    contracts: dict[int, OptionContract] = {}
    lot_sizes: list[int] = []
    for row in selected:
        token = int(float(row["FinInstrmId"]))
        contracts[token] = OptionContract(
            token=token,
            underlying_token=int(float(row.get("UndrlygFinInstrmId", "0") or 0)),
            instrument_name="OPTIDX",
            symbol="BANKNIFTY",
            expiry=expiry,
            strike_price=Decimal(row.get("StrkPric", "0") or "0") / divisor,
            option_type=row["OptnTp"].strip().upper(),
        )
        raw_lot = row.get("NewBrdLotQty") or row.get("MinLot") or "0"
        try:
            lot = int(float(raw_lot))
            if lot > 0:
                lot_sizes.append(lot)
        except ValueError:
            pass
    sides = {contract.option_type for contract in contracts.values()}
    if sides != {"CE", "PE"}:
        raise RuntimeError(f"monthly expiry {expiry} is missing a CE or PE side")
    if not lot_sizes:
        raise RuntimeError("could not determine BANKNIFTY lot size from contract master")
    lot_size = max(set(lot_sizes), key=lot_sizes.count)
    return contracts, lot_size, expiry
