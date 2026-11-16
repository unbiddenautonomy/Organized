"""Fetch and validate the daily NSE FO contract master before live inference.

The ERP credential is deliberately read at runtime and is never written to the
configuration, downloaded file, token manifest, or logs.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
import gzip
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


NSE_EPOCH = datetime(1980, 1, 1)
REQUIRED_COLUMNS = {
    "FinInstrmId",
    "UndrlygFinInstrmId",
    "FinInstrmNm",
    "TckrSymb",
    "XpryDt",
    "StrkPric",
    "OptnTp",
}
CONTRACT_NAME = re.compile(r"^NSE_FO_contract_(\d{2})(\d{2})(\d{4})\.csv\.gz$")
Fetcher = Callable[[str, dict[str, str], bytes, float], bytes]


@dataclass(frozen=True, slots=True)
class ContractBootstrapResult:
    contract_file: Path
    file_date: date
    downloaded: bool
    manifest_file: Path
    nearest_expiry: date
    nearest_option_tokens: int
    pipeline_monthly_expiry: date
    pipeline_option_tokens: int
    front_future_token: int
    front_future_expiry: date


def _parse_nse_date(value: str) -> date:
    text = str(value).strip()
    try:
        return (NSE_EPOCH + timedelta(seconds=int(float(text)))).date()
    except ValueError:
        return datetime.fromisoformat(text).date()


def _file_date(path: Path) -> date:
    match = CONTRACT_NAME.match(path.name)
    if not match:
        raise ValueError(f"unexpected FO contract filename: {path.name}")
    return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))


def _candidate_dates(asof: date, lag_days: int, lookback_days: int) -> list[date]:
    values: list[date] = []
    cursor = asof - timedelta(days=lag_days)
    for _ in range(lookback_days + 1):
        if cursor.weekday() < 5:
            values.append(cursor)
        cursor -= timedelta(days=1)
    return values


def _default_fetcher(url: str, headers: dict[str, str], payload: bytes, timeout: float) -> bytes:
    request = Request(url, data=payload, headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured internal ERP endpoint
        return response.read()


def _decode_base64(value: str) -> bytes:
    text = value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    text = "".join(text.split())
    try:
        return base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("ERP response file content is not valid base64") from exc


def _find_encoded_file(value: Any) -> bytes:
    if isinstance(value, str):
        return _decode_base64(value)
    if isinstance(value, list) and all(isinstance(item, int) and 0 <= item <= 255 for item in value):
        return bytes(value)
    if isinstance(value, dict):
        preferred = (
            "file_content", "fileContent", "content", "base64", "file_data",
            "fileData", "file", "data", "body",
        )
        for key in preferred:
            if key in value:
                try:
                    return _find_encoded_file(value[key])
                except ValueError:
                    pass
    raise ValueError("ERP success response did not contain downloadable file content")


def decode_erp_response(raw: bytes) -> bytes:
    """Return gzip bytes from either a raw response or the ERP JSON envelope."""
    if raw.startswith(b"\x1f\x8b"):
        return raw
    try:
        envelope = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("ERP response was neither gzip data nor valid JSON") from exc
    if not isinstance(envelope, dict):
        raise ValueError("ERP response JSON must be an object")
    if str(envelope.get("status", "")).strip().lower() not in {"success", "ok", "200"}:
        message = str(envelope.get("message", "contract download failed"))
        raise RuntimeError(f"ERP contract download failed: {message}")
    content = _find_encoded_file(envelope.get("data"))
    if not content.startswith(b"\x1f\x8b"):
        raise ValueError("ERP file content is not a gzip archive")
    return content


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"contract file missing columns: {', '.join(sorted(missing))}")
            rows = list(reader)
    except (OSError, EOFError) as exc:
        raise ValueError(f"invalid gzip contract file {path}") from exc
    if not rows:
        raise ValueError(f"contract file is empty: {path}")
    return rows


def _token_manifest(rows: list[dict[str, str]], asof: date, price_divisor: float) -> dict[str, Any]:
    symbol_rows: list[dict[str, Any]] = []
    futures: list[dict[str, Any]] = []
    divisor = Decimal(str(price_divisor))
    for row in rows:
        if str(row.get("DelFlg", "N")).strip().upper() == "Y":
            continue
        if str(row.get("TckrSymb", "")).strip().upper() != "BANKNIFTY":
            continue
        expiry_text = str(row.get("XpryDt", "")).strip()
        token_text = str(row.get("FinInstrmId", "")).strip()
        if not expiry_text or not token_text:
            continue
        expiry = _parse_nse_date(expiry_text)
        if expiry < asof:
            continue
        instrument = str(row.get("FinInstrmNm", "")).strip().upper()
        record = {
            "token": int(float(token_text)),
            "underlying_token": int(float(str(row.get("UndrlygFinInstrmId", "0") or "0"))),
            "expiry": expiry,
        }
        if instrument == "FUTIDX":
            futures.append(record)
        elif instrument == "OPTIDX" and str(row.get("OptnTp", "")).strip().upper() in {"CE", "PE"}:
            raw_lot = row.get("NewBrdLotQty") or row.get("MinLot") or "0"
            record.update({
                "option_type": str(row["OptnTp"]).strip().upper(),
                "strike": float(Decimal(str(row.get("StrkPric", "0") or "0")) / divisor),
                "lot_size": int(float(raw_lot or 0)),
            })
            symbol_rows.append(record)
    if not symbol_rows:
        raise RuntimeError("contract master has no non-expired BANKNIFTY option contracts")
    if not futures:
        raise RuntimeError("contract master has no non-expired BANKNIFTY future contract")

    expiries = sorted({row["expiry"] for row in symbol_rows})
    nearest = expiries[0]
    monthly = min(
        expiry for expiry in expiries
        if expiry == max(value for value in expiries if value.year == expiry.year and value.month == expiry.month)
    )
    nearest_rows = sorted(
        (row for row in symbol_rows if row["expiry"] == nearest),
        key=lambda row: (row["strike"], row["option_type"], row["token"]),
    )
    monthly_rows = sorted(
        (row for row in symbol_rows if row["expiry"] == monthly),
        key=lambda row: (row["strike"], row["option_type"], row["token"]),
    )
    for label, selected in (("nearest", nearest_rows), ("monthly", monthly_rows)):
        if {row["option_type"] for row in selected} != {"CE", "PE"}:
            raise RuntimeError(f"{label} BANKNIFTY expiry is missing CE or PE contracts")
    front_future = min(futures, key=lambda row: (row["expiry"], row["token"]))

    def serialise(row: dict[str, Any]) -> dict[str, Any]:
        return {key: (value.isoformat() if isinstance(value, date) else value) for key, value in row.items()}

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "asof_date": asof.isoformat(),
        "symbol": "BANKNIFTY",
        "nearest_expiry": nearest.isoformat(),
        "nearest_expiry_tokens": [serialise(row) for row in nearest_rows],
        "pipeline_monthly_expiry": monthly.isoformat(),
        "pipeline_monthly_tokens": [serialise(row) for row in monthly_rows],
        "front_future": serialise(front_future),
    }


def _auth_token(config: dict[str, Any]) -> str:
    env_name = str(config.get("contract_auth_token_env", "ERP_AUTH_TOKEN"))
    token = os.environ.get(env_name, "").strip()
    token_file = str(config.get("contract_auth_token_file", "")).strip()
    if not token and token_file:
        token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(
            f"a fresh FO contract file is required; set the {env_name} environment variable "
            "or configure contract_auth_token_file"
        )
    return token


def _download(config: dict[str, Any], filename: str, fetcher: Fetcher) -> bytes:
    token = _auth_token(config)
    payload = json.dumps({
        "outer": str(config.get("contract_download_outer", "Common")),
        "segment": str(config.get("contract_download_segment", "FO")),
        "path": str(config.get("contract_download_path", "")),
        "filename": filename,
    }, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "auth-token": token,
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
    }
    raw = fetcher(
        str(config["contract_download_url"]), headers, payload,
        float(config.get("contract_download_timeout_seconds", 30)),
    )
    return decode_erp_response(raw)


def ensure_latest_contract_file(
    config: dict[str, Any], *, asof: date | None = None, fetcher: Fetcher | None = None,
) -> ContractBootstrapResult:
    """Select/download the latest expected master and update ``config`` in place."""
    asof = asof or date.today()
    lag_days = int(config.get("contract_file_lag_days", 1))
    lookback_days = int(config.get("contract_download_lookback_days", 7))
    candidates = _candidate_dates(asof, lag_days, lookback_days)
    if not candidates:
        raise ValueError("contract download lookback produced no trading-day candidates")
    directory = Path(config.get("contract_download_directory", Path(__file__).parent / "contracts")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    filename_template = str(config.get("contract_filename_template", "NSE_FO_contract_{date:%d%m%Y}.csv.gz"))
    paths = [(day, directory / filename_template.format(date=day)) for day in candidates]

    selected: tuple[date, Path] | None = None
    downloaded = False
    force_download = bool(config.get("contract_force_download", False))
    expected_day, expected_path = paths[0]
    if expected_path.is_file() and not force_download:
        _token_manifest(_read_rows(expected_path), asof, float(config.get("price_divisor", 100.0)))
        selected = (expected_day, expected_path)
    elif bool(config.get("contract_download_enabled", True)):
        fetch = fetcher or _default_fetcher
        errors: list[str] = []
        for candidate_day, candidate_path in paths:
            if candidate_path.is_file() and not force_download:
                try:
                    _token_manifest(
                        _read_rows(candidate_path), asof,
                        float(config.get("price_divisor", 100.0)),
                    )
                    selected = (candidate_day, candidate_path)
                    break
                except (ValueError, RuntimeError) as exc:
                    errors.append(f"{candidate_path.name}: {exc}")
                    continue
            try:
                content = _download(config, candidate_path.name, fetch)
                temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
                temporary.write_bytes(content)
                _token_manifest(
                    _read_rows(temporary), asof,
                    float(config.get("price_divisor", 100.0)),
                )
                temporary.replace(candidate_path)
                selected = (candidate_day, candidate_path)
                downloaded = True
                break
            except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
                candidate_path.with_suffix(candidate_path.suffix + ".tmp").unlink(missing_ok=True)
                errors.append(f"{candidate_path.name}: {exc}")
        if selected is None:
            raise RuntimeError("could not obtain a valid recent FO contract master; " + " | ".join(errors))
    else:
        raise FileNotFoundError(f"latest expected FO contract master is missing: {expected_path}")

    file_day, contract_path = selected
    rows = _read_rows(contract_path)
    manifest = _token_manifest(rows, asof, float(config.get("price_divisor", 100.0)))
    manifest["contract_file"] = str(contract_path)
    manifest["contract_file_date"] = file_day.isoformat()
    manifest_path = directory / "banknifty_latest_expiry_tokens.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)

    config["contract_file"] = str(contract_path)
    config["contract_asof_date"] = asof.isoformat()
    nearest_tokens = manifest["nearest_expiry_tokens"]
    monthly_tokens = manifest["pipeline_monthly_tokens"]
    future = manifest["front_future"]
    return ContractBootstrapResult(
        contract_file=contract_path,
        file_date=file_day,
        downloaded=downloaded,
        manifest_file=manifest_path,
        nearest_expiry=date.fromisoformat(manifest["nearest_expiry"]),
        nearest_option_tokens=len(nearest_tokens),
        pipeline_monthly_expiry=date.fromisoformat(manifest["pipeline_monthly_expiry"]),
        pipeline_option_tokens=len(monthly_tokens),
        front_future_token=int(future["token"]),
        front_future_expiry=date.fromisoformat(future["expiry"]),
    )


def _load_config(path: Path) -> dict[str, Any]:
    local = json.loads(path.resolve().read_text(encoding="utf-8"))
    source_path = Path(local["source_config"]).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update(local)
    source["source_config"] = str(source_path)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure the latest NSE FO contract master and export BANKNIFTY tokens",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parent / "live_config.json",
    )
    parser.add_argument("--asof", type=date.fromisoformat, help="optional YYYY-MM-DD override")
    args = parser.parse_args()
    result = ensure_latest_contract_file(_load_config(args.config), asof=args.asof)
    print(json.dumps({
        "contract_file": str(result.contract_file),
        "contract_file_date": result.file_date.isoformat(),
        "downloaded": result.downloaded,
        "nearest_expiry": result.nearest_expiry.isoformat(),
        "nearest_option_tokens": result.nearest_option_tokens,
        "pipeline_monthly_expiry": result.pipeline_monthly_expiry.isoformat(),
        "pipeline_option_tokens": result.pipeline_option_tokens,
        "front_future_token": result.front_future_token,
        "front_future_expiry": result.front_future_expiry.isoformat(),
        "token_manifest": str(result.manifest_file),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
