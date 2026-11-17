from __future__ import annotations

import base64
import csv
from datetime import date
import json
from pathlib import Path

from voltr.instruments.download import decode_erp_response, ensure_latest_contract_file
from voltr.instruments.testing import contract_gzip


def config(directory: Path) -> dict:
    return {
        "contract_download_enabled": True,
        "contract_download_url": "http://erp.invalid/v1/file-content?api_type=erp",
        "contract_download_directory": str(directory),
        "contract_file_lag_days": 1,
        "contract_download_lookback_days": 7,
        "contract_auth_token_env": "TEST_ERP_AUTH_TOKEN",
        "price_divisor": 100,
    }


def test_decode_nested_erp_base64_response():
    content = contract_gzip()
    response = json.dumps({
        "status": "success",
        "data": {"file_content": base64.b64encode(content).decode("ascii")},
    }).encode("utf-8")
    assert decode_erp_response(response) == content


def test_downloads_expected_prior_business_day_and_builds_both_token_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ERP_AUTH_TOKEN", "runtime-secret")
    content = contract_gzip()
    calls = []

    def fetcher(url, headers, payload, timeout):
        calls.append(json.loads(payload))
        assert headers["auth-token"] == "runtime-secret"
        return json.dumps({
            "status": "success",
            "data": {"content": base64.b64encode(content).decode("ascii")},
        }).encode("utf-8")

    settings = config(tmp_path)
    result = ensure_latest_contract_file(
        settings, asof=date(2026, 10, 22), fetcher=fetcher,
    )

    assert calls == [{
        "outer": "Common", "segment": "FO", "path": "",
        "filename": "NSE_FO_contract_21102026.csv.gz",
    }]
    assert result.downloaded is True
    assert result.nearest_expiry == date(2026, 10, 27)
    assert result.nearest_option_tokens == 2
    assert result.pipeline_monthly_expiry == date(2026, 10, 29)
    assert result.pipeline_option_tokens == 2
    assert result.front_future_token == 301
    assert settings["contract_file"] == str(result.contract_file)
    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert {row["token"] for row in manifest["nearest_expiry_tokens"]} == {101, 102}
    assert {row["token"] for row in manifest["pipeline_monthly_tokens"]} == {201, 202}
    assert "runtime-secret" not in result.manifest_file.read_text(encoding="utf-8")


def test_existing_latest_file_requires_no_network_or_auth(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_ERP_AUTH_TOKEN", raising=False)
    target = tmp_path / "NSE_FO_contract_21102026.csv.gz"
    target.write_bytes(contract_gzip())

    def no_fetch(*_args):
        raise AssertionError("fresh local contract file must not trigger a download")

    result = ensure_latest_contract_file(
        config(tmp_path), asof=date(2026, 10, 22), fetcher=no_fetch,
    )
    assert result.downloaded is False
    assert result.contract_file == target


def test_force_download_refreshes_an_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ERP_AUTH_TOKEN", "runtime-secret")
    target = tmp_path / "NSE_FO_contract_21102026.csv.gz"
    target.write_bytes(contract_gzip())
    settings = config(tmp_path)
    settings["contract_force_download"] = True
    requested = []

    def fetcher(_url, _headers, payload, _timeout):
        requested.append(json.loads(payload)["filename"])
        return contract_gzip()

    result = ensure_latest_contract_file(
        settings, asof=date(2026, 10, 22), fetcher=fetcher,
    )
    assert requested == [target.name]
    assert result.downloaded is True


def test_monday_uses_friday_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ERP_AUTH_TOKEN", "runtime-secret")
    requested = []

    def fetcher(_url, _headers, payload, _timeout):
        requested.append(json.loads(payload)["filename"])
        return contract_gzip()

    ensure_latest_contract_file(config(tmp_path), asof=date(2026, 10, 26), fetcher=fetcher)
    assert requested == ["NSE_FO_contract_23102026.csv.gz"]
