"""Resolve checkout-local paths and the selected Windows multicast adapter."""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import subprocess


def relative_to_config(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_path.resolve().parent / path).resolve()


def local_ipv4_adapters() -> list[dict[str, str]]:
    """Return active, non-loopback Windows adapter IPv4 addresses."""
    if os.name != "nt":
        raise RuntimeError("Automatic adapter discovery requires Windows; set an explicit interface IP")
    script = (
        "$up = @(Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object -ExpandProperty ifIndex); "
        "@(Get-NetIPAddress -AddressFamily IPv4 | "
        "Where-Object { $up -contains $_.InterfaceIndex -and $_.IPAddress -notmatch '^(127\\.|169\\.254\\.)' } | "
        "Select-Object InterfaceAlias,IPAddress) | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=15, check=True,
    )
    payload = json.loads(result.stdout.strip() or "[]")
    rows = payload if isinstance(payload, list) else [payload]
    return sorted(
        ({"alias": str(row["InterfaceAlias"]), "ip": str(row["IPAddress"])} for row in rows),
        key=lambda row: (row["alias"], row["ip"]),
    )


def resolve_interface(value: str | None, *, environment: str) -> str:
    selected = os.environ.get(environment, value or "auto").strip()
    try:
        return str(ipaddress.IPv4Address(selected))
    except ipaddress.AddressValueError:
        pass
    rows = local_ipv4_adapters()
    matches = rows if selected.lower() == "auto" else [row for row in rows if row["alias"].lower() == selected.lower()]
    if len(matches) == 1:
        return matches[0]["ip"]
    available = ", ".join(f"{row['alias']}={row['ip']}" for row in rows) or "none"
    raise RuntimeError(
        f"Cannot select multicast adapter {selected!r}: {len(matches)} matches; "
        f"available adapters: {available}. Set {environment} to an adapter alias or IPv4 address."
    )


def load_source_config(path: Path) -> dict:
    """Load standalone engine settings relative to their own JSON file."""
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if os.environ.get("VOLTR_DB_PASSWORD"):
        config["db_password"] = os.environ["VOLTR_DB_PASSWORD"]
    for key in ("contract_file", "spot_model_dir", "iv_model_dir", "pnl_model_dir", "hmm_model_dir",
                "expanded_hybrid_model_file", "rv120_model_file", "intraday_training_database",
                "intraday_artifact_dir", "eod_artifact_dir"):
        if config.get(key):
            config[key] = str(relative_to_config(config[key], path))
    config["interface_ip"] = resolve_interface(config.get("interface_ip"), environment="VOLTR_FO_INTERFACE")
    config["cm_interface_ip"] = resolve_interface(
        config.get("cm_interface_ip", config["interface_ip"]), environment="VOLTR_CM_INTERFACE"
    )
    return config
