"""Machine-specific settings: config-relative paths and the multicast network adapter."""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def relative_to_config(value: str | Path, config_path: Path) -> Path:
    """Resolve `value` against the folder of the config file that names it."""
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
    """IPv4 of the adapter named by `environment`, by `value`, or the only active one ("auto")."""
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


# Settings that differ per PC. Only these may be set in the git-ignored
# machine.local.json next to live_config.json; strategy settings stay shared.
MACHINE_LOCAL_KEYS = frozenset({"interface_ip", "cm_interface_ip", "contract_download_url", "dashboard_port"})


def machine_local_overrides(live_config: Path) -> dict[str, Any]:
    """Return this PC's settings from machine.local.json, or {} if it is absent."""
    machine = Path(live_config).with_name("machine.local.json")
    if not machine.is_file():
        return {}
    overrides = json.loads(machine.read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise ValueError(f"{machine} must be a JSON object")
    unknown = set(overrides) - MACHINE_LOCAL_KEYS
    if unknown:
        raise ValueError(f"{machine} may only override {sorted(MACHINE_LOCAL_KEYS)}; got {sorted(unknown)}")
    return overrides
