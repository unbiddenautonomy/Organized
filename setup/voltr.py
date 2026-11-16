"""Configure and validate a portable BANKNIFTY paper-trading checkout.

This command never starts a market receiver or changes a scheduled task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "pipelines" / "families"
APP = FAMILIES / "ivrv" / "engine" / "Inference"
LIVE = APP / "live"
SOURCE = FAMILIES / "ivrv" / "standalone"
PCR = FAMILIES / "PCR"
WEEKS = FAMILIES / "weeks43"
MANIFEST = Path(__file__).with_name("model_manifest.json")
PIPELINES = ("ivrv", "pcr", "43weeks", "rv120", "legacy-spot-iv")


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure(args: argparse.Namespace) -> int:
    machine_file = LIVE / "machine.local.json"
    machine = json.loads(machine_file.read_text(encoding="utf-8")) if machine_file.is_file() else {}
    values = {
        "interface_ip": args.fo_interface,
        "cm_interface_ip": args.cm_interface,
        "contract_download_url": args.contract_url,
        "dashboard_port": args.dashboard_port,
    }
    changed = False
    for key, value in values.items():
        if value is not None and machine.get(key) != value:
            machine[key] = value
            changed = True
    if changed:
        _save_json(machine_file, machine)
    print(f"machine settings: {machine_file if machine else 'portable defaults'}")

    source_file = SOURCE / "config.json"
    if source_file.exists():
        source = json.loads(source_file.read_text(encoding="utf-8"))
        print(f"preserved local source config: {source_file}")
    else:
        source = json.loads((SOURCE / "config.example.json").read_text(encoding="utf-8"))
        _save_json(source_file, source)
        print(f"created local source config: {source_file}")
    if args.legacy_contract_file:
        source["contract_file"] = str(Path(args.legacy_contract_file).expanduser().resolve())
        _save_json(source_file, source)
    print("No strategy thresholds or existing credentials were reset.")
    return 0


def _model_group(relative: str) -> str | None:
    path = relative.replace("\\", "/")
    if "/standalone/models/rv120/" in path:
        return "rv120"
    if "/families/PCR/" in path:
        return "pcr"
    if "/families/weeks43/" in path:
        return "43weeks"
    if "/families/ivrv/" in path or "/families/rising_rv/" in path:
        return "ivrv"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_models(group: str, errors: list[str]) -> int:
    if not MANIFEST.is_file():
        errors.append(f"missing saved model manifest: {MANIFEST}")
        return 0
    files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    selected = {name: sha for name, sha in files.items() if _model_group(name) == group}
    if not selected:
        errors.append(f"no model artifacts listed for {group}")
    for name, expected in selected.items():
        path = ROOT / name
        if not path.is_file():
            errors.append(f"missing model: {name}")
        elif _sha256(path) != expected:
            errors.append(f"model hash mismatch: {name}")
    return len(selected)


def _preflight_ivrv(errors: list[str], notes: list[str]) -> None:
    sys.path.insert(0, str(APP))
    from live.live_inference import load_live_config
    from inference_pipeline import FixedStrikeInferencePipeline
    from live.lzo_codec import Lzo1zDecoder

    config = load_live_config(LIVE / "live_config.json")
    if config["enabled_pipelines"] != ["ivrv"]:
        notes.append(f"enabled scheduled workers: {config['enabled_pipelines']}")
    for key in ("source_root", "source_config", "inference_config", "lzo_dll"):
        if not Path(config[key]).exists():
            errors.append(f"missing {key}: {config[key]}")
    if not errors:
        FixedStrikeInferencePipeline(config["inference_config"])
        Lzo1zDecoder(config["lzo_dll"])
    notes.append(f"FO/CM adapter: {config['interface_ip']} / {config['cm_interface_ip']}")
    if not Path(config["contract_file"]).is_file():
        if config.get("contract_download_enabled"):
            token_name = config.get("contract_auth_token_env", "ERP_AUTH_TOKEN")
            notes.append(f"live contract download requires {token_name} and access to {config['contract_download_url']}")
        else:
            errors.append(f"missing contract master: {config['contract_file']}")


def _preflight_pcr(errors: list[str]) -> None:
    sys.path.insert(0, str(FAMILIES))
    from PCR.runtime import Models, load_config

    load_config()
    Models()


def _preflight_weeks(errors: list[str]) -> None:
    sys.path.insert(0, str(WEEKS))
    from weeks43_engine import ModelRunner

    config = json.loads((WEEKS / "config.json").read_text(encoding="utf-8"))
    if not config.get("enabled"):
        errors.append("43Weeks config.json disables its worker")
    ModelRunner(WEEKS)


def _preflight_rv120(errors: list[str], notes: list[str]) -> None:
    import joblib

    folder = SOURCE / "models" / "rv120"
    rv = joblib.load(folder / "rv_intraday_120m_model.joblib")
    hybrid = joblib.load(folder / "expanded_hybrid_models.joblib")
    if rv.get("horizon_min") != 120 or hybrid.get("secondary_horizon_min") != 120:
        errors.append("RV/hybrid saved horizon is not 120 minutes")
    config_file = SOURCE / "config.json"
    config = json.loads((config_file if config_file.is_file() else SOURCE / "config.example.json").read_text())
    if not config.get("contract_file"):
        notes.append("RV/hybrid live start requires a current contract file in local config or --contract-file")


def _preflight_legacy(errors: list[str], notes: list[str]) -> None:
    config_file = SOURCE / "config.json"
    if not config_file.is_file():
        errors.append("run setup configure to create the ignored older-engine config.json")
        return
    sys.path.insert(0, str(SOURCE))
    from portable_config import load_source_config

    config = load_source_config(config_file)
    for key in ("spot_model_dir", "iv_model_dir", "pnl_model_dir", "hmm_model_dir"):
        if not Path(config[key]).is_dir():
            errors.append(f"missing older-engine package {key}: {config[key]}")
    if not config.get("contract_file"):
        notes.append("older engine needs a current contract file or a configured PostgreSQL contract table")


def preflight(args: argparse.Namespace) -> int:
    requested = PIPELINES if args.pipeline == "all" else (args.pipeline,)
    report = {}
    for pipeline in requested:
        errors: list[str] = []
        notes: list[str] = []
        checked = verify_models(pipeline, errors) if pipeline != "legacy-spot-iv" else 0
        try:
            if pipeline == "ivrv":
                _preflight_ivrv(errors, notes)
            elif pipeline == "pcr":
                _preflight_pcr(errors)
            elif pipeline == "43weeks":
                _preflight_weeks(errors)
            elif pipeline == "rv120":
                _preflight_rv120(errors, notes)
            else:
                _preflight_legacy(errors, notes)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        report[pipeline] = {"offline_ready": not errors, "model_files_checked": checked,
                            "errors": errors, "notes": notes}
    print(json.dumps(report, indent=2))
    return 0 if all(item["offline_ready"] for item in report.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure", help="write ignored machine settings; preserve existing local credentials")
    setup.add_argument("--fo-interface", help="adapter alias, IP or auto")
    setup.add_argument("--cm-interface", help="adapter alias, IP or auto")
    setup.add_argument("--contract-url", help="machine's contract service URL")
    setup.add_argument("--dashboard-port", type=int)
    setup.add_argument("--legacy-contract-file", help="current FO master for the older standalone engines")
    check = commands.add_parser("preflight", help="verify local models and configuration without receiving market data")
    check.add_argument("--pipeline", choices=(*PIPELINES, "all"), default="ivrv")
    args = parser.parse_args()
    return configure(args) if args.command == "configure" else preflight(args)


if __name__ == "__main__":
    raise SystemExit(main())
