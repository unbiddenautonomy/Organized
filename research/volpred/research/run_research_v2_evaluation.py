"""Run the hash-authorized Research V2 development/validation evaluation."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.research_v2_evaluation import (
    development_validation_gate,
    estimate_cscv_pbo,
    full_trial_count,
    load_authorized_ledgers,
    plateau_gate,
    primary_tail_gate,
    sha256_file,
    summarize_ledgers,
    tail_sample_sufficiency,
    trial_registry_counts,
    validate_decision_coverage,
)


AUTHORIZATION = ROOT / "STRATEGY_EVALUATION_AUTHORIZATION_V2.json"
AUTHORIZATION_SIDECAR = ROOT / "STRATEGY_EVALUATION_AUTHORIZATION_V2.json.sha256"
TRIALS = ROOT / "TRIALS.csv"
OUT = ROOT / "outputs" / "research_v2" / "evaluation"


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_authorization_sidecar(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PermissionError("authorization sidecar must contain exactly one nonempty line")
    match = re.fullmatch(r"([0-9a-fA-F]{64}) \*([^\\/]+)", lines[0])
    if match is None:
        raise PermissionError("authorization sidecar must use '<sha256> *<basename>' format")
    digest, basename = match.groups()
    if basename != AUTHORIZATION.name:
        raise PermissionError("authorization sidecar names an unexpected file")
    return digest.lower(), basename


def verify_authorization_integrity(authorization: dict[str, Any]) -> None:
    expected_authorization_hash, _ = parse_authorization_sidecar(
        AUTHORIZATION_SIDECAR.read_text(encoding="utf-8")
    )
    if sha256_file(AUTHORIZATION) != expected_authorization_hash:
        raise PermissionError("evaluation authorization hash mismatch")
    inventory = authorization.get("integrity_inventory", {})
    for relative, expected in inventory.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise PermissionError(f"authorization-bound integrity mismatch: {relative}")
    if authorization.get("pseudo_authorized") is not False:
        raise PermissionError("pseudo must remain unauthorized")
    ledgers = authorization.get("ledger_sha256", {})
    if len(ledgers) != 14 or any("pseudo" in str(path).lower() for path in ledgers):
        raise PermissionError("authorization must bind exactly 14 non-pseudo ledgers")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"evaluation output directory is not empty: {OUT}")
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    verify_authorization_integrity(authorization)
    paths = [Path(path) for path in authorization["ledger_sha256"]]
    ledger = load_authorized_ledgers(paths, AUTHORIZATION, allow_pseudo=False)
    trial_counts = trial_registry_counts(TRIALS)
    substantive_trials = full_trial_count(TRIALS)
    if substantive_trials != trial_counts["substantive_trials"]:
        raise AssertionError("substantive trial count mismatch")
    coverage = validate_decision_coverage(ledger)
    if coverage != {
        segment: {
            "eligible_decisions_per_trial": int(values["eligible_decisions_per_trial"]),
            "trial_count": int(values["trial_count"]),
            "coverage_pass": True,
            "absolute_decision_coverage_verified": True,
            "decision_universe_sha256": values["decision_universe_sha256"],
        }
        for segment, values in authorization["coverage"].items()
    }:
        raise AssertionError("computed ledger coverage differs from authorization")
    summary = summarize_ledgers(
        ledger, substantive_trials,
        assumption_metadata={
            "margin_verified": False, "lot_size_verified": False, "execution_verified": False,
        },
    )
    pbo = {
        segment: estimate_cscv_pbo(ledger, segment=segment, blocks=8)
        for segment in ("development", "validation")
    }
    gates = {
        "plateau": {
            segment: plateau_gate(summary, segment=segment)
            for segment in ("development", "validation")
        },
        "observed_tail": {
            segment: primary_tail_gate(summary, segment=segment)
            for segment in ("development", "validation")
        },
        "sample_sufficiency": {
            segment: tail_sample_sufficiency(summary, segment=segment)
            for segment in ("development", "validation")
        },
        "development_validation": development_validation_gate(summary),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "segment_trial_summary.csv", index=False)
    write_json(OUT / "coverage.json", coverage)
    write_json(OUT / "pbo.json", pbo)
    write_json(OUT / "trial_counts.json", trial_counts)
    write_json(OUT / "gates.json", gates)
    manifest = {
        "authorization": AUTHORIZATION.relative_to(ROOT).as_posix(),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "evaluator_sha256": sha256_file(ROOT / "research" / "research_v2_evaluation.py"),
        "runner_sha256": sha256_file(Path(__file__)),
        "trials_sha256": sha256_file(TRIALS),
        "ledger_count": len(paths), "pseudo_opened": False,
        "substantive_trial_count_for_dsr": substantive_trials,
        "administrative_trial_records_excluded_from_dsr": trial_counts["administrative_records"],
        "outputs": [
            "segment_trial_summary.csv", "coverage.json", "pbo.json",
            "trial_counts.json", "gates.json", "manifest.json", "SHA256SUMS",
        ],
        "reproduction_command": "py -3.13 research/run_research_v2_evaluation.py",
        "interpretation": "research-only; not verified, live eligible, or acceptance",
    }
    write_json(OUT / "manifest.json", manifest)
    hashed = [path for path in sorted(OUT.iterdir()) if path.name != "SHA256SUMS"]
    (OUT / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)} *{path.relative_to(ROOT).as_posix()}\n" for path in hashed
    ), encoding="utf-8")
    print(json.dumps(clean_json({"coverage": coverage, "pbo": pbo, "gates": gates,
                                 "trial_counts": trial_counts}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
