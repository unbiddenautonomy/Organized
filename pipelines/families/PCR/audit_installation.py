"""Verify deployment provenance and save a reviewable installation inventory."""
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_result(path):
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    result = {k: suite.attrib[k] for k in ("tests", "failures", "errors", "skipped", "time")}
    assert int(result["failures"]) == 0 and int(result["errors"]) == 0, result
    return result


def main():
    expected = json.loads((ROOT / "model_files.sha256.json").read_text())
    assert all(digest(ROOT/name) == sha for name, sha in expected.items()), "Copied model artifacts changed"
    before = json.loads((ROOT / "integration_backup" / "hashes_before.json").read_text())
    original_files = {name:{"before":sha, "after":digest(Path(name))} for name,sha in before.items()}
    for name, pair in original_files.items():
        if not name.endswith("live_inference.py"):
            assert pair["before"] == pair["after"], name
    def summary(path):
        with path.open(encoding="utf-8") as f:
            return next(csv.DictReader(f))
    reference = summary(ROOT / "reference" / "max3_original" / "summary.csv")
    reproduction = summary(ROOT / "backtests" / "installation_reference_check" / "summary.csv")
    assert reference == reproduction, "Historical reference differs"
    result = {"installed_at":datetime.now().isoformat(), "paper_only":True,
              "source_model_directory":r"E:\PCR\hypothesis testing\output\zero_week_focus_stack_zero_brokerage\models",
              "production_endpoint":"http://127.0.0.1:8780/PCR",
              "activation":"Next normal launch of the existing paper trading application",
              "model_files_verified":len(expected), "original_files":original_files,
              "pcr_tests":test_result(ROOT/"validation"/"pcr_tests.xml"),
              "original_app_tests":test_result(ROOT/"validation"/"original_app_tests.xml"),
              "reference_summary_exact_match":True, "reference_summary":reproduction,
              "browser_validation":{"page_loaded":True,"four_model_outputs_visible":True,
                                    "three_position_capacity_visible":True,"cutoff_save_success":True,
                                    "console_errors":0,"screenshot":"validation/dashboard.jpg"},
              "prospective_live_market_session_executed":False,
              "config":json.loads((ROOT/"config.json").read_text()),
              "limitations":["Historical reference contains 178 synthetic exits.",
                             "Historical test has been repeatedly reviewed and is not a fresh evaluation.",
                             "Live LTP fills and current configured fees can differ from historical results."]}
    files = [p for p in ROOT.rglob("*") if p.is_file() and not {"__pycache__", ".pytest_cache", "logs", "replays"}.intersection(p.relative_to(ROOT).parts)
             and p.name not in {"deployment_manifest.json", "installation_audit.json", "bridge_status.json"}]
    result["files_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"):digest(p) for p in files}
    (ROOT / "deployment_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    compact = {k:v for k,v in result.items() if k != "files_sha256"}
    (ROOT / "validation" / "installation_audit.json").write_text(json.dumps(compact,indent=2),encoding="utf-8")
    print(json.dumps({"model_files_verified":len(expected),"inventory_files":len(files),
                      "pcr_tests":result["pcr_tests"],"original_app_tests":result["original_app_tests"],
                      "reference_exact_match":True},indent=2))


if __name__ == "__main__":
    main()
