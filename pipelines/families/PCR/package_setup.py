"""Copy the approved PCR artifacts once; preserve a rollback copy of the host."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("VOLTR_PCR_IMPORT_SOURCE", ROOT / "import_source"))


def main():
    package_source = SOURCE / "output" / "zero_week_focus_stack_zero_brokerage"
    for folder in (package_source / "models").iterdir():
        if folder.is_dir():
            destination = ROOT / "models" / folder.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite model package {destination}")
            shutil.copytree(folder, destination)
    for name in ("thresholds.json", "run_metadata.json", "model_family_selection_2021.csv"):
        dest = ROOT / ("training_metadata.json" if name == "run_metadata.json" else name)
        shutil.copy2(package_source / name, dest)
    research = ROOT / "research"
    research.mkdir(exist_ok=True)
    scripts = ["train_win_classifier.py", "run_expiry_week_shared_stack.py", "optimize_expiry_week_shared_stack.py",
               "run_zero_week_focus_stack.py", "run_straddle_target_stop_pairs.py", "run_target_3000_stop_1500_max3.py"]
    for name in scripts:
        shutil.copy2(SOURCE / name, research / name)
    reference = ROOT / "reference"
    reference.mkdir(exist_ok=True)
    shutil.copy2(package_source / "all_test_candidates.csv.gz", reference / "all_test_candidates.csv.gz")
    results = SOURCE / "output" / "target_3000_stop_1500_max3"
    shutil.copytree(results, reference / "max3_original")
    backup = ROOT / "integration_backup"
    backup.mkdir(exist_ok=True)
    app = ROOT.parent / "ivrv" / "engine" / "Inference"
    preserved = [app / "live" / "live_inference.py", app / "live" / "live_config.json", app / "config.yaml", app / "live" / "live_dashboard.html"]
    baseline = {}
    for file in preserved:
        shutil.copy2(file, backup / file.name)
        baseline[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (backup / "hashes_before.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "lightgbm", "xgboost")}
    (ROOT / "requirements.lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items()) + "\n", encoding="utf-8")
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "models").rglob("*") if p.is_file()}
    (ROOT / "model_files.sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Copied four model packages and reproducibility sources to {ROOT}")


if __name__ == "__main__":
    main()
