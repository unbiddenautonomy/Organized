"""Refactor parity checks: capture a baseline, then compare each step against it.

    python tools/parity/run.py capture --label baseline
    python tools/parity/run.py check --label step03 --against baseline

Every output goes to runtime/parity/<label>/ (gitignored). The run refuses to
finish if anything in the repo outside runtime/parity changed while it ran.

Checks
  A tests      the four suites, file by file where needed
  B replay     simulations/run_replay_20261102.py on both prepared days
  C livefeed   synthetic NSE packets through the real IVRV live path
  D preflight  setup/voltr.py preflight --pipeline all
  E dashboard  HTTP routes of the IVRV dashboard served from a livefeed run
  F inventory  every gitignored file, by content hash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_ROOT = REPO / "runtime" / "parity"
DAYS = {
    "2026-11-02": REPO.parent / "data" / "replays" / "banknifty" / "2026-11-02" / "2026-11-02",
    "2026-11-03": REPO.parent / "data" / "replays" / "banknifty" / "2026-11-03" / "2026-11-03",
}
# Location-specific: refactor steps update these and nothing else.
FAMILIES = REPO / "pipelines" / "families"
SUITES = [
    ("ivrv", FAMILIES / "ivrv" / "engine" / "Inference", ["."]),
    ("pcr", FAMILIES / "PCR", ["test_pcr.py"]),
    ("weeks43", FAMILIES / "weeks43", ["test_completion_audit.py", "test_entry_controls.py", "test_weeks43.py"]),
    ("standalone", FAMILIES / "ivrv" / "standalone", ["."]),
    ("voltr", REPO / "tests", ["."]),  # added in step 2; absent from the baseline
]
REPLAY_SCRIPT = REPO / "simulations" / "run_replay_20261102.py"
PREFLIGHT_SCRIPT = REPO / "setup" / "voltr.py"

# Fields that record when or where a run happened, not what it decided.
VOLATILE_KEYS = {"server_time", "written_at", "generated_at", "processing_delay", "processing_delay_seconds",
                 "received_at", "wall_time", "run_id", "output", "pid", "elapsed_seconds", "updated_at"}
ENV = dict(os.environ, POSITION_ALERT_DISABLED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items()) if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        return float(f"{value:.12g}")
    if isinstance(value, str):
        # Absolute paths differ between runs; keep only the part after the run folder.
        return re.sub(r".*runtime[\\/]+parity[\\/]+[^\\/]+[\\/]+", "<run>/", value)
    return value


def digest_file(path: Path) -> dict:
    """Content hash after removing volatile fields; per-record hashes for jsonl."""
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(hashlib.sha256(json.dumps(canonical(json.loads(line)), sort_keys=True)
                                              .encode()).hexdigest()[:16])
        whole = hashlib.sha256("\n".join(records).encode()).hexdigest()
        return {"sha256": whole, "records": len(records), "record_hashes": records}
    if path.suffix == ".json":
        data = canonical(json.loads(path.read_text(encoding="utf-8")))
        return {"sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}
    if path.suffix == ".log":
        return {"skipped": "log text carries wall-clock stamps"}
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def digest_tree(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): digest_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def py(*args, cwd=REPO, timeout=1800):
    return subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=ENV, capture_output=True,
                          text=True, timeout=timeout)


def check_tests(out: Path) -> dict:
    result = {}
    for name, cwd, targets in SUITES:
        counts = {"passed": 0, "failed": 0, "errors": 0}
        for target in targets:
            run = py("-m", "pytest", "-q", "-p", "no:cacheprovider", target, cwd=cwd)
            (out / f"tests_{name}.txt").open("a", encoding="utf-8").write(run.stdout + run.stderr)
            if run.returncode:
                # Some worker tests race a subprocess (test_completion_audit was seen flaky at baseline).
                # Rerun once; a real failure fails twice. Both runs stay in the log.
                counts.setdefault("reruns", []).append(target)
                run = py("-m", "pytest", "-q", "-p", "no:cacheprovider", target, cwd=cwd)
                (out / f"tests_{name}.txt").open("a", encoding="utf-8").write(
                    "\n=== RERUN ===\n" + run.stdout + run.stderr)
            tail = (run.stdout.strip().splitlines() or [""])[-1]
            for key, pattern in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"), ("errors", r"(\d+) error")):
                m = re.search(pattern, tail)
                counts[key] += int(m.group(1)) if m else 0
        result[name] = {k: v for k, v in counts.items() if k != "reruns"}
        if counts.get("reruns"):
            with (out / "test_reruns.json").open("a", encoding="utf-8") as f:
                f.write(json.dumps({name: counts["reruns"]}) + "\n")
    return result


def check_replay(out: Path) -> dict:
    result = {}
    for day, folder in DAYS.items():
        target = out / "replay" / day
        run = py(REPLAY_SCRIPT, "--data", folder, "--output", target)
        (out / f"replay_{day}.txt").write_text(run.stdout[-20000:] + run.stderr[-20000:], encoding="utf-8")
        if run.returncode:
            result[day] = {"error": f"exit {run.returncode}"}
            continue
        result[day] = digest_tree(target)
    return result


def check_livefeed(out: Path) -> dict:
    result = {}
    for day, folder in DAYS.items():
        target = out / "livefeed" / day
        run = py(HERE / "live_feed.py", "--data", folder, "--output", target)
        (out / f"livefeed_{day}.txt").write_text(run.stdout[-20000:] + run.stderr[-20000:], encoding="utf-8")
        result[day] = {"error": f"exit {run.returncode}"} if run.returncode else digest_tree(target)
    return result


def check_preflight(out: Path) -> dict:
    run = py(PREFLIGHT_SCRIPT, "preflight", "--pipeline", "all")
    (out / "preflight.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
    try:
        report = json.loads(run.stdout)
    except json.JSONDecodeError:
        return {"error": "preflight did not print JSON", "exit": run.returncode}
    return {"exit": run.returncode, "report": canonical(report)}


def check_dashboard(out: Path) -> dict:
    run = py(HERE / "dashboard_probe.py", "--logs", out / "livefeed" / "2026-11-02" / "logs")
    (out / "dashboard.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
    try:
        return json.loads(run.stdout)
    except json.JSONDecodeError:
        return {"error": "probe failed", "exit": run.returncode}


def check_inventory() -> dict:
    sys.path.insert(0, str(HERE))
    import inventory
    return inventory.build()


def repo_state() -> str:
    """Hash of git status plus the ignored inventory, excluding parity outputs."""
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO,
                            capture_output=True, text=True).stdout
    status = "\n".join(l for l in status.splitlines() if "runtime/parity/" not in l)
    inv = check_inventory()
    return hashlib.sha256((status + json.dumps(inv, sort_keys=True)).encode()).hexdigest()


CHECKS = {"A_tests": check_tests, "B_replay": check_replay, "C_livefeed": check_livefeed,
          "D_preflight": check_preflight, "E_dashboard": check_dashboard}


def capture(label: str, resume: bool = False) -> Path:
    """Run every check into runtime/parity/<label>/.

    A partial summary is saved after each check, so an interrupted run can be
    finished with --resume instead of starting again.
    """
    out = OUT_ROOT / label
    partial = out / "summary.partial.json"
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f"{out} already exists; pick a new label or pass --resume")
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(partial.read_text(encoding="utf-8")) if resume and partial.exists() else {}
    summary.setdefault("label", label)
    summary.setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
    before = repo_state()
    for name, fn in CHECKS.items():
        if name in summary:
            print(f"{name}: kept from earlier run", flush=True)
            continue
        t0 = time.time()
        summary[name] = fn(out)
        partial.write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    summary["F_inventory"] = check_inventory()
    after = repo_state()
    summary["repo_unchanged_by_run"] = before == after
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    if before != after:
        print("WARNING: the repo changed while the parity run was going (outside runtime/parity)")
    return out


def differences(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = []
        for key in sorted(set(a) | set(b)):
            if key in {"label", "started"}:
                continue
            if key not in a or key not in b:
                diffs.append(f"{path}/{key}: only in {'current' if key in b else 'baseline'}")
            else:
                diffs += differences(a[key], b[key], f"{path}/{key}")
        return diffs
    if isinstance(a, list) and isinstance(b, list) and path.endswith("record_hashes"):
        if a == b:
            return []
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return [f"{path}: first differing record #{first + 1} (baseline {len(a)}, current {len(b)})"]
    return [] if a == b else [f"{path}: baseline {json.dumps(a)[:120]} != current {json.dumps(b)[:120]}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--label", required=True)
    c.add_argument("--resume", action="store_true")
    k = sub.add_parser("check")
    k.add_argument("--label", required=True)
    k.add_argument("--against", default="baseline")
    k.add_argument("--resume", action="store_true")
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    args = parser.parse_args()
    if args.command == "compare":
        a, b = args.a, args.b
    else:
        capture(args.label, args.resume)
        if args.command == "capture":
            return 0
        a, b = args.against, args.label
    base = json.loads((OUT_ROOT / a / "summary.json").read_text(encoding="utf-8"))
    cur = json.loads((OUT_ROOT / b / "summary.json").read_text(encoding="utf-8"))
    diffs = differences(base, cur)
    report = OUT_ROOT / b / f"diff_vs_{a}.txt"
    report.write_text("\n".join(diffs) or "IDENTICAL", encoding="utf-8")
    print("IDENTICAL" if not diffs else f"{len(diffs)} differences (see {report}):\n" + "\n".join(diffs[:40]))
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
