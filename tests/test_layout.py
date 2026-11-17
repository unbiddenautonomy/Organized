"""The layering rules from docs/ARCHITECTURE.md, checked on every test run."""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The PCR reference backtest reruns the original study, whose code lives in research/pcr.
SYS_PATH_ALLOWED = {"pipelines/pcr/backtest.py"}


def modules(folder):
    for path in sorted((REPO / folder).rglob("*.py")):
        relative = path.relative_to(REPO).as_posix()
        if "/tests/" not in relative:
            yield relative, ast.parse(path.read_text(encoding="utf-8"))


def imported(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_voltr_never_imports_a_pipeline():
    offenders = [(name, module) for name, tree in modules("voltr") for module in imported(tree)
                 if module == "pipelines" or module.startswith("pipelines.")]
    assert offenders == []


def test_pipelines_never_import_each_other():
    offenders = []
    for name, tree in modules("pipelines"):
        own = name.split("/")[1]
        for module in imported(tree):
            parts = module.split(".")
            if parts[0] == "pipelines" and len(parts) > 1 and parts[1] != own:
                offenders.append((name, module))
    assert offenders == []


def test_no_sys_path_edits_in_shared_or_pipeline_code():
    offenders = [name for folder in ("voltr", "pipelines") for name, tree in modules(folder)
                 if name not in SYS_PATH_ALLOWED and any(
                     isinstance(n, ast.Attribute) and n.attr == "path" and isinstance(n.value, ast.Name) and n.value.id == "sys"
                     for n in ast.walk(tree))]
    assert offenders == []
