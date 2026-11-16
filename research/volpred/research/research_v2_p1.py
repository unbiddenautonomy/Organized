"""P0 HAR-Ridge versus preregistered P1 HARQ ElasticNet, predictive only.

Reads only development and walk-forward-validation rows from the frozen common
120-minute panel.  Pseudo-holdout rows are never selected.  No strategy or PnL
logic is present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from itertools import combinations
from math import erf, sqrt
from pathlib import Path
from typing import Iterable
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

try:
    from . import research_v2_splits as split_guards
except ImportError:  # direct script execution
    import research_v2_splits as split_guards


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs/research_v2/common/predictive_dataset_120m.parquet"
OUT = ROOT / "outputs/research_v2/p1"
SPLITS_FILE = ROOT / "SPLITS.md"
SPLITS_HASH_FILE = ROOT / "SPLITS.sha256"
COMMON_MANIFEST = ROOT / "outputs/research_v2/common/manifest.json"
COMMON_HASHES = ROOT / "outputs/research_v2/common/SHA256SUMS"
GLOBAL_TRIALS = ROOT / "TRIALS.csv"
SEED = 7319
P0_FEATURES = ["p0_log_rv_5", "p0_log_rv_15", "p0_log_rv_60"]
P1_FEATURES = P0_FEATURES + [
    "p1_log_rq_15", "p1_log_rq_60",
    "p1_downside_rv_15", "p1_upside_rv_15",
    "p1_downside_rv_60", "p1_upside_rv_60",
    "p1_jump_rv_60", "p1_abs_return", "p1_tod_sin", "p1_tod_cos",
]
P0_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
P1_ALPHAS = (0.0001, 0.001, 0.01, 0.1)
P1_L1_RATIOS = (0.0, 0.25, 0.5)
ALLOWED_SPLITS = ("development", "walk_forward_validation")
TRIAL_COLUMNS = [
    "trial_id", "phase", "fold", "model", "alpha", "l1_ratio", "train_rows",
    "test_rows", "train_max_label_end", "test_first_decision", "purge_ok",
    "embargo_ok", "status", "error", "qlike", "log_mse", "seed",
]
ROOT_TRIAL_COLUMNS = [
    "trial_id", "hypothesis", "registered_at", "status", "structure",
    "horizon_min", "decision_time", "dte_band", "parameter_grid",
    "outcome_opened", "notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_inventory(path: Path) -> dict[str, str]:
    inventory = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        inventory[name.lstrip("* ")] = digest
    return inventory


def verify_frozen_foundations(
    dataset: Path = DATASET,
    *,
    splits_file: Path = SPLITS_FILE,
    splits_hash_file: Path = SPLITS_HASH_FILE,
    manifest_path: Path = COMMON_MANIFEST,
    hashes_path: Path = COMMON_HASHES,
) -> dict:
    expected_split = splits_hash_file.read_text(encoding="ascii").strip().split()[0]
    actual_split = split_guards.split_protocol_hash(splits_file)
    if actual_split != expected_split:
        raise PermissionError("SPLITS.md checksum mismatch")
    inventory = _hash_inventory(hashes_path)
    if sha256_file(manifest_path) != inventory.get("manifest.json"):
        raise PermissionError("common manifest checksum mismatch")
    dataset_hash = sha256_file(dataset)
    if dataset_hash != inventory.get(dataset.name):
        raise PermissionError("predictive dataset checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("splits_sha256") != actual_split:
        raise PermissionError("manifest split checksum mismatch")
    if manifest.get("dataset_sha256") != dataset_hash:
        raise PermissionError("manifest dataset checksum mismatch")
    if manifest.get("p0_features") != P0_FEATURES or manifest.get("p1_features") != P1_FEATURES:
        raise PermissionError("manifest feature contract mismatch")
    return {"splits_sha256": actual_split, "dataset_sha256": dataset_hash,
            "manifest_sha256": sha256_file(manifest_path)}


def load_allowed_panel(path: Path = DATASET) -> pd.DataFrame:
    """Select only the two authorized split labels at the storage scan."""
    columns = [
        "decision_time", "decision_date", "label_end_at", "split",
        "rv_fwd_120", "target", *P1_FEATURES,
    ]
    frame = split_guards.read_restricted_parquet(
        path, allowed_splits=[split_guards.DEVELOPMENT, split_guards.WALK_FORWARD],
        columns=columns, context="P1 predictive research",
    ).sort_values("decision_time").reset_index(drop=True)
    if frame.empty or not set(frame["split"]).issubset(ALLOWED_SPLITS):
        raise RuntimeError("authorized split selection failed closed")
    for column in ("decision_time", "decision_date", "label_end_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert("Asia/Kolkata")
    split_guards.assert_no_pseudo_holdout(frame, context="P1 predictive research")
    split_guards.assert_calendar_partition(frame)
    numeric = ["rv_fwd_120", "target", *P1_FEATURES]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise RuntimeError("nonfinite model input in authorized panel")
    if not frame["rv_fwd_120"].gt(0).all():
        raise RuntimeError("target variance must be positive")
    return frame


@dataclass(frozen=True)
class Fold:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray
    test_blocks: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    embargo_dates: tuple[pd.Timestamp, ...]


def _date_values(frame: pd.DataFrame) -> np.ndarray:
    return frame["decision_time"].dt.normalize().drop_duplicates().sort_values().to_numpy()


def _purged_train_mask(
    frame: pd.DataFrame,
    test_dates: Iterable[pd.Timestamp],
    all_sessions: Iterable[pd.Timestamp],
) -> tuple[np.ndarray, tuple[tuple[pd.Timestamp, pd.Timestamp], ...], tuple[pd.Timestamp, ...]]:
    normalized_test = pd.DatetimeIndex(test_dates).sort_values()
    session_index = pd.DatetimeIndex(all_sessions).sort_values()
    test_set = set(normalized_test)
    mask = ~frame["decision_time"].dt.normalize().isin(test_set).to_numpy()

    positions = sorted(session_index.get_loc(day) for day in normalized_test)
    runs: list[list[int]] = []
    for position in positions:
        if not runs or position != runs[-1][-1] + 1:
            runs.append([position])
        else:
            runs[-1].append(position)

    blocks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    embargo: list[pd.Timestamp] = []
    decision = frame["decision_time"]
    label_end = frame["label_end_at"]
    for run in runs:
        start_day, end_day = session_index[run[0]], session_index[run[-1]]
        block_rows = frame[decision.dt.normalize().isin(session_index[run])]
        start = block_rows["decision_time"].min()
        end = block_rows["decision_time"].max()
        blocks.append((start, end))
        mask &= ~((decision < start) & (label_end >= start)).to_numpy()
        next_position = run[-1] + 1
        if next_position < len(session_index):
            embargo_day = session_index[next_position]
            embargo.append(embargo_day)
            mask &= decision.dt.normalize().ne(embargo_day).to_numpy()
    return mask, tuple(blocks), tuple(embargo)


def assert_fold_integrity(frame: pd.DataFrame, fold: Fold) -> None:
    train, test = frame.iloc[fold.train_index], frame.iloc[fold.test_index]
    if train.empty or test.empty:
        raise AssertionError("empty train or test fold")
    test_dates = set(test["decision_time"].dt.normalize())
    if train["decision_time"].dt.normalize().isin(test_dates).any():
        raise AssertionError("test session leaked into training")
    for start, _ in fold.test_blocks:
        if ((train["decision_time"] < start) & (train["label_end_at"] >= start)).any():
            raise AssertionError("training label overlaps test block")
    if train["decision_time"].dt.normalize().isin(fold.embargo_dates).any():
        raise AssertionError("post-test embargo session leaked into training")


def cpcv_folds(development: pd.DataFrame) -> list[Fold]:
    sessions = pd.DatetimeIndex(_date_values(development))
    _, guarded_folds = split_guards.cpcv_8_choose_2(sessions)
    folds: list[Fold] = []
    decision_dates = development["decision_time"].dt.normalize()
    for guarded in guarded_folds:
        left, right = guarded.test_groups
        test_dates = pd.DatetimeIndex(guarded.test_dates).tz_localize("Asia/Kolkata")
        guarded_train = split_guards.purged_embargoed_train(
            development, test_dates=test_dates, all_sessions=sessions,
        )
        test_mask = decision_dates.isin(test_dates).to_numpy()
        train_mask = development.index.isin(guarded_train.index)
        _, blocks, embargo = _purged_train_mask(development, test_dates, sessions)
        fold = Fold(
            f"g{left + 1}_g{right + 1}", np.flatnonzero(train_mask),
            np.flatnonzero(test_mask), blocks, embargo,
        )
        split_guards.assert_purge_embargo(
            development.iloc[fold.train_index], development.iloc[fold.test_index],
            all_sessions=sessions,
        )
        assert_fold_integrity(development, fold)
        folds.append(fold)
    if len(folds) != 28:
        raise AssertionError("CPCV must contain exactly 28 two-of-eight folds")
    return folds


def qlike(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, float)
    forecast = np.asarray(forecast, float)
    if np.any(actual <= 0) or np.any(forecast <= 0) or not np.isfinite(forecast).all():
        return float("inf")
    ratio = actual / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def log_mse(actual_log: np.ndarray, forecast: np.ndarray) -> float:
    if np.any(np.asarray(forecast) <= 0):
        return float("inf")
    return float(np.mean((np.asarray(actual_log, float) - np.log(forecast)) ** 2))


@dataclass
class LogVarianceModel:
    scaler: StandardScaler
    estimator: object
    smearing: float
    features: tuple[str, ...]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.scaler.transform(frame[list(self.features)].to_numpy(float))
        log_forecast = np.asarray(self.estimator.predict(transformed), float)
        return np.maximum(np.exp(np.clip(log_forecast, -50, 20)) * self.smearing, 1e-12)


def fit_model(
    train: pd.DataFrame,
    features: list[str],
    *,
    model: str,
    alpha: float,
    l1_ratio: float | None = None,
) -> LogVarianceModel:
    scaler = StandardScaler()
    x = scaler.fit_transform(train[features].to_numpy(float))
    y = train["target"].to_numpy(float)
    if model == "ridge":
        estimator = Ridge(alpha=alpha)
    elif model == "elastic_net":
        estimator = ElasticNet(
            alpha=alpha, l1_ratio=float(l1_ratio), fit_intercept=True,
            max_iter=20_000, tol=1e-6, selection="cyclic", random_state=SEED,
        )
    else:
        raise ValueError(f"unknown model {model}")
    estimator.fit(x, y)
    residual = y - np.asarray(estimator.predict(x), float)
    smearing = float(np.mean(np.exp(np.clip(residual, -50, 50))))
    if not np.isfinite(smearing) or smearing <= 0:
        raise RuntimeError("invalid training-only smearing correction")
    return LogVarianceModel(scaler, estimator, smearing, tuple(features))


def _lock_handle(handle) -> None:
    """Acquire a non-blocking exclusive one-byte registry lock or fail closed."""
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("global trial registry is locked") from exc
    else:
        import fcntl
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("global trial registry is locked") from exc


def _unlock_handle(handle) -> None:
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_global_trial(path: Path, row: dict[str, object]) -> None:
    """Append one exact-schema row under an exclusive lock; never rewrite."""
    if not path.exists():
        raise RuntimeError("global TRIALS.csv must already exist")
    with path.open("r+", newline="", encoding="utf-8") as handle:
        _lock_handle(handle)
        try:
            handle.seek(0)
            header = next(csv.reader(handle), [])
            if header != ROOT_TRIAL_COLUMNS:
                raise RuntimeError("global TRIALS.csv schema mismatch")
            handle.seek(0, os.SEEK_END)
            csv.DictWriter(handle, fieldnames=ROOT_TRIAL_COLUMNS).writerow(
                {column: row.get(column, "") for column in ROOT_TRIAL_COLUMNS}
            )
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_handle(handle)


def register_interrupted_attempts(
    detailed_path: Path,
    global_path: Path = GLOBAL_TRIALS,
    *,
    registered_at: str = "2026-10-13T18:38:36+05:30",
) -> int:
    """Idempotently preserve the pre-remediation interrupted runner attempts."""
    with detailed_path.open(newline="", encoding="utf-8") as handle:
        legacy = list(csv.DictReader(handle))
    with global_path.open(newline="", encoding="utf-8") as handle:
        existing = {row["trial_id"] for row in csv.DictReader(handle)}
    appended = 0
    for number, source in enumerate(legacy, 1):
        fingerprint = hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        trial_id = f"P1-LEGACY-INTERRUPTED-{number:04d}-{fingerprint}"
        if trial_id in existing:
            continue
        row = {
            "trial_id": trial_id,
            "hypothesis": "P0" if source.get("model") == "ridge" else "P1",
            "registered_at": registered_at,
            "status": "interrupted_pre_remediation",
            "structure": "predictive_model_fit",
            "horizon_min": 120,
            "decision_time": source.get("fold", ""),
            "dte_band": "NA",
            "parameter_grid": f"alpha={source.get('alpha')};l1_ratio={source.get('l1_ratio')}",
            "outcome_opened": True,
            "notes": (
                "legacy_runner_run=pre_remediation_20261013; "
                f"old_trial_id={source.get('trial_id')}; old_status={source.get('status')}; "
                f"phase={source.get('phase')}; error={source.get('error')}"
            ),
        }
        append_global_trial(global_path, row)
        existing.add(trial_id)
        appended += 1
    return appended


class TrialLogger:
    def __init__(self, path: Path, root_compatible_path: Path, run_id: str,
                 global_registry_path: Path = GLOBAL_TRIALS):
        self.path = path
        self.root_compatible_path = root_compatible_path
        self.run_id = run_id
        self.global_registry_path = global_registry_path
        self.failures = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as handle:
                if next(csv.reader(handle), []) != TRIAL_COLUMNS:
                    raise RuntimeError("existing trials.csv schema mismatch; refusing overwrite")
        else:
            with path.open("x", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS).writeheader()
        self.root_columns = ROOT_TRIAL_COLUMNS
        if root_compatible_path.exists():
            with root_compatible_path.open("r", newline="", encoding="utf-8") as handle:
                if next(csv.reader(handle), []) != self.root_columns:
                    raise RuntimeError("existing root-compatible trial schema mismatch")
        else:
            with root_compatible_path.open("x", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.root_columns).writeheader()

    def add(self, **record: object) -> None:
        record = dict(record)
        record["trial_id"] = f"{self.run_id}:{record['trial_id']}"
        root_row = {
            "trial_id": record["trial_id"],
            "hypothesis": "P0" if str(record.get("model")) == "ridge" else "P1",
            "registered_at": self.run_id,
            "status": record.get("status"),
            "structure": "predictive_model_fit",
            "horizon_min": 120,
            "decision_time": record.get("fold"),
            "dte_band": "NA",
            "parameter_grid": f"alpha={record.get('alpha')};l1_ratio={record.get('l1_ratio')}",
            "outcome_opened": True,
            "notes": f"phase={record.get('phase')}; error={record.get('error', '')}",
        }
        # The global registry is the first durable write and is lock-protected.
        append_global_trial(self.global_registry_path, root_row)
        row = {column: record.get(column, "") for column in TRIAL_COLUMNS}
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS).writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        if record.get("status") == "failure":
            self.failures += 1
        with self.root_compatible_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.root_columns).writerow(root_row)


def _trial(
    logger: TrialLogger,
    *,
    trial_id: str,
    phase: str,
    fold: str,
    model: str,
    alpha: float,
    l1_ratio: float | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    fit_and_predict,
) -> tuple[LogVarianceModel | None, np.ndarray | None, dict]:
    base = dict(
        trial_id=trial_id, phase=phase, fold=fold, model=model, alpha=alpha,
        l1_ratio="" if l1_ratio is None else l1_ratio, train_rows=len(train),
        test_rows=len(test), train_max_label_end=train["label_end_at"].max(),
        test_first_decision=test["decision_time"].min(), purge_ok=True,
        embargo_ok=True, seed=SEED,
    )
    try:
        fitted, forecast = fit_and_predict()
        metrics = {
            "qlike": qlike(test["rv_fwd_120"].to_numpy(float), forecast),
            "log_mse": log_mse(test["target"].to_numpy(float), forecast),
        }
        logger.add(**base, status="success", error="", **metrics)
        return fitted, forecast, metrics
    except Exception as exc:  # failure must remain in the immutable trial log
        logger.add(**base, status="failure", error=f"{type(exc).__name__}: {exc}")
        return None, None, {"qlike": float("inf"), "log_mse": float("inf")}


def inner_ridge_selection(train: pd.DataFrame, fold_name: str, phase: str, logger: TrialLogger) -> float:
    sessions = pd.DatetimeIndex(_date_values(train))
    cutoff = max(1, int(np.floor(len(sessions) * 0.8)))
    inner_test_dates = sessions[cutoff:]
    if inner_test_dates.empty:
        raise RuntimeError("insufficient sessions for inner Ridge selection")
    mask, blocks, embargo = _purged_train_mask(train, inner_test_dates, sessions)
    test_mask = train["decision_time"].dt.normalize().isin(inner_test_dates).to_numpy()
    fold = Fold(f"{fold_name}_inner", np.flatnonzero(mask), np.flatnonzero(test_mask), blocks, embargo)
    assert_fold_integrity(train, fold)
    inner_train, inner_test = train.iloc[fold.train_index], train.iloc[fold.test_index]
    scores = []
    for alpha in P0_ALPHAS:
        _, _, metric = _trial(
            logger, trial_id=f"{phase}:{fold_name}:P0:a={alpha}", phase=f"{phase}_p0_inner",
            fold=fold_name, model="ridge", alpha=alpha, l1_ratio=None,
            train=inner_train, test=inner_test,
            fit_and_predict=lambda a=alpha: _fit_predict(inner_train, inner_test, P0_FEATURES, "ridge", a),
        )
        scores.append((metric["qlike"], metric["log_mse"], -alpha, alpha))
    winner = min(scores)
    if not np.isfinite(winner[0]):
        raise RuntimeError("all inner Ridge candidates failed")
    return float(winner[-1])


def _fit_predict(train, test, features, model, alpha, l1_ratio=None):
    fitted = fit_model(train, features, model=model, alpha=alpha, l1_ratio=l1_ratio)
    return fitted, fitted.predict(test)


def run_cpcv(development: pd.DataFrame, logger: TrialLogger) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for fold in cpcv_folds(development):
        train, test = development.iloc[fold.train_index], development.iloc[fold.test_index]
        selected_alpha = inner_ridge_selection(train, fold.name, "cpcv", logger)
        _, baseline_forecast, baseline_metric = _trial(
            logger, trial_id=f"cpcv:{fold.name}:P0:selected={selected_alpha}",
            phase="cpcv_p0_outer", fold=fold.name, model="ridge", alpha=selected_alpha,
            l1_ratio=None, train=train, test=test,
            fit_and_predict=lambda: _fit_predict(train, test, P0_FEATURES, "ridge", selected_alpha),
        )
        rows.append({"fold": fold.name, "model": "P0", "alpha": selected_alpha,
                     "l1_ratio": np.nan, **baseline_metric})
        for alpha in P1_ALPHAS:
            for l1_ratio in P1_L1_RATIOS:
                _, _, metric = _trial(
                    logger,
                    trial_id=f"cpcv:{fold.name}:P1:a={alpha}:l1={l1_ratio}",
                    phase="cpcv_p1_outer", fold=fold.name, model="elastic_net",
                    alpha=alpha, l1_ratio=l1_ratio, train=train, test=test,
                    fit_and_predict=lambda a=alpha, l=l1_ratio: _fit_predict(
                        train, test, P1_FEATURES, "elastic_net", a, l
                    ),
                )
                rows.append({"fold": fold.name, "model": "P1", "alpha": alpha,
                             "l1_ratio": l1_ratio, **metric})
    metrics = pd.DataFrame(rows)
    p1 = metrics[metrics.model.eq("P1")].copy()
    p1["qlike_rank"] = p1.groupby("fold")["qlike"].rank(method="average")
    p1["log_mse_rank"] = p1.groupby("fold")["log_mse"].rank(method="average")
    summary = p1.groupby(["alpha", "l1_ratio"], as_index=False).agg(
        median_qlike_rank=("qlike_rank", "median"),
        median_log_mse_rank=("log_mse_rank", "median"),
        median_qlike=("qlike", "median"), median_log_mse=("log_mse", "median"),
        failed_folds=("qlike", lambda x: int((~np.isfinite(x)).sum())),
    )
    summary = summary.sort_values(
        ["median_qlike_rank", "median_log_mse_rank", "alpha", "l1_ratio"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    selected = summary.iloc[0]
    baseline = metrics[metrics.model.eq("P0")].set_index("fold")
    chosen_folds = p1[
        p1.alpha.eq(selected.alpha) & p1.l1_ratio.eq(selected.l1_ratio)
    ].set_index("fold")
    wins = int((chosen_folds["qlike"] < baseline["qlike"]).sum())
    plateau = plateau_pass(summary, float(selected.alpha), float(selected.l1_ratio))
    selected_successes = int(np.isfinite(chosen_folds["qlike"]).sum())
    all_outer_success = bool(
        metrics.groupby("model")["qlike"].apply(lambda x: np.isfinite(x).all()).all()
    )
    zero_failures = logger.failures == 0
    decision = {
        "alpha": float(selected.alpha), "l1_ratio": float(selected.l1_ratio),
        "qlike_wins_vs_fold_selected_p0": wins, "required_wins": 20,
        "plateau_pass": bool(plateau), "selected_successful_folds": selected_successes,
        "all_cpcv_outer_fits_successful": all_outer_success,
        "zero_logged_cpcv_failures": zero_failures,
        "cpcv_pass": bool(
            wins >= 20 and plateau and selected_successes == 28
            and all_outer_success and zero_failures
        ),
    }
    return metrics, {"summary": summary, "selection": decision}


def plateau_pass(summary: pd.DataFrame, alpha: float, l1_ratio: float, tolerance: float = 0.02) -> bool:
    row = summary[summary.alpha.eq(alpha) & summary.l1_ratio.eq(l1_ratio)]
    if len(row) != 1 or not np.isfinite(row.iloc[0].median_qlike):
        return False
    alpha_pos, l1_pos = P1_ALPHAS.index(alpha), P1_L1_RATIOS.index(l1_ratio)
    neighbor_keys = []
    for pos in (alpha_pos - 1, alpha_pos + 1):
        if 0 <= pos < len(P1_ALPHAS):
            neighbor_keys.append((P1_ALPHAS[pos], l1_ratio))
    for pos in (l1_pos - 1, l1_pos + 1):
        if 0 <= pos < len(P1_L1_RATIOS):
            neighbor_keys.append((alpha, P1_L1_RATIOS[pos]))
    selected_loss = float(row.iloc[0].median_qlike)
    values = []
    for a, l1 in neighbor_keys:
        neighbor = summary[summary.alpha.eq(a) & summary.l1_ratio.eq(l1)]
        if len(neighbor) != 1:
            return False
        values.append(float(neighbor.iloc[0].median_qlike))
    return bool(values and all(np.isfinite(x) and x <= selected_loss * (1.0 + tolerance) for x in values))


def expanding_month_folds(panel: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    development = panel[panel.split.eq("development")]
    validation = panel[panel.split.eq("walk_forward_validation")]
    folds = []
    months = validation["decision_time"].dt.tz_localize(None).dt.to_period("M")
    for month in sorted(months.unique()):
        test = validation[months.eq(month)].copy()
        first = test["decision_time"].min()
        candidates = panel[
            panel.split.isin(ALLOWED_SPLITS)
            & (panel["decision_time"] < first)
            & (panel["label_end_at"] < first)
        ].copy()
        test_dates = pd.DatetimeIndex(test["decision_time"].dt.normalize().unique()).sort_values()
        # The one-session post-test embargo cannot enter a forward-only fit.
        train = candidates
        if train.empty or test.empty:
            raise AssertionError("empty expanding fold")
        if not (train["label_end_at"] < first).all():
            raise AssertionError("expanding-fold purge failure")
        folds.append((str(month), train, test))
    if len(folds) != 18:
        raise AssertionError("validation must contain 18 monthly folds")
    return folds


def run_walk_forward(
    panel: pd.DataFrame,
    selected: dict,
    logger: TrialLogger,
) -> pd.DataFrame:
    predictions = []
    for month, train, test in expanding_month_folds(panel):
        p0_alpha = inner_ridge_selection(train, month, "walk_forward", logger)
        _, p0_forecast, _ = _trial(
            logger, trial_id=f"walk_forward:{month}:P0:selected={p0_alpha}",
            phase="walk_forward_p0_outer", fold=month, model="ridge", alpha=p0_alpha,
            l1_ratio=None, train=train, test=test,
            fit_and_predict=lambda: _fit_predict(train, test, P0_FEATURES, "ridge", p0_alpha),
        )
        _, p1_forecast, _ = _trial(
            logger, trial_id=f"walk_forward:{month}:P1:a={selected['alpha']}:l1={selected['l1_ratio']}",
            phase="walk_forward_p1_outer", fold=month, model="elastic_net",
            alpha=selected["alpha"], l1_ratio=selected["l1_ratio"], train=train, test=test,
            fit_and_predict=lambda: _fit_predict(
                train, test, P1_FEATURES, "elastic_net", selected["alpha"], selected["l1_ratio"]
            ),
        )
        if p0_forecast is None or p1_forecast is None:
            raise RuntimeError(f"walk-forward fit failed for {month}")
        cut_low, cut_high = train["p0_log_rv_60"].quantile([1 / 3, 2 / 3])
        regime = np.where(
            test["p0_log_rv_60"] <= cut_low, "low",
            np.where(test["p0_log_rv_60"] <= cut_high, "middle", "high"),
        )
        predictions.append(pd.DataFrame({
            "decision_time": test["decision_time"].to_numpy(),
            "label_end_at": test["label_end_at"].to_numpy(),
            "month": month, "regime": regime,
            "actual_variance": test["rv_fwd_120"].to_numpy(float),
            "actual_log_variance": test["target"].to_numpy(float),
            "p0_forecast": p0_forecast, "p1_forecast": p1_forecast,
            "p0_alpha": p0_alpha,
        }))
    return pd.concat(predictions, ignore_index=True)


def hac_dm_one_sided(loss_difference: np.ndarray, lag: int = 120) -> dict:
    """H1 is positive mean(P0 loss - P1 loss), Bartlett Newey-West."""
    values = np.asarray(loss_difference, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return {"statistic": np.nan, "p_value": np.nan, "lag": lag, "n": n}
    centered = values - values.mean()
    gamma0 = float(centered @ centered / n)
    long_run = gamma0
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / n)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    se = sqrt(max(long_run, 1e-30) / n)
    statistic = float(values.mean() / se)
    p_value = float(0.5 * (1.0 - erf(statistic / sqrt(2.0))))
    return {"statistic": statistic, "p_value": p_value, "lag": lag, "n": n}


def calibration(frame: pd.DataFrame, forecast_col: str, lag: int = 120) -> dict:
    y = frame["actual_log_variance"].to_numpy(float)
    x = np.column_stack([np.ones(len(frame)), np.log(frame[forecast_col].to_numpy(float))])
    inv = np.linalg.pinv(x.T @ x)
    beta = inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((2, 2), float)
    score = x * residual[:, None]
    n = len(frame)
    meat += score.T @ score
    for offset in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        cross = score[offset:].T @ score[:-offset]
        meat += weight * (cross + cross.T)
    covariance = inv @ meat @ inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    intercept_ok = bool(beta[0] - 1.96 * se[0] <= 0 <= beta[0] + 1.96 * se[0])
    slope_ok = bool(beta[1] - 1.96 * se[1] <= 1 <= beta[1] + 1.96 * se[1])
    return {
        "intercept": float(beta[0]), "intercept_se_hac": float(se[0]),
        "slope": float(beta[1]), "slope_se_hac": float(se[1]),
        "mean_actual_forecast_ratio": float(frame.actual_variance.mean() / frame[forecast_col].mean()),
        "failure": not (intercept_ok and slope_ok),
    }


def _loss_rows(frame: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for model, column in (("P0", "p0_forecast"), ("P1", "p1_forecast")):
        rows.append({
            "segment": label, "model": model, "rows": len(frame),
            "qlike": qlike(frame.actual_variance, frame[column]),
            "log_mse": log_mse(frame.actual_log_variance, frame[column]),
        })
    return rows


def evaluate_gate(predictions: pd.DataFrame, cpcv_decision: dict) -> dict:
    y = predictions.actual_variance.to_numpy(float)
    p0, p1 = predictions.p0_forecast.to_numpy(float), predictions.p1_forecast.to_numpy(float)
    p0_loss = y / p0 - np.log(y / p0) - 1.0
    p1_loss = y / p1 - np.log(y / p1) - 1.0
    aggregate_improvement = 1.0 - p1_loss.mean() / p0_loss.mean()
    periods = {
        "2023_H1": predictions[(predictions.decision_time >= "2023-01-01") & (predictions.decision_time < "2023-07-01")],
        "2023_H2": predictions[(predictions.decision_time >= "2023-07-01") & (predictions.decision_time < "2024-01-01")],
        "2024_H1": predictions[(predictions.decision_time >= "2024-01-01") & (predictions.decision_time < "2024-07-01")],
    }
    subperiod_lower = {
        name: qlike(frame.actual_variance, frame.p1_forecast)
        < qlike(frame.actual_variance, frame.p0_forecast)
        for name, frame in periods.items()
    }
    dm = hac_dm_one_sided(p0_loss - p1_loss, lag=120)
    calibration_groups = {"aggregate": predictions, **periods}
    calibration_rows = []
    failures = {"P0": 0, "P1": 0}
    for segment, frame in calibration_groups.items():
        for model, column in (("P0", "p0_forecast"), ("P1", "p1_forecast")):
            result = calibration(frame, column)
            failures[model] += int(result["failure"])
            calibration_rows.append({"segment": segment, "model": model, **result})
    monthly = predictions.assign(loss_improvement=p0_loss - p1_loss).groupby("month").loss_improvement.sum()
    total_improvement = float(monthly.sum())
    concentration = float(monthly.max() / total_improvement) if total_improvement > 0 else float("inf")
    checks = {
        "cpcv_selection": bool(cpcv_decision["cpcv_pass"]),
        "aggregate_qlike_improvement_at_least_2pct": bool(aggregate_improvement >= 0.02),
        "p1_lower_qlike_all_subperiods": bool(all(subperiod_lower.values())),
        "hac_dm_one_sided_p_below_0_05": bool(dm["p_value"] < 0.05),
        "all_forecasts_positive_finite": bool(
            np.isfinite(p0).all() and np.isfinite(p1).all() and (p0 > 0).all() and (p1 > 0).all()
        ),
        "p1_calibration_failures_no_more_than_p0": failures["P1"] <= failures["P0"],
        "max_month_share_at_most_35pct": bool(concentration <= 0.35),
    }
    return {
        "predictive_gate": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "aggregate_qlike_improvement": float(aggregate_improvement),
        "subperiod_lower_qlike": subperiod_lower, "hac_dm": dm,
        "calibration_failures": failures, "calibration_rows": calibration_rows,
        "max_month_improvement_share": concentration,
    }


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    try:
        con.register("artifact", frame)
        con.execute("COPY artifact TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    foundations = verify_frozen_foundations(args.dataset)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    logger = TrialLogger(
        args.output / "trials.csv", args.output / "root_compatible_trials.csv", run_id,
        GLOBAL_TRIALS,
    )
    panel = load_allowed_panel(args.dataset)
    development = panel[panel.split.eq("development")].reset_index(drop=True)

    cpcv_metrics, cpcv = run_cpcv(development, logger)
    cpcv_metrics.to_csv(args.output / "cpcv_fold_metrics.csv", index=False)
    cpcv["summary"].to_csv(args.output / "cpcv_candidate_summary.csv", index=False)
    (args.output / "selected_model.json").write_text(
        json.dumps(cpcv["selection"], indent=2), encoding="utf-8"
    )

    if not cpcv["selection"]["cpcv_pass"]:
        gate = {
            "predictive_gate": "FAIL", "stage": "CPCV_DEVELOPMENT",
            "reason": "P1 did not satisfy the frozen CPCV win/plateau/all-fold-success gate; walk-forward validation remained unopened.",
            "checks": {"cpcv_selection": False}, "selected_model": cpcv["selection"],
            "seed": SEED, **foundations, "authorized_development_rows": len(development),
            "walk_forward_rows_accessed": 0, "pseudo_holdout_accessed": False,
        }
        (args.output / "predictive_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(json.dumps(gate, indent=2))
        return

    predictions = run_walk_forward(panel, cpcv["selection"], logger)
    save_parquet(predictions, args.output / "walk_forward_predictions.parquet")

    metric_rows = _loss_rows(predictions, "aggregate")
    naive_time = predictions.decision_time.dt.tz_localize(None)
    period_masks = {
        "2023_H1": (naive_time >= "2023-01-01") & (naive_time < "2023-07-01"),
        "2023_H2": (naive_time >= "2023-07-01") & (naive_time < "2024-01-01"),
        "2024_H1": (naive_time >= "2024-01-01") & (naive_time < "2024-07-01"),
    }
    for name, mask in period_masks.items():
        metric_rows.extend(_loss_rows(predictions[mask], name))
    pd.DataFrame(metric_rows).to_csv(args.output / "subperiod_metrics.csv", index=False)
    regime_rows = []
    for regime, frame in predictions.groupby("regime", sort=True):
        regime_rows.extend(_loss_rows(frame, str(regime)))
    pd.DataFrame(regime_rows).to_csv(args.output / "regime_metrics.csv", index=False)

    monthly_rows = []
    for month, frame in predictions.groupby("month", sort=True):
        monthly_rows.extend(_loss_rows(frame, str(month)))
    pd.DataFrame(monthly_rows).to_csv(args.output / "monthly_metrics.csv", index=False)

    gate = evaluate_gate(predictions, cpcv["selection"])
    pd.DataFrame(gate.pop("calibration_rows")).to_csv(args.output / "calibration.csv", index=False)
    gate.update({
        "seed": SEED, **foundations,
        "authorized_rows": len(panel), "development_rows": len(development),
        "walk_forward_rows": len(predictions), "pseudo_holdout_accessed": False,
        "selected_model": cpcv["selection"],
    })
    (args.output / "predictive_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
