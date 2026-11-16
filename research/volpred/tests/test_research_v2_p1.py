from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import csv

from research.research_v2_p1 import (
    P0_FEATURES,
    P1_FEATURES,
    TrialLogger,
    append_global_trial,
    cpcv_folds,
    fit_model,
    hac_dm_one_sided,
    plateau_pass,
    qlike,
    verify_frozen_foundations,
)


def synthetic_panel(days: int = 24, rows_per_day: int = 4) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=days, tz="Asia/Kolkata")
    rows = []
    for day_number, day in enumerate(dates):
        for minute in range(rows_per_day):
            decision = day + pd.Timedelta(hours=10, minutes=5 * minute)
            values = {name: np.sin(day_number + minute + index) for index, name in enumerate(P1_FEATURES)}
            target = -10 + 0.1 * values[P0_FEATURES[0]]
            rows.append({
                "decision_time": decision,
                "decision_date": day,
                "label_end_at": decision + pd.Timedelta(minutes=120),
                "split": "development",
                "rv_fwd_120": np.exp(target),
                "target": target,
                **values,
            })
    return pd.DataFrame(rows)


def test_registered_feature_contract_is_exact():
    assert P0_FEATURES == ["p0_log_rv_5", "p0_log_rv_15", "p0_log_rv_60"]
    assert len(P1_FEATURES) == 13
    assert len(set(P1_FEATURES)) == len(P1_FEATURES)


def test_cpcv_has_28_folds_and_enforces_purge_and_one_session_embargo():
    panel = synthetic_panel()
    folds = cpcv_folds(panel)
    assert len(folds) == 28
    for fold in folds:
        train, test = panel.iloc[fold.train_index], panel.iloc[fold.test_index]
        assert set(train.decision_time.dt.normalize()).isdisjoint(test.decision_time.dt.normalize())
        assert not train.decision_time.dt.normalize().isin(fold.embargo_dates).any()
        for start, _ in fold.test_blocks:
            assert not ((train.decision_time < start) & (train.label_end_at >= start)).any()


def test_qlike_is_zero_for_exact_positive_forecast_and_rejects_nonpositive():
    actual = np.array([1.0, 2.0, 3.0])
    assert qlike(actual, actual) == pytest.approx(0.0)
    assert np.isinf(qlike(actual, np.array([1.0, 0.0, 3.0])))


def test_training_only_smearing_model_is_deterministic_and_positive():
    panel = synthetic_panel(days=12)
    first = fit_model(panel, P1_FEATURES, model="elastic_net", alpha=0.01, l1_ratio=0.25)
    second = fit_model(panel, P1_FEATURES, model="elastic_net", alpha=0.01, l1_ratio=0.25)
    assert np.all(first.predict(panel) > 0)
    np.testing.assert_allclose(first.predict(panel), second.predict(panel), rtol=0, atol=0)


def test_hac_dm_direction_and_one_sided_probability():
    differential = np.linspace(0.1, 0.2, 500)
    result = hac_dm_one_sided(differential, lag=120)
    assert result["statistic"] > 0
    assert result["p_value"] < 0.05


def test_plateau_requires_every_adjacent_grid_cell_within_two_percent():
    rows = []
    for alpha in (0.0001, 0.001, 0.01, 0.1):
        for l1 in (0.0, 0.25, 0.5):
            rows.append({"alpha": alpha, "l1_ratio": l1, "median_qlike": 1.01})
    summary = pd.DataFrame(rows)
    summary.loc[summary.alpha.eq(0.001) & summary.l1_ratio.eq(0.25), "median_qlike"] = 1.0
    assert plateau_pass(summary, 0.001, 0.25)
    summary.loc[summary.alpha.eq(0.01) & summary.l1_ratio.eq(0.25), "median_qlike"] = 1.03
    assert not plateau_pass(summary, 0.001, 0.25)


def test_foundation_verification_fails_closed_on_tampered_split(tmp_path):
    split = tmp_path / "SPLITS.md"
    split.write_text("tampered", encoding="utf-8")
    split_hash = tmp_path / "SPLITS.sha256"
    split_hash.write_text("0" * 64 + " *SPLITS.md\n", encoding="ascii")
    with pytest.raises(PermissionError, match="SPLITS.md checksum mismatch"):
        verify_frozen_foundations(
            tmp_path / "dataset.parquet", splits_file=split,
            splits_hash_file=split_hash, manifest_path=tmp_path / "manifest.json",
            hashes_path=tmp_path / "SHA256SUMS",
        )


def test_trial_logger_appends_and_never_erases_failures(tmp_path):
    detailed = tmp_path / "trials.csv"
    compatible = tmp_path / "root.csv"
    global_registry = tmp_path / "global" / "TRIALS.csv"
    global_registry.parent.mkdir()
    global_registry.write_text(
        "trial_id,hypothesis,registered_at,status,structure,horizon_min,decision_time,dte_band,parameter_grid,outcome_opened,notes\n",
        encoding="utf-8",
    )
    first = TrialLogger(detailed, compatible, "run1", global_registry)
    first.add(trial_id="x", phase="cpcv", fold="f1", model="elastic_net",
              alpha=0.1, l1_ratio=0.5, status="failure", error="intentional")
    second = TrialLogger(detailed, compatible, "run2", global_registry)
    second.add(trial_id="y", phase="cpcv", fold="f2", model="ridge",
               alpha=1.0, l1_ratio="", status="success", error="")
    with detailed.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["failure", "success"]
    assert [row["trial_id"].split(":", 1)[0] for row in rows] == ["run1", "run2"]
    with global_registry.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_global_registry_lock_failure_is_fail_closed(tmp_path, monkeypatch):
    registry = tmp_path / "TRIALS.csv"
    registry.write_text(
        "trial_id,hypothesis,registered_at,status,structure,horizon_min,decision_time,dte_band,parameter_grid,outcome_opened,notes\n",
        encoding="utf-8",
    )
    import research.research_v2_p1 as module
    monkeypatch.setattr(module, "_lock_handle", lambda handle: (_ for _ in ()).throw(RuntimeError("locked")))
    with pytest.raises(RuntimeError, match="locked"):
        append_global_trial(registry, {"trial_id": "must-not-appear"})
    assert "must-not-appear" not in registry.read_text(encoding="utf-8")
