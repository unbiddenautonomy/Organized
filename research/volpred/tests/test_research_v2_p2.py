from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from research.research_v2_p2 import (
    HAC_LAG,
    P0_ALPHAS,
    P0_FEATURES,
    P2_FEATURES,
    P2_GRID,
    SEED,
    cpcv_pairs,
    claim_pseudo_holdout_once,
    development_groups,
    fit_p2,
    fit_frozen_p2,
    frozen_p2_params,
    hac_dm,
    positive_forecast,
    pseudo_stability_verdict,
    purged_train_test,
    qlike,
    root_trial_rows,
    verify_approved_inputs,
)


def small_frame(days: int = 16) -> pd.DataFrame:
    sessions = pd.bdate_range("2021-01-04", periods=days, tz="Asia/Kolkata")
    rows = []
    for day_number, day in enumerate(sessions):
        for minute in (615, 620):
            decision = day + pd.Timedelta(minutes=minute)
            row = {
                "decision_time": decision,
                "decision_date": day,
                "label_end_at": decision + pd.Timedelta(minutes=120),
                "split": "development",
                "rv_fwd_120": 1e-5 + day_number * 1e-7,
                "target": np.log(1e-5 + day_number * 1e-7),
            }
            for number, feature in enumerate(P2_FEATURES):
                row[feature] = 0.01 * number + 0.001 * day_number
            rows.append(row)
    return pd.DataFrame(rows)


def test_registered_constants_and_grid_are_exact():
    assert SEED == 7319
    assert HAC_LAG == 120
    assert P0_ALPHAS == (0.01, 0.1, 1.0, 10.0, 100.0)
    expected = set(itertools.product((2, 3), (0.03, 0.05), (1.0, 10.0)))
    observed = {(x["max_depth"], x["learning_rate"], x["l2_regularization"]) for x in P2_GRID}
    assert observed == expected
    assert len(P2_GRID) == 8
    assert set(P0_FEATURES).issubset(P2_FEATURES)
    assert frozen_p2_params() == {
        "max_depth": 3, "learning_rate": 0.05, "l2_regularization": 1.0
    }


def test_frozen_split_manifest_and_dataset_hashes_are_approved():
    verified = verify_approved_inputs()
    assert verified["splits_sha256"] == verified["manifest_splits_sha256"]
    assert verified["dataset_sha256"] == verified["manifest_dataset_sha256"]
    assert verified["manifest_sha256"] == verified["inventory_manifest_sha256"]


def test_development_partition_has_eight_balanced_groups_and_28_pairs():
    frame = small_frame(19)
    groups, sessions = development_groups(frame)
    assert len(groups) == 8
    assert max(map(len, groups)) - min(map(len, groups)) <= 1
    assert sum(map(len, groups)) == len(sessions) == 19
    assert len(cpcv_pairs()) == 28
    assert len(set(cpcv_pairs())) == 28


def test_purge_and_one_session_embargo_are_enforced():
    frame = small_frame(10)
    sessions = frame["decision_date"].drop_duplicates().sort_values().to_numpy()
    tests = sessions[3:5]
    first_test = frame.loc[frame["decision_date"].eq(tests[0]), "decision_time"].min()
    preceding_index = frame.index[frame["decision_date"].eq(sessions[2])][-1]
    frame.loc[preceding_index, "label_end_at"] = first_test
    train, test, metadata = purged_train_test(frame, sessions, tests)
    assert preceding_index not in train.index
    assert set(test["decision_date"].unique()) == set(tests)
    assert not train["decision_date"].eq(sessions[5]).any()
    assert metadata["purged_or_embargoed_rows"] >= 3
    assert str(pd.Timestamp(sessions[5]).date()) in metadata["removed_dates"]
    before = train["decision_time"].lt(first_test)
    assert train.loc[before, "label_end_at"].lt(first_test).all()


def test_p2_configuration_has_no_test_driven_early_stopping():
    frame = small_frame(20)
    params = {"max_depth": 2, "learning_rate": 0.03, "l2_regularization": 10.0}
    model, smear = fit_p2(frame, params)
    assert model.early_stopping is False
    assert model.max_iter == 200
    assert model.random_state == 7319
    assert np.isfinite(smear)
    forecast, failures = positive_forecast(model.predict(frame[P2_FEATURES]), smear)
    assert failures == 0
    assert np.isfinite(forecast).all() and (forecast > 0).all()


def test_frozen_fit_uses_exact_feature_order_and_rejects_unauthorized_split():
    frame = small_frame(20)
    model, smear = fit_frozen_p2(frame)
    assert model.max_depth == 3
    assert model.learning_rate == 0.05
    assert model.l2_regularization == 1.0
    assert model.max_iter == 200 and model.early_stopping is False
    assert np.isfinite(smear)
    unauthorized = frame.copy()
    unauthorized.loc[unauthorized.index[-1], "split"] = "pseudo_holdout_previously_exposed"
    with pytest.raises(PermissionError, match="unauthorized split"):
        fit_frozen_p2(unauthorized)


def test_nonfinite_and_overflowing_raw_log_forecasts_count_before_replacement():
    values = np.array([0.0, np.inf, np.nan, 1_000.0, -1_000.0])
    forecast, failures = positive_forecast(values, smear=0.0)
    assert failures == 4
    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()


def test_root_trial_mapping_preserves_exact_schema_and_run_identity():
    trial = {
        "trial_id": "CPCV-01-P2-d2", "stage": "outer_cpcv", "fold_id": "CPCV-01",
        "model": "P2_HistGradientBoosting", "params": "{}", "status": "ok",
        "train_rows": 10, "test_rows": 2, "qlike": 1.0, "log_mse": 2.0,
        "calibration_failures": 0, "error": "",
    }
    first = root_trial_rows([trial], "run-a")[0]
    second = root_trial_rows([trial], "run-b")[0]
    assert first["trial_id"] != second["trial_id"]
    assert first["trial_id"].startswith("P2::run-a::")
    assert list(first) == [
        "trial_id", "hypothesis", "registered_at", "status", "structure",
        "horizon_min", "decision_time", "dte_band", "parameter_grid",
        "outcome_opened", "notes",
    ]


def test_qlike_and_hac_dm_direction():
    actual = np.full(500, 2.0)
    better = np.full(500, 2.05)
    worse = np.full(500, 4.0)
    difference = qlike(actual, better) - qlike(actual, worse)
    result = hac_dm(difference, lag=120)
    assert result["mean_difference"] < 0
    assert result["statistic"] < 0
    assert result["one_sided_pvalue"] < 0.05


def test_purge_rejects_unknown_test_dates():
    frame = small_frame(10)
    sessions = frame["decision_date"].drop_duplicates().sort_values().to_numpy()
    with pytest.raises(ValueError, match="test dates"):
        purged_train_test(frame, sessions, [pd.Timestamp("2030-01-01", tz="Asia/Kolkata")])


def test_atomic_pseudo_claim_rejects_existing_marker_and_preserves_consumed(tmp_path):
    marker = tmp_path / "PSEUDO_HOLDOUT_OPENED.json"
    first = claim_pseudo_holdout_once(marker)
    assert first["consumed"] is True
    persisted = json.loads(marker.read_text(encoding="utf-8"))
    assert persisted["status"] == "STARTED_AND_CONSUMED"
    with pytest.raises(PermissionError, match="already consumed"):
        claim_pseudo_holdout_once(marker)
    assert json.loads(marker.read_text(encoding="utf-8")) == persisted


def test_atomic_pseudo_claim_allows_only_one_concurrent_invocation(tmp_path):
    marker = tmp_path / "PSEUDO_HOLDOUT_OPENED.json"

    def attempt() -> str:
        try:
            claim_pseudo_holdout_once(marker)
            return "claimed"
        except PermissionError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ["claimed", "refused"]
    assert json.loads(marker.read_text(encoding="utf-8"))["consumed"] is True


def test_second_pseudo_claim_is_permanently_refused(tmp_path):
    marker = tmp_path / "PSEUDO_HOLDOUT_OPENED.json"
    claim_pseudo_holdout_once(marker)
    for _ in range(2):
        with pytest.raises(PermissionError):
            claim_pseudo_holdout_once(marker)
    assert marker.exists()


def test_pseudo_stability_verdict_is_exact_two_state_and_fail_closed():
    size = 1000
    actual = np.linspace(1.0, 2.0, size)
    stable = pd.DataFrame({
        "decision_time": pd.date_range("2024-07-01", periods=size, freq="min", tz="Asia/Kolkata"),
        "actual_variance": actual,
        "p0_forecast": actual * 1.5,
        "p2_forecast": actual * 1.01,
    })
    stable.attrs["p0_raw_calibration_failures"] = 1
    stable.attrs["p2_raw_calibration_failures"] = 0
    assert pseudo_stability_verdict(stable)["verdict"] == "research_stable"
    failed = stable.copy()
    failed["p2_forecast"] = failed["p0_forecast"]
    assert pseudo_stability_verdict(failed)["verdict"] == "research_failed"
    with pytest.raises(ValueError, match="missing/empty"):
        pseudo_stability_verdict(pd.DataFrame())
