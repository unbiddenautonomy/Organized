from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import duckdb
import joblib
import numpy as np

import train_fixed_atm_horizons as base
import train_high_win_rate_models as high
import train_tail_risk_veto as tail


def main() -> None:
    output = Path("artifacts/2026_tail_veto_overlay")
    output.mkdir(parents=True, exist_ok=True)
    prepared = Path("artifacts/frozen_models_2026_holdout/prepared_2026_holdout.parquet")
    frame = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(prepared)]).fetchdf()
    frame, base_features = base.add_entry_features(frame)
    frame, tail_features = tail.add_tail_features(frame, base_features)
    # Score the complete chronological signal stream. Label/exact-exit
    # availability is future information, so it must not control which scores
    # enter the rolling veto history.
    frame = frame.sort_values("datetime").reset_index(drop=True)
    usable, target = high.horizon_data(frame, 30)
    usable = usable.sort_values("datetime")

    entry_artifact = joblib.load(
        "artifacts/frozen_models_2026_holdout/frozen_classifier_30m.joblib"
    )
    veto_artifact = joblib.load("artifacts/tail_risk_veto/tail_veto_30m.joblib")
    all_probability = entry_artifact["model"].predict_proba(
        frame[entry_artifact["features"]]
    )[:, 1]
    all_veto_probability = veto_artifact["model"].predict_proba(
        frame[veto_artifact["features"]]
    )[:, 1]

    history = deque(
        [float(value) for value in veto_artifact["rolling_score_history"]],
        maxlen=int(veto_artifact["rolling_score_window"]),
    )
    veto_thresholds = []
    vetoed = []
    for score in all_veto_probability:
        threshold = float(np.quantile(np.asarray(history, dtype=float), 0.80))
        veto_thresholds.append(threshold)
        vetoed.append(bool(score >= threshold))
        history.append(float(score))
    vetoed = np.asarray(vetoed, dtype=bool)
    exact_index = usable.index.to_numpy(dtype=int)
    probability = all_probability[exact_index]
    vetoed = vetoed[exact_index]
    entry_signal = probability >= float(entry_artifact["probability_threshold"])
    accepted = entry_signal & ~vetoed
    gated_score = np.where(accepted, probability, -np.inf)

    before, before_trades = base.non_overlapping_strategy(
        usable, usable[target].to_numpy(), probability, 30,
        float(entry_artifact["probability_threshold"]),
    )
    after, after_trades = base.non_overlapping_strategy(
        usable, usable[target].to_numpy(), gated_score, 30,
        float(entry_artifact["probability_threshold"]),
    )
    result = {
        "entry_model": "frozen 30-minute Extra Trees classifier",
        "tail_veto": veto_artifact["model_name"],
        "entry_threshold": entry_artifact["probability_threshold"],
        "rolling_veto_window": veto_artifact["rolling_score_window"],
        "rolling_veto_percentile": 0.80,
        "rolling_history_update_universe": "all chronological signals, before exact-exit label filtering",
        "candidate_signals": int(entry_signal.sum()),
        "vetoed_candidate_signals": int((entry_signal & vetoed).sum()),
        "before_veto": before,
        "after_veto": after,
        "research_status": "diagnostic on already-consumed 2026 labels; not a new holdout",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    before_trades.to_csv(output / "trades_before_veto.csv", index=False)
    after_trades.to_csv(output / "trades_after_veto.csv", index=False)
    (output / "REPORT.md").write_text(
        "\n".join(
            [
                "# 2026 Tail Veto Overlay Diagnostic",
                "",
                f"Before veto: {before['trades']} trades, {before['win_rate_after_slippage']:.2%} wins, {before['net_points_after_2pt_slippage']:.1f} net points, profit factor {before['profit_factor']:.2f}, max drawdown {before['maximum_drawdown_points']:.1f}.",
                "",
                f"After veto: {after['trades']} trades, {after['win_rate_after_slippage']:.2%} wins, {after['net_points_after_2pt_slippage']:.1f} net points, profit factor {after['profit_factor']:.2f}, max drawdown {after['maximum_drawdown_points']:.1f}.",
                "",
                f"The veto rejected {int((entry_signal & vetoed).sum())} of {int(entry_signal.sum())} candidate signals before non-overlap selection.",
                "",
                "This is a diagnostic on already-consumed 2026 labels and is not a fresh holdout result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
