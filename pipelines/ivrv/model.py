from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


IDENTITY_COLUMNS = ["row_id", "datetime", "Date", "dte_flag", "expiry", "entry_strike"]


def _numeric_matrix(frame: pd.DataFrame, features: list[str], strict: bool) -> pd.DataFrame:
    missing = [name for name in features if name not in frame.columns]
    if strict and missing:
        preview = ", ".join(missing[:12])
        suffix = " ..." if len(missing) > 12 else ""
        raise ValueError(f"Missing {len(missing)} required features: {preview}{suffix}")
    return (
        frame.reindex(columns=features)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def _future_from_log_ratio(current: np.ndarray, prediction: np.ndarray, clip: float) -> np.ndarray:
    return np.asarray(current, dtype=float) * np.exp(np.clip(np.asarray(prediction, dtype=float), -clip, clip))


def _inverse_log_variance(prediction: np.ndarray) -> np.ndarray:
    return np.sqrt(np.exp(np.clip(np.asarray(prediction, dtype=float), -20.0, 5.0)))


def _add_rv_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in list(result.columns):
        if name.startswith(("rv_trailing_", "rq_trailing_")):
            derived = f"log_{name}"
            if derived not in result:
                value = pd.to_numeric(result[name], errors="coerce")
                result[derived] = np.log(np.maximum(value, 1e-8))
        elif name.startswith("spot_return_"):
            derived = f"abs_{name}"
            if derived not in result:
                result[derived] = pd.to_numeric(result[name], errors="coerce").abs()
    if "entry_iv_weighted" in result and "rv_trailing_120m" in result:
        rv = pd.to_numeric(result["rv_trailing_120m"], errors="coerce")
        iv = pd.to_numeric(result["entry_iv_weighted"], errors="coerce")
        result["iv_minus_rv_120m"] = iv - rv
        result["iv_to_rv_120m_ratio"] = iv / rv.clip(lower=1e-8)
    return result


def _add_no120_event_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the engineered inputs used by the packaged no-120m event model."""
    result = frame.copy()

    def add(name: str, values: Any) -> None:
        result[name] = values.replace([np.inf, -np.inf], np.nan)

    iv = pd.to_numeric(result.get("entry_atm_iv_combined"), errors="coerce")
    for horizon in (5, 15, 30, 60):
        rv_name = f"rv_trailing_{horizon}m"
        if rv_name not in result:
            continue
        rv = pd.to_numeric(result[rv_name], errors="coerce")
        add(f"engineered_iv_minus_rv_{horizon}m", iv - rv)
        add(f"engineered_iv_to_rv_ratio_{horizon}m", iv / rv.where(rv.abs() > 1e-8))
        add(f"engineered_log_rv_{horizon}m", np.log(rv.clip(lower=1e-8)))
    add("engineered_log_entry_iv", np.log(iv.clip(lower=1e-8)))

    available = [h for h in (5, 15, 30, 60) if f"rv_trailing_{h}m" in result]
    for short, long in zip(available[:-1], available[1:]):
        add(
            f"engineered_rv_term_slope_{short}m_{long}m",
            pd.to_numeric(result[f"rv_trailing_{short}m"], errors="coerce")
            - pd.to_numeric(result[f"rv_trailing_{long}m"], errors="coerce"),
        )
    if "entry_straddle_price" in result and "spot_price" in result:
        spot = pd.to_numeric(result["spot_price"], errors="coerce")
        add(
            "engineered_straddle_to_spot",
            pd.to_numeric(result["entry_straddle_price"], errors="coerce")
            / spot.where(spot.abs() > 1e-8),
        )
    if "call_oi" in result and "put_oi" in result:
        call_oi = pd.to_numeric(result["call_oi"], errors="coerce")
        put_oi = pd.to_numeric(result["put_oi"], errors="coerce")
        add("engineered_put_call_oi_ratio", put_oi / call_oi.where(call_oi.abs() > 1e-8))
        total_oi = put_oi + call_oi
        add(
            "engineered_put_call_oi_imbalance",
            (put_oi - call_oi) / total_oi.where(total_oi.abs() > 1e-8),
        )
    return result


def _load_packaged_event_model(path: Path) -> dict[str, Any]:
    """Load a trusted joblib whose custom estimator lives beside the artifact."""
    component_path = path.with_name("model_components.py")
    if not component_path.is_file():
        raise FileNotFoundError(component_path)
    previous = sys.modules.get("model_components")
    spec = importlib.util.spec_from_file_location("model_components", component_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model components from {component_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_components"] = module
    try:
        spec.loader.exec_module(module)
        payload = joblib.load(path)
    finally:
        if previous is None:
            sys.modules.pop("model_components", None)
        else:
            sys.modules["model_components"] = previous
    return payload


def _check_alignment(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"{label} has {len(other)} rows; expected {len(reference)}")
    shared = [name for name in IDENTITY_COLUMNS if name in reference.columns and name in other.columns]
    for name in shared:
        left = reference[name].astype(str).fillna("<NA>").reset_index(drop=True)
        right = other[name].astype(str).fillna("<NA>").reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"{label} is not row-aligned on {name!r}")


@dataclass
class PipelineConfig:
    paths: dict[str, Path]
    prediction_clip_log_ratio: float
    convergence_probability_threshold: float
    strict_feature_contract: bool
    annual_trading_days: int
    minutes_per_trading_day: int
    horizon_minutes: int
    event_model_path: Path
    event_current_rv_feature: str
    event_gap_decimal: float
    event_profit_target_inr: float
    event_stop_loss_inr: float
    event_horizon_minutes: int
    rising_rv_model_path: Path
    rising_rv_probability_threshold: float


class FixedStrikeInferencePipeline:
    """Stack saved RV and fixed-strike IV models without refitting them."""

    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path or Path(__file__).with_name("config.yaml")).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings = raw["inference"]
        event = raw["rv_iv_sharp_decline_signal"]
        rising = raw["iv_gt_rv_120m_signal"]
        self.config = PipelineConfig(
            paths={name: (path.parent / value).resolve() for name, value in raw["models"].items()},
            prediction_clip_log_ratio=float(settings["prediction_clip_log_ratio"]),
            convergence_probability_threshold=float(settings["convergence_probability_threshold"]),
            strict_feature_contract=bool(settings["strict_feature_contract"]),
            annual_trading_days=int(settings["annual_trading_days"]),
            minutes_per_trading_day=int(settings["minutes_per_trading_day"]),
            horizon_minutes=int(settings["horizon_minutes"]),
            event_model_path=(path.parent / event["model_path"]).resolve(),
            event_current_rv_feature=str(event["current_rv_feature"]),
            event_gap_decimal=float(event["iv_rv_gap_volatility_points"])
            / float(event["volatility_points_per_decimal"]),
            event_profit_target_inr=float(event["profit_target_inr"]),
            event_stop_loss_inr=float(event["stop_loss_inr"]),
            event_horizon_minutes=int(event["horizon_minutes"]),
            rising_rv_model_path=(path.parent / rising["model_path"]).resolve(),
            rising_rv_probability_threshold=float(rising["probability_threshold"]),
        )
        absent = [str(value) for value in self.config.paths.values() if not value.exists()]
        if not self.config.event_model_path.exists():
            absent.append(str(self.config.event_model_path))
        if not self.config.rising_rv_model_path.exists():
            absent.append(str(self.config.rising_rv_model_path))
        if absent:
            raise FileNotFoundError("Missing model artifacts:\n" + "\n".join(absent))
        self.artifacts = {name: joblib.load(value) for name, value in self.config.paths.items()}
        event_payload = _load_packaged_event_model(self.config.event_model_path)
        if any("120m" in name.lower() for name in event_payload["features"]):
            raise RuntimeError("No-120m event model unexpectedly contains a 120-minute input feature")
        self.artifacts["rv_iv_sharp_decline_event"] = event_payload
        rising_payload = joblib.load(self.config.rising_rv_model_path)
        expected = json.loads(self.config.rising_rv_model_path.with_name("features.json").read_text())
        if rising_payload["features"] != expected or len(expected) != 19:
            raise ValueError("Rising-RV model feature contract differs from features.json")
        self.artifacts["iv_gt_rv_rising_120m"] = rising_payload

    def _matrix(self, frame: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
        return _numeric_matrix(frame, list(payload["features"]), self.config.strict_feature_contract)

    def _predict_rv(self, frame: pd.DataFrame) -> np.ndarray:
        payload = self.artifacts["rv_xgboost"]
        engineered = _add_rv_engineered_features(frame)
        raw = payload["model"].predict(self._matrix(engineered, payload))
        return _inverse_log_variance(raw) * float(payload["calibration"])

    def _predict_leg(self, side: str, frame: pd.DataFrame, output: pd.DataFrame) -> None:
        point = self.artifacts[f"{side}_hist_gradient_boosting"]
        current_name = f"entry_iv_{side}"
        if current_name not in frame:
            raise ValueError(f"Missing current-IV field {current_name!r}")
        current = pd.to_numeric(frame[current_name], errors="coerce").to_numpy(float)
        raw = point["estimator"].predict(self._matrix(frame, point))
        output[f"pred_{side}_log_iv_ratio_hgb"] = raw
        output[f"pred_{side}_iv_120m_hgb"] = _future_from_log_ratio(
            current, raw, self.config.prediction_clip_log_ratio
        )
        for quantile in (10, 50, 90):
            payload = self.artifacts[f"{side}_lightgbm_q{quantile}"]
            raw_q = payload["estimator"].predict(self._matrix(frame, payload))
            output[f"pred_{side}_log_iv_ratio_q{quantile}"] = raw_q
            output[f"pred_{side}_iv_120m_q{quantile}"] = _future_from_log_ratio(
                current, raw_q, self.config.prediction_clip_log_ratio
            )

    def _predict_combined(self, frame: pd.DataFrame, output: pd.DataFrame) -> None:
        direct_names = {
            "combined_error_correction_ridge": "ridge",
            "combined_monotonic_xgboost": "xgboost",
            "combined_direct_lightgbm": "lightgbm",
        }
        for artifact_name, short_name in direct_names.items():
            payload = self.artifacts[artifact_name]
            output[f"pred_combined_iv_delta_{short_name}"] = payload["model"].predict(
                self._matrix(frame, payload)
            )

        ensemble = self.artifacts["combined_three_model_ensemble"]
        matrix = self._matrix(frame, ensemble)
        ensemble_delta = np.zeros(len(frame), dtype=float)
        for name, model in ensemble["models"].items():
            ensemble_delta += float(ensemble["weights"][name]) * model.predict(matrix)
        output["pred_combined_iv_delta_ensemble"] = ensemble_delta

        classifier = self.artifacts["combined_convergence_classifier"]
        probability = classifier["model"].predict_proba(self._matrix(frame, classifier))[:, 1]
        output["pred_convergence_probability_classifier"] = probability
        output["pred_converged_classifier"] = probability >= self.config.convergence_probability_threshold

        quantiles = self.artifacts["combined_quantile_lightgbm"]
        quantile_matrix = self._matrix(frame, quantiles)
        for key, model in quantiles["models"].items():
            label = int(round(float(key) * 100))
            output[f"pred_combined_iv_delta_q{label}"] = model.predict(quantile_matrix)

        ngboost = self.artifacts["combined_ngboost"]
        ng_matrix = self._matrix(frame, ngboost)
        transformed = ngboost["imputer"].transform(ng_matrix)
        standardized_mean = ngboost["model"].predict(transformed)
        output["pred_combined_iv_delta_ngboost_mean"] = (
            float(ngboost["target_center"]) + float(ngboost["target_scale"]) * standardized_mean
        )
        try:
            distribution = ngboost["model"].pred_dist(transformed)
            scale = np.asarray(distribution.params["scale"], dtype=float)
            output["pred_combined_iv_delta_ngboost_scale"] = abs(float(ngboost["target_scale"])) * scale
        except (AttributeError, KeyError, TypeError):
            output["pred_combined_iv_delta_ngboost_scale"] = np.nan

    def _predict_sharp_decline_event(self, frame: pd.DataFrame, output: pd.DataFrame) -> None:
        payload = self.artifacts["rv_iv_sharp_decline_event"]
        engineered = _add_no120_event_features(frame)
        components = payload["model"].predict(self._matrix(engineered, payload))
        predicted_rv_change = np.asarray(components[:, 0], dtype=float)
        predicted_iv_change = np.asarray(components[:, 1], dtype=float)
        target = payload["target_definition"]
        rv_margin = -float(target["minimum_rv_drop"]) - predicted_rv_change
        relative_margin = (
            predicted_iv_change
            - predicted_rv_change
            - float(target["minimum_rv_drop_beyond_iv"])
        )
        score = np.minimum(rv_margin, relative_margin)
        predicts_sharp_decline = score >= float(payload["threshold"])

        rv_name = self.config.event_current_rv_feature
        if rv_name not in engineered:
            raise ValueError(f"Event gate requires current-RV field {rv_name!r}")
        current_rv = pd.to_numeric(engineered[rv_name], errors="coerce").to_numpy(float)
        current_iv = pd.to_numeric(
            engineered["entry_atm_iv_combined"], errors="coerce"
        ).to_numpy(float)
        gap = current_iv - current_rv
        gate = np.isfinite(current_iv) & np.isfinite(current_rv) & (
            gap <= self.config.event_gap_decimal
        )

        output["pred_no120_rv_change_120m"] = predicted_rv_change
        output["pred_no120_iv_change_120m"] = predicted_iv_change
        output["pred_no120_event_score"] = score
        output["pred_no120_sharp_rv_decline"] = predicts_sharp_decline
        output["pred_no120_gate_current_rv"] = current_rv
        output["pred_no120_gate_iv_minus_rv"] = gap
        output["pred_no120_gate_max_gap"] = self.config.event_gap_decimal
        output["pred_no120_gate_iv_near_or_below_rv"] = gate
        output["pred_short_straddle_profit_target_inr"] = self.config.event_profit_target_inr
        output["pred_short_straddle_stop_loss_inr"] = self.config.event_stop_loss_inr
        output["pred_short_straddle_horizon_minutes"] = self.config.event_horizon_minutes

    def _predict_rising_rv_convergence(self, frame: pd.DataFrame, output: pd.DataFrame) -> None:
        """Score the fixed entry-strike, rising-RV research model independently."""
        payload = self.artifacts["iv_gt_rv_rising_120m"]
        required = ["entry_iv_call", "entry_iv_put", "rv_sample_30m", "rv_sample_60m",
                    "rv_sample_120m", "spot_price", "entry_strike", "days_to_expiry",
                    "delta_call", "delta_put", "gamma_call", "gamma_put", "vega_call",
                    "vega_put", "call_oi", "put_oi", "future_open", "datetime"]
        missing = [name for name in required if name not in frame]
        if missing:
            output["pred_rising_rv_current_iv_points"] = np.nan
            output["pred_rising_rv_current_rv_points"] = np.nan
            output["pred_rising_rv_eligible"] = False
            output["pred_rising_rv_convergence_probability"] = np.nan
            output["pred_rising_rv_convergence_signal"] = False
            return
        call = pd.to_numeric(frame["entry_iv_call"], errors="coerce")
        put = pd.to_numeric(frame["entry_iv_put"], errors="coerce")
        spot = pd.to_numeric(frame["spot_price"], errors="coerce")
        strike = pd.to_numeric(frame["entry_strike"], errors="coerce")
        stamp = pd.to_datetime(frame["datetime"], errors="coerce")
        current_iv = 50.0 * (call + put)
        current_rv = pd.to_numeric(frame["rv_sample_120m"], errors="coerce")
        features = pd.DataFrame(index=frame.index)
        features["current_iv"] = current_iv
        features["rv_30"] = pd.to_numeric(frame["rv_sample_30m"], errors="coerce")
        features["rv_60"] = pd.to_numeric(frame["rv_sample_60m"], errors="coerce")
        features["iv_skew"] = 100.0 * (put - call)
        features["spot_price"] = spot
        features["strike"] = strike
        features["moneyness"] = (spot - strike) / spot
        features["days_to_expiry"] = pd.to_numeric(frame["days_to_expiry"], errors="coerce")
        features["minute_of_day"] = stamp.dt.hour * 60 + stamp.dt.minute
        features["weekday"] = stamp.dt.weekday
        for name in payload["features"]:
            if name not in features:
                features[name] = pd.to_numeric(frame[name], errors="coerce")
        matrix = self._matrix(features, payload)
        valid = (np.isfinite(current_iv) & np.isfinite(current_rv) &
                 (current_iv > current_rv) & np.isfinite(matrix.to_numpy(float)).all(axis=1))
        probability = np.full(len(frame), np.nan)
        if valid.any():
            probability[valid] = payload["model"].predict_proba(matrix.loc[valid])[:, 1]
        output["pred_rising_rv_current_iv_points"] = current_iv.to_numpy(float)
        output["pred_rising_rv_current_rv_points"] = current_rv.to_numpy(float)
        output["pred_rising_rv_eligible"] = valid
        output["pred_rising_rv_convergence_probability"] = probability
        output["pred_rising_rv_convergence_signal"] = valid & (probability >= self.config.rising_rv_probability_threshold)

    @staticmethod
    def _converged(entry_iv: np.ndarray, future_iv: np.ndarray, rv: np.ndarray) -> np.ndarray:
        return np.abs(future_iv - rv) < np.abs(entry_iv - rv)

    def predict(
        self,
        combined_features: pd.DataFrame,
        leg_features: pd.DataFrame | None = None,
        rv_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return all component forecasts and convergence diagnostics.

        Frames must be row-aligned. A single wide feature frame can be supplied
        as ``combined_features`` when it contains all three saved schemas.
        """
        combined = combined_features.reset_index(drop=True).copy()
        legs = (leg_features if leg_features is not None else combined_features).reset_index(drop=True).copy()
        rv_frame = (rv_features if rv_features is not None else combined_features).reset_index(drop=True).copy()
        _check_alignment(combined, legs, "leg_features")
        _check_alignment(combined, rv_frame, "rv_features")

        output = pd.DataFrame(index=combined.index)
        for name in IDENTITY_COLUMNS:
            if name in combined:
                output[name] = combined[name]

        rv_pred = self._predict_rv(rv_frame)
        output["pred_rv_120m_xgboost"] = rv_pred

        if "entry_atm_iv_combined" not in combined:
            raise ValueError("combined_features must contain 'entry_atm_iv_combined'")
        entry_combined = pd.to_numeric(combined["entry_atm_iv_combined"], errors="coerce").to_numpy(float)
        combined["rv_pred_120m"] = rv_pred
        combined["entry_gap_to_rv_pred_combined"] = entry_combined - rv_pred
        output["entry_atm_iv_combined"] = entry_combined
        output["entry_gap_to_pred_rv"] = np.abs(entry_combined - rv_pred)

        self._predict_combined(combined, output)
        self._predict_leg("call", legs, output)
        self._predict_leg("put", legs, output)

        event_frame = combined.copy()
        for source in (legs, rv_frame):
            missing = [name for name in source.columns if name not in event_frame]
            if missing:
                event_frame = pd.concat(
                    [event_frame, source.loc[:, missing].reset_index(drop=True)], axis=1
                )
        self._predict_sharp_decline_event(event_frame, output)
        self._predict_rising_rv_convergence(event_frame, output)

        delta_columns = [
            "pred_combined_iv_delta_ridge",
            "pred_combined_iv_delta_xgboost",
            "pred_combined_iv_delta_lightgbm",
            "pred_combined_iv_delta_ensemble",
            "pred_combined_iv_delta_ngboost_mean",
            "pred_combined_iv_delta_q10",
            "pred_combined_iv_delta_q50",
            "pred_combined_iv_delta_q90",
        ]
        for delta_name in delta_columns:
            future_name = delta_name.replace("_delta_", "_120m_")
            output[future_name] = entry_combined + output[delta_name].to_numpy(float)

        output["pred_combined_iv_120m_from_legs_hgb"] = (
            output["pred_call_iv_120m_hgb"] + output["pred_put_iv_120m_hgb"]
        ) / 2.0
        for quantile in (10, 50, 90):
            output[f"pred_combined_iv_120m_from_legs_q{quantile}"] = (
                output[f"pred_call_iv_120m_q{quantile}"] + output[f"pred_put_iv_120m_q{quantile}"]
            ) / 2.0

        forecast_map = {
            "direct_lightgbm": "pred_combined_iv_120m_lightgbm",
            "combined_ensemble": "pred_combined_iv_120m_ensemble",
            "ngboost_mean": "pred_combined_iv_120m_ngboost_mean",
            "leg_hgb_average": "pred_combined_iv_120m_from_legs_hgb",
            "combined_q50": "pred_combined_iv_120m_q50",
            "leg_q50_average": "pred_combined_iv_120m_from_legs_q50",
        }
        for label, column in forecast_map.items():
            future = output[column].to_numpy(float)
            output[f"pred_exit_gap_to_rv_{label}"] = np.abs(future - rv_pred)
            output[f"pred_gap_improvement_{label}"] = output["entry_gap_to_pred_rv"] - output[f"pred_exit_gap_to_rv_{label}"]
            output[f"pred_converged_{label}"] = self._converged(entry_combined, future, rv_pred)

        current_rv_name = self.config.event_current_rv_feature
        if current_rv_name not in event_frame:
            raise ValueError(f"Strategy rules require current-RV field {current_rv_name!r}")
        current_rv = pd.to_numeric(event_frame[current_rv_name], errors="coerce").to_numpy(float)
        predicted_iv = output["pred_combined_iv_120m_lightgbm"].to_numpy(float)
        predicted_iv_declines = predicted_iv < entry_combined
        predicted_iv_moves_toward_rv = output[
            "pred_converged_direct_lightgbm"
        ].to_numpy(bool)
        finite_current = np.isfinite(entry_combined) & np.isfinite(current_rv)
        near_gap = self.config.event_gap_decimal
        rule_iv_above = (
            finite_current
            & (entry_combined > current_rv)
            & predicted_iv_declines
            & predicted_iv_moves_toward_rv
        )
        rule_near = finite_current & (np.abs(entry_combined - current_rv) <= near_gap)
        rule_iv_below_fast_rv = (
            finite_current
            & (entry_combined < current_rv)
            & output["pred_no120_sharp_rv_decline"].to_numpy(bool)
        )
        output["pred_strategy_rule_iv_above_rv_iv_declines_to_rv"] = rule_iv_above
        output["pred_strategy_rule_iv_rv_within_half_point"] = rule_near
        output["pred_strategy_rule_iv_below_rv_fast_rv_decline"] = rule_iv_below_fast_rv
        output["pred_short_straddle_120m_signal"] = (
            rule_iv_above | rule_near | rule_iv_below_fast_rv
        )

        lower = output["pred_combined_iv_120m_q10"].to_numpy(float)
        upper = output["pred_combined_iv_120m_q90"].to_numpy(float)
        output["pred_combined_iv_interval_width"] = upper - lower
        entry_gap = output["entry_gap_to_pred_rv"].to_numpy(float)
        output["pred_converged_entire_combined_q10_q90_interval"] = (
            (np.abs(lower - rv_pred) < entry_gap) & (np.abs(upper - rv_pred) < entry_gap)
        )

        vote_columns = [
            "pred_converged_direct_lightgbm",
            "pred_converged_leg_hgb_average",
            "pred_converged_classifier",
        ]
        votes = output[vote_columns].astype(float)
        output["pred_convergence_vote_count"] = votes.sum(axis=1).astype(int)
        output["pred_convergence_vote_fraction"] = votes.mean(axis=1)
        output["pred_convergence_all_three_agree"] = votes.nunique(axis=1).eq(1)

        sigma = rv_pred * np.sqrt(
            self.config.horizon_minutes
            / (self.config.annual_trading_days * self.config.minutes_per_trading_day)
        )
        output["pred_sigma_120"] = sigma
        spot_column = "spot_price" if "spot_price" in rv_frame else "entry_spot_price"
        if spot_column in rv_frame:
            spot = pd.to_numeric(rv_frame[spot_column], errors="coerce").to_numpy(float)
            for label, z_score in (("1sigma", 1.0), ("90pct", 1.645)):
                output[f"pred_spot_range_{label}_low"] = spot * np.exp(-z_score * sigma)
                output[f"pred_spot_range_{label}_high"] = spot * np.exp(z_score * sigma)

        return output


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the saved fixed-strike IV/RV inference stack")
    parser.add_argument("--combined-features", required=True, help="CSV containing combined-IV model features")
    parser.add_argument("--leg-features", help="CSV containing call/put model features; defaults to combined CSV")
    parser.add_argument("--rv-features", help="CSV containing RV model features; defaults to combined CSV")
    parser.add_argument("--output", required=True, help="Destination prediction CSV")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()

    combined = _read_csv(args.combined_features)
    legs = _read_csv(args.leg_features) if args.leg_features else None
    rv = _read_csv(args.rv_features) if args.rv_features else None
    pipeline = FixedStrikeInferencePipeline(args.config)
    predictions = pipeline.predict(combined, legs, rv)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    print(f"Wrote {len(predictions):,} inference rows to {destination}")


if __name__ == "__main__":
    main()

