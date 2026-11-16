"""Live adapter for the stacked HMM + P&L + adverse-excursion models."""
from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


class LivePnlInferenceAdapter:
    def __init__(self, config: dict) -> None:
        root = Path(config["pnl_model_dir"])
        self.horizons = tuple(int(x) for x in config.get("paper_prediction_horizons", [5, 15, 30, 45, 60]))
        self.pnl = {h: joblib.load(root / "pnl_models" / f"horizon_{h}m" / "production_bundle.joblib") for h in self.horizons}
        self.mae = {h: joblib.load(root / "mae_models" / f"horizon_{h}m" / "production_bundle.joblib") for h in self.horizons}
        hmm_root = Path(config.get("hmm_model_dir", root / "hmm_horizon_warmup"))
        self.hmm_bundles = {h: joblib.load(hmm_root / f"hmm_horizon_{h}.joblib") for h in self.horizons}
        self.hmm_priors: dict[int, np.ndarray] = {}
        self.last_hmm_stamps: dict[int, datetime] = {}
        self.last_hmm_fields: dict[int, dict] = {}

    @staticmethod
    def _finite(value: object, default: float = math.nan) -> float:
        try:
            number = float(value)
            return number if math.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    def _hmm_features(self, live: dict, horizon: int, stamp: datetime) -> dict:
        bundle = self.hmm_bundles[horizon]
        values = np.asarray([[self._finite(live.get(name)) for name in bundle["features"]]], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            transformed = bundle["scaler"].transform(bundle["imputer"].transform(values))
        model = bundle["hmm"]
        log_emission = np.asarray(model._compute_log_likelihood(transformed)[0], dtype=float)
        likelihood = np.exp(log_emission - np.max(log_emission))
        previous_stamp = self.last_hmm_stamps.get(horizon)
        reset = previous_stamp is None or previous_stamp.date() != stamp.date() or (stamp-previous_stamp).total_seconds()!=60
        prior = np.asarray(model.startprob_ if reset else self.hmm_priors[horizon] @ model.transmat_, dtype=float)
        posterior = prior * likelihood
        posterior /= posterior.sum()
        self.hmm_priors[horizon] = posterior
        self.last_hmm_stamps[horizon] = stamp
        state = int(np.argmax(posterior))
        peak = float(np.max(log_emission))
        emission_loglik = peak + float(np.log(np.sum(prior * np.exp(log_emission - peak))))
        future = posterior @ model.transmat_
        return {
            "hmm_p0": float(posterior[0]), "hmm_p1": float(posterior[1]), "hmm_p2": float(posterior[2]),
            "hmm_state": state, "hmm_emission_loglik": emission_loglik,
            "hmm_in_distribution": float(emission_loglik >= float(bundle["drift_cutoff"])),
            "hmm_future_p0": float(future[0]), "hmm_future_p1": float(future[1]), "hmm_future_p2": float(future[2]),
            "hmm_persistence": float(model.transmat_[state, state]),
        }

    def score(self, spot_event: dict, iv_event: dict) -> dict[str, dict]:
        common_live = dict(iv_event.get("live_features", {}))
        current_spot = self._finite(spot_event.get("current_spot"))
        current_iv = self._finite(iv_event.get("current_iv"))
        stamp = datetime.fromisoformat(str(iv_event.get("exchange_minute")))
        output: dict[str, dict] = {}
        for horizon in self.horizons:
            sf = spot_event.get("predictions", {}).get(str(horizon), {})
            vf = iv_event.get("predictions", {}).get(str(horizon), {})
            if not sf or not vf: continue
            live = dict(common_live)
            hmm_live = iv_event.get("hmm_features", {}).get(str(horizon), {})
            if self.last_hmm_stamps.get(horizon) != stamp:
                self.last_hmm_fields[horizon] = self._hmm_features(hmm_live, horizon, stamp)
            live.update(self.last_hmm_fields[horizon])
            predicted_spot = self._finite(sf.get("predicted_spot"))
            predicted_return = self._finite(sf.get("predicted_return"))
            predicted_iv = self._finite(vf.get("predicted_iv"))
            predicted_iv_change = self._finite(vf.get("predicted_iv_change"))
            row = dict(live)
            row.update({
                "current_spot": current_spot, "current_iv": current_iv, "predicted_spot": predicted_spot,
                "predicted_return": predicted_return, "spot_predicted_move_points": predicted_spot-current_spot,
                "spot_predicted_abs_move_points": abs(predicted_spot-current_spot),
                "spot_predicted_direction": float(np.sign(predicted_spot-current_spot)),
                "predicted_iv": predicted_iv, "predicted_iv_change": predicted_iv_change,
                "predicted_move_to_credit": abs(predicted_spot-current_spot) / max(self._finite(row.get("straddle_value"), 1.0), 1.0),
                "predicted_iv_change_per_sqrt_minute": predicted_iv_change / math.sqrt(horizon),
                "forecast_short_vol_support": float(predicted_iv_change <= 0.00025), "horizon": float(horizon),
            })
            pnl_bundle, mae_bundle = self.pnl[horizon], self.mae[horizon]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pnl_x = pnl_bundle["imputer"].transform(np.asarray([[self._finite(row.get(n)) for n in pnl_bundle["features"]]], dtype=float))
                mae_x = mae_bundle["imputer"].transform(np.asarray([[self._finite(row.get(n)) for n in mae_bundle["features"]]], dtype=float))
            quantiles = sorted(float(pnl_bundle[k].predict(pnl_x)[0]) for k in ("q10_model", "q50_model", "q90_model"))
            output[str(horizon)] = {
                "expected_pnl": float(pnl_bundle["mean_model"].predict(pnl_x)[0]),
                "pnl_q10": quantiles[0], "pnl_q50": quantiles[1], "pnl_q90": quantiles[2],
                "profit_probability": float(pnl_bundle["profit_model"].predict_proba(pnl_x)[0, 1]),
                "mae_q50": max(0.0, float(mae_bundle["q50_model"].predict(mae_x)[0])),
                "mae_q90": max(0.0, float(mae_bundle["q90_model"].predict(mae_x)[0])),
                "hmm_state": int(row["hmm_state"]), "hmm_in_distribution": bool(row["hmm_in_distribution"]),
                "hmm_emission_loglik": float(row["hmm_emission_loglik"]),
            }
        return output
