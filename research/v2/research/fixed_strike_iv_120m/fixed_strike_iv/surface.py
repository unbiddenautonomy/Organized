from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_KEYS = ["datetime", "dte_flag", "expiry"]


def _quadratic_fit(x: np.ndarray, y: np.ndarray, valid: np.ndarray, minimum_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Batched, scaled fit of IV = a + b*x + c*x^2."""
    counts = valid.sum(axis=1)
    scale = np.max(np.where(valid, np.abs(x), 0.0), axis=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = x / scale[:, None]
    design = np.stack((np.ones_like(z), z, z * z), axis=2)
    weights = valid.astype(float)
    xtx = np.einsum("gpi,gp,gpj->gij", design, weights, design)
    xty = np.einsum("gpi,gp,gp->gi", design, weights, np.where(valid, y, 0.0))
    eligible = counts >= minimum_points
    scaled = np.full((len(x), 3), np.nan)
    if eligible.any():
        try:
            scaled[eligible] = np.linalg.solve(xtx[eligible], xty[eligible][..., None])[..., 0]
        except np.linalg.LinAlgError:
            for row in np.flatnonzero(eligible):
                scaled[row] = np.linalg.lstsq(design[row, valid[row]], y[row, valid[row]], rcond=None)[0]
    fitted = np.einsum("gpi,gi->gp", design, np.nan_to_num(scaled))
    squared = np.where(valid, (y - fitted) ** 2, 0.0)
    rmse = np.full(len(x), np.nan)
    rmse[eligible] = np.sqrt(squared[eligible].sum(axis=1) / counts[eligible])
    coefs = scaled.copy()
    coefs[:, 1] /= scale
    coefs[:, 2] /= scale * scale
    return coefs, rmse


def make_entry_rows(day: pd.DataFrame, minimum_points: int, forward_column: str) -> pd.DataFrame:
    frame = day.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    frame = frame.sort_values(GROUP_KEYS + ["strike"])
    frame["_size"] = frame.groupby(GROUP_KEYS, observed=True)["strike"].transform("size")
    frame = frame[frame["_size"] == 5].copy()
    if frame.empty:
        return frame
    frame["_group"] = frame.groupby(GROUP_KEYS, sort=False, observed=True).ngroup()
    frame = frame.sort_values(["_group", "strike"])
    groups = int(frame["_group"].max()) + 1
    strike = frame["strike"].to_numpy(float).reshape(groups, 5)
    forward = frame[forward_column].to_numpy(float).reshape(groups, 5)[:, 0]
    safe_forward = np.where(forward > 0, forward, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(strike / safe_forward[:, None])

    surface = {}
    for side in ("call", "put"):
        iv = frame[f"iv_{side}"].to_numpy(float).reshape(groups, 5)
        converged = frame[f"iv_{side}_converged"].astype(str).str.lower().eq("true").to_numpy().reshape(groups, 5)
        valid = np.isfinite(iv) & np.isfinite(x) & (iv > 0) & converged
        coefs, rmse = _quadratic_fit(x, iv, valid, minimum_points)
        surface[f"surface_atm_iv_{side}"] = coefs[:, 0]
        surface[f"surface_skew_{side}"] = coefs[:, 1]
        surface[f"surface_smile_{side}"] = 2.0 * coefs[:, 2]
        surface[f"surface_rmse_{side}"] = rmse
        surface[f"surface_points_{side}"] = valid.sum(axis=1)

    target_atm = frame["atm_strike"].to_numpy(float).reshape(groups, 5)[:, 0]
    atm_position = np.abs(strike - target_atm[:, None]).argmin(axis=1)
    flat_position = np.arange(groups) * 5 + atm_position
    entry = frame.iloc[flat_position].copy().reset_index(drop=True)
    entry["entry_atm_exact"] = entry["strike"].to_numpy(float) == target_atm
    entry["entry_strike_distance_from_atm"] = entry["strike"].to_numpy(float) - target_atm
    for name, values in surface.items():
        entry[name] = values

    rename = {
        "strike": "entry_strike", "spot_price": "entry_spot_price", "future_open": "entry_future_price",
        "Future_oi": "entry_future_oi", "call_price": "entry_call_price", "put_price": "entry_put_price",
        "call_oi": "entry_call_oi", "put_oi": "entry_put_oi", "iv_call": "entry_iv_call",
        "iv_put": "entry_iv_put", "iv_call_converged": "entry_iv_call_converged",
        "iv_put_converged": "entry_iv_put_converged", "delta_call": "entry_delta_call",
        "delta_put": "entry_delta_put", "vega_call": "entry_vega_call", "vega_put": "entry_vega_put",
        "gamma_call": "entry_gamma_call", "gamma_put": "entry_gamma_put", "theta_call": "entry_theta_call",
        "theta_put": "entry_theta_put",
    }
    entry = entry.rename(columns=rename).drop(columns=["_size", "_group"], errors="ignore")
    entry["entry_straddle_price"] = entry["entry_call_price"] + entry["entry_put_price"]
    safe_entry_forward = entry["entry_future_price"].where(entry["entry_future_price"] > 0)
    entry["entry_log_moneyness"] = np.log(entry["entry_strike"] / safe_entry_forward)
    return entry

