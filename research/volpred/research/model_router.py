"""Config-driven routing and train-only calibration for volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


EPS = 1e-12


@dataclass
class CalibratedForecast:
    mean: float
    q90: float
    slope: float
    intercept: float
    residual_sd: float


def fit_log_calibration(train: pd.DataFrame, forecast_col: str, bucket: int) -> tuple[float,float,float,float]:
    z=train[train["time_bucket"].eq(bucket)].dropna(subset=[forecast_col,"actual_var"])
    if len(z)<200: z=train.dropna(subset=[forecast_col,"actual_var"])
    x=np.log(z[forecast_col].to_numpy(float).clip(min=EPS)); y=np.log(z["actual_var"].to_numpy(float).clip(min=EPS))
    X=np.column_stack([np.ones(len(x)),x]); beta=np.linalg.pinv(X.T@X)@X.T@y
    residual=y-X@beta; smear=float(np.mean(np.exp(residual))); q90=float(np.quantile(residual,.90))
    return float(beta[0]),float(beta[1]),smear,q90


def calibrated_forecast(row: pd.Series, train: pd.DataFrame, forecast_col: str) -> CalibratedForecast:
    a,b,smear,q90=fit_log_calibration(train,forecast_col,int(row["time_bucket"]))
    raw=a+b*np.log(max(float(row[forecast_col]),EPS)); residual=np.log(train.actual_var.clip(lower=EPS))-np.log(train[forecast_col].clip(lower=EPS))
    return CalibratedForecast(float(np.exp(raw)*smear),float(np.exp(raw+q90)),b,a,float(residual.std(ddof=1)))


def route_column(entry_time: pd.Timestamp, chain_valid: bool, arm: str) -> str:
    if entry_time.strftime("%H:%M") == "09:15": return "opening_extended"
    if chain_valid:
        return "Realized_GARCH_style" if arm == "style" else "Realized_GARCH_extended"
    return "HAR"


def dm_daily(frame: pd.DataFrame, a: str, b: str) -> dict:
    z=frame.copy()
    if "date" not in z: z["date"]=pd.to_datetime(z["datetime"]).dt.normalize()
    z["la"]=z.actual_var/z[a]-np.log(z.actual_var/z[a])-1; z["lb"]=z.actual_var/z[b]-np.log(z.actual_var/z[b])-1
    d=z.groupby("date").apply(lambda x: float((x.la-x.lb).mean()),include_groups=False).to_numpy(); n=len(d)
    se=float(np.std(d,ddof=1)/np.sqrt(n)); t=float(np.mean(d)/se) if se else 0.; return {"days":n,"mean_loss_a_minus_b":float(np.mean(d)),"t":t,"p_two_sided":float(2*stats.t.sf(abs(t),n-1))}

