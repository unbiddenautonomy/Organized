from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import chi2, norm, rankdata, t as student_t


ROOT = Path(__file__).resolve().parents[1]
ATM_FILE = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm.csv"
SPOT_FILE = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
OUT = ROOT / "outputs" / "atm_iv_rv_analysis"
HORIZONS = (5, 15, 30, 60)
TRAIN_END = pd.Timestamp("2022-12-31 23:59:59")
EVAL_START = pd.Timestamp("2023-01-01")
EVAL_END = pd.Timestamp("2023-12-31 23:59:59")
HOLDOUT_START = pd.Timestamp("2024-01-01")
MIN_PRICE = 5.0
EPS = 1e-12


def markdown_table(df: pd.DataFrame, decimals: int = 5) -> str:
    view = df.copy()
    for c in view.select_dtypes(include=[np.number]).columns:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    headers = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.itertuples(index=False, name=None)]
    widths = [max(len(headers[j]), *(len(row[j]) for row in rows)) for j in range(len(headers))]
    line1 = "| " + " | ".join(headers[j].ljust(widths[j]) for j in range(len(headers))) + " |"
    line2 = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |" for row in rows]
    return "\n".join([line1, line2] + body)


def load_inputs():
    con = duckdb.connect()
    atm_path = ATM_FILE.as_posix().replace("'", "''")
    source_atm_rows = con.execute(f"SELECT count(*) FROM read_csv('{atm_path}', header=true, sample_size=100000)").fetchone()[0]
    atm = con.execute(f"""
        SELECT datetime, expiry, strike, days_to_expiry, spot_price,
               call_price, put_price, call_oi, put_oi,
               iv_call, iv_put, iv_call_converged, iv_put_converged
        FROM read_csv('{atm_path}', header=true, sample_size=100000)
        QUALIFY row_number() OVER (PARTITION BY datetime ORDER BY expiry, days_to_expiry) = 1
        ORDER BY datetime
    """).df()
    con.close()
    spot = pd.read_csv(SPOT_FILE, parse_dates=["datetime", "Date"])
    atm["datetime"] = pd.to_datetime(atm["datetime"])
    atm["expiry"] = pd.to_datetime(atm["expiry"])
    return atm, spot, source_atm_rows


def black76_price(forward, strike, maturity, sigma, call):
    sqrt_t = np.sqrt(np.maximum(maturity, 1e-16))
    vt = sigma * sqrt_t
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * maturity) / vt
    d2 = d1 - vt
    c = forward * ndtr(d1) - strike * ndtr(d2)
    p = strike * ndtr(-d2) - forward * ndtr(-d1)
    price = np.where(call, c, p)
    vega = forward * np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * sqrt_t
    return price, vega


def implied_vol(forward, strike, maturity, price, call, initial):
    arrays = np.broadcast_arrays(forward, strike, maturity, price, initial)
    forward, strike, maturity, price, initial = [np.asarray(x, float) for x in arrays]
    intrinsic = np.where(call, np.maximum(forward - strike, 0), np.maximum(strike - forward, 0))
    upper = np.where(call, forward, strike)
    valid = (
        np.isfinite(forward) & np.isfinite(strike) & np.isfinite(maturity) & np.isfinite(price)
        & (forward > 0) & (strike > 0) & (maturity > 0)
        & (price >= intrinsic - 0.10) & (price < upper)
    )
    sigma = np.clip(np.where(np.isfinite(initial), initial, 0.25), 0.005, 4.0)
    for _ in range(30):
        model, vega = black76_price(forward, strike, maturity, sigma, call)
        step = np.divide(model - price, vega, out=np.zeros_like(sigma), where=vega > 1e-9)
        sigma = np.clip(sigma - step, 0.001, 5.0)
    model, _ = black76_price(forward, strike, maturity, sigma, call)
    ok = valid & (np.abs(model - price) <= np.maximum(0.10, price * 2e-4))
    return np.where(ok, sigma, np.nan), ok


def recalculate_atm_iv(atm: pd.DataFrame):
    k = atm["strike"].to_numpy(float)
    call_price = atm["call_price"].to_numpy(float)
    put_price = atm["put_price"].to_numpy(float)
    forward = k + call_price - put_price  # parity forward, discount-factor-one approximation
    maturity = atm["days_to_expiry"].to_numpy(float) / 365.0
    call_iv, call_ok = implied_vol(
        forward, k, maturity, call_price, np.ones(len(atm), bool), atm["iv_call"].to_numpy(float)
    )
    put_iv, put_ok = implied_vol(
        forward, k, maturity, put_price, np.zeros(len(atm), bool), atm["iv_put"].to_numpy(float)
    )
    fixed_iv = 0.5 * (call_iv + put_iv)
    spot = atm["spot_price"].to_numpy(float)
    basis = np.log(forward / spot)
    quality_steps = {
        "total_atm_rows": np.ones(len(atm), bool),
        "positive_maturity": maturity > 0,
        "prices_at_least_5": (call_price >= MIN_PRICE) & (put_price >= MIN_PRICE),
        "positive_call_put_oi": (atm["call_oi"].to_numpy(float) > 0) & (atm["put_oi"].to_numpy(float) > 0),
        "both_fixed_ivs_solved": call_ok & put_ok,
        "forward_within_5pct_of_spot": np.isfinite(basis) & (np.abs(basis) <= 0.05),
        "fixed_iv_between_3_and_300pct": np.isfinite(fixed_iv) & (fixed_iv >= 0.03) & (fixed_iv <= 3.0),
    }
    quality = np.ones(len(atm), bool)
    attrition = []
    for name, condition in quality_steps.items():
        quality &= condition
        attrition.append({"filter": name, "rows_remaining": int(quality.sum()), "share_remaining": float(quality.mean())})
    out = pd.DataFrame({
        "datetime": atm["datetime"], "expiry": atm["expiry"], "strike": k, "spot_atm_row": spot,
        "days_to_expiry": atm["days_to_expiry"].to_numpy(float), "forward": forward,
        "forward_log_basis": basis, "source_iv_call": atm["iv_call"].to_numpy(float),
        "source_iv_put": atm["iv_put"].to_numpy(float), "fixed_iv_call": call_iv,
        "fixed_iv_put": put_iv, "atm_iv_calendar": fixed_iv,
        "call_put_gap_before": np.abs(atm["iv_call"].to_numpy(float) - atm["iv_put"].to_numpy(float)),
        "call_put_gap_after": np.abs(call_iv - put_iv), "call_price": call_price, "put_price": put_price,
        "call_oi": atm["call_oi"].to_numpy(float), "put_oi": atm["put_oi"].to_numpy(float),
        "quality_ok": quality,
    })
    return out, pd.DataFrame(attrition)


def prepare_spot(spot: pd.DataFrame):
    spot = spot.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    spot["date"] = spot["datetime"].dt.normalize()
    spot["minute"] = spot["datetime"].dt.hour * 60 + spot["datetime"].dt.minute - 555
    gap = spot["datetime"].diff().dt.total_seconds().to_numpy()
    same_day = spot["date"].eq(spot["date"].shift()).to_numpy()
    values = spot["spot_price"].to_numpy(float)
    previous = np.roll(values, 1)
    ratio = np.divide(values, previous, out=np.full_like(values, np.nan), where=(values > 0) & (previous > 0))
    ret = np.log(ratio)
    valid = same_day & (gap >= 59) & (gap <= 61) & np.isfinite(ret)
    ret[~valid] = np.nan
    spot["return"] = ret
    return spot


def seasonality_profile(spot):
    use = (spot["datetime"] <= TRAIN_END) & spot["return"].notna() & spot["minute"].between(0, 375)
    sq = spot.loc[use, "return"].pow(2)
    capped = sq.clip(upper=sq.quantile(0.995))
    profile = pd.DataFrame({"minute": spot.loc[use, "minute"], "sq": capped}).groupby("minute")["sq"].mean()
    profile = profile.reindex(range(376)).interpolate(limit_direction="both")
    profile = profile.rolling(9, center=True, min_periods=1).mean()
    return profile / profile.loc[1:375].mean()


def rolling_variances(spot, seasonal, horizons):
    n = len(spot)
    minute = spot["minute"].to_numpy(int)
    weights = seasonal.to_numpy()[np.clip(minute, 0, 375)]
    r2 = spot["return"].to_numpy(float) ** 2
    result = {}
    for h in sorted(set(horizons)):
        future = np.full(n, np.nan)
        past = np.full(n, np.nan)
        for _, idx in spot.groupby("date", sort=False).indices.items():
            idx = np.asarray(idx)
            vals, w = r2[idx], weights[idx]
            bad = (~np.isfinite(vals)) | (~np.isfinite(w))
            cv = np.r_[0.0, np.cumsum(np.where(bad, 0.0, vals))]
            cw = np.r_[0.0, np.cumsum(np.where(bad, 0.0, w))]
            cb = np.r_[0, np.cumsum(bad.astype(int))]
            m = len(idx)
            starts = np.arange(m)
            ends = starts + h + 1
            ok = ends <= m
            s, e = starts[ok] + 1, ends[ok]
            good = (cb[e] - cb[s]) == 0
            loc = starts[ok][good]
            future[idx[loc]] = 252 * 375 * (cv[e[good]] - cv[s[good]]) / (cw[e[good]] - cw[s[good]])
            e2 = np.arange(1, m + 1)
            s2 = e2 - h
            ok2 = s2 >= 0
            ss, ee = s2[ok2], e2[ok2]
            good2 = (cb[ee] - cb[ss]) == 0
            loc2 = (ee - 1)[good2]
            past[idx[loc2]] = 252 * 375 * (cv[ee[good2]] - cv[ss[good2]]) / (cw[ee[good2]] - cw[ss[good2]])
        result[f"future_var_{h}"] = future
        result[f"past_var_{h}"] = past
    return result


def remaining_trading_minutes(dt: pd.DatetimeIndex, expiry: pd.DatetimeIndex, spot_dates):
    dates = np.array(sorted(pd.DatetimeIndex(spot_dates).normalize().unique()), dtype="datetime64[D]")
    current = dt.normalize().to_numpy(dtype="datetime64[D]")
    exp = expiry.normalize().to_numpy(dtype="datetime64[D]")
    future_sessions = np.maximum(0, np.searchsorted(dates, exp, side="right") - np.searchsorted(dates, current, side="right"))
    current_remaining = np.maximum(0, 930 - (dt.hour.to_numpy() * 60 + dt.minute.to_numpy()))
    return current_remaining + 375 * future_sessions


def causal_regime(past60, dates):
    regime = np.full(len(past60), "unclassified", dtype=object)
    history = []
    for _, idx in pd.Series(np.arange(len(dates))).groupby(dates).groups.items():
        idx = np.asarray(list(idx), int)
        if history:
            hist = np.concatenate(history[-60:])
            hist = hist[np.isfinite(hist)]
            if len(hist) >= 1000:
                q1, q2 = np.quantile(hist, [1 / 3, 2 / 3])
                regime[idx] = np.where(past60[idx] < q1, "calm", np.where(past60[idx] > q2, "volatile", "normal"))
        history.append(past60[idx][np.isfinite(past60[idx])])
    return regime


def cluster_ols(X, y, clusters):
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, clusters = X[keep], y[keep], np.asarray(clusters)[keep]
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    unique = np.unique(clusters)
    for g in unique:
        use = clusters == g
        score = X[use].T @ resid[use]
        meat += np.outer(score, score)
    scale = (len(unique) / max(len(unique) - 1, 1)) * ((len(y) - 1) / max(len(y) - X.shape[1], 1))
    cov = scale * bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * student_t.sf(np.abs(tstat), max(len(unique) - 1, 1))
    fitted = X @ beta
    resid = y - fitted
    smear = float(np.mean(np.exp(resid)))
    ssr = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return beta, cov, se, p, smear, 1 - ssr / sst


def predict_log_variance(X_train, y_train, day_train, X_eval):
    beta, cov, se, p, smear, r2 = cluster_ols(X_train, np.log(y_train), day_train)
    pred = np.exp(X_eval @ beta) * smear
    return pred, beta, cov, se, p, r2


def qlike(y, f):
    ratio = np.maximum(y, EPS) / np.maximum(f, EPS)
    return ratio - np.log(ratio) - 1


def dm_daily(loss_a, loss_b, day):
    daily = pd.DataFrame({"day": day, "d": loss_a - loss_b}).groupby("day")["d"].mean().to_numpy()
    n = len(daily)
    centered = daily - daily.mean()
    lag = min(5, n - 1)
    lrvar = np.dot(centered, centered) / n
    for ell in range(1, lag + 1):
        gamma = np.dot(centered[ell:], centered[:-ell]) / n
        lrvar += 2 * (1 - ell / (lag + 1)) * gamma
    se = math.sqrt(max(lrvar, 0) / n)
    stat = daily.mean() / se if se > 0 else np.nan
    p = 2 * norm.sf(abs(stat)) if np.isfinite(stat) else np.nan
    return float(daily.mean()), float(stat), float(p)


def bootstrap_ci(values, days, fn=np.mean, reps=1000, seed=17):
    grouped = pd.DataFrame({"day": days, "value": values}).groupby("day")["value"].apply(np.asarray)
    arrays = grouped.to_list()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        chosen = rng.integers(0, len(arrays), len(arrays))
        sample = np.concatenate([arrays[i] for i in chosen])
        estimates.append(fn(sample))
    return np.quantile(estimates, [0.025, 0.975])


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adj = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def time_bucket(end_minutes):
    return pd.cut(end_minutes, [-np.inf, 600, 720, 870, np.inf], labels=["open_to_1000", "1000_to_1200", "1200_to_1430", "after_1430"])


def dte_bucket(hours):
    return pd.cut(hours, [-np.inf, 24, 72, 168, 336, np.inf], labels=["0_24h", "24_72h", "72_168h", "168_336h", "over_336h"])


def run():
    OUT.mkdir(exist_ok=True)
    print("Loading ATM and full spot inputs...", flush=True)
    atm_raw, spot, source_atm_rows = load_inputs()
    atm, attrition = recalculate_atm_iv(atm_raw)
    del atm_raw
    spot = prepare_spot(spot)
    seasonal = seasonality_profile(spot)
    seasonal.rename("variance_weight").to_csv(OUT / "intraday_seasonality.csv", index_label="minute_from_0915")
    windows = rolling_variances(spot, seasonal, list(HORIZONS) + [15, 60])
    for name, values in windows.items():
        spot[name] = values

    position = pd.Index(spot["datetime"]).get_indexer(atm["datetime"])
    matched = position >= 0
    if not matched.all():
        raise RuntimeError(f"{(~matched).sum()} ATM timestamps do not match the spot timeline")
    for col in ["spot_price", "minute", "date", "past_var_5", "past_var_15", "past_var_60"] + [f"future_var_{h}" for h in HORIZONS]:
        atm[col] = spot[col].to_numpy()[position]

    dt = pd.DatetimeIndex(atm["datetime"])
    expiry = pd.DatetimeIndex(atm["expiry"])
    rem = remaining_trading_minutes(dt, expiry, spot["date"])
    atm["remaining_trading_minutes"] = rem
    atm["iv_trade_var"] = atm["atm_iv_calendar"] ** 2 * (atm["days_to_expiry"] / 365.0) * (252 * 375 / np.maximum(rem, 1))
    atm["iv_trade_vol"] = np.sqrt(atm["iv_trade_var"])
    atm["dte_hours"] = atm["days_to_expiry"] * 24
    atm["dte_bucket"] = dte_bucket(atm["dte_hours"])
    atm["year"] = dt.year
    atm["split"] = np.where(dt <= TRAIN_END, "train", np.where(dt <= EVAL_END, "evaluation", "holdout"))
    atm["regime"] = causal_regime(spot["past_var_60"].to_numpy(), spot["date"].to_numpy())[position]
    for h in HORIZONS:
        atm[f"future_rv_{h}"] = np.sqrt(atm[f"future_var_{h}"])
    for h in (5, 15, 60):
        atm[f"past_rv_{h}"] = np.sqrt(atm[f"past_var_{h}"])

    train_date = dt <= TRAIN_END
    eval_date = (dt >= EVAL_START) & (dt <= EVAL_END)
    base_quality = atm["quality_ok"].to_numpy(bool) & np.isfinite(atm["iv_trade_var"]) & (atm["iv_trade_var"] > 0) & (rem > 0)
    audit_rows = attrition.to_dict("records")
    audit_rows.append({"filter": "matched_to_full_spot_timeline", "rows_remaining": int(matched.sum()), "share_remaining": float(matched.mean())})

    descriptive_rows, premium_rows, calibration_rows, incremental_rows = [], [], [], []
    bias_rows, robustness_rows, lookup_rows = [], [], []
    chart_data = {}

    for h in HORIZONS:
        future_var = atm[f"future_var_{h}"].to_numpy(float)
        har_cols = [atm[f"past_var_{x}"].to_numpy(float) for x in (5, 15, 60)]
        bias_valid = base_quality & np.isfinite(future_var) & (future_var > 0)
        btr, bev = bias_valid & train_date, bias_valid & eval_date
        valid = bias_valid.copy()
        for x in har_cols:
            valid &= np.isfinite(x) & (x > 0)
        tr, ev = valid & train_date, valid & eval_date
        audit_rows.append({"filter": f"h{h}_common_train", "rows_remaining": int(tr.sum()), "share_remaining": float(tr.mean())})
        audit_rows.append({"filter": f"h{h}_common_evaluation_2023", "rows_remaining": int(ev.sum()), "share_remaining": float(ev.mean())})
        if tr.sum() < 1000 or ev.sum() < 500:
            raise RuntimeError(f"Insufficient valid sample at horizon {h}: train={tr.sum()}, eval={ev.sum()}")

        ivv = atm["iv_trade_var"].to_numpy(float)
        day = atm["date"].to_numpy()
        X_har_tr = np.column_stack([np.ones(tr.sum())] + [np.log(x[tr]) for x in har_cols])
        X_har_ev = np.column_stack([np.ones(ev.sum())] + [np.log(x[ev]) for x in har_cols])
        har_pred, _, _, _, _, _ = predict_log_variance(X_har_tr, future_var[tr], day[tr], X_har_ev)
        X_iv_tr = np.column_stack([np.ones(tr.sum()), np.log(ivv[tr])])
        X_iv_ev = np.column_stack([np.ones(ev.sum()), np.log(ivv[ev])])
        iv_pred, _, _, _, _, _ = predict_log_variance(X_iv_tr, future_var[tr], day[tr], X_iv_ev)
        X_cal_tr = np.column_stack([np.ones(btr.sum()), np.log(ivv[btr])])
        X_cal_ev = np.column_stack([np.ones(bev.sum()), np.log(ivv[bev])])
        _, cal_beta, cal_cov, cal_se, cal_p, cal_r2 = predict_log_variance(X_cal_tr, future_var[btr], day[btr], X_cal_ev)
        X_comb_tr = np.column_stack([np.ones(tr.sum()), np.log(ivv[tr])] + [np.log(x[tr]) for x in har_cols])
        X_comb_ev = np.column_stack([np.ones(ev.sum()), np.log(ivv[ev])] + [np.log(x[ev]) for x in har_cols])
        comb_pred, comb_beta, comb_cov, comb_se, comb_p, comb_r2 = predict_log_variance(X_comb_tr, future_var[tr], day[tr], X_comb_ev)

        loss_raw_common, loss_iv, loss_har, loss_comb = [qlike(future_var[ev], x) for x in (ivv[ev], iv_pred, har_pred, comb_pred)]
        dm_mean, dm_stat, dm_p = dm_daily(loss_har, loss_comb, day[ev])
        rv_vol, iv_vol = np.sqrt(future_var[bev]), np.sqrt(ivv[bev])
        log_ratio = np.log(rv_vol / iv_vol)
        mean_ci = bootstrap_ci(log_ratio, day[bev], np.mean)
        median_ci = bootstrap_ci(log_ratio, day[bev], np.median)

        for variable, arr in [("ATM_IV_trade", iv_vol), ("future_RV", rv_vol)]:
            qs = np.quantile(arr, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            descriptive_rows.append({
                "horizon_min": h, "variable": variable, "n": len(arr), "mean": arr.mean(), "std": arr.std(),
                "p01": qs[0], "p05": qs[1], "p25": qs[2], "median": qs[3], "p75": qs[4], "p95": qs[5], "p99": qs[6], "max": arr.max(),
            })
        premium_rows.append({
            "horizon_min": h, "n_eval": int(bev.sum()), "mean_log_rv_over_iv": log_ratio.mean(),
            "mean_log_ratio_ci_low": mean_ci[0], "mean_log_ratio_ci_high": mean_ci[1],
            "median_log_rv_over_iv": np.median(log_ratio), "median_log_ratio_ci_low": median_ci[0],
            "median_log_ratio_ci_high": median_ci[1], "pct_iv_gt_rv": np.mean(iv_vol > rv_vol),
            "mae_vol": np.mean(np.abs(iv_vol - rv_vol)), "rmse_vol": np.sqrt(np.mean((iv_vol - rv_vol) ** 2)),
            "raw_iv_qlike": qlike(future_var[bev], ivv[bev]).mean(), "train_median_calibration_multiplier": np.exp(np.median(np.log(np.sqrt(future_var[btr]) / np.sqrt(ivv[btr])))),
        })

        # Log-vol calibration is equivalent to the variance regression with coefficients divided consistently.
        vol_beta = np.array([cal_beta[0] / 2, cal_beta[1]])
        vol_cov = cal_cov.copy(); vol_cov[0, :] /= 2; vol_cov[:, 0] /= 2
        vol_se = np.sqrt(np.diag(vol_cov))
        slope_z = (vol_beta[1] - 1) / vol_se[1]
        slope_p1 = 2 * norm.sf(abs(slope_z))
        diff = np.array([vol_beta[0], vol_beta[1] - 1])
        joint = float(diff.T @ np.linalg.pinv(vol_cov) @ diff)
        calibration_rows.append({
            "horizon_min": h, "train_n": int(btr.sum()), "intercept_log_vol": vol_beta[0],
            "intercept_ci_low": vol_beta[0] - 1.96 * vol_se[0], "intercept_ci_high": vol_beta[0] + 1.96 * vol_se[0],
            "slope_log_vol": vol_beta[1], "slope_ci_low": vol_beta[1] - 1.96 * vol_se[1], "slope_ci_high": vol_beta[1] + 1.96 * vol_se[1],
            "r2_log_variance": cal_r2, "slope_equals_1_pvalue": slope_p1, "joint_intercept0_slope1_pvalue": chi2.sf(joint, 2),
        })
        incremental_rows.append({
            "horizon_min": h, "train_n": int(tr.sum()), "eval_n": int(ev.sum()),
            "combined_iv_coef": comb_beta[1], "combined_iv_ci_low": comb_beta[1] - 1.96 * comb_se[1],
            "combined_iv_ci_high": comb_beta[1] + 1.96 * comb_se[1], "combined_iv_pvalue": comb_p[1],
            "iv_only_qlike": loss_iv.mean(), "har_qlike": loss_har.mean(), "combined_qlike": loss_comb.mean(),
            "combined_improvement_vs_har": loss_har.mean() - loss_comb.mean(),
            "combined_improvement_pct": 1 - loss_comb.mean() / loss_har.mean(),
            "dm_daily_mean_loss_improvement": dm_mean, "dm_stat": dm_stat, "dm_pvalue": dm_p,
        })

        end_minute = 555 + atm["minute"].to_numpy(float) + h
        tb = time_bucket(end_minute)
        db = atm["dte_bucket"].astype(str).to_numpy()
        eval_frame = pd.DataFrame({"time_bucket": np.asarray(tb).astype(str)[bev], "dte_bucket": db[bev], "log_ratio": log_ratio})
        grouped = eval_frame.groupby(["time_bucket", "dte_bucket"], observed=True)["log_ratio"]
        for (time_name, dte_name), values in grouped:
            bias_rows.append({"horizon_min": h, "time_bucket_by_window_end": time_name, "dte_bucket": dte_name,
                              "n": len(values), "mean_log_rv_over_iv": values.mean(), "median_rv_over_iv": np.exp(values.median())})

        train_frame = pd.DataFrame({"time_bucket": np.asarray(tb).astype(str)[btr], "dte_bucket": db[btr],
                                    "ratio": np.sqrt(future_var[btr] / ivv[btr])})
        for (time_name, dte_name), values in train_frame.groupby(["time_bucket", "dte_bucket"], observed=True)["ratio"]:
            lookup_rows.append({"horizon_min": h, "time_bucket_by_window_end": time_name, "dte_bucket": dte_name,
                                "n": len(values), "median_multiplier": values.median(), "mean_multiplier": values.mean()})

        for dimension, vals in [
            ("year", atm["year"].astype(str).to_numpy()), ("regime", atm["regime"].astype(str).to_numpy()),
            ("expiry_day", np.where(atm["days_to_expiry"].to_numpy(float) < 1, "expiry", "non_expiry")),
        ]:
            allowed = bias_valid & (dt <= EVAL_END)
            tmp = pd.DataFrame({"value": vals[allowed], "log_ratio": np.log(np.sqrt(future_var[allowed] / ivv[allowed]))})
            for value, series in tmp.groupby("value")["log_ratio"]:
                robustness_rows.append({"horizon_min": h, "dimension": dimension, "value": value, "n": len(series),
                                        "mean_log_rv_over_iv": series.mean(), "median_rv_over_iv": np.exp(series.median())})
        history = bias_valid & (dt <= EVAL_END)
        chart_data[h] = {"iv": iv_vol, "rv": rv_vol, "ratio": rv_vol / iv_vol, "day": pd.to_datetime(day[bev]),
                         "time_bucket": np.asarray(tb).astype(str)[bev], "dte_bucket": db[bev],
                         "history_ratio": np.sqrt(future_var[history] / ivv[history]), "history_day": pd.to_datetime(day[history])}

    incremental = pd.DataFrame(incremental_rows)
    incremental["combined_iv_pvalue_holm"] = holm(incremental["combined_iv_pvalue"])
    quality_summary = pd.DataFrame([
        {"metric": "Source ATM expiry-specific rows", "value": source_atm_rows},
        {"metric": "ATM rows", "value": len(atm)},
        {"metric": "Additional monthly expiries removed", "value": source_atm_rows - len(atm)},
        {"metric": "ATM trading days", "value": atm["datetime"].dt.normalize().nunique()},
        {"metric": "Full spot timestamps", "value": len(spot)},
        {"metric": "Full spot trading days", "value": spot["date"].nunique()},
        {"metric": "Quality-valid ATM rows", "value": int(atm["quality_ok"].sum())},
        {"metric": "Share ATM rows dropped", "value": float(1 - atm["quality_ok"].mean())},
        {"metric": "Median call-put IV gap before forward fix", "value": float(atm["call_put_gap_before"].median())},
        {"metric": "Median call-put IV gap after forward fix", "value": float(atm["call_put_gap_after"].median())},
        {"metric": "Mean call-put IV gap before forward fix", "value": float(atm["call_put_gap_before"].mean())},
        {"metric": "Mean call-put IV gap after forward fix", "value": float(atm["call_put_gap_after"].mean())},
        {"metric": "Share gap over 3 vol points before", "value": float((atm["call_put_gap_before"] > 0.03).mean())},
        {"metric": "Share gap over 3 vol points after", "value": float((atm["call_put_gap_after"] > 0.03).mean())},
    ])
    tables = {
        "data_audit": pd.DataFrame(audit_rows), "descriptives": pd.DataFrame(descriptive_rows),
        "premium": pd.DataFrame(premium_rows), "calibration": pd.DataFrame(calibration_rows),
        "incremental_value": incremental, "bias_map": pd.DataFrame(bias_rows),
        "robustness": pd.DataFrame(robustness_rows), "seasonality_calibration_lookup": pd.DataFrame(lookup_rows),
        "data_quality_summary": quality_summary,
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)

    panel_cols = [
        "datetime", "expiry", "strike", "spot_price", "forward", "days_to_expiry", "dte_hours", "dte_bucket",
        "atm_iv_calendar", "iv_trade_vol", "iv_trade_var", "call_put_gap_before", "call_put_gap_after",
        "quality_ok", "regime", "split", "past_rv_5", "past_rv_15", "past_rv_60",
    ] + [f"future_rv_{h}" for h in HORIZONS] + [f"future_var_{h}" for h in HORIZONS]
    atm[panel_cols].to_csv(OUT / "cleaned_atm_iv_rv_panel.csv", index=False)
    make_charts(chart_data, tables["bias_map"])
    write_report(tables, atm)
    print("Analysis complete", flush=True)
    print(incremental.to_string(index=False), flush=True)


def make_charts(data, bias_map):
    plt.style.use("seaborn-v0_8-whitegrid")
    primary = data[15]
    bins = pd.qcut(primary["iv"], 10, duplicates="drop")
    binned = pd.DataFrame({"bin": bins, "iv": primary["iv"], "rv": primary["rv"]}).groupby("bin", observed=True)[["iv", "rv"]].mean()
    fig, ax = plt.subplots(figsize=(6.6, 5.2)); ax.scatter(binned["iv"], binned["rv"], s=55, color="#1f4e78")
    lo, hi = min(binned.min()), max(binned.max()); ax.plot([lo, hi], [lo, hi], "--", color="#b22222")
    ax.set(title="ATM IV deciles versus 15-minute realized volatility", xlabel="Mean trading-time ATM IV", ylabel="Mean future RV")
    fig.tight_layout(); fig.savefig(OUT / "binned_scatter_15m.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    order = ["open_to_1000", "1000_to_1200", "1200_to_1430", "after_1430"]
    for h, d in data.items():
        s = pd.DataFrame({"bucket": d["time_bucket"], "ratio": d["ratio"]}).groupby("bucket")["ratio"].median().reindex(order)
        ax.plot(order, s, marker="o", label=f"{h} min")
    ax.axhline(1, color="black", linestyle="--", linewidth=1); ax.legend(); ax.tick_params(axis="x", rotation=20)
    ax.set(title="Median RV/IV by target-window end time", ylabel="Median RV/IV")
    fig.tight_layout(); fig.savefig(OUT / "time_of_day_rv_iv.png", dpi=170); plt.close(fig)

    d = data[15]
    rolling = pd.DataFrame({"date": d["history_day"], "premium": np.log(d["history_ratio"])}).set_index("date").resample("ME").mean()
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(rolling.index, rolling["premium"], color="#1f4e78")
    ax.axhline(0, color="black", linestyle="--", linewidth=1); ax.set(title="Monthly mean log(RV/IV), 15-minute horizon", ylabel="Mean log ratio")
    fig.tight_layout(); fig.savefig(OUT / "monthly_premium_15m.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8)); arrays = [np.log(data[h]["ratio"]) for h in HORIZONS]
    ax.boxplot(arrays, tick_labels=[str(h) for h in HORIZONS], showfliers=False); ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title="Distribution of log(RV/IV)", xlabel="Horizon (minutes)", ylabel="log(RV/IV)")
    fig.tight_layout(); fig.savefig(OUT / "rv_iv_ratio_distribution.png", dpi=170); plt.close(fig)

    dte_order = ["0_24h", "24_72h", "72_168h", "168_336h", "over_336h"]
    heat = bias_map[bias_map["horizon_min"] == 15].pivot(index="time_bucket_by_window_end", columns="dte_bucket", values="median_rv_over_iv").reindex(index=order, columns=dte_order)
    fig, ax = plt.subplots(figsize=(8, 4.5)); im = ax.imshow(heat.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right"); ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Median RV/IV bias map, 15-minute horizon")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat.iloc[i, j]): ax.text(j, i, f"{heat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Median RV/IV"); fig.tight_layout(); fig.savefig(OUT / "bias_heatmap_15m.png", dpi=170); plt.close(fig)


def write_report(tables, panel):
    inc = tables["incremental_value"].set_index("horizon_min")
    prem = tables["premium"].set_index("horizon_min")
    cal = tables["calibration"].set_index("horizon_min")
    p = inc.loc[15]; pp = prem.loc[15]; pc = cal.loc[15]
    adds = (p["combined_iv_pvalue_holm"] < 0.05) and (p["combined_improvement_vs_har"] > 0) and (p["dm_pvalue"] < 0.05)
    multiplier = pp["train_median_calibration_multiplier"]
    verdict = (
        f"Use ATM IV as an input to the 15-minute σ_h forecast, with an initial training-sample multiplier of {multiplier:.3f}."
        if adds else
        f"Do not use ATM IV as an incremental 15-minute σ_h input at this stage; retain it as a pricing benchmark. The training-sample calibration multiplier is {multiplier:.3f}."
    )
    rob15 = tables["robustness"][tables["robustness"]["horizon_min"] == 15]
    regime15 = rob15[rob15["dimension"] == "regime"].set_index("value")
    year15 = rob15[rob15["dimension"] == "year"].set_index("value")
    surface_next = (
        "ATM IV passes the incremental-value test, so testing five-strike slope and curvature next is justified."
        if adds else "ATM IV does not pass the incremental-value test, so slope and curvature are not the next priority."
    )
    lines = [
        "# ATM IV versus intraday realized volatility",
        "",
        "## Decision",
        "",
        verdict,
        " The decision requires all three conditions: a Holm-adjusted significant IV coefficient, positive QLIKE improvement over HAR, and a significant daily Diebold–Mariano test in 2023.",
        " " + surface_next + " A premium-selling or buying rule cannot be approved from these data alone because bid/ask spreads and execution costs are unavailable.",
        "",
        "## 1. Does ATM IV add information beyond trailing RV?",
        "",
        markdown_table(tables["incremental_value"][["horizon_min", "combined_iv_coef", "combined_iv_ci_low", "combined_iv_ci_high", "combined_iv_pvalue_holm", "har_qlike", "combined_qlike", "combined_improvement_pct", "dm_pvalue"]], 6),
        "",
        "## 2. How biased is IV?",
        "",
        markdown_table(tables["premium"][["horizon_min", "mean_log_rv_over_iv", "median_log_rv_over_iv", "pct_iv_gt_rv", "mae_vol", "rmse_vol", "raw_iv_qlike", "train_median_calibration_multiplier"]], 6),
        "",
        "## Calibration",
        "",
        markdown_table(tables["calibration"][["horizon_min", "intercept_log_vol", "slope_log_vol", "slope_ci_low", "slope_ci_high", "r2_log_variance", "slope_equals_1_pvalue", "joint_intercept0_slope1_pvalue"]], 6),
        "",
        "## 3. Where does the bias change?",
        "",
        "The detailed evaluation-sample time-of-day × DTE table is in `bias_map.csv`. The reusable training-only multipliers are in `seasonality_calibration_lookup.csv`. Time buckets are defined by when each target window ends.",
        f" At 15 minutes, median RV/IV is {regime15.loc['calm', 'median_rv_over_iv']:.3f} in calm regimes and {regime15.loc['volatile', 'median_rv_over_iv']:.3f} in volatile regimes. It rises from {year15.loc['2022', 'median_rv_over_iv']:.3f} in 2022 to {year15.loc['2023', 'median_rv_over_iv']:.3f} in 2023, so the premium is not stable enough to use one unconditional multiplier indefinitely. Near-expiry late-day cells can reach or exceed RV/IV = 1.",
        "",
        "## Data audit",
        "",
        markdown_table(tables["data_quality_summary"], 6),
        "",
        "## Statistical design",
        "",
        "- ATM option rows: exact `strike == atm_strike` monthly contracts.",
        "- Forward: `K + call price − put price`, with discount factor assumed to be 1.",
        "- IV: call and put IV re-solved with Black-76 and averaged at ATM.",
        "- RV: continuous one-minute spot returns from the original dataset, seasonally adjusted using a profile estimated through 2022.",
        "- Day count: calendar-time option variance converted to average trading-minute variance using observed sessions remaining to expiry.",
        "- Baseline: HAR-style regression using trailing 5-, 15-, and 60-minute realized variance.",
        "- Inference: day-clustered regression covariance; daily Diebold–Mariano comparison of QLIKE losses.",
        "- Training: 2019–2022. Evaluation: 2023. The 2024–2025 holdout is excluded from all reported metrics and model selection.",
        "",
        "## Data limitations",
        "",
        "- Option prices are last-traded values, not synchronized bid/ask midquotes.",
        "- The parity-forward calculation uses a unit discount factor because no risk-free curve is present.",
        "- Transaction costs and executable spreads are unavailable, so the premium results are not a trading-rule backtest.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
