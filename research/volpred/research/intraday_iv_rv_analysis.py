from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import rankdata, t as student_t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_5strikes.csv"
OUT = ROOT / "outputs" / "iv_rv_analysis"
HORIZONS = (5, 15, 30, 60)
POSITIONS = (-2, -1, 0, 1, 2)
TRAIN_END = pd.Timestamp("2022-12-31 23:59:59")
EVAL_START = pd.Timestamp("2023-01-01")
EVAL_END = pd.Timestamp("2023-12-31 23:59:59")
HOLDOUT_START = pd.Timestamp("2024-01-01")
MIN_OPTION_PRICE = 5.0
MAX_PARITY_MAD_REL = 0.002
EPS = 1e-12


def load_source() -> pd.DataFrame:
    con = duckdb.connect()
    path = SOURCE.as_posix().replace("'", "''")
    query = f"""
        SELECT datetime, expiry, strike, days_to_expiry, spot_price,
               call_price, put_price, call_oi, put_oi,
               iv_call, iv_put, iv_call_converged, iv_put_converged
        FROM read_csv('{path}', header=true, sample_size=100000)
        ORDER BY datetime, expiry, strike
    """
    df = con.execute(query).df()
    con.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def black76_price(forward, strike, maturity, sigma, is_call):
    sqrt_t = np.sqrt(np.maximum(maturity, 1e-16))
    vol_t = sigma * sqrt_t
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * maturity) / vol_t
    d2 = d1 - vol_t
    call = forward * ndtr(d1) - strike * ndtr(d2)
    put = strike * ndtr(-d2) - forward * ndtr(-d1)
    return np.where(is_call, call, put), forward * np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * sqrt_t


def implied_vol_newton(forward, strike, maturity, price, is_call, initial):
    forward, strike, maturity, price, initial = np.broadcast_arrays(
        np.asarray(forward, float), np.asarray(strike, float), np.asarray(maturity, float),
        np.asarray(price, float), np.asarray(initial, float)
    )
    intrinsic = np.where(is_call, np.maximum(forward - strike, 0), np.maximum(strike - forward, 0))
    upper = np.where(is_call, forward, strike)
    valid = (
        np.isfinite(forward) & np.isfinite(strike) & np.isfinite(maturity) & np.isfinite(price)
        & (forward > 0) & (strike > 0) & (maturity > 0)
        & (price >= intrinsic - 0.10) & (price < upper)
    )
    sigma = np.clip(np.where(np.isfinite(initial), initial, 0.25), 0.01, 4.0)
    for _ in range(25):
        model, vega = black76_price(forward, strike, maturity, sigma, is_call)
        step = np.divide(model - price, vega, out=np.zeros_like(sigma), where=vega > 1e-8)
        sigma = np.clip(sigma - step, 0.001, 5.0)
    model, _ = black76_price(forward, strike, maturity, sigma, is_call)
    converged = valid & (np.abs(model - price) <= np.maximum(0.10, price * 2e-4))
    return np.where(converged, sigma, np.nan), converged


def reshape_groups(df: pd.DataFrame):
    sizes = df.groupby(["datetime", "expiry"], sort=False).size().to_numpy()
    if len(sizes) == 0 or not np.all(sizes == 5):
        raise ValueError("Source does not contain exactly five rows per datetime/expiry group")
    n = len(sizes)
    fields = {}
    for col in [
        "strike", "spot_price", "days_to_expiry", "call_price", "put_price",
        "call_oi", "put_oi", "iv_call", "iv_put",
    ]:
        fields[col] = df[col].to_numpy(float).reshape(n, 5)
    dt = df["datetime"].to_numpy().reshape(n, 5)[:, 0]
    expiry = df["expiry"].to_numpy().reshape(n, 5)[:, 0]
    return pd.DatetimeIndex(dt), pd.DatetimeIndex(expiry), fields


def prepare_chain(dt, expiry, f):
    k = f["strike"]
    call = f["call_price"]
    put = f["put_price"]
    synth = k + call - put  # zero-rate parity forward; robustly aggregated below
    forward = np.nanmedian(synth, axis=1)
    parity_mad = np.nanmedian(np.abs(synth - forward[:, None]), axis=1)
    parity_mad_rel = parity_mad / forward
    forward_atm_idx = np.argmin(np.abs(np.log(k / forward[:, None])), axis=1)
    central_forward_atm = forward_atm_idx == 2

    t_years = np.nanmedian(f["days_to_expiry"], axis=1) / 365.0
    t_matrix = np.broadcast_to(t_years[:, None], k.shape)
    fw_matrix = np.broadcast_to(forward[:, None], k.shape)
    call_iv, call_ok = implied_vol_newton(fw_matrix, k, t_matrix, call, np.ones_like(k, bool), f["iv_call"])
    put_iv, put_ok = implied_vol_newton(fw_matrix, k, t_matrix, put, np.zeros_like(k, bool), f["iv_put"])

    selected_iv = np.where(k < forward[:, None], put_iv, call_iv)
    selected_price = np.where(k < forward[:, None], put, call)
    selected_oi = np.where(k < forward[:, None], f["put_oi"], f["call_oi"])
    # Forward ATM is required to be the center row. Average separately solved call/put IV there.
    selected_iv[:, 2] = 0.5 * (call_iv[:, 2] + put_iv[:, 2])
    selected_price[:, 2] = np.minimum(call[:, 2], put[:, 2])
    selected_oi[:, 2] = np.minimum(f["call_oi"][:, 2], f["put_oi"][:, 2])
    atm_cp_gap = np.abs(call_iv[:, 2] - put_iv[:, 2])

    iv_valid = np.all(np.isfinite(selected_iv), axis=1)
    price_valid = np.all(selected_price >= MIN_OPTION_PRICE, axis=1)
    oi_valid = np.all(selected_oi > 0, axis=1)
    maturity_valid = t_years > 0
    parity_valid = np.isfinite(parity_mad_rel) & (parity_mad_rel <= MAX_PARITY_MAD_REL)
    quality = central_forward_atm & iv_valid & price_valid & oi_valid & maturity_valid & parity_valid

    atm_iv = selected_iv[:, 2]
    standardized_moneyness = np.log(k / forward[:, None]) / (atm_iv[:, None] * np.sqrt(t_years[:, None]))
    return {
        "forward": forward,
        "parity_mad_rel": parity_mad_rel,
        "forward_atm_idx": forward_atm_idx,
        "selected_iv": selected_iv,
        "selected_price": selected_price,
        "selected_oi": selected_oi,
        "atm_cp_gap": atm_cp_gap,
        "t_years": t_years,
        "z": standardized_moneyness,
        "quality": quality,
        "checks": {
            "groups": len(dt),
            "forward_atm_is_center": int(central_forward_atm.sum()),
            "all_five_iv_solved": int(iv_valid.sum()),
            "all_five_price_ge_5": int(price_valid.sum()),
            "all_five_positive_oi": int(oi_valid.sum()),
            "parity_mad_le_20bp": int(parity_valid.sum()),
            "common_quality_groups": int(quality.sum()),
            "atm_call_put_gap_gt_3volpts": int(np.sum(np.isfinite(atm_cp_gap) & (atm_cp_gap > 0.03))),
        },
    }


def build_spot_returns(dt, spot):
    frame = pd.DataFrame({"datetime": dt, "spot": spot})
    frame["date"] = frame["datetime"].dt.normalize()
    frame["minute"] = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute - (9 * 60 + 15)
    gap = frame["datetime"].diff().dt.total_seconds().to_numpy()
    same_day = frame["date"].eq(frame["date"].shift()).to_numpy()
    ret = np.log(frame["spot"].to_numpy() / np.roll(frame["spot"].to_numpy(), 1))
    valid = same_day & (gap >= 59) & (gap <= 61) & np.isfinite(ret)
    ret[~valid] = np.nan
    frame["ret"] = ret
    return frame


def estimate_seasonality(frame):
    train = frame["datetime"] <= TRAIN_END
    valid = train & frame["ret"].notna() & frame["minute"].between(0, 375)
    sq = frame.loc[valid, "ret"].pow(2)
    cap = sq.quantile(0.995)
    tmp = pd.DataFrame({"minute": frame.loc[valid, "minute"], "sq": sq.clip(upper=cap)})
    profile = tmp.groupby("minute")["sq"].mean().reindex(range(376))
    profile = profile.interpolate(limit_direction="both").rolling(9, center=True, min_periods=1).mean()
    profile = profile / profile.loc[1:375].mean()
    return profile


def window_variances(frame, seasonal, horizon):
    n = len(frame)
    future = np.full(n, np.nan)
    past = np.full(n, np.nan)
    r2 = frame["ret"].to_numpy() ** 2
    minute = frame["minute"].to_numpy(int)
    weights = np.where((minute >= 0) & (minute <= 375), seasonal.to_numpy()[np.clip(minute, 0, 375)], np.nan)
    for _, idx in frame.groupby("date", sort=False).indices.items():
        idx = np.asarray(idx)
        vals = r2[idx]
        w = weights[idx]
        bad = (~np.isfinite(vals)) | (~np.isfinite(w))
        cv = np.r_[0.0, np.cumsum(np.where(bad, 0.0, vals))]
        cw = np.r_[0.0, np.cumsum(np.where(bad, 0.0, w))]
        cb = np.r_[0, np.cumsum(bad.astype(int))]
        m = len(idx)
        # At i, future uses returns i+1 ... i+h.
        starts = np.arange(m)
        ends = starts + horizon + 1
        ok = ends <= m
        s = starts[ok] + 1
        e = ends[ok]
        good = (cb[e] - cb[s]) == 0
        loc = starts[ok][good]
        vv = cv[e[good]] - cv[s[good]]
        ww = cw[e[good]] - cw[s[good]]
        future[idx[loc]] = 252.0 * 375.0 * vv / ww
        # At i, past uses returns i-h+1 ... i.
        e2 = np.arange(1, m + 1)
        s2 = e2 - horizon
        ok2 = s2 >= 0
        s2v, e2v = s2[ok2], e2[ok2]
        good2 = (cb[e2v] - cb[s2v]) == 0
        loc2 = (e2[ok2] - 1)[good2]
        vv2 = cv[e2v[good2]] - cv[s2v[good2]]
        ww2 = cw[e2v[good2]] - cw[s2v[good2]]
        past[idx[loc2]] = 252.0 * 375.0 * vv2 / ww2
    return future, past


def remaining_trading_minutes(dt, expiry):
    dates = np.array(sorted(pd.DatetimeIndex(dt.normalize().unique()).values), dtype="datetime64[D]")
    current = dt.normalize().to_numpy(dtype="datetime64[D]")
    exp = expiry.normalize().to_numpy(dtype="datetime64[D]")
    left = np.searchsorted(dates, current, side="right")
    right = np.searchsorted(dates, exp, side="right")
    future_sessions = np.maximum(0, right - left)
    minute_of_day = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
    current_remaining = np.maximum(0, 15 * 60 + 30 - minute_of_day)
    return current_remaining + 375 * future_sessions


def corr_metrics(x, y, day):
    lx, ly = np.log(x), np.log(y)
    pearson = np.corrcoef(lx, ly)[0, 1]
    spearman = np.corrcoef(rankdata(lx), rankdata(ly))[0, 1]
    tmp = pd.DataFrame({"x": lx, "y": ly, "day": day})
    dx = tmp["x"] - tmp.groupby("day")["x"].transform("mean")
    dy = tmp["y"] - tmp.groupby("day")["y"].transform("mean")
    within = np.corrcoef(dx, dy)[0, 1]
    return pearson, spearman, within


def qlike(y, forecast):
    ratio = np.maximum(y, EPS) / np.maximum(forecast, EPS)
    return ratio - np.log(ratio) - 1.0


def fit_ols_hac(X, y, lag):
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[keep], y[keep]
    inv = np.linalg.pinv(X.T @ X)
    beta = inv @ X.T @ y
    resid = y - X @ beta
    xe = X * resid[:, None]
    S = xe.T @ xe
    for ell in range(1, min(lag, len(y) - 1) + 1):
        weight = 1.0 - ell / (lag + 1.0)
        gamma = xe[ell:].T @ xe[:-ell]
        S += weight * (gamma + gamma.T)
    cov = inv @ S @ inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * student_t.sf(np.abs(tstat), max(len(y) - X.shape[1], 1))
    smear = float(np.mean(np.exp(resid)))
    return beta, se, p, smear


def predict_log_model(X_train, y_train, X_eval, lag):
    beta, se, p, smear = fit_ols_hac(X_train, np.log(y_train), lag)
    pred = np.exp(X_eval @ beta) * smear
    return pred, beta, se, p


def bootstrap_day_ci(loss_base, loss_model, days, seed=7, reps=1000):
    tmp = pd.DataFrame({"d": days, "diff": loss_base - loss_model}).groupby("d")["diff"].mean()
    values = tmp.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def factor_values(log_iv_var, z):
    X = np.stack([np.ones_like(z), z, z * z], axis=2)
    xtx = np.einsum("nki,nkj->nij", X, X)
    xty = np.einsum("nki,nk->ni", X, log_iv_var)
    beta = np.full((len(z), 3), np.nan)
    good = (
        np.all(np.isfinite(xtx), axis=(1, 2))
        & np.all(np.isfinite(xty), axis=1)
        & (np.abs(np.linalg.det(xtx)) > 1e-12)
    )
    beta[good] = np.linalg.solve(xtx[good], xty[good, :, None]).squeeze(-1)
    return beta


def run_analysis():
    OUT.mkdir(exist_ok=True)
    print("Loading five-strike monthly dataset...", flush=True)
    raw = load_source()
    dt, expiry, fields = reshape_groups(raw)
    del raw
    print(f"Loaded {len(dt):,} timestamp/expiry groups", flush=True)

    chain = prepare_chain(dt, expiry, fields)
    spot = fields["spot_price"][:, 0]
    frame = build_spot_returns(dt, spot)
    seasonal = estimate_seasonality(frame)
    seasonal.rename("seasonal_variance_weight").to_csv(OUT / "intraday_seasonality.csv", index_label="minute_from_0915")

    rem_minutes = remaining_trading_minutes(dt, expiry)
    iv_calendar_var = chain["selected_iv"] ** 2
    iv_trade_var = iv_calendar_var * chain["t_years"][:, None] * (252.0 * 375.0 / np.maximum(rem_minutes[:, None], 1))
    factors = factor_values(np.log(np.maximum(iv_trade_var, EPS)), chain["z"])

    train_date = dt <= TRAIN_END
    eval_date = (dt >= EVAL_START) & (dt <= EVAL_END)
    holdout = dt >= HOLDOUT_START
    base_quality = chain["quality"] & (~holdout) & (rem_minutes > 0)
    days = dt.normalize()

    metric_rows = []
    regression_rows = []
    factor_rows = []
    primary_losses = {}

    for h in HORIZONS:
        future_var, past_var = window_variances(frame, seasonal, h)
        common = base_quality & np.isfinite(future_var) & np.isfinite(past_var) & (future_var > 0) & (past_var > 0)
        tr = common & train_date
        ev = common & eval_date
        print(f"Horizon {h}: common={common.sum():,}, train={tr.sum():,}, eval={ev.sum():,}", flush=True)
        if tr.sum() < 1000 or ev.sum() < 100:
            raise RuntimeError(f"Insufficient common sample for {h}-minute horizon")

        Xb_tr = np.column_stack([np.ones(tr.sum()), np.log(past_var[tr])])
        Xb_ev = np.column_stack([np.ones(ev.sum()), np.log(past_var[ev])])
        baseline_pred, b_beta, b_se, b_p = predict_log_model(Xb_tr, future_var[tr], Xb_ev, h)
        baseline_loss = qlike(future_var[ev], baseline_pred)

        for j, pos in enumerate(POSITIONS):
            ivv = iv_trade_var[:, j]
            pearson, spearman, within = corr_metrics(ivv[ev], future_var[ev], days[ev])
            raw_loss = qlike(future_var[ev], ivv[ev])
            X_tr = np.column_stack([np.ones(tr.sum()), np.log(ivv[tr]), np.log(past_var[tr])])
            X_ev = np.column_stack([np.ones(ev.sum()), np.log(ivv[ev]), np.log(past_var[ev])])
            pred, beta, se, p = predict_log_model(X_tr, future_var[tr], X_ev, h)
            model_loss = qlike(future_var[ev], pred)
            ci_low, ci_high = bootstrap_day_ci(baseline_loss, model_loss, days[ev])
            metric_rows.append({
                "horizon_min": h, "position": pos, "train_n": int(tr.sum()), "eval_n": int(ev.sum()),
                "pearson_log": pearson, "spearman_log": spearman, "within_day_pearson_log": within,
                "mean_log_iv_over_rv": float(np.mean(np.log(ivv[ev] / future_var[ev]))),
                "raw_iv_qlike": float(np.mean(raw_loss)), "baseline_qlike": float(np.mean(baseline_loss)),
                "encompassing_qlike": float(np.mean(model_loss)),
                "qlike_improvement_vs_trailing": float(np.mean(baseline_loss - model_loss)),
                "qlike_improvement_ci_low": ci_low, "qlike_improvement_ci_high": ci_high,
            })
            regression_rows.append({
                "horizon_min": h, "position": pos,
                "intercept": beta[0], "iv_coef": beta[1], "iv_hac_se": se[1], "iv_pvalue": p[1],
                "trailing_rv_coef": beta[2], "trailing_rv_hac_se": se[2], "trailing_rv_pvalue": p[2],
            })
            if h == 15:
                primary_losses[pos] = (baseline_loss, model_loss, days[ev])

        Xf_tr = np.column_stack([
            np.ones(tr.sum()), np.log(past_var[tr]), factors[tr, 0], factors[tr, 1], factors[tr, 2]
        ])
        Xf_ev = np.column_stack([
            np.ones(ev.sum()), np.log(past_var[ev]), factors[ev, 0], factors[ev, 1], factors[ev, 2]
        ])
        fpred, fbeta, fse, fp = predict_log_model(Xf_tr, future_var[tr], Xf_ev, h)
        floss = qlike(future_var[ev], fpred)
        fci_low, fci_high = bootstrap_day_ci(baseline_loss, floss, days[ev])
        factor_rows.append({
            "horizon_min": h, "train_n": int(tr.sum()), "eval_n": int(ev.sum()),
            "baseline_qlike": float(np.mean(baseline_loss)), "factor_qlike": float(np.mean(floss)),
            "qlike_improvement_vs_trailing": float(np.mean(baseline_loss - floss)),
            "improvement_ci_low": fci_low, "improvement_ci_high": fci_high,
            "level_coef": fbeta[2], "level_pvalue": fp[2],
            "slope_coef": fbeta[3], "slope_pvalue": fp[3],
            "curvature_coef": fbeta[4], "curvature_pvalue": fp[4],
        })

    metrics = pd.DataFrame(metric_rows)
    regressions = pd.DataFrame(regression_rows)
    regressions["iv_pvalue_holm"] = holm_adjust(regressions["iv_pvalue"])
    factors_out = pd.DataFrame(factor_rows)
    metrics.to_csv(OUT / "strike_metrics.csv", index=False)
    regressions.to_csv(OUT / "encompassing_regressions.csv", index=False)
    factors_out.to_csv(OUT / "factor_model.csv", index=False)

    quality = dict(chain["checks"])
    quality.update({
        "train_end": str(TRAIN_END), "evaluation_period": "2023-01-01 to 2023-12-31",
        "holdout_start_untouched": str(HOLDOUT_START), "min_option_price": MIN_OPTION_PRICE,
        "max_parity_mad_relative": MAX_PARITY_MAD_REL,
        "median_parity_mad_bp": float(np.nanmedian(chain["parity_mad_rel"]) * 10000),
        "median_atm_call_put_gap_vol_points": float(np.nanmedian(chain["atm_cp_gap"]) * 100),
    })
    (OUT / "quality_audit.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    make_plots(metrics, factors_out, seasonal)
    write_report(metrics, regressions, factors_out, quality)
    print("Analysis complete", flush=True)
    print(json.dumps(quality, indent=2), flush=True)


def make_plots(metrics, factors_out, seasonal):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(seasonal.index, seasonal.values, color="#1f4e78", linewidth=1.6)
    ax.set(title="Intraday variance seasonality (training sample)", xlabel="Minutes from 09:15", ylabel="Relative variance weight")
    fig.tight_layout()
    fig.savefig(OUT / "intraday_seasonality.png", dpi=160)
    plt.close(fig)

    pivot = metrics.pivot(index="position", columns="horizon_min", values="qlike_improvement_vs_trailing")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Forward RV horizon (minutes)")
    ax.set_ylabel("Strike position relative to forward ATM")
    ax.set_title("QLIKE improvement over trailing-RV benchmark")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Positive is better")
    fig.tight_layout()
    fig.savefig(OUT / "qlike_improvement_heatmap.png", dpi=160)
    plt.close(fig)


def write_report(metrics, regressions, factors_out, quality):
    primary = metrics[metrics["horizon_min"] == 15].sort_values("position")
    preg = regressions[regressions["horizon_min"] == 15].sort_values("position")
    pfactor = factors_out[factors_out["horizon_min"] == 15].iloc[0]
    best = primary.loc[primary["qlike_improvement_vs_trailing"].idxmax()]
    lines = [
        "# Intraday IV–RV statistical analysis",
        "",
        "## Design",
        "",
        "- Source: five-strike monthly-expiry dataset.",
        "- Parity forward: median of `K + call - put` across the five strikes (zero-rate approximation).",
        "- IV: Black-76 re-solved from the parity forward; OTM put below forward, OTM call above forward, average separately solved call/put IV at forward ATM.",
        "- Common sample: all five IVs solved, all selected prices at least ₹5, positive OI, forward ATM is the center strike, and parity MAD no more than 20 bp of forward.",
        "- RV: seasonally adjusted forward realized variance using minute spot returns. Windows never cross the close.",
        "- IV/RV convention: calendar-time IV total variance converted to average trading-minute variance using remaining observed trading minutes to expiry.",
        "- Training: through 2022. Evaluation: calendar 2023. Data from 2024 onward is untouched holdout.",
        "- Primary horizon: 15 minutes. Other horizons are robustness checks.",
        "",
        "## Data quality",
        "",
        f"- Total five-strike groups: {quality['groups']:,}",
        f"- Common quality groups before horizon labels: {quality['common_quality_groups']:,}",
        f"- Median parity dispersion: {quality['median_parity_mad_bp']:.2f} bp",
        f"- Median parity-corrected ATM call/put IV gap: {quality['median_atm_call_put_gap_vol_points']:.2f} vol points",
        "",
        "## Primary 15-minute result",
        "",
        f"The best individual strike position by out-of-sample QLIKE improvement is {int(best['position']):+d}. "
        f"Its improvement over the trailing-RV regression is {best['qlike_improvement_vs_trailing']:.6f} "
        f"with a day-bootstrap 95% interval of [{best['qlike_improvement_ci_low']:.6f}, {best['qlike_improvement_ci_high']:.6f}].",
        "",
        primary[["position", "eval_n", "pearson_log", "spearman_log", "within_day_pearson_log", "baseline_qlike", "encompassing_qlike", "qlike_improvement_vs_trailing", "qlike_improvement_ci_low", "qlike_improvement_ci_high"]].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Encompassing regression, 15 minutes",
        "",
        "`log RV_future = a + b1 log IV + b2 log RV_past`. HAC standard errors use a 15-observation lag. Holm adjustment covers all 20 strike/horizon IV-coefficient tests.",
        "",
        preg[["position", "iv_coef", "iv_hac_se", "iv_pvalue", "iv_pvalue_holm", "trailing_rv_coef", "trailing_rv_pvalue"]].to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Level, slope and curvature factor model",
        "",
        f"At 15 minutes, the five-IV factor model changes QLIKE versus trailing RV by {pfactor['qlike_improvement_vs_trailing']:.6f} "
        f"with a day-bootstrap interval of [{pfactor['improvement_ci_low']:.6f}, {pfactor['improvement_ci_high']:.6f}].",
        "",
        factors_out.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Interpretation limits",
        "",
        "- The parity-forward calculation assumes discount factor 1 because the source does not contain a risk-free curve. The short monthly maturity makes this a small but nonzero approximation.",
        "- The filtered source contains only timestamps with a complete spot-centered five-strike set. Results apply to that common-support sample.",
        "- Intraday RV is noisy. Statistical significance and day-bootstrap loss improvement matter more than small rank differences between strikes.",
        "- The 2024–2025 holdout was not inspected for model selection or reported performance.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_analysis()
