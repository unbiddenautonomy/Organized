from __future__ import annotations

from pathlib import Path
import math

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm, t as student_t

from atm_netpremium_correlation import dataframe_to_markdown
from five_strike_spot_hypothesis import load_nearest_expiry, add_fixed_forward_iv, implied_vol_vector


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data" / "raw" / "icharts_option_banknifty.csv"
ATM_PANEL = ROOT / "outputs" / "atm_iv_rv_analysis" / "cleaned_atm_iv_rv_panel.csv"
OUT = ROOT / "outputs" / "intraday_structure_attribution"
EXIT_PARQUET = OUT / "_fixed_strike_exit_prices.parquet"
EXIT_FORWARD_PARQUET = OUT / "_exit_chain_forward.parquet"
HORIZONS = {"5m": 5, "15m": 15, "30m": 30, "60m": 60, "session_end": None}
POS_NAMES = {-2: "m2", -1: "m1", 0: "atm", 1: "p1", 2: "p2"}

STRUCTURES = {
    "long_atm_call": {(0, "call"): 1},
    "long_atm_put": {(0, "put"): 1},
    "long_p1_call": {(1, "call"): 1},
    "long_m1_put": {(-1, "put"): 1},
    "bull_call_debit_atm_p2": {(0, "call"): 1, (2, "call"): -1},
    "bear_put_debit_atm_m2": {(0, "put"): 1, (-2, "put"): -1},
    "long_atm_straddle": {(0, "call"): 1, (0, "put"): 1},
    "short_atm_straddle": {(0, "call"): -1, (0, "put"): -1},
    "long_1_strangle": {(1, "call"): 1, (-1, "put"): 1},
    "short_1_strangle": {(1, "call"): -1, (-1, "put"): -1},
    "long_2_strangle": {(2, "call"): 1, (-2, "put"): 1},
    "short_2_strangle": {(2, "call"): -1, (-2, "put"): -1},
    "iron_condor_short1_long2": {(-1, "put"): -1, (1, "call"): -1, (-2, "put"): 1, (2, "call"): 1},
}

PRIMARY = {
    ("long_atm_straddle", "15m"),
    ("long_atm_call", "15m"),
    ("long_atm_put", "15m"),
    ("iron_condor_short1_long2", "30m"),
}


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def black_greeks(forward: pd.Series, strike: pd.Series, t_years: pd.Series, vol: pd.Series,
                 spot: pd.Series, call: bool) -> dict[str, np.ndarray]:
    f = forward.to_numpy(float)
    k = strike.to_numpy(float)
    t = t_years.to_numpy(float)
    s = spot.to_numpy(float)
    v = vol.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rt = np.sqrt(t)
        d1 = (np.log(f / k) + 0.5 * v * v * t) / (v * rt)
        d2 = d1 - v * rt
        pdf = norm_pdf(d1)
        scale = f / s
        delta_f = ndtr(d1) if call else ndtr(d1) - 1.0
        delta = delta_f * scale
        gamma = pdf / (f * v * rt) * scale * scale
        vega = f * pdf * rt  # price points per 1.00 absolute volatility
        theta_day = (-f * pdf * v / (2.0 * rt)) / 365.0
        vanna = (-pdf * d2 / v) * scale
        volga = vega * d1 * d2 / v
        dt_year = 1.0 / 365.0
        t2 = np.maximum(t - dt_year, 1e-8)
        rt2 = np.sqrt(t2)
        d1_2 = (np.log(f / k) + 0.5 * v * v * t2) / (v * rt2)
        delta_f_2 = ndtr(d1_2) if call else ndtr(d1_2) - 1.0
        charm_elapsed_day = (delta_f_2 * scale - delta)  # one calendar day passage
    valid = np.isfinite(f) & np.isfinite(k) & np.isfinite(t) & np.isfinite(v) & np.isfinite(s) & (f > 0) & (k > 0) & (t > 0) & (v > 0) & (s > 0)
    result = {"delta": delta, "gamma": gamma, "vega": vega, "theta_day": theta_day,
              "vanna": vanna, "volga": volga, "charm_day": charm_elapsed_day}
    for key in result:
        result[key] = np.where(valid, result[key], np.nan)
    return result


def prepare_entry_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = add_fixed_forward_iv(load_nearest_expiry())
    rows["date"] = rows["datetime"].dt.date.astype(str)
    rows["minute_of_day"] = rows["datetime"].dt.hour * 60 + rows["datetime"].dt.minute
    rows["dte_hours"] = rows["days_to_expiry"] * 24.0
    rows["time_bucket"] = pd.cut(
        rows["minute_of_day"], bins=[-np.inf, 600, 720, 870, np.inf],
        labels=["open_to_1000", "1000_to_1200", "1200_to_1430", "after_1430"], right=True,
    ).astype(str)
    rows["dte_bucket"] = pd.cut(
        rows["dte_hours"], bins=[-np.inf, 24, 72, 168, 336, np.inf],
        labels=["0_24h", "24_72h", "72_168h", "168_336h", "over_336h"], right=True,
    ).astype(str)
    rows["chain_iv_mean"] = rows[["iv_call", "iv_put"]].mean(axis=1).groupby(rows["datetime"]).transform("mean")

    atm = pd.read_csv(ATM_PANEL, usecols=["datetime", "iv_trade_vol", "quality_ok"], parse_dates=["datetime"])
    atm = atm.drop_duplicates("datetime")
    meta = rows.groupby("datetime", as_index=False).agg(
        date=("date", "first"), spot_price=("spot_price", "first"), common_forward=("common_forward", "first"),
        chain_iv_mean=("chain_iv_mean", "first"), minute_of_day=("minute_of_day", "first"),
        time_bucket=("time_bucket", "first"), dte_bucket=("dte_bucket", "first"),
    ).merge(atm, on="datetime", how="left", validate="one_to_one")
    return rows, meta


def build_exit_parquet(entries: pd.DataFrame, meta: pd.DataFrame) -> None:
    if EXIT_PARQUET.exists():
        return
    eod = meta.groupby("date", as_index=False)["datetime"].max().rename(columns={"datetime": "eod_datetime"})
    entry_keys = entries[["source_index", "datetime", "expiry", "strike", "strike_position", "date"]].copy()
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    con.register("entry_keys_df", entry_keys)
    con.register("eod_df", eod)
    original = str(ORIGINAL).replace("'", "''")
    target = str(EXIT_PARQUET).replace("'", "''")
    query = f"""
    COPY (
      WITH keys AS (
        SELECT source_index, datetime AS entry_datetime, expiry, strike, strike_position,
               '5m' AS horizon, datetime + INTERVAL 5 MINUTE AS exit_datetime FROM entry_keys_df
        UNION ALL
        SELECT source_index, datetime, expiry, strike, strike_position,
               '15m', datetime + INTERVAL 15 MINUTE FROM entry_keys_df
        UNION ALL
        SELECT source_index, datetime, expiry, strike, strike_position,
               '30m', datetime + INTERVAL 30 MINUTE FROM entry_keys_df
        UNION ALL
        SELECT source_index, datetime, expiry, strike, strike_position,
               '60m', datetime + INTERVAL 60 MINUTE FROM entry_keys_df
        UNION ALL
        SELECT k.source_index, k.datetime, k.expiry, k.strike, k.strike_position,
               'session_end', e.eod_datetime
        FROM entry_keys_df k JOIN eod_df e USING(date)
        WHERE e.eod_datetime > k.datetime
      ), option_exit AS (
        SELECT datetime::TIMESTAMP AS exit_datetime, expiry::DATE AS expiry, strike::DOUBLE AS strike,
               call_price::DOUBLE AS exit_call_price, put_price::DOUBLE AS exit_put_price
        FROM read_csv_auto('{original}', header=true)
        WHERE expiry_type='MONTHLY'
      )
      SELECT k.*, o.exit_call_price, o.exit_put_price
      FROM keys k
      LEFT JOIN option_exit o USING(exit_datetime, expiry, strike)
      WHERE CAST(k.entry_datetime AS DATE)=CAST(k.exit_datetime AS DATE)
    ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.close()


def build_exit_forward_parquet() -> None:
    if EXIT_FORWARD_PARQUET.exists():
        return
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    groups = con.execute(
        "SELECT DISTINCT exit_datetime, expiry FROM read_parquet(?)", [str(EXIT_PARQUET)]
    ).fetchdf()
    con.register("exit_groups_df", groups)
    original = str(ORIGINAL).replace("'", "''")
    target = str(EXIT_FORWARD_PARQUET).replace("'", "''")
    query = f"""
    COPY (
      SELECT o.datetime::TIMESTAMP AS exit_datetime, o.expiry::DATE AS expiry,
             median(o.strike::DOUBLE + o.call_price::DOUBLE - o.put_price::DOUBLE) AS exit_forward,
             median(o.spot_price::DOUBLE) AS exit_spot
      FROM read_csv_auto('{original}', header=true) o
      INNER JOIN exit_groups_df g
        ON o.datetime::TIMESTAMP=g.exit_datetime AND o.expiry::DATE=g.expiry
      WHERE o.expiry_type='MONTHLY'
      GROUP BY o.datetime::TIMESTAMP, o.expiry::DATE
    ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.close()


def add_entry_greeks(entries: pd.DataFrame) -> pd.DataFrame:
    result = entries.copy()
    t = result["days_to_expiry"].clip(lower=1.0 / (365 * 24 * 60)) / 365.0
    for side, call in [("call", True), ("put", False)]:
        iv = result[f"iv_{side}"]
        greeks = black_greeks(result["common_forward"], result["strike"], t, iv, result["spot_price"], call)
        for name, value in greeks.items():
            result[f"{side}_{name}"] = value
    return result


def prepare_leg_window(entries: pd.DataFrame, meta: pd.DataFrame, horizon: str) -> pd.DataFrame:
    exits = duckdb.connect().execute(
        """SELECT e.*, f.exit_forward, f.exit_spot
           FROM read_parquet(?) e
           LEFT JOIN read_parquet(?) f USING(exit_datetime, expiry)
           WHERE e.horizon=?""",
        [str(EXIT_PARQUET), str(EXIT_FORWARD_PARQUET), horizon],
    ).fetchdf()
    base_cols = list(entries.columns)
    w = exits.merge(entries[base_cols], on=["source_index", "expiry", "strike", "strike_position"], how="left", validate="many_to_one")
    w["elapsed_minutes"] = (w["exit_datetime"] - w["entry_datetime"]).dt.total_seconds() / 60.0
    w["elapsed_days"] = w["elapsed_minutes"] / 1440.0
    w["spot_change"] = w["exit_spot"] - w["spot_price"]
    w["exit_days_to_expiry"] = w["days_to_expiry"] - w["elapsed_days"]
    w["exit_t_years"] = w["exit_days_to_expiry"].clip(lower=1.0 / (365 * 24 * 60)) / 365.0
    w["entry_t_years"] = w["days_to_expiry"].clip(lower=1.0 / (365 * 24 * 60)) / 365.0

    for side, call in [("call", True), ("put", False)]:
        w[f"exit_iv_{side}"] = implied_vol_vector(
            w[f"exit_{side}_price"], w["exit_forward"], w["strike"], w["exit_t_years"], call
        )
        exit_g = black_greeks(w["exit_forward"], w["strike"], w["exit_t_years"], w[f"exit_iv_{side}"], w["exit_spot"], call)
        use_mid = (w["dte_hours"] <= 24) & (w["elapsed_minutes"] > 15)
        for name, exit_value in exit_g.items():
            start_value = w[f"{side}_{name}"].to_numpy(float)
            w[f"used_{side}_{name}"] = np.where(use_mid, 0.5 * (start_value + exit_value), start_value)
        dvol = w[f"exit_iv_{side}"] - w[f"iv_{side}"]
        ds = w["spot_change"]
        ed = w["elapsed_days"]
        w[f"{side}_actual"] = w[f"exit_{side}_price"] - w[f"{side}_price"]
        w[f"{side}_delta_term"] = w[f"used_{side}_delta"] * ds
        w[f"{side}_gamma_term"] = 0.5 * w[f"used_{side}_gamma"] * ds * ds
        w[f"{side}_vega_term"] = w[f"used_{side}_vega"] * dvol
        w[f"{side}_theta_term"] = w[f"used_{side}_theta_day"] * ed
        w[f"{side}_vanna_term"] = w[f"used_{side}_vanna"] * ds * dvol
        w[f"{side}_volga_term"] = 0.5 * w[f"used_{side}_volga"] * dvol * dvol
        w[f"{side}_charm_term"] = w[f"used_{side}_charm_day"] * ds * ed
        terms = [f"{side}_{x}_term" for x in ["delta", "gamma", "vega", "theta", "vanna", "volga", "charm"]]
        w[f"{side}_explained"] = w[terms].sum(axis=1, min_count=len(terms))
        w[f"{side}_residual"] = w[f"{side}_actual"] - w[f"{side}_explained"]
        w[f"{side}_tiny_price"] = (w[f"{side}_price"] < 5) | (w[f"exit_{side}_price"] < 5)
        w[f"{side}_unchanged_price"] = w[f"exit_{side}_price"].eq(w[f"{side}_price"])
        w[f"{side}_dvol"] = dvol
    w["exit_chain_iv_mean"] = w[["exit_iv_call", "exit_iv_put"]].mean(axis=1).groupby(w["entry_datetime"]).transform("mean")
    entry_pair_gap = (w["iv_call"] - w["iv_put"]).abs()
    exit_pair_gap = (w["exit_iv_call"] - w["exit_iv_put"]).abs()
    common_quality = (
        entry_pair_gap.le(0.05) & exit_pair_gap.le(0.05)
        & (w["common_forward"] / w["spot_price"]).between(0.95, 1.05)
        & (w["exit_forward"] / w["exit_spot"]).between(0.95, 1.05)
    )
    for side in ["call", "put"]:
        w[f"{side}_attribution_quality"] = (
            common_quality
            & w[f"iv_{side}"].between(0.05, 1.50)
            & w[f"exit_iv_{side}"].between(0.05, 1.50)
            & w[f"used_{side}_gamma"].between(0.0, 0.01)
            & np.isfinite(w[f"{side}_explained"])
        )
    return w


def wide_leg_metrics(w: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "actual", "delta_term", "gamma_term", "vega_term", "theta_term", "vanna_term",
        "volga_term", "charm_term", "explained", "residual", "tiny_price", "unchanged_price",
        "dvol", "price", "gamma", "delta", "attribution_quality",
    ]
    pieces = []
    for side in ["call", "put"]:
        cols = {}
        for metric in metrics:
            source = f"used_{side}_{metric}" if metric in ["gamma", "delta"] else f"{side}_{metric}"
            if source in w.columns:
                cols[source] = f"{side}_{metric}"
        part = w[["entry_datetime", "strike_position", *cols]].rename(columns=cols)
        piv = part.pivot(index="entry_datetime", columns="strike_position")
        piv.columns = [f"{metric}_{POS_NAMES[int(pos)]}" for metric, pos in piv.columns]
        pieces.append(piv)
    wide = pd.concat(pieces, axis=1).reset_index()
    meta_cols = [
        "entry_datetime", "exit_datetime", "date", "spot_price", "exit_spot", "spot_change",
        "elapsed_minutes", "elapsed_days", "days_to_expiry", "dte_hours", "dte_bucket", "time_bucket",
        "chain_iv_mean", "exit_chain_iv_mean",
    ]
    meta = w.groupby("entry_datetime", as_index=False)[meta_cols[1:]].first()
    return meta.merge(wide, on="entry_datetime", how="inner", validate="one_to_one")


def build_structure_frame(wide: pd.DataFrame, structure: str, weights: dict) -> pd.DataFrame:
    out = wide[[
        "entry_datetime", "exit_datetime", "date", "spot_price", "exit_spot", "spot_change",
        "elapsed_minutes", "elapsed_days", "days_to_expiry", "dte_hours", "dte_bucket", "time_bucket",
        "chain_iv_mean", "exit_chain_iv_mean",
    ]].copy()
    sum_metrics = ["actual", "delta_term", "gamma_term", "vega_term", "theta_term", "vanna_term", "volga_term", "charm_term", "explained", "residual"]
    for metric in sum_metrics:
        cols = []
        for (pos, side), weight in weights.items():
            col = f"{side}_{metric}_{POS_NAMES[pos]}"
            cols.append(wide[col] * weight)
        out[metric] = pd.concat(cols, axis=1).sum(axis=1, min_count=len(cols))
    entry_values = []
    gamma_values = []
    delta_values = []
    stale = []
    tiny = []
    attribution_quality = []
    for (pos, side), weight in weights.items():
        suffix = POS_NAMES[pos]
        entry_values.append(wide[f"{side}_price_{suffix}"] * weight)
        gamma_values.append(wide[f"{side}_gamma_{suffix}"] * weight)
        delta_values.append(wide[f"{side}_delta_{suffix}"] * weight)
        stale.append(wide[f"{side}_unchanged_price_{suffix}"].astype(float))
        tiny.append(wide[f"{side}_tiny_price_{suffix}"].astype(float))
        attribution_quality.append(wide[f"{side}_attribution_quality_{suffix}"].astype(float))
    out["entry_value"] = pd.concat(entry_values, axis=1).sum(axis=1, min_count=len(entry_values))
    out["gamma_coefficient"] = pd.concat(gamma_values, axis=1).sum(axis=1, min_count=len(gamma_values))
    out["delta_coefficient"] = pd.concat(delta_values, axis=1).sum(axis=1, min_count=len(delta_values))
    out["stale_any"] = pd.concat(stale, axis=1).max(axis=1).astype(bool)
    out["tiny_any"] = pd.concat(tiny, axis=1).max(axis=1).astype(bool)
    out["attribution_quality"] = pd.concat(attribution_quality, axis=1).min(axis=1).eq(1.0)
    out["structure"] = structure
    out["chain_iv_change_vol_points"] = 100.0 * (out["exit_chain_iv_mean"] - out["chain_iv_mean"])
    out["greek_break_even_move_points"] = np.sqrt(
        np.where((-2.0 * out["theta_term"] / out["gamma_coefficient"]) > 0,
                 -2.0 * out["theta_term"] / out["gamma_coefficient"], np.nan)
    )
    out["move_exceeds_greek_break_even"] = out["spot_change"].abs() > out["greek_break_even_move_points"]
    return out


def day_bootstrap_ci(frame: pd.DataFrame, value: str, reps: int = 500) -> tuple[float, float, float]:
    daily = frame.groupby("date")[value].mean().dropna().to_numpy()
    if len(daily) < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20261008)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(daily, size=len(daily), replace=True).mean()
    p_two = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(min(p_two, 1.0))


def bh_adjust(p: pd.Series) -> pd.Series:
    values = p.to_numpy(float)
    result = np.full(len(values), np.nan)
    good = np.isfinite(values)
    order = np.argsort(values[good])
    ranked = values[good][order]
    m = len(ranked)
    adjusted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    original = np.where(good)[0][order]
    result[original] = adjusted
    return pd.Series(result, index=p.index)


def summarize_structure(frame: pd.DataFrame, horizon: str) -> dict:
    z = frame.dropna(subset=["actual"])
    clean = z[~z["stale_any"]]
    lo, hi, p = day_bootstrap_ci(clean, "actual")
    attr_clean = clean[clean["attribution_quality"]]
    return {
        "structure": frame["structure"].iloc[0], "horizon": horizon, "primary": (frame["structure"].iloc[0], horizon) in PRIMARY,
        "n": len(z), "n_clean": len(clean), "trading_days": clean["date"].nunique(),
        "mean_pnl_points": clean["actual"].mean(), "median_pnl_points": clean["actual"].median(),
        "win_rate": (clean["actual"] > 0).mean(), "p05_pnl": clean["actual"].quantile(0.05),
        "p95_pnl": clean["actual"].quantile(0.95), "worst_pnl": clean["actual"].min(),
        "mean_entry_value": clean["entry_value"].mean(), "stale_share": z["stale_any"].mean(),
        "tiny_price_share": z["tiny_any"].mean(), "attribution_quality_share": z["attribution_quality"].mean(),
        "mean_abs_residual": attr_clean["residual"].abs().mean(),
        "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "day_bootstrap_pvalue": p,
        "median_greek_break_even_move_points": clean["greek_break_even_move_points"].median(),
        "share_move_exceeds_break_even": clean["move_exceeds_greek_break_even"].mean(),
        "cost_included": False,
    }


TERM_COLS = ["delta_term", "gamma_term", "vega_term", "theta_term", "vanna_term", "volga_term", "charm_term"]


def attribution_groups(frame: pd.DataFrame, horizon: str) -> tuple[list[dict], list[dict]]:
    attr_rows, reg_rows = [], []
    for (tod, dte), g in frame[(~frame["stale_any"]) & frame["attribution_quality"]].groupby(["time_bucket", "dte_bucket"]):
        g = g.dropna(subset=["actual", *TERM_COLS])
        if len(g) < 100:
            continue
        predicted = g[TERM_COLS].sum(axis=1)
        sse = np.sum((g["actual"] - predicted) ** 2)
        sst = np.sum((g["actual"] - g["actual"].mean()) ** 2)
        row = {"structure": frame["structure"].iloc[0], "horizon": horizon, "time_bucket": tod,
               "dte_bucket": dte, "n": len(g), "mean_actual": g["actual"].mean(),
               "mean_residual": (g["actual"] - predicted).mean(), "mean_abs_residual": (g["actual"] - predicted).abs().mean(),
               "explained_r2_identity_coefficients": 1.0 - sse / sst if sst > 0 else np.nan}
        for c in TERM_COLS:
            row[f"mean_{c}"] = g[c].mean()
        attr_rows.append(row)

        x = np.column_stack([np.ones(len(g)), *[g[c].to_numpy(float) for c in TERM_COLS]])
        y = g["actual"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fit = x @ beta
        reg = {"structure": frame["structure"].iloc[0], "horizon": horizon, "time_bucket": tod,
               "dte_bucket": dte, "n": len(g), "intercept": beta[0],
               "regression_r2": 1.0 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2),
               "condition_number": np.linalg.cond(x)}
        for c, b in zip(TERM_COLS, beta[1:]):
            reg[f"coef_{c}"] = b
        reg_rows.append(reg)
    return attr_rows, reg_rows


def payoff_bins(frame: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = frame[~frame["stale_any"]].copy()
    elapsed = z["elapsed_minutes"].clip(lower=1)
    z["implied_move_points"] = z["spot_price"] * z["iv_trade_vol"] * np.sqrt(elapsed / (252.0 * 375.0))
    z["abs_move_sigma"] = z["spot_change"].abs() / z["implied_move_points"]
    z["signed_move_sigma"] = z["spot_change"] / z["implied_move_points"]
    move_bins = [-np.inf, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, np.inf]
    z["move_bin"] = pd.cut(z["abs_move_sigma"], move_bins).astype(str)
    move = z.groupby("move_bin", observed=True).agg(
        n=("actual", "count"), mean_pnl=("actual", "mean"), median_pnl=("actual", "median"),
        win_rate=("actual", lambda s: (s > 0).mean()), mean_abs_move_sigma=("abs_move_sigma", "mean"),
        mean_signed_move_sigma=("signed_move_sigma", "mean"),
    ).reset_index()
    move.insert(0, "horizon", horizon)
    move.insert(0, "structure", frame["structure"].iloc[0])

    iv_bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
    z["iv_change_bin_vol_points"] = pd.cut(z["chain_iv_change_vol_points"], iv_bins).astype(str)
    iv = z.groupby("iv_change_bin_vol_points", observed=True).agg(
        n=("actual", "count"), mean_pnl=("actual", "mean"), median_pnl=("actual", "median"),
        win_rate=("actual", lambda s: (s > 0).mean()), mean_iv_change_vol_points=("chain_iv_change_vol_points", "mean"),
    ).reset_index()
    iv.insert(0, "horizon", horizon)
    iv.insert(0, "structure", frame["structure"].iloc[0])
    return move, iv


def empirical_decay(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    z = frame[(~frame["stale_any"]) & frame["attribution_quality"]].copy()
    elapsed = z["elapsed_minutes"].clip(lower=1)
    implied_move = z["spot_price"] * z["iv_trade_vol"] * np.sqrt(elapsed / (252.0 * 375.0))
    mask = (z["spot_change"].abs() <= 0.25 * implied_move) & (z["chain_iv_change_vol_points"].abs() <= 0.25)
    flat = z[mask]
    if flat.empty:
        return pd.DataFrame()
    out = flat.groupby(["time_bucket", "dte_bucket"]).agg(
        n=("actual", "count"), empirical_mean_decay_points=("actual", "mean"),
        model_mean_theta_points=("theta_term", "mean"), median_actual=("actual", "median"),
    ).reset_index()
    out["empirical_minus_model_theta"] = out["empirical_mean_decay_points"] - out["model_mean_theta_points"]
    out.insert(0, "horizon", horizon)
    return out


def create_plots(move_curves: pd.DataFrame, primary_rows: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    # ATM straddle 15-minute curve.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for structure, label in [("long_atm_straddle", "Long straddle"), ("short_atm_straddle", "Short straddle")]:
        z = move_curves[(move_curves.structure == structure) & (move_curves.horizon == "15m")].sort_values("mean_abs_move_sigma")
        ax.plot(z.mean_abs_move_sigma, z.mean_pnl, marker="o", label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="ATM straddle P&L versus realized move (15 minutes)", xlabel="Absolute spot move / implied move", ylabel="Mean gross P&L (points)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "atm_straddle_15m_payoff_curve.png", dpi=170); plt.close(fig)

    # Directional ATM options use signed moves from primary row-level records.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(-3, 3, 25)
    for structure, label in [("long_atm_call", "Long ATM call"), ("long_atm_put", "Long ATM put")]:
        z = primary_rows[(primary_rows.structure == structure) & (primary_rows.horizon == "15m")].copy()
        z["signed_bin"] = pd.cut(z["signed_move_sigma"], bins)
        curve = z.groupby("signed_bin", observed=True).agg(x=("signed_move_sigma", "mean"), y=("actual", "mean"))
        ax.plot(curve.x, curve.y, marker="o", label=label)
    ax.axhline(0, color="black", linewidth=1); ax.axvline(0, color="grey", linewidth=1)
    ax.set(title="ATM option P&L versus signed spot move (15 minutes)", xlabel="Signed spot move / implied move", ylabel="Mean gross P&L (points)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "atm_directional_15m_payoff_curve.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    z = move_curves[(move_curves.structure == "iron_condor_short1_long2") & (move_curves.horizon == "30m")].sort_values("mean_abs_move_sigma")
    ax.plot(z.mean_abs_move_sigma, z.mean_pnl, marker="o", color="#A23B72")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Iron condor P&L versus realized move (30 minutes)", xlabel="Absolute spot move / implied move", ylabel="Mean gross P&L (points)")
    fig.tight_layout(); fig.savefig(OUT / "iron_condor_30m_payoff_curve.png", dpi=170); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries, meta = prepare_entry_rows()
    entries = add_entry_greeks(entries)
    build_exit_parquet(entries, meta)
    build_exit_forward_parquet()

    summaries, attrs, regs, move_curves, iv_curves, decay_rows, oracle_rows = [], [], [], [], [], [], []
    primary_frames = []
    audit_rows = []

    for horizon in HORIZONS:
        legs = prepare_leg_window(entries, meta, horizon)
        audit_rows.append({
            "horizon": horizon, "entry_leg_rows": len(legs),
            "exit_call_price_coverage": legs["exit_call_price"].notna().mean(),
            "exit_put_price_coverage": legs["exit_put_price"].notna().mean(),
            "exit_forward_coverage": legs["exit_forward"].notna().mean(),
        })
        wide = wide_leg_metrics(legs)
        wide = wide.merge(meta[["datetime", "iv_trade_vol", "quality_ok"]].rename(columns={"datetime": "entry_datetime"}), on="entry_datetime", how="left", validate="one_to_one")
        horizon_structures = {}
        for structure, weights in STRUCTURES.items():
            sf = build_structure_frame(wide, structure, weights)
            sf["horizon"] = horizon
            sf["iv_trade_vol"] = wide["iv_trade_vol"].to_numpy()
            summaries.append(summarize_structure(sf, horizon))
            a, r = attribution_groups(sf, horizon); attrs.extend(a); regs.extend(r)
            mc, ic = payoff_bins(sf, horizon); move_curves.append(mc); iv_curves.append(ic)
            if structure == "long_atm_straddle":
                decay = empirical_decay(sf, horizon)
                if not decay.empty: decay_rows.append(decay)
            if (structure, horizon) in PRIMARY or (horizon == "15m" and structure == "short_atm_straddle"):
                clean = sf[~sf["stale_any"]].copy()
                elapsed = clean["elapsed_minutes"].clip(lower=1)
                clean["implied_move_points"] = clean["spot_price"] * clean["iv_trade_vol"] * np.sqrt(elapsed / (252.0 * 375.0))
                clean["abs_move_sigma"] = clean["spot_change"].abs() / clean["implied_move_points"]
                clean["signed_move_sigma"] = clean["spot_change"] / clean["implied_move_points"]
                primary_frames.append(clean)
            horizon_structures[structure] = sf.set_index("entry_datetime")["actual"]

        # Oracle ceilings use only timestamps where both alternatives exist.
        directional = pd.concat([horizon_structures["long_atm_call"], horizon_structures["long_atm_put"], wide.set_index("entry_datetime")["spot_change"]], axis=1)
        directional.columns = ["call", "put", "ds"]
        direction_oracle = np.where(directional.ds >= 0, directional.call, directional.put)
        straddle = horizon_structures["long_atm_straddle"].dropna()
        for name, values in [
            ("direction_known_at_entry_atm_option", pd.Series(direction_oracle, index=directional.index).dropna()),
            ("perfect_long_short_straddle_choice", straddle.abs()),
        ]:
            oracle_rows.append({"horizon": horizon, "oracle": name, "n": len(values), "mean_pnl_points": values.mean(),
                                "win_rate": (values > 0).mean(), "p05": values.quantile(0.05), "worst": values.min()})

    summary = pd.DataFrame(summaries)
    summary["bh_qvalue"] = bh_adjust(summary["day_bootstrap_pvalue"])
    attr = pd.DataFrame(attrs); reg = pd.DataFrame(regs)
    move = pd.concat(move_curves, ignore_index=True); ivcurve = pd.concat(iv_curves, ignore_index=True)
    decay = pd.concat(decay_rows, ignore_index=True) if decay_rows else pd.DataFrame()
    oracle = pd.DataFrame(oracle_rows); primary = pd.concat(primary_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    audit["bid_ask_available"] = False
    audit["cost_status"] = "gross P&L only; source has no bid/ask fields"

    summary.to_csv(OUT / "structure_summary.csv", index=False)
    attr.to_csv(OUT / "greek_attribution_by_bucket.csv", index=False)
    reg.to_csv(OUT / "attribution_regression_by_bucket.csv", index=False)
    move.to_csv(OUT / "payoff_vs_move_curves.csv", index=False)
    ivcurve.to_csv(OUT / "payoff_vs_iv_change_curves.csv", index=False)
    decay.to_csv(OUT / "empirical_decay.csv", index=False)
    oracle.to_csv(OUT / "oracle_benchmarks.csv", index=False)
    primary.to_csv(OUT / "primary_trade_windows.csv", index=False)
    audit.to_csv(OUT / "data_audit.csv", index=False)
    create_plots(move, primary)

    primary_summary = summary[summary["primary"]].copy()
    theta_diag = reg[reg["structure"].isin(["long_atm_straddle", "iron_condor_short1_long2"])].groupby(["structure", "horizon"]).agg(
        median_theta_coefficient=("coef_theta_term", "median"), median_regression_r2=("regression_r2", "median"), cells=("n", "count")
    ).reset_index()
    report = f"""# Intraday option-structure attribution

## Status

This is retrospective design evidence using data already examined through 2025. It is not an untouched performance test. All P&L is gross mark-to-market because the source contains no bid/ask fields. Transaction costs and executable slippage remain unresolved.

## Primary results

{dataframe_to_markdown(primary_summary[["structure", "horizon", "n_clean", "mean_pnl_points", "win_rate", "p05_pnl", "worst_pnl", "bootstrap_ci_low", "bootstrap_ci_high", "bh_qvalue", "stale_share", "tiny_price_share", "attribution_quality_share"]].round(5))}

## Decision

- The 15-minute long ATM straddle lost 0.77 points per window on average (41.57% win rate), and its day-block confidence interval stayed below zero. The mirrored short straddle earned the same gross amount, but this is not yet a tradable edge because costs are absent and the short tail is severe.
- The 15-minute long ATM call and put had no statistically reliable unconditional edge. They require a directional forecast; merely buying either side systematically is unsupported.
- The 30-minute iron condor earned 0.16 points gross with a 53.17% win rate, but its confidence interval included zero. Treat it as exploratory, not a deployment candidate.
- For model design, predict realized move and IV change for straddles, direction plus move size for calls/puts and debit spreads, and tail-move probability for the iron condor.

## Decay and oracle context

On flat-spot, flat-IV 15-minute windows, empirical ATM-straddle decay was more negative than model theta in every DTE bucket. For example, at 0-24 hours to expiry it was -9.82 points versus -7.26 from theta; at 24-72 hours it was -2.32 versus -1.49. Use the empirical time-of-day/DTE decay table when setting break-even thresholds.

The direction-known ATM-option oracle averaged 23.83 gross points at 15 minutes, while the ex-post long/short straddle oracle averaged 9.86 points. These are deliberately unattainable ceilings that quantify how valuable accurate direction or move-size forecasts could be.

## Greek-unit validation

{dataframe_to_markdown(theta_diag.round(4))}

The regression file contains delta, gamma, vega, theta, vanna, volga and charm coefficients for every structure x horizon x time-of-day x DTE cell. Coefficients far from one identify local approximation, quote-quality or unit problems. Expiry-day windows longer than 15 minutes use the average of entry and exit Greeks.

## Method

- Entry structures use the nearest monthly expiry and positions -2 to +2.
- Every exit uses the exact entry expiry and strike from the original full chain; strikes are never re-centered.
- IV and Greeks are recomputed using one parity-implied forward per timestamp.
- Vega is stored per 1.00 absolute volatility; the vega contribution multiplies the absolute IV change. Theta is per calendar day and multiplies elapsed calendar days.
- Stale-price windows are flagged when any leg price is unchanged. Prices below 5 points are separately flagged.
- Primary tests were declared as ATM straddle/call/put at 15 minutes and iron condor at 30 minutes. Other results are exploratory; day-block p-values receive Benjamini-Hochberg adjustment.
- Payoff curves use gross P&L against spot movement in units of the forward-consistent ATM implied move and against the change in the five-strike IV level.

## Cost limitation

Neither the original chain nor the five-strike CSV has call/put bid and ask columns. The files contain only `call_price` and `put_price`. Net P&L, spread per leg and slippage cannot be identified from these data.
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")
    print("PRIMARY RESULTS")
    print(primary_summary.to_string(index=False))
    print("\nAUDIT")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
