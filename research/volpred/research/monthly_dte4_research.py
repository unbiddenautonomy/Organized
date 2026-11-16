from __future__ import annotations

"""Isolated research backtest for monthly BANKNIFTY options with DTE >= 4.

This script deliberately does not modify any incumbent model or artifact.  It
builds one entry per day, holds the entry ATM straddle to the same session's
last quote, and evaluates paper-derived Greek P&L forecasts in rolling monthly
expiry-cycle folds.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from economic_evaluation import charge_components, performance, write_parquet


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "monthly_dte4_research"
FEATURES = ROOT / "outputs" / "spot_magnitude_models" / "spot_magnitude_feature_panel.parquet"
IV_FEATURES = ROOT / "outputs" / "log_iv_change_models" / "log_iv_feature_panel.parquet"
EXITS = ROOT / "outputs" / "intraday_structure_attribution" / "_fixed_strike_exit_prices.parquet"
SPOT = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
SEED = 7319
PRIMARY_BPS = 25.0
MIN_TRAIN_CYCLES = 24
VALIDATION_CYCLES = 6
ALPHAS = (0.1, 1.0, 10.0, 100.0)
EPS = 1e-12


@dataclass(frozen=True)
class Candidate:
    candidate: str
    formula: str
    direction: str
    hurdle: str = "predicted rupee edge > entry-time estimated round-trip cost"


CANDIDATES = (
    Candidate(
        "vrp_gamma_theta_short",
        "0.5*Gamma*S^2*forecast_variance + Theta*elapsed_days; enter short only when this expected long P&L is sufficiently negative",
        "short_only",
    ),
    Candidate(
        "iv_change_vega_both",
        "Vega*forecast_change_in_decimal_IV",
        "long_or_short",
    ),
    Candidate(
        "combined_greeks_short",
        "0.5*Gamma*S^2*forecast_variance + Theta*elapsed_days + Vega*forecast_change_in_decimal_IV",
        "short_only",
    ),
    Candidate(
        "combined_greeks_both",
        "0.5*Gamma*S^2*forecast_variance + Theta*elapsed_days + Vega*forecast_change_in_decimal_IV",
        "long_or_short",
    ),
    Candidate(
        "always_short_benchmark",
        "unconditional short ATM straddle from the daily entry to session end",
        "benchmark",
        "none",
    ),
)


def load_daily_panel() -> tuple[pd.DataFrame, dict[str, float]]:
    con = duckdb.connect()
    query = """
    WITH eligible AS (
        SELECT *, row_number() OVER (PARTITION BY date ORDER BY datetime) AS rn
        FROM read_parquet(?)
        WHERE quality_ok
          AND strike_position = 0
          AND days_to_expiry >= 4
          AND days_to_expiry <= 45
          AND minute_of_day >= 585
          AND minute_of_day < 600
    ), entries AS (
        SELECT * EXCLUDE (rn) FROM eligible WHERE rn = 1
    ), exits AS (
        SELECT source_index, exit_datetime, exit_call_price, exit_put_price
        FROM read_parquet(?)
        WHERE horizon = 'session_end'
    ), end_iv AS (
        SELECT date, datetime AS end_iv_datetime, iv_atm AS end_iv_atm,
               row_number() OVER (PARTITION BY date ORDER BY datetime DESC) AS rn
        FROM read_parquet(?)
        WHERE iv_atm > 0
    )
    SELECT e.*, x.exit_datetime, x.exit_call_price, x.exit_put_price,
           i.end_iv_datetime, i.end_iv_atm,
           v.log_iv_atm, v.iv_atm_zscore, v.vol_of_vol_log_30
    FROM entries e
    LEFT JOIN exits x USING (source_index)
    LEFT JOIN end_iv i ON e.date = i.date AND i.rn = 1
    LEFT JOIN read_parquet(?) v ON e.datetime = v.datetime
    ORDER BY e.datetime
    """
    frame = con.execute(
        query, [str(FEATURES), str(EXITS), str(IV_FEATURES), str(IV_FEATURES)]
    ).fetchdf()
    con.close()

    raw_rows = len(frame)
    exit_missing = frame["exit_call_price"].isna() | frame["exit_put_price"].isna()
    end_iv_missing = frame["end_iv_atm"].isna()
    same_day = frame["exit_datetime"].dt.normalize().eq(frame["datetime"].dt.normalize())
    exit_minute = frame["exit_datetime"].dt.hour * 60 + frame["exit_datetime"].dt.minute
    close_exit = same_day & exit_minute.ge(15 * 60 + 15)
    audit = {
        "eligible_daily_entries": raw_rows,
        "missing_session_end_fixed_strike_exit": int(exit_missing.sum()),
        "missing_session_end_iv": int(end_iv_missing.sum()),
        "exit_coverage": float((~exit_missing).mean()),
        "end_iv_coverage": float((~end_iv_missing).mean()),
        "same_day_exit_violations": int((~same_day & ~exit_missing).sum()),
        "available_exit_before_1515": int((~close_exit & ~exit_missing).sum()),
        "close_exit_coverage": float((close_exit & ~exit_missing).mean()),
    }
    # Intraday-only contract: never carry overnight, and do not relabel an
    # incomplete historical session's last midday quote as a closing exit.
    frame = frame[~exit_missing & ~end_iv_missing & close_exit].copy()
    frame["entry_premium"] = frame["call_price"] + frame["put_price"]
    frame["exit_premium"] = frame["exit_call_price"] + frame["exit_put_price"]
    frame["long_pnl_points"] = frame["exit_premium"] - frame["entry_premium"]
    frame["elapsed_days"] = (
        frame["exit_datetime"] - frame["datetime"]
    ).dt.total_seconds() / 86400.0
    frame.loc[~frame["iv_atm"].gt(0), "iv_atm"] = np.nan
    frame["target_dlog_iv"] = np.log(frame["end_iv_atm"] / frame["iv_atm"])
    frame["expiry_cycle"] = pd.to_datetime(frame["expiry"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["lot_size"] = np.where(frame["date"] <= pd.Timestamp("2024-12-24"), 15, 30)

    spot = pd.read_csv(SPOT, usecols=["datetime", "spot_price"], parse_dates=["datetime"])
    spot = spot.sort_values("datetime")
    spot["date"] = spot["datetime"].dt.normalize()
    spot["spot_price"] = spot["spot_price"].where(spot["spot_price"].gt(0))
    spot["return"] = spot.groupby("date")["spot_price"].transform(lambda s: np.log(s).diff())
    entry_map = frame.set_index("date")[["datetime", "exit_datetime"]]
    spot = spot.merge(entry_map, left_on="date", right_index=True, how="inner", validate="many_to_one")
    spot = spot[(spot["datetime_x"] > spot["datetime_y"]) & (spot["datetime_x"] <= spot["exit_datetime"])]
    future = spot.groupby("date").agg(
        future_session_variance=("return", lambda x: float(np.nansum(x.to_numpy(float) ** 2))),
        session_end_spot=("spot_price", "last"),
    )
    frame = frame.merge(future, on="date", how="left", validate="one_to_one")
    frame = frame[frame["future_session_variance"].gt(0)].copy()
    frame["log_future_session_variance"] = np.log(frame["future_session_variance"].clip(lower=EPS))
    frame["target_session_return"] = np.log(frame["session_end_spot"] / frame["spot_price"])
    frame["dte_bucket"] = pd.cut(
        frame["days_to_expiry"], [4, 8, 15, 22, 46], right=False,
        labels=["4-7", "8-14", "15-21", "22-45"],
    ).astype(str)
    return frame.sort_values("datetime").reset_index(drop=True), audit


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    ok = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0) & (predicted > 0)
    if not ok.any():
        return np.inf
    ratio = np.clip(actual[ok], EPS, None) / np.clip(predicted[ok], EPS, None)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
) -> tuple[np.ndarray, float]:
    cols = features + [target]
    tr = train.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    sc = score.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    pred = np.full(len(score), np.nan)
    if len(tr) < 100 or sc.empty:
        return pred, np.nan
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(tr[features], tr[target])
    pred[score.index.get_indexer(sc.index)] = model.predict(sc[features])
    resid_var = float(np.var(tr[target].to_numpy() - model.predict(tr[features]), ddof=1))
    return pred, resid_var


def choose_models(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rv_sets = {
        "HAR": ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60"],
        "HAR_IV": ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60", "log_iv_atm"],
    }
    iv_features = ["days_to_expiry", "log_iv_atm", "iv_atm_zscore", "overnight_return", "vol_of_vol_log_30"]
    rows: list[dict] = []
    for name, features in rv_sets.items():
        for alpha in ALPHAS:
            log_pred, resid_var = fit_predict(train, validation, features, "log_future_session_variance", alpha)
            level = np.exp(np.clip(log_pred + 0.5 * resid_var, -30.0, -1.0))
            ok = np.isfinite(level) & validation["future_session_variance"].notna().to_numpy()
            rows.append({"model_type": "rv", "model": name, "alpha": alpha,
                         "validation_loss": qlike(validation.loc[ok, "future_session_variance"].to_numpy(), level[ok]) if ok.any() else np.inf})
    for alpha in ALPHAS:
        pred, _ = fit_predict(train, validation, iv_features, "target_dlog_iv", alpha)
        ok = np.isfinite(pred) & validation["target_dlog_iv"].notna().to_numpy()
        loss = float(np.mean(np.abs(validation.loc[ok, "target_dlog_iv"].to_numpy() - pred[ok]))) if ok.any() else np.inf
        rows.append({"model_type": "iv", "model": "compact_IV", "alpha": alpha, "validation_loss": loss})
    tuning = pd.DataFrame(rows)
    chosen = {}
    for kind in ("rv", "iv"):
        best = tuning[tuning["model_type"].eq(kind)].sort_values(["validation_loss", "model", "alpha"]).iloc[0]
        chosen[kind] = best.to_dict()
    return chosen, tuning


def walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = np.array(sorted(frame["expiry_cycle"].dropna().unique()))
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    rv_sets = {
        "HAR": ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60"],
        "HAR_IV": ["log_rv_past_5", "log_rv_past_15", "log_rv_past_60", "log_iv_atm"],
    }
    iv_features = ["days_to_expiry", "log_iv_atm", "iv_atm_zscore", "overnight_return", "vol_of_vol_log_30"]
    for pos in range(MIN_TRAIN_CYCLES + VALIDATION_CYCLES, len(cycles)):
        test_cycle = cycles[pos]
        validation_cycles = cycles[pos - VALIDATION_CYCLES:pos]
        train_cycles = cycles[:pos - VALIDATION_CYCLES]
        train = frame[frame["expiry_cycle"].isin(train_cycles)].copy()
        validation = frame[frame["expiry_cycle"].isin(validation_cycles)].copy()
        test = frame[frame["expiry_cycle"].eq(test_cycle)].copy()
        if train["expiry_cycle"].nunique() < MIN_TRAIN_CYCLES or test.empty:
            continue
        chosen, _ = choose_models(train, validation)
        refit = pd.concat([train, validation], ignore_index=True)
        rv_name, rv_alpha = chosen["rv"]["model"], float(chosen["rv"]["alpha"])
        log_rv, resid_var = fit_predict(refit, test, rv_sets[rv_name], "log_future_session_variance", rv_alpha)
        dlog_iv, _ = fit_predict(refit, test, iv_features, "target_dlog_iv", float(chosen["iv"]["alpha"]))
        out = test.copy()
        out["pred_future_variance"] = np.exp(np.clip(log_rv + 0.5 * resid_var, -30.0, -1.0))
        out["pred_dlog_iv"] = dlog_iv
        out["pred_d_iv"] = out["iv_atm"] * np.expm1(out["pred_dlog_iv"])
        out["outer_test_cycle"] = pd.Timestamp(test_cycle)
        out["selected_rv_model"] = rv_name
        out["selected_rv_alpha"] = rv_alpha
        out["selected_iv_alpha"] = float(chosen["iv"]["alpha"])
        predictions.append(out)
        fold_rows.append({
            "test_cycle": pd.Timestamp(test_cycle),
            "train_start": train["datetime"].min(), "train_end": train["datetime"].max(),
            "validation_start": validation["datetime"].min(), "validation_end": validation["datetime"].max(),
            "test_start": test["datetime"].min(), "test_end": test["datetime"].max(),
            "train_cycles": train["expiry_cycle"].nunique(), "validation_cycles": validation["expiry_cycle"].nunique(),
            "test_rows": len(test), "selected_rv_model": rv_name, "selected_rv_alpha": rv_alpha,
            "selected_iv_alpha": float(chosen["iv"]["alpha"]),
        })
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows)


def estimated_cost(frame: pd.DataFrame, action: np.ndarray, bps: float) -> np.ndarray:
    return charge_components(
        frame["entry_premium"].to_numpy(), frame["entry_premium"].to_numpy(), action,
        frame["lot_size"].to_numpy(), frame["date"], slippage_bps=bps, current_2026=False,
    )["total_cost"].to_numpy()


def build_trades(predictions: pd.DataFrame) -> pd.DataFrame:
    z = predictions.dropna(subset=["pred_future_variance", "pred_d_iv"]).copy()
    gamma_theta = (
        0.5 * z["gamma_atm"] * z["spot_price"].pow(2) * z["pred_future_variance"]
        + z["theta_atm"] * z["elapsed_days"]
    )
    vega = z["vega_atm"] * z["pred_d_iv"]
    scores = {
        "vrp_gamma_theta_short": gamma_theta,
        "iv_change_vega_both": vega,
        "combined_greeks_short": gamma_theta + vega,
        "combined_greeks_both": gamma_theta + vega,
    }
    trade_frames = []
    long_cost = estimated_cost(z, np.ones(len(z), int), PRIMARY_BPS)
    short_cost = estimated_cost(z, -np.ones(len(z), int), PRIMARY_BPS)
    for spec in CANDIDATES:
        if spec.direction == "benchmark":
            expected_rupees = np.full(len(z), np.nan)
            action = np.full(len(z), -1, dtype=int)
        else:
            expected_rupees = scores[spec.candidate].to_numpy(float) * z["lot_size"].to_numpy(float)
            action = np.where(expected_rupees > long_cost, 1, np.where(expected_rupees < -short_cost, -1, 0))
            if spec.direction == "short_only":
                action[action > 0] = 0
        t = z[action != 0].copy()
        t["action"] = action[action != 0]
        t["candidate"] = spec.candidate
        t["expected_pnl_rupees"] = expected_rupees[action != 0]
        t["selection_cost_rupees"] = np.where(action[action != 0] > 0, long_cost[action != 0], short_cost[action != 0])
        trade_frames.append(t)
    return pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()


def attach_costs(trades: pd.DataFrame, bps: float) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    c = charge_components(
        trades["entry_premium"].to_numpy(), trades["exit_premium"].to_numpy(), trades["action"].to_numpy(),
        trades["lot_size"].to_numpy(), trades["date"], slippage_bps=bps, current_2026=False,
    )
    out = trades.copy().reset_index(drop=True)
    for col in c.columns:
        out[col] = c[col].to_numpy()
    out["gross_pnl_rupees"] = out["action"] * out["long_pnl_points"] * out["lot_size"]
    out["net_pnl_rupees"] = out["gross_pnl_rupees"] - out["total_cost"]
    out["cost_bps"] = bps
    return out


def cycle_inference(trades: pd.DataFrame, n_boot: int = 5000) -> dict[str, float]:
    cycle = trades.groupby("expiry_cycle")["net_pnl_rupees"].sum().sort_index()
    n = len(cycle)
    if n < 5 or cycle.std(ddof=1) <= 0:
        return {"cycles": n, "mean_cycle_net": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "one_sided_p": np.nan, "mde_80pct": np.nan}
    x = cycle.to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    t_stat = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    return {
        "cycles": n, "mean_cycle_net": x.mean(),
        "ci_low": np.percentile(draws, 2.5), "ci_high": np.percentile(draws, 97.5),
        "one_sided_p": float(student_t.sf(t_stat, df=n - 1)),
        "mde_80pct": float((norm.ppf(.95) + norm.ppf(.80)) * x.std(ddof=1) / np.sqrt(n)),
    }


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    valid = pvalues.dropna().sort_values()
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    running = 0.0
    m = len(valid)
    for rank, (idx, value) in enumerate(valid.items()):
        running = max(running, min(1.0, (m - rank) * value))
        out.loc[idx] = running
    return out


def summarize(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf_rows, inference_rows = [], []
    all_costed = []
    for bps in (10.0, 25.0, 50.0):
        costed = attach_costs(trades, bps)
        all_costed.append(costed)
        for spec in CANDIDATES:
            group = costed[costed["candidate"].eq(spec.candidate)]
            metrics = performance(group)
            if group.empty:
                metrics.update({"trades": 0, "days": 0, "total_gross": 0.0, "total_cost": 0.0,
                                "total_net": 0.0, "hit_rate": np.nan})
            perf_rows.append({"candidate": spec.candidate, "cost_bps": bps, **metrics})
            inference_rows.append({"candidate": spec.candidate, "cost_bps": bps, **cycle_inference(group)})
    perf = pd.DataFrame(perf_rows)
    inference = pd.DataFrame(inference_rows)
    primary = inference["cost_bps"].eq(PRIMARY_BPS)
    inference.loc[primary, "holm_p"] = holm_adjust(inference.loc[primary, "one_sided_p"])
    all_trades = pd.concat(all_costed, ignore_index=True)
    stability = all_trades.groupby(["candidate", "cost_bps", "dte_bucket", all_trades["datetime"].dt.year.rename("year")], observed=True).agg(
        trades=("net_pnl_rupees", "size"), days=("date", "nunique"), cycles=("expiry_cycle", "nunique"),
        total_net=("net_pnl_rupees", "sum"), mean_net=("net_pnl_rupees", "mean"),
        hit_rate=("net_pnl_rupees", lambda x: float((x > 0).mean())),
    ).reset_index()
    return perf, inference, stability


def report(frame: pd.DataFrame, audit: dict, folds: pd.DataFrame, perf: pd.DataFrame, inference: pd.DataFrame) -> str:
    primary = perf[perf["cost_bps"].eq(PRIMARY_BPS)].merge(
        inference[inference["cost_bps"].eq(PRIMARY_BPS)], on=["candidate", "cost_bps"], how="left"
    )
    stress = perf.pivot(index="candidate", columns="cost_bps", values="total_net").reset_index()
    tested = primary[~primary["candidate"].eq("always_short_benchmark")].merge(
        stress[["candidate", 50.0]], on="candidate", how="left"
    )
    passed = tested[
        tested["cycles"].ge(30) & tested["ci_low"].gt(0) & tested["holm_p"].lt(.05)
        & tested[50.0].gt(0)
    ]
    decision = (
        "Promotable shadow candidates: " + ", ".join(passed["candidate"].tolist())
        if len(passed) else
        "No candidate passes the predeclared promotion gate. Keep every DTE >= 4 challenger out of production."
    )
    return "\n".join([
        "# Monthly DTE >= 4 session-end signal research",
        "",
        "The incumbent models and artifacts were not modified. This is an isolated retrospective shadow study.",
        "",
        "## Research status",
        "",
        f"**Decision:** {decision}",
        "",
        "No finite historical backtest mathematically proves future profitability. The pricing identities motivate the hypotheses; the rolling expiry-cycle results test them on the available data.",
        "",
        "## Research basis and equations",
        "",
        "Carr and Wu (2009) define the variance risk premium from the difference between option-replicated risk-neutral variance and subsequently realized variance. Bakshi and Kapadia (2003) connect volatility-risk-premium sign and magnitude to delta-hedged option gains. Corsi (2009) motivates the heterogeneous autoregressive realized-volatility forecast. Sankar, Ramachandran and Lukose (2020) provide India-specific evidence that variance risk is priced and that lagged continuous variance, rather than realized jumps, predicts short-term synthetic variance-swap returns.",
        "",
        "For the available ATM data, the causal session-end approximation tested here is:",
        "",
        "`E[long straddle P&L] = 0.5*Gamma*S^2*forecast_variance + Theta*elapsed_days + Vega*forecast_delta_IV`.",
        "",
        "An action is allowed only when the forecast rupee edge exceeds an entry-time estimate of the complete round-trip cost. This is a Greek/ATM proxy; exact model-free VRP requires a liquid full OTM strike strip and synchronized executable quotes.",
        "",
        "Primary sources: Carr & Wu, https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1359527 ; Bakshi & Kapadia, https://doi.org/10.1093/rfs/hhg002 ; Corsi, https://doi.org/10.1093/jjfinec/nbp001 ; Sankar et al., https://doi.org/10.1016/j.iref.2020.06.010 .",
        "",
        f"Eligible daily entries: {audit['eligible_daily_entries']}; any same-day fixed-strike exit coverage: {audit['exit_coverage']:.2%}; verified >=15:15 close-exit coverage: {audit['close_exit_coverage']:.2%}; session-end IV coverage: {audit['end_iv_coverage']:.2%}.",
        f"Outer expiry-cycle folds: {len(folds)}; first test cycle: {folds.test_cycle.min() if len(folds) else 'NA'}; last: {folds.test_cycle.max() if len(folds) else 'NA'}.",
        "",
        "## Primary 25 bps results",
        "",
        primary.to_string(index=False),
        "",
        "## Cost sensitivity: total net rupees",
        "",
        stress.to_string(index=False),
        "",
        "## Interpretation rules",
        "",
        "A candidate is not promotable unless its 25 bps cycle-bootstrap lower confidence bound is positive, its Holm-adjusted one-sided p-value is below 0.05, it remains profitable at 50 bps, and at least 30 independent outer expiry cycles are available.",
        "",
        "## Limitations",
        "",
        "- Historical bid/ask and volume are absent; 10/25/50 bps are modeled execution stresses.",
        "- The source ends in May 2025 and all historical periods have influenced earlier research; these are nested retrospective results, not an untouched holdout.",
        "- The engineered panel has nearest-monthly ATM/five-strike data, not a full executable option strip. This study therefore tests a Greek/ATM-IV proxy, not a model-free variance-swap replication.",
        "- Missing session-end labels are disclosed in data_audit.csv; label availability is a possible survivorship bias.",
        "- Every evaluated trade exits on its entry date at or after 15:15. There is no overnight carry.",
        "- Same-day session-end holding reduces the two-hour cost mismatch but does not test multi-day or expiry-held positions.",
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(x) for x in CANDIDATES]).to_csv(OUT / "candidate_registry.csv", index=False)
    frame, audit = load_daily_panel()
    pd.DataFrame([audit]).to_csv(OUT / "data_audit.csv", index=False)
    predictions, folds = walk_forward(frame)
    folds.to_csv(OUT / "fold_manifest.csv", index=False)
    write_parquet(predictions, OUT / "oos_predictions.parquet")
    selected = build_trades(predictions)
    write_parquet(selected, OUT / "selected_trades_before_costs.parquet")
    primary_ledger = attach_costs(selected, PRIMARY_BPS)
    primary_ledger = primary_ledger[primary_ledger["candidate"].eq("combined_greeks_short")].copy()
    primary_ledger["profitable"] = primary_ledger["net_pnl_rupees"].gt(0)
    primary_ledger[[
        "date", "datetime", "exit_datetime", "expiry", "days_to_expiry", "dte_bucket",
        "entry_premium", "exit_premium", "expected_pnl_rupees", "gross_pnl_rupees",
        "total_cost", "net_pnl_rupees", "profitable",
    ]].to_csv(OUT / "signal_day_results_25bps.csv", index=False)
    perf, inference, stability = summarize(selected)
    perf.to_csv(OUT / "performance.csv", index=False)
    inference.to_csv(OUT / "cycle_inference.csv", index=False)
    stability.to_csv(OUT / "stability.csv", index=False)
    (OUT / "analysis_report.md").write_text(report(frame, audit, folds, perf, inference), encoding="utf-8")
    print((OUT / "analysis_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
