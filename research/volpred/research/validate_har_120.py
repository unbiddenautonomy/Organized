"""Fixed audit of the deployed 120-minute HAR forecast.

This script does not tune or select a replacement model.  It uses the existing
walk-forward forecasts, applies calibration/coverage checks to HAR, and runs
one predeclared comparison of the 50/50 HAR/Enhanced forecast against HAR.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from economic_evaluation import PRIMARY_SLIPPAGE_BPS, attach_pnl, performance
from enhanced_rv_model import EPS, qlike
from validate_120_rv_model import dm_hac, hac_regression, mean_hac_test


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs" / "enhanced_rv_models" / "validation_120" / "oos_120m.parquet"
TRADE_PANEL = ROOT / "outputs" / "economic_evaluation" / "economic_trade_panel.parquet"
OUT = ROOT / "outputs" / "har_120_validation"


def read_parquet(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    result = con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
    con.close()
    return result


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute("COPY frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    con.close()


def mz_test(frame: pd.DataFrame) -> dict:
    z = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual", "har"])
    y = np.log(z["actual"].to_numpy(float).clip(min=EPS))
    p = np.log(z["har"].to_numpy(float).clip(min=EPS))
    x = np.column_stack([np.ones(len(z)), p])
    beta, cov = hac_regression(y, x)
    delta = beta - np.array([0.0, 1.0])
    wald = float(delta @ np.linalg.pinv(cov) @ delta)
    ratio = z["actual"].to_numpy(float) / z["har"].to_numpy(float)
    level = mean_hac_test(ratio, null=1.0, alternative="two-sided")
    return {
        "n": len(z), "intercept": float(beta[0]), "slope": float(beta[1]),
        "wald": wald, "mz_pvalue": float(stats.chi2.sf(wald, 2)),
        "mean_actual_over_forecast": float(level["mean"]),
        "ratio_pvalue": float(level["pvalue"]),
    }


def coverage_test(frame: pd.DataFrame) -> dict:
    hits = (frame["actual"].to_numpy(float) > frame["har_q90"].to_numpy(float)).astype(int)
    n = len(hits); x = int(hits.sum()); expected = 0.10
    phat = x / n if n else np.nan
    if not n:
        return {"n": 0}
    ll0 = x * np.log(expected) + (n - x) * np.log(1 - expected)
    ll1 = x * np.log(max(phat, EPS)) + (n - x) * np.log(max(1 - phat, EPS))
    lr_uc = float(-2 * (ll0 - ll1))
    counts = np.zeros((2, 2), dtype=int)
    for before, after in zip(hits[:-1], hits[1:]):
        counts[before, after] += 1
    p01 = counts[0, 1] / max(counts[0].sum(), 1)
    p11 = counts[1, 1] / max(counts[1].sum(), 1)
    pall = counts[:, 1].sum() / max(counts.sum(), 1)

    def llterm(count: int, probability: float) -> float:
        return count * np.log(max(probability, EPS))

    ll_ind0 = llterm(int(counts[:, 1].sum()), pall) + llterm(int(counts[:, 0].sum()), 1 - pall)
    ll_ind1 = (
        llterm(counts[0, 1], p01) + llterm(counts[0, 0], 1 - p01)
        + llterm(counts[1, 1], p11) + llterm(counts[1, 0], 1 - p11)
    )
    lr_ind = float(-2 * (ll_ind0 - ll_ind1))
    return {
        "n": n, "exceedances": x, "hit_rate": float(phat),
        "kupiec_lr": lr_uc, "kupiec_pvalue": float(stats.chi2.sf(lr_uc, 1)),
        "transition_counts": counts.tolist(), "christoffersen_lr": lr_ind,
        "christoffersen_pvalue": float(stats.chi2.sf(lr_ind, 1)),
        "conditional_coverage_pvalue": float(stats.chi2.sf(lr_uc + lr_ind, 2)),
    }


def calibration_tables(grid: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    primary = grid[grid["datetime"].dt.strftime("%H:%M").eq("11:15")].copy()
    overall = mz_test(primary)
    overall["timestamp"] = "11:15"
    rows = []
    for bucket, part in grid.groupby("time_bucket", sort=True):
        result = mz_test(part)
        result["time_bucket"] = int(bucket)
        result["start_time"] = part["datetime"].dt.strftime("%H:%M").min()
        result["end_time"] = part["datetime"].dt.strftime("%H:%M").max()
        rows.append(result)
    return overall, pd.DataFrame(rows)


def quarterly_qlike(grid: pd.DataFrame) -> pd.DataFrame:
    z = grid.copy()
    z["quarter"] = z["datetime"].dt.to_period("Q").astype(str)
    z["har_qlike"] = qlike(z["actual"].to_numpy(), z["har"].to_numpy())
    return z.groupby("quarter", as_index=False).agg(
        days=("date", "nunique"), observations=("date", "size"),
        har_qlike=("har_qlike", "mean"), mean_actual=("actual", "mean"),
        mean_forecast=("har", "mean"),
    )


def combination_test(grid: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    z = grid.copy()
    z["combination"] = 0.5 * z["har"] + 0.5 * z["enhanced"]
    z["har_loss"] = qlike(z["actual"].to_numpy(), z["har"].to_numpy())
    z["combination_loss"] = qlike(z["actual"].to_numpy(), z["combination"].to_numpy())
    daily = z.groupby("date", as_index=False).agg(
        har_loss=("har_loss", "mean"), combination_loss=("combination_loss", "mean")
    )
    daily["D"] = daily["har_loss"] - daily["combination_loss"]
    test = dm_hac(daily["D"].to_numpy())
    test.update({
        "days": len(daily), "weight_enhanced": 0.5,
        "har_qlike": float(daily["har_loss"].mean()),
        "combination_qlike": float(daily["combination_loss"].mean()),
        "improvement_ratio": float(daily["har_loss"].mean() / daily["combination_loss"].mean() - 1),
        "decision_alpha_0_05": "PASS" if test["pvalue"] < 0.05 else "DO_NOT_REPLACE_HAR",
    })
    return test, daily


def signal_backtest(pred: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    trades = read_parquet(TRADE_PANEL)
    trades = trades[trades["horizon_min"].eq(120)].copy()
    trades["join_minute"] = trades["datetime"].dt.floor("min")
    forecast = pred[["datetime", "har", "har_q90"]].rename(columns={"datetime": "join_minute"})
    joined = trades.merge(forecast, on="join_minute", how="inner", validate="many_to_one")
    minute_from_open = (joined["join_minute"].dt.hour * 60 + joined["join_minute"].dt.minute) - 555
    fixed = joined[minute_from_open.mod(120).eq(60)].sort_values("datetime").copy()
    fixed["variance_spread"] = fixed["iv_implied_total_variance"] - fixed["har_q90"]
    selected = fixed[fixed["variance_spread"].gt(0)].copy()
    selected["action"] = -1
    # The fixed grid makes the 120-minute holding windows non-overlapping.
    scored = attach_pnl(selected, PRIMARY_SLIPPAGE_BPS)
    result = performance(scored)
    result.update({
        "rule": "SHORT when same-window implied variance > HAR q90",
        "grid": "10:15 and 12:15 (120-minute non-overlapping)",
        "eligible_rows": len(fixed), "signal_rows": len(selected),
        "slippage_bps": PRIMARY_SLIPPAGE_BPS,
        "sample_start": str(fixed["date"].min().date()) if len(fixed) else None,
        "sample_end": str(fixed["date"].max().date()) if len(fixed) else None,
    })
    return result, scored


def live_status() -> dict:
    path = ROOT / "runtime" / "predictions.jsonl"
    if not path.exists():
        return {"status": "no_runtime_file", "mature_120m_records": 0}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            forecast = value.get("forecasts", {}).get("120", {})
            if forecast.get("available"):
                rows.append({"bar_timestamp": value.get("bar_timestamp"), **forecast})
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    if not rows:
        return {"status": "no_120m_records", "mature_120m_records": 0}
    frame = pd.DataFrame(rows)
    stamp_col = "bar_timestamp"
    frame[stamp_col] = pd.to_datetime(frame[stamp_col], errors="coerce")
    timezone = frame[stamp_col].dt.tz
    cutoff = pd.Timestamp.now(tz=timezone) - pd.Timedelta(minutes=120)
    mature = frame[frame[stamp_col].le(cutoff)]
    return {
        "status": "insufficient_for_2026_calibration" if len(mature) < 30 else "ready_to_score",
        "records": len(frame), "mature_120m_records": len(mature),
        "first_prediction": str(frame[stamp_col].min()), "last_prediction": str(frame[stamp_col].max()),
        "note": "A 2026 MZ test needs matched realized 120-minute variance and a meaningful daily sample.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = read_parquet(PREDICTIONS).sort_values("datetime")
    grid = pred[pred["datetime"].dt.minute.mod(15).eq(0)].copy()
    overall, bucket = calibration_tables(grid)

    minute_from_open = (pred["datetime"].dt.hour * 60 + pred["datetime"].dt.minute) - 555
    every_120 = pred[minute_from_open.mod(120).eq(60)].copy()
    once_daily = pred[pred["datetime"].dt.strftime("%H:%M").eq("11:15")].copy()
    coverage = {
        "every_120_minutes": coverage_test(every_120),
        "once_per_day_1115": coverage_test(once_daily),
    }
    quarters = quarterly_qlike(grid)
    combination, combination_daily = combination_test(grid)
    pnl, trades = signal_backtest(pred)
    results = {
        "sample": {"start": str(pred["date"].min().date()), "end": str(pred["date"].max().date()),
                   "days": int(pred["date"].nunique()), "rows": len(pred)},
        "har_calibration_primary": overall,
        "har_nonoverlap_coverage": coverage,
        "fixed_50_50_combination_hac_dm": combination,
        "signal_pnl": pnl,
        "live_2026": live_status(),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    bucket.to_csv(OUT / "calibration_by_time_bucket.csv", index=False)
    quarters.to_csv(OUT / "quarterly_har_qlike.csv", index=False)
    combination_daily.to_csv(OUT / "combination_daily_losses.csv", index=False)
    write_parquet(trades, OUT / "har_q90_signal_trades.parquet")

    def fmt(value: float) -> str:
        return "NA" if value is None or not np.isfinite(value) else f"{value:.4f}"

    lines = [
        "# Deployed HAR 120-minute audit", "",
        f"Sample: {results['sample']['start']} to {results['sample']['end']} ({results['sample']['days']} days).", "",
        "## Calibration", "",
        f"At 11:15, Mincer-Zarnowitz p={overall['mz_pvalue']:.4f}; mean actual/forecast={overall['mean_actual_over_forecast']:.4f} (p={overall['ratio_pvalue']:.4f}).", "",
        "## Non-overlapping q90 coverage", "",
        f"Every 120 minutes: hit rate={coverage['every_120_minutes']['hit_rate']:.3%}, Kupiec p={coverage['every_120_minutes']['kupiec_pvalue']:.4f}, Christoffersen p={coverage['every_120_minutes']['christoffersen_pvalue']:.4f}.",
        f"Once per day at 11:15: hit rate={coverage['once_per_day_1115']['hit_rate']:.3%}, Kupiec p={coverage['once_per_day_1115']['kupiec_pvalue']:.4f}, Christoffersen p={coverage['once_per_day_1115']['christoffersen_pvalue']:.4f}.", "",
        "## Fixed 50/50 forecast combination", "",
        f"One-sided daily HAC-DM p={combination['pvalue']:.4f}; QLIKE change={100*combination['improvement_ratio']:.2f}%; decision={combination['decision_alpha_0_05']}.", "",
        "## Economic signal", "",
        f"The fixed rule produced {pnl['trades']} trades. Net P&L after historical charges and 25 bps slippage was Rs {fmt(pnl['total_net'])}; mean/trade Rs {fmt(pnl['mean_net'])}; hit rate {fmt(pnl['hit_rate'])}; daily Sharpe {fmt(pnl['daily_sharpe'])}.", "",
        "## 2026 live data", "",
        f"Status: {results['live_2026']['status']}. Mature 120-minute runtime records: {results['live_2026']['mature_120m_records']}. A valid 2026 MZ result is deferred until realized windows and enough live days exist.", "",
        "Detailed bucket and quarterly tables are stored beside this report.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
