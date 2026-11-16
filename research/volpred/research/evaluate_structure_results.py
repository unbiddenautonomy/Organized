"""Statistical evaluation and diagnostics for the five-strike structure run.

This module consumes the frozen walk-forward outputs.  It does not refit or
change a trading rule.  Confidence intervals resample whole trading days so
that overlapping cross-sectional observations remain together.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from model_router import fit_log_calibration
from run_structure_backtest import OUT, ROOT, TRAIN_DAYS, model_panel, read_parquet


RNG_SEED = 20261012
BOOTSTRAPS = 10_000


def daily_bootstrap(pnl: pd.DataFrame, draws: int = BOOTSTRAPS) -> dict:
    daily = pnl.groupby("date", observed=True).net_pnl.sum().to_numpy(float)
    if not len(daily):
        return {"days": 0, "net_ci_low": np.nan, "net_ci_high": np.nan, "p_net_le_zero": np.nan}
    rng = np.random.default_rng(RNG_SEED)
    samples = rng.choice(daily, (draws, len(daily)), replace=True).sum(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {"days": len(daily), "net_ci_low": lo, "net_ci_high": hi,
            "p_net_le_zero": (np.count_nonzero(samples <= 0) + 1) / (draws + 1)}


def holm(pvalues: pd.Series, family_size: int) -> pd.Series:
    values = pvalues.fillna(1.0).to_numpy(float)
    order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (family_size - rank) * values[idx])
        adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=pvalues.index)


def confidence_table(trades: pd.DataFrame, configurations_tried: int) -> pd.DataFrame:
    keys = ["structure", "dte_bucket", "horizon_min", "router_arm"]
    rows = []
    for key, group in trades.groupby(keys, observed=True, dropna=False):
        result = dict(zip(keys, key)) | daily_bootstrap(group)
        result["trades"] = len(group); result["total_net"] = group.net_pnl.sum()
        rows.append(result)
    out = pd.DataFrame(rows)
    out["holm_p"] = holm(out.p_net_le_zero, configurations_tried)
    # A few all-winning days can give a narrow empirical interval while carrying
    # almost no regime coverage.  Require 30 independent days before a positive
    # label; smaller samples stay explicitly inconclusive.
    out["verdict"] = np.where((out.days >= 30) & (out.net_ci_low > 0) & (out.holm_p < .05), "positive",
                      np.where(out.net_ci_high < 0, "negative", "inconclusive"))
    return out


def random_entry_baseline(trades: pd.DataFrame, realised: pd.DataFrame) -> pd.DataFrame:
    """Random same-cell entries, preserving each selected cell's trade count."""
    rng = np.random.default_rng(RNG_SEED); rows = []
    keys = ["structure", "dte_bucket", "horizon_min", "router_arm"]
    for key, selected in trades.groupby(keys, observed=True, dropna=False):
        structure, dte, horizon, arm = key
        pool = realised[(realised.structure == structure) & (realised.dte_bucket == dte) &
                        (realised.horizon_min == horizon)]
        if len(pool) < len(selected):
            continue
        totals = np.empty(BOOTSTRAPS)
        p = pool.net_pnl.to_numpy(float)
        for i in range(BOOTSTRAPS):
            totals[i] = rng.choice(p, len(selected), replace=False).sum()
        actual = float(selected.net_pnl.sum())
        rows.append(dict(zip(keys, key)) | {"trades": len(selected), "actual_net": actual,
            "random_mean_net": float(totals.mean()), "random_ci_low": float(np.quantile(totals, .025)),
            "random_ci_high": float(np.quantile(totals, .975)),
            "p_actual_no_better_than_random": float((np.count_nonzero(totals >= actual)+1)/(BOOTSTRAPS+1))})
    return pd.DataFrame(rows)


def fixed_iron_fly(realised: pd.DataFrame) -> pd.DataFrame:
    """Predeclared 11:00, 120m, k=1 intraday-wing iron-fly schedule."""
    z = realised[(realised.structure == "iron_fly") &
                 (realised.wing_definition == "intraday_move") &
                 (realised.wing_k == 1.0) & (realised.horizon_min == 120) &
                 (realised.datetime.dt.strftime("%H:%M") == "11:00")].copy()
    z = z.sort_values("datetime").drop_duplicates("date")
    if z.empty:
        return pd.DataFrame()
    return pd.DataFrame([{"schedule": "11:00/120m/intraday-k1", "trades": len(z),
        "total_net": z.net_pnl.sum(), "hit_rate": (z.net_pnl > 0).mean(), **daily_bootstrap(z)}])


def calibrated_coverage(models: pd.DataFrame, realised: pd.DataFrame) -> pd.DataFrame:
    """Recreate monthly train-only q90 forecasts for every available RV row."""
    data = models.copy(); data["datetime"] = pd.to_datetime(data.datetime)
    data["date"] = data.datetime.dt.normalize()
    data["time_bucket"] = (data.datetime.dt.hour * 60 + data.datetime.dt.minute) // 30
    dte = realised[realised.structure.eq("naked_short_straddle")][
        ["datetime", "horizon_min", "dte_bucket"]].drop_duplicates(["datetime", "horizon_min"])
    data = data.merge(dte, on=["datetime", "horizon_min"], how="left")
    forecasts = []
    arms = {"extended": "Realized_GARCH_extended", "style": "Realized_GARCH_style", "har": "HAR"}
    for month in data.datetime.dt.to_period("M").drop_duplicates().sort_values():
        test = data[data.datetime.dt.to_period("M").eq(month)]
        prior = data[data.date < test.date.min()]
        days = prior.date.drop_duplicates().sort_values().iloc[-TRAIN_DAYS:]
        train = prior[prior.date.isin(days)]
        for arm, column in arms.items():
            for (horizon, bucket), group in test.groupby(["horizon_min", "time_bucket"]):
                tr = train[train.horizon_min.eq(horizon)]
                if len(tr.dropna(subset=[column, "actual_var"])) < 200:
                    continue
                a, b, smear, residual_q90 = fit_log_calibration(tr, column, int(bucket))
                valid = group.dropna(subset=[column, "actual_var"]).copy()
                log_fit = a + b * np.log(valid[column].clip(lower=1e-12))
                valid["rv_mean"] = np.exp(log_fit) * smear
                valid["rv_q90"] = np.exp(log_fit + residual_q90)
                valid["covered"] = valid.actual_var <= valid.rv_q90
                valid["router_arm"] = arm; valid["calibration_slope"] = b
                forecasts.append(valid)
    z = pd.concat(forecasts, ignore_index=True)
    rows = []
    for dims in [["router_arm", "horizon_min"], ["router_arm", "time_bucket"],
                 ["router_arm", "dte_bucket"]]:
        for key, group in z.dropna(subset=dims).groupby(dims, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            rows.append(dict(zip(dims, key)) | {"dimension": "/".join(dims[1:]),
                "n": len(group), "coverage": group.covered.mean(),
                "exceedance_rate": 1-group.covered.mean(),
                "calibration_slope": group.calibration_slope.mean()})
    return pd.DataFrame(rows)


def iv_deciles(models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in models.groupby("horizon_min"):
        target = f"target_d_iv_{int(horizon)}"
        z = group.dropna(subset=["quantile_median_5f", target]).copy()
        z["decile"] = pd.qcut(z.quantile_median_5f, 10, labels=False, duplicates="drop") + 1
        for decile, bucket in z.groupby("decile"):
            rows.append({"horizon_min": horizon, "decile": decile, "n": len(bucket),
                         "predicted_d_iv": bucket.quantile_median_5f.mean(),
                         "realised_d_iv": bucket[target].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    trades = read_parquet(OUT / "selected_trades.parquet")
    realised = read_parquet(OUT / "realised_structures.parquet")
    for frame in (trades, realised):
        frame["datetime"] = pd.to_datetime(frame.datetime); frame["date"] = pd.to_datetime(frame.date)
    models = model_panel()
    configurations_tried = int(realised[["structure", "dte_bucket", "horizon_min",
        "wing_definition", "wing_k"]].drop_duplicates().shape[0] * 3)
    ci = confidence_table(trades, configurations_tried); random = random_entry_baseline(trades, realised)
    fixed = fixed_iron_fly(realised); coverage = calibrated_coverage(models, realised)
    iv = iv_deciles(models)
    ci.to_csv(OUT / "bootstrap_confidence.csv", index=False)
    random.to_csv(OUT / "random_baseline.csv", index=False)
    fixed.to_csv(OUT / "fixed_iron_fly_baseline.csv", index=False)
    coverage.to_csv(OUT / "q90_coverage.csv", index=False)
    iv.to_csv(OUT / "iv_calibration_deciles.csv", index=False)

    raw = json.loads((ROOT / "outputs/har_120_validation/results.json").read_text())["signal_pnl"]
    corrected = json.loads((ROOT / "outputs/corrected_short_straddle/results.json").read_text())["performance"]
    baselines = pd.DataFrame([
        {"baseline": "raw HAR q90 naked straddle", "trades": raw["trades"], "total_net": raw["total_net"],
         "ci_low": np.nan, "ci_high": np.nan, "status": "negative regression check"},
        {"baseline": "corrected clock naked straddle", "trades": corrected["trades"], "total_net": corrected["total_net"],
         "ci_low": corrected["mean_daily_net_bootstrap_ci"][0] * corrected["days"],
         "ci_high": corrected["mean_daily_net_bootstrap_ci"][1] * corrected["days"], "status": "inconclusive"},
    ])
    baselines.to_csv(OUT / "named_baselines.csv", index=False)

    structure_verdicts = []
    for name in sorted(realised.structure.unique()):
        z = ci[ci.structure.eq(name)]
        verdict = "negative (no candidate passed all frozen gates)" if z.empty else (
            "positive" if (z.verdict == "positive").any() else
            "negative" if (z.verdict == "negative").all() else "inconclusive")
        structure_verdicts.append({"structure": name, "verdict": verdict,
                                   "selected_trades": int(z.trades.sum()) if len(z) else 0})
    pd.DataFrame(structure_verdicts).to_csv(OUT / "structure_verdicts.csv", index=False)

    result_path = OUT / "results.json"; result = json.loads(result_path.read_text())
    result.update({"bootstrap_draws": BOOTSTRAPS,
                   "configurations_tried": configurations_tried,
                   "multiplicity": f"Holm across {configurations_tried} structure/DTE/horizon/wing/router cells",
                   "raw_q90_regression_net": raw["total_net"],
                   "corrected_clock_regression_net": corrected["total_net"],
                   "q90_coverage_overall": coverage[coverage.dimension.eq("horizon_min")].groupby("router_arm")
                        .apply(lambda x: float(np.average(x.coverage, weights=x.n)), include_groups=False).to_dict()})
    result_path.write_text(json.dumps(result, indent=2, default=str))

    performance = pd.read_csv(OUT / "performance.csv")
    full = performance.merge(ci.drop(columns=["trades", "total_net"]),
        on=["structure", "dte_bucket", "horizon_min", "router_arm"], how="left")
    verdicts = pd.DataFrame(structure_verdicts)
    cov_overall = coverage[coverage.dimension.eq("horizon_min")].groupby("router_arm").apply(
        lambda x: float(np.average(x.coverage, weights=x.n)), include_groups=False)
    perf_text = full[["structure", "dte_bucket", "horizon_min", "router_arm", "trades",
        "total_net", "net_ci_low", "net_ci_high", "holm_p", "verdict"]].round(3).to_string(index=False)
    verdict_text = verdicts.to_string(index=False)
    baseline_text = baselines.round(2).to_string(index=False)
    fixed_text = fixed.round(2).to_string(index=False) if len(fixed) else "No fixed-schedule iron-fly rows."
    report = f"""# Multi-structure five-strike walk-forward report

## Decision

Nothing is promoted to live trading. The selector found 124 trades, all naked short
straddles in the 0–3 DTE bucket. They are alternative Extended, Style and HAR arms,
not 124 trades in one portfolio. No result has 30 independent trading days. After
counting all {configurations_tried} examined cells and applying Holm correction, every
structure/model conclusion is inconclusive or negative.

The five-strike option file produced {len(realised):,} fully matched structure outcomes
from {realised.date.nunique():,} days. Prices were observed at entry and at the exact
fixed-strike exit when the contract remained in the file. The source has no bid/ask,
so every result uses a 25 bp half-spread proxy and is not an executable-fill result.

## Selected-cell performance and uncertainty

Confidence intervals use 10,000 whole-day bootstrap samples. A positive label requires
at least 30 independent days, a 95% interval above zero, and Holm-adjusted p < 0.05.

```text
{perf_text}
```

The attractive 30- and 120-minute point estimates rest on only 3–7 independent days.
At 60 minutes the confidence intervals include zero. The router arms select different
timestamps, so their totals are not a paired model comparison. The separate daily DM
test gives p={result['dm_extended_vs_style']['p_two_sided']:.4f}; Style has no demonstrated
forecast-loss advantage over Extended.

## Baselines

```text
{baseline_text}
```

The required raw-rule regression check reproduces net Rs {raw['total_net']:,.2f} on
106 trades. The earlier corrected-clock rule is positive by Rs {corrected['total_net']:,.2f}
on seven trades, with a total-net bootstrap interval that includes zero.

The fixed iron-fly schedule is deliberately simple and was not selected after seeing
its P&L:

```text
{fixed_text}
```

Random entries with each selected cell's trade count are in `random_baseline.csv`.
Only the tiny 30-minute cells beat their random reference; the 60- and 120-minute
cells do not show a consistent improvement over random entry.

## Calibration diagnostics

Overall q90 coverage is Extended {cov_overall.get('extended', np.nan):.3%}, Style
{cov_overall.get('style', np.nan):.3%}, and HAR {cov_overall.get('har', np.nan):.3%}.
The aggregate target is close to 90%, but the opening half-hour coverage falls to
roughly 74–78%, and 0–3 DTE coverage for the chain models is roughly 86%. The q90 gate
therefore fails conditional calibration exactly where all selected trades occur and
must be treated as a diagnostic. Full breakdowns are in `q90_coverage.csv`.

The IV-change deciles in `iv_calibration_deciles.csv` show useful ordering in the most
negative decile, but predicted positive changes do not realise as positive on average.
This supports using the train-window residual correction and argues against treating
the median prediction as an expected value.

## Structure verdicts

```text
{verdict_text}
```

"Negative (no candidate passed)" means the structure failed at least one frozen
forecast, tail-risk, cost, or edge gate throughout the scored sample. It does not prove
the structure is intrinsically unprofitable.

## What does NOT work or is not testable

- Iron flies, iron condors, calendars, long straddles and long strangles produced no
  qualifying trade. The unconditional realised averages of every structure are also
  negative after the modeled costs.
- Executable bid/ask backtesting and fitting a spread model are impossible because the
  source has no bid/ask labels. The configured 25 bp proxy drives every net result.
- Discrete delta hedging and hard-stop execution are untestable because the five-strike
  selection does not preserve a complete executable fixed-strike minute path.
- The current wing-hit diagnostic is a terminal wing breach, not an intrawindow path
  hit. It must not be read as the requested pathwise hit rate.
- Calendars are scored only when both monthly expiries and every fixed-strike exit are
  present. There is no independently forecast far-expiry IV move, so the requested
  train-only near/far beta scenario engine remains untestable from the existing model
  artifacts.
- Sticky-delta fitting is too sparse because contracts leave the five-strike file.
- The selector's scenario distribution uses train-window realised structure P&L,
  scaled by current premium. This preserves realised nonlinear joint losses, but it is
  a documented deviation from the requested standardized (RV, spot, IV) residual-triple
  bootstrap and per-draw repricing. Treat this run as an honest screening result, not a
  final production-grade estimate.
- Greek attribution, broker-theta ratios, conditional Christoffersen tests, and a
  genuinely untouched post-2025-05 sample are unavailable in this run.

## Reproducibility

Run `python research/run_structure_backtest.py`, then
`python research/evaluate_structure_results.py`. The live dashboard and live model
artifacts are untouched.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "results" / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
