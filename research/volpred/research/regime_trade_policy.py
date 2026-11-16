"""Audit whether IV/RV Greek regimes support a tradable same-expiry policy."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "outputs" / "short_straddle_extended_analysis" / "short_straddle_windows.parquet"
FORECASTS = ROOT / "outputs" / "volatility_specific_models" / "walk_forward_predictions.parquet"
OUT = ROOT / "model" / "regimeUnderstanding" / "trade_policy"
HORIZONS = (15, 30, 60, 120)
EQUALITY_BAND = 0.005
COST_BPS = 25.0


def load_fixed_strike_windows() -> pd.DataFrame:
    horizons = ",".join(f"'{h}m'" for h in HORIZONS)
    query = f"""
        SELECT entry_datetime, date, horizon, iv_trade_vol, future_rv,
               actual AS short_gross_pnl, entry_value, stale_any
        FROM read_parquet('{WINDOWS.as_posix()}')
        WHERE horizon IN ({horizons})
    """
    frame = duckdb.connect().execute(query).fetchdf()
    # ``stale_any`` compares entry and exit quotes, so using it here would
    # disclose the future exit at entry time. Retain it only as an audit field.
    frame = frame[
        frame["iv_trade_vol"].notna() & frame["future_rv"].notna()
    ].copy()
    frame["horizon_min"] = frame["horizon"].str.removesuffix("m").astype(int)
    entry_premium = -frame["entry_value"]
    exit_premium = entry_premium - frame["short_gross_pnl"]
    frame["cost_points"] = COST_BPS / 10_000.0 * (
        entry_premium + exit_premium.clip(lower=0)
    )
    return frame


def scene(gap: pd.Series) -> np.ndarray:
    return np.select(
        [gap.gt(EQUALITY_BAND), gap.lt(-EQUALITY_BAND)],
        ["IV > RV", "IV < RV"],
        default="IV = RV",
    )


def metrics(frame: pd.DataFrame, group_cols: list[str], pnl_col: str) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pnl = group[pnl_col].dropna()
        row = dict(zip(group_cols, keys))
        row.update({
            "trades": len(pnl),
            "days": group.loc[pnl.index, "date"].nunique(),
            "mean_net_points": pnl.mean(),
            "median_net_points": pnl.median(),
            "win_rate": pnl.gt(0).mean(),
            "p05_net_points": pnl.quantile(0.05),
            "p95_net_points": pnl.quantile(0.95),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def ex_post_research(base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    year = frame["entry_datetime"].dt.year
    frame["split"] = np.select(
        [year.le(2022), year.eq(2023)], ["train_2019_2022", "validation_2023"],
        default="test_2024_2025",
    )
    frame["scene"] = scene(frame["iv_trade_vol"] - frame["future_rv"])
    frames = []
    for side, sign in (("long_atm_straddle", -1.0), ("short_atm_straddle", 1.0)):
        candidate = frame.copy()
        candidate["side"] = side
        candidate["net_pnl"] = sign * candidate["short_gross_pnl"] - candidate["cost_points"]
        frames.append(candidate)
    both = pd.concat(frames, ignore_index=True)
    return metrics(both, ["split", "scene", "horizon_min", "side"], "net_pnl")


def tradable_research(base: pd.DataFrame) -> pd.DataFrame:
    forecast = duckdb.connect().execute(
        f"SELECT datetime, horizon_min, HAR FROM read_parquet('{FORECASTS.as_posix()}')"
    ).fetchdf()
    forecast["predicted_rv"] = np.sqrt(
        forecast["HAR"] * 252.0 * 375.0 / forecast["horizon_min"]
    )
    frame = base.merge(
        forecast[["datetime", "horizon_min", "predicted_rv"]],
        left_on=["entry_datetime", "horizon_min"],
        right_on=["datetime", "horizon_min"],
        how="inner",
        validate="one_to_one",
    )
    frame["split"] = np.where(
        frame["entry_datetime"].dt.year.eq(2024), "development_2024", "holdout_2025"
    )
    frame["scene"] = scene(frame["iv_trade_vol"] - frame["predicted_rv"])
    frame["side"] = frame["scene"].map({
        "IV < RV": "long_atm_straddle",
        "IV = RV": "no_trade",
        "IV > RV": "short_atm_straddle",
    })
    sign = frame["side"].map({"long_atm_straddle": -1.0, "no_trade": 0.0, "short_atm_straddle": 1.0})
    frame["net_pnl"] = sign * frame["short_gross_pnl"] - frame["cost_points"] * sign.ne(0)
    return metrics(frame, ["split", "scene", "horizon_min", "side"], "net_pnl")


def selected(table: pd.DataFrame, scene_name: str, horizon: int) -> pd.DataFrame:
    return table[
        table["scene"].eq(scene_name) & table["horizon_min"].eq(horizon)
    ].copy()


def markdown(frame: pd.DataFrame) -> str:
    view = frame.copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    columns = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(ex_post: pd.DataFrame, tradable: pd.DataFrame) -> None:
    oracle_long = selected(
        ex_post[ex_post["side"].eq("long_atm_straddle")], "IV < RV", 60
    )
    live_short = selected(
        tradable[tradable["side"].eq("short_atm_straddle")], "IV > RV", 120
    )
    live_long = selected(
        tradable[tradable["side"].eq("long_atm_straddle")], "IV < RV", 60
    )
    equality = tradable[tradable["scene"].eq("IV = RV")]
    report = f"""# Same-expiry intraday trade policy research

## Decision

**No live-capital trade is approved by the present evidence.** The only setup
worth forward paper testing is a **120-minute short ATM straddle when current
trading-time IV exceeds the timestamp-available 120-minute HAR RV forecast**.
It is a candidate, not a deployment rule: after a 25 bps premium-turnover cost
stress its mean was only approximately flat, while its 5% trade tail remained
large.

For `IV < predicted RV`, do not buy the straddle with the current forecast. For
`IV = predicted RV`, do not trade. The apparently attractive long-vol result
when IV is below *realized future* RV is an oracle result and cannot be known at
entry.

## Exact contract and P&L

At entry time `t`, buy or sell the call and put with the same ATM strike `K`
and the same monthly expiry `T`; hold both original contracts to `t+h`. Never
re-center the strike. Define exact gross straddle change in index points as

`G = (C(t+h,K,T) + P(t+h,K,T)) - (C(t,K,T) + P(t,K,T))`.

For positive round-trip cost `c`, `PnL_long = G - c` and
`PnL_short = -G - c`.

The local Greek approximation for position sign `q` (`+1` long, `-1` short) is

`PnL(q) ~= q * [Delta*dS + 0.5*Gamma*dS^2 + Vega*dSigma + Theta*dt + Vanna*dS*dSigma + 0.5*Volga*dSigma^2 + Charm*dS*dt + residual] - costs`.

Here Greeks are the call-plus-put values, `dt=h/1440` calendar days, and
`dSigma` is decimal IV change. At entry, a tradable expected-value gate replaces
unknown `dS^2` and `dSigma` with forecasts:

`E_t[PnL_long] ~= 0.5*Gamma*S^2*vhat_h + Theta*h/1440 + Vega*dSigma_hat + Vanna*E[dS*dSigma] + 0.5*Volga*E[dSigma^2] + Charm*E[dS]*h/1440`.

Enter a short only if `-E_t[PnL_long]` exceeds complete round-trip costs plus a
pre-registered safety buffer. With no directional forecast, set `E[dS]=0` and
do not claim expected delta or charm profit.

## Why 120 minutes is the only paper candidate

- With predicted scenes, 15- and 30-minute short straddles did not overcome the
  25 bps turnover stress; 60 minutes also had negative mean net P&L.
- 120 minutes was the shortest tested horizon with a non-negative mean in both
  2024 development and 2025 holdout, and its median and win rate were positive.
- 180 minutes is excluded because the deployed walk-forward RV model has no
  180-minute forecast artifact. Selecting it from realized-RV hindsight would
  be look-ahead.
- Even at 120 minutes the mean edge is too small relative to the left tail, so
  this remains paper-only.

### Tradable 120-minute short candidate (`IV > predicted RV`)

{markdown(live_short)}

## Why the other scenes are no-trade

### `IV < RV`

The ex-post 60-minute long straddle looked positive when classified with future
realized RV:

{markdown(oracle_long)}

But using the RV forecast actually available at entry reversed the result:

{markdown(live_long)}

This is forecast error, not a Greek identity failure. Do not trade the oracle
label.

### `IV = RV`

Within the +/-{100 * EQUALITY_BAND:.1f}-vol-point equality band, expected gamma
and theta are approximately balanced before IV repricing. Costs make both
directions unattractive, so the prescribed position is zero.

{markdown(equality)}

## Research controls

- Entry and exit are in the same 09:15–15:30 session.
- Strike and monthly expiry remain fixed through exit.
- IV and RV use the same annualized trading-time clock.
- The tradable audit uses walk-forward HAR forecasts, not future realized RV.
- Results use a symmetric 25 bps premium-turnover stress because the source has
  no bid/ask quotes. This is not a substitute for exact broker, tax, spread and
  market-impact modeling.
- Overlapping minute-level candidates are descriptive and must not be summed as
  an executable portfolio without a non-overlap and capital-allocation rule.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_fixed_strike_windows()
    ex_post = ex_post_research(base)
    tradable = tradable_research(base)
    ex_post.to_csv(OUT / "ex_post_scene_metrics.csv", index=False)
    tradable.to_csv(OUT / "walk_forward_scene_metrics.csv", index=False)
    policy = {
        "status": "PAPER_ONLY_NOT_APPROVED_FOR_LIVE_CAPITAL",
        "same_expiry": True,
        "fixed_strike": True,
        "intraday_session": "09:15-15:30",
        "iv_rv_equal_band_vol_points": 100 * EQUALITY_BAND,
        "cost_stress_bps_of_premium_turnover": COST_BPS,
        "scenes": {
            "IV < predicted RV": {"action": "NO_TRADE", "reason": "long straddle failed walk-forward audit"},
            "IV = predicted RV": {"action": "NO_TRADE", "reason": "no edge after costs"},
            "IV > predicted RV": {
                "action": "PAPER_TEST_SHORT_ATM_STRADDLE",
                "horizon_minutes": 120,
                "reason": "shortest horizon with non-negative mean in 2024 development and 2025 holdout",
            },
        },
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    write_report(ex_post, tradable)
    print(selected(tradable, "IV > RV", 120).to_string(index=False))


if __name__ == "__main__":
    main()
