"""Test same-expiry directional long options and defined-risk debit spreads.

The study deliberately reports two cases:
1. oracle_sign: the future direction is supplied ex post (not tradable), and
2. forecast_sign: direction comes from a timestamp-available walk-forward Ridge model.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "directional_debit_spreads"
FEATURE_FILE = ROOT / "outputs" / "spot_magnitude_models" / "spot_magnitude_feature_panel.parquet"
SLIPPAGE_BPS = 25.0
ROLLING_TRAIN_DAYS = 60
EQUALITY_BAND = 0.005

DIRECTION_FEATURES = [
    "return_past_5", "return_past_15", "return_past_60",
    "abs_return_past_5", "abs_return_past_15", "abs_return_past_60",
    "log_rv_past_5", "log_rv_past_15", "log_rv_past_60",
    "overnight_return", "log_iv_trade", "iv_skew", "forward_log_basis",
    "oi_imbalance", "time_fraction", "time_sin", "time_cos", "sqrt_dte",
]

# Positive weight buys an option; negative weight sells one. Entry strikes stay
# fixed until exit and every leg uses the same monthly expiry.
STRUCTURES: dict[str, dict[tuple[int, str], int]] = {
    "long_atm_call": {(0, "call"): 1},
    "long_atm_put": {(0, "put"): 1},
    "bull_call_spread_1": {(0, "call"): 1, (1, "call"): -1},
    "bear_put_spread_1": {(0, "put"): 1, (-1, "put"): -1},
    "bull_call_spread_2": {(0, "call"): 1, (2, "call"): -1},
    "bear_put_spread_2": {(0, "put"): 1, (-2, "put"): -1},
    "bull_call_plus_put_1": {(0, "call"): 1, (-1, "put"): 1},
    "bear_put_plus_call_1": {(0, "put"): 1, (1, "call"): 1},
    "bull_call_plus_put_2": {(0, "call"): 1, (-2, "put"): 1},
    "bear_put_plus_call_2": {(0, "put"): 1, (2, "call"): 1},
}

FAMILIES = {
    "naked_atm": ("long_atm_call", "long_atm_put"),
    "debit_spread_1": ("bull_call_spread_1", "bear_put_spread_1"),
    "debit_spread_2": ("bull_call_spread_2", "bear_put_spread_2"),
    "bought_safety_1": ("bull_call_plus_put_1", "bear_put_plus_call_1"),
    "bought_safety_2": ("bull_call_plus_put_2", "bear_put_plus_call_2"),
}


def read_feature_panel() -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute("SELECT * FROM read_parquet(?)", [str(FEATURE_FILE)]).fetchdf()
    con.close()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def direction_predictions(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune on 2023, then refit monthly on the preceding 60 sessions."""
    prediction_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict] = []
    for horizon in smb.HORIZONS:
        target = f"future_spot_{horizon}"
        columns = DIRECTION_FEATURES + [target, "spot_price"]
        valid = panel["quality_ok"].fillna(False) & np.isfinite(panel[columns]).all(axis=1)
        data = panel.loc[valid].copy()
        data["signed_return"] = np.log(data[target] / data["spot_price"])

        development = data[data["year"].le(2022)]
        validation = data[data["year"].eq(2023)]
        best_alpha, best_mse = None, np.inf
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(development[DIRECTION_FEATURES], development["signed_return"])
            pred = model.predict(validation[DIRECTION_FEATURES])
            mse = float(np.mean((validation["signed_return"].to_numpy() - pred) ** 2))
            if mse < best_mse:
                best_alpha, best_mse = alpha, mse

        evaluation = data[data["year"].ge(2024)].copy()
        month_key = evaluation["datetime"].dt.to_period("M")
        monthly: list[pd.DataFrame] = []
        for month in month_key.drop_duplicates().sort_values():
            test = evaluation[month_key.eq(month)].copy()
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-ROLLING_TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)]
            if len(train) < 1000 or test.empty:
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
            model.fit(train[DIRECTION_FEATURES], train["signed_return"])
            piece = test[["datetime", "date", "spot_price", "signed_return"]].copy()
            piece["horizon_min"] = horizon
            piece["predicted_return"] = model.predict(test[DIRECTION_FEATURES])
            piece["alpha"] = best_alpha
            monthly.append(piece)
        predictions = pd.concat(monthly, ignore_index=True)
        prediction_frames.append(predictions)

        for year, part in predictions.groupby(predictions["date"].dt.year):
            y = part["signed_return"].to_numpy()
            p = part["predicted_return"].to_numpy()
            denom = np.sum(y**2)
            diagnostic_rows.append({
                "horizon_min": horizon,
                "year": int(year),
                "observations": len(part),
                "alpha_selected_on_2023": best_alpha,
                "validation_mse": best_mse,
                "r2_vs_zero": 1.0 - np.sum((y - p) ** 2) / denom if denom > 0 else np.nan,
                "direction_hit_rate": np.mean(np.sign(y) == np.sign(p)),
                "return_correlation": np.corrcoef(y, p)[0, 1] if len(y) > 1 else np.nan,
                "mae_return": np.mean(np.abs(y - p)),
            })
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(diagnostic_rows)


def build_structure_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    entries, _ = smb.prepare_entry_rows()
    entries = smb.add_entry_greeks(entries)
    smb.build_120m_exit_cache(entries)
    forecasts = smb.forecasts()
    frames: list[pd.DataFrame] = []
    for horizon in smb.HORIZONS:
        wide = smb.wide_legs(smb.leg_panel(entries, smb.load_exits(horizon)))
        for name, weights in STRUCTURES.items():
            frames.append(smb.structure_panel(wide, forecasts, name, weights, horizon))
    out = smb.attach_entry_iv(pd.concat(frames, ignore_index=True))
    out = out.merge(
        predictions[["datetime", "horizon_min", "signed_return", "predicted_return"]],
        left_on=["entry_datetime", "horizon_min"], right_on=["datetime", "horizon_min"],
        how="inner", validate="many_to_one",
    ).drop(columns="datetime")

    numeric = [
        "gross_pnl_points", "entry_turnover_points", "exit_turnover_points",
        "buy_turnover_points", "sell_turnover_points", "delta", "gamma", "vega",
        "theta_day", "pred_annualized_variance", "pred_d_iv", "lot_size",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[
        out["gross_pnl_points"].notna()
        & out["entry_turnover_points"].gt(0)
        & out["exit_turnover_points"].gt(0)
        & ~out["entry_tiny_any"]
    ].copy()
    entry_minute = out["entry_datetime"].dt.hour * 60 + out["entry_datetime"].dt.minute
    exit_minute = out["exit_datetime"].dt.hour * 60 + out["exit_datetime"].dt.minute
    out = out[
        entry_minute.between(9 * 60 + 15, 15 * 60 + 30)
        & exit_minute.between(9 * 60 + 15, 15 * 60 + 30)
        & out["entry_datetime"].dt.normalize().eq(out["exit_datetime"].dt.normalize())
    ].copy()

    # This legacy forecast column is cumulative variance over the horizon.
    out["predicted_horizon_variance"] = out["pred_annualized_variance"].clip(lower=0)
    out["predicted_rv"] = np.sqrt(
        out["predicted_horizon_variance"] * smb.TRADING_MINUTES_YEAR / out["horizon_min"]
    )
    iv_rv_gap = out["iv_trade_vol"] - out["predicted_rv"]
    out["scene"] = np.select(
        [iv_rv_gap.gt(EQUALITY_BAND), iv_rv_gap.lt(-EQUALITY_BAND)],
        ["IV > RV", "IV < RV"], default="IV = RV",
    )
    out["predicted_spot_change"] = out["spot_price"] * np.expm1(out["predicted_return"])
    out["expected_delta"] = out["delta"] * out["predicted_spot_change"]
    out["expected_gamma"] = (
        0.5 * out["gamma"] * out["spot_price"].pow(2) * out["predicted_horizon_variance"]
    )
    out["expected_theta"] = out["theta_day"] * out["horizon_min"] / 1440.0
    out["expected_vega"] = out["vega"] * out["pred_d_iv"]
    out["expected_pnl_points"] = out[[
        "expected_delta", "expected_gamma", "expected_theta", "expected_vega"
    ]].sum(axis=1)

    estimate = out.copy()
    estimate["exit_turnover_points"] = estimate["entry_turnover_points"]
    estimate["buy_turnover_points"] = estimate["entry_turnover_points"]
    estimate["sell_turnover_points"] = estimate["entry_turnover_points"]
    out["estimated_round_trip_cost"] = smb.charge_components(
        estimate, SLIPPAGE_BPS, current_2026=True
    )["total_cost"].to_numpy()
    out["expected_net_rupees"] = (
        out["expected_pnl_points"] * out["lot_size"] - out["estimated_round_trip_cost"]
    )
    out["expected_edge_gate"] = out["expected_net_rupees"].gt(0)
    return out


def choose_direction(panel: pd.DataFrame, family: str, signal_mode: str, edge_gate: bool) -> pd.DataFrame:
    bull, bear = FAMILIES[family]
    signal = panel["signed_return"] if signal_mode == "oracle_sign" else panel["predicted_return"]
    selected = panel[
        (panel["structure"].eq(bull) & signal.gt(0))
        | (panel["structure"].eq(bear) & signal.lt(0))
    ].copy()
    if edge_gate:
        selected = selected[selected["expected_edge_gate"]]
    return smb.select_non_overlapping(selected)


def score(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Exit-derived quote flags are diagnostics, not observable entry filters.
    clean = panel.copy()
    rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    for horizon in smb.HORIZONS:
        horizon_panel = clean[clean["horizon_min"].eq(horizon)]
        scene_panels = {"ALL": horizon_panel}
        scene_panels.update({name: group for name, group in horizon_panel.groupby("scene", observed=True)})
        for scene, scene_panel in scene_panels.items():
            for family in FAMILIES:
                for signal_mode in ("oracle_sign", "forecast_sign"):
                    for gate in ("sign_only", "sign_plus_expected_edge"):
                        chosen = choose_direction(
                            scene_panel, family, signal_mode, edge_gate=gate.endswith("expected_edge")
                        )
                        executed = smb.execute(chosen, SLIPPAGE_BPS, current_2026=True)
                        executed["family"] = family
                        executed["signal_mode"] = signal_mode
                        executed["gate"] = gate
                        executed["evaluation_scene"] = scene
                        if not executed.empty:
                            trade_frames.append(executed)
                        for year, label in ((2024, "development_2024"), (2025, "holdout_2025")):
                            part = executed[executed["date"].dt.year.eq(year)]
                            metrics = smb.metrics(part)
                            rows.append({
                                "scene": scene, "family": family, "horizon_min": horizon,
                                "signal_mode": signal_mode, "gate": gate, "period": label,
                                **metrics,
                                "p05_net": part["net_pnl_rupees"].quantile(0.05) if not part.empty else np.nan,
                                "mean_debit_points": part["entry_value"].mean() if not part.empty else np.nan,
                            })
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else clean.iloc[:0].copy()
    return pd.DataFrame(rows), trades


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    index = ["scene", "family", "horizon_min", "signal_mode", "gate"]
    wide = summary.pivot_table(
        index=index, columns="period",
        values=["trades", "mean_net", "hit_rate", "profit_factor", "p05_net", "net_pnl"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
    wide = wide.reset_index()
    for column in (
        "trades_development_2024", "trades_holdout_2025",
        "mean_net_development_2024", "mean_net_holdout_2025",
    ):
        if column not in wide:
            wide[column] = np.nan
    wide["robust_positive"] = (
        wide["trades_development_2024"].ge(20)
        & wide["trades_holdout_2025"].ge(10)
        & wide["mean_net_development_2024"].gt(0)
        & wide["mean_net_holdout_2025"].gt(0)
    )
    return wide.sort_values(
        ["signal_mode", "scene", "robust_positive", "mean_net_holdout_2025"],
        ascending=[True, True, False, False],
    )


def write_report(comparison: pd.DataFrame, diagnostics: pd.DataFrame, panel: pd.DataFrame) -> None:
    cols = [
        "scene", "family", "horizon_min", "signal_mode", "gate", "robust_positive",
        "trades_development_2024", "mean_net_development_2024", "hit_rate_development_2024",
        "trades_holdout_2025", "mean_net_holdout_2025", "hit_rate_holdout_2025", "p05_net_holdout_2025",
    ]
    passing = comparison[comparison["robust_positive"]][cols]
    top = comparison.groupby(["signal_mode", "scene"], observed=True).head(5)[cols]
    forecast_pass = passing[passing["signal_mode"].eq("forecast_sign")]
    decision = (
        "At least one forecast-driven configuration passed the stated robustness screen. "
        "It remains a paper-trading candidate because the direction model and execution assumptions require live validation."
        if not forecast_pass.empty else
        "No forecast-driven long-option or debit-spread configuration passed the stated robustness screen. "
        "The structures can work when direction is supplied correctly, but the available direction forecast does not justify deployment."
    )
    report = f"""# Directional call, put and debit-spread test

## Decision

{decision}

The defined-risk variants are:

- bullish: buy the entry ATM call and sell the call one or two strikes above ATM;
- bearish: buy the entry ATM put and sell the put one or two strikes below ATM.

The farther option is sold, not bought. It reduces the initial debit and caps
maximum profit. Both the naked long option and the debit spread already have a
maximum loss equal to the debit plus costs. The `bought_safety_1` and
`bought_safety_2` families test the user's literal bought-safety alternative:
an ATM call plus a one- or two-strike OTM put for a bullish signal, and an ATM
put plus a one- or two-strike OTM call for a bearish signal. This adds reversal
protection but also adds premium and negative theta.

## P&L equations

For a bull call spread with lower strike `K0` and upper strike `K1`:

`PnL_T = [(C_T(K0)-C_0(K0)) - (C_T(K1)-C_0(K1))] * lot - costs`.

At expiry, before costs: `max(S_T-K0,0) - max(S_T-K1,0) - net_debit`.
Maximum loss is `net_debit`; maximum profit is `(K1-K0)-net_debit`.

For a bear put spread with upper strike `K0` and lower strike `K-1`:

`PnL_T = [(P_T(K0)-P_0(K0)) - (P_T(K-1)-P_0(K-1))] * lot - costs`.

At expiry, before costs: `max(K0-S_T,0) - max(K-1-S_T,0) - net_debit`.
Maximum loss is `net_debit`; maximum profit is `(K0-K-1)-net_debit`.

The timestamp entry estimate is:

`E[PnL] = Delta*dS_hat + 0.5*Gamma*S^2*vhat_h + Theta*h/1440 + Vega*dIV_hat - costs`.

Here `dS_hat = S*(exp(rhat_h)-1)`. `vhat_h` is cumulative forecast variance,
and `dIV_hat` is the separate IV-change forecast.

## Direction tests

- `oracle_sign` uses the realized future sign only. It is a structural ceiling,
  not a tradable result.
- `forecast_sign` uses a Ridge signed-return forecast. Alpha is selected on
  2023, then the model is refit monthly using only the preceding 60 sessions.
- `sign_only` trades every nonzero signal. `sign_plus_expected_edge` also
  requires forecast Greek P&L to exceed complete estimated costs.
- Results are shown for all timestamps and separately for `IV < predicted RV`,
  `IV = predicted RV`, and `IV > predicted RV`; equality is the same
  +/-{100 * EQUALITY_BAND:.1f}-vol-point band used in the regime study.
- Costs use the repository's current charge schedule plus {SLIPPAGE_BPS:.0f}
  bps slippage on premium turnover.
- All legs use one monthly expiry, strikes stay fixed, and each {min(smb.HORIZONS)}
  to {max(smb.HORIZONS)} minute exit remains inside 09:15-15:30.

## Direction-model diagnostics

{dataframe_to_markdown(diagnostics.round(6))}

## Configurations passing the robustness rule

{dataframe_to_markdown(passing.round(4)) if not passing.empty else "No configuration passed."}

The rule requires at least 20 non-overlapping 2024 trades, at least 10 2025
holdout trades, and positive mean net P&L in both periods.

## Leading configurations

{dataframe_to_markdown(top.round(4))}

## Limitations

The 2024-2025 period is a reused research evaluation period, not an untouched
test. Quote data do not provide executable bid/ask depth. The direction model
uses overlapping targets during fitting, while evaluated trades are forced to
be non-overlapping. Results therefore remain forward-paper evidence only.

Coverage: {len(panel):,} clean directional structure-windows from
{panel['entry_datetime'].min()} through {panel['entry_datetime'].max()}.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feature_panel = read_feature_panel()
    predictions, diagnostics = direction_predictions(feature_panel)
    panel = build_structure_panel(predictions)
    summary, trades = score(panel)
    comparison = comparison_table(summary)

    diagnostics.to_csv(OUT / "direction_model_diagnostics.csv", index=False)
    summary.to_csv(OUT / "strategy_performance.csv", index=False)
    comparison.to_csv(OUT / "strategy_comparison.csv", index=False)
    smb.write_parquet(predictions, OUT / "walk_forward_direction_predictions.parquet")
    smb.write_parquet(trades, OUT / "nonoverlapping_trades.parquet")
    write_report(comparison, diagnostics, panel)

    forecast_pass = comparison[
        comparison["signal_mode"].eq("forecast_sign") & comparison["robust_positive"]
    ]
    policy = {
        "status": "FORWARD_PAPER_TEST_ONLY" if not forecast_pass.empty else "NO_DEPLOYABLE_DIRECTIONAL_SIGNAL",
        "same_monthly_expiry": True,
        "fixed_entry_strikes": True,
        "intraday_session": "09:15-15:30",
        "cost_model": "current encoded charges plus 25 bps premium-turnover slippage",
        "forecast_passing_configurations": forecast_pass.to_dict(orient="records"),
        "oracle_results_are_tradable": False,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2, default=str), encoding="utf-8")
    print(comparison[comparison["robust_positive"]].to_string(index=False))


if __name__ == "__main__":
    main()
