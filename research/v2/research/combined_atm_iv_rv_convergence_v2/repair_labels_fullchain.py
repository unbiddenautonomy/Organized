"""Rebuild fixed-strike exit labels from the full option chain.

The original table looks up the t+120 exit in the five-strike file, so an entry whose
strike drifted out of the ATM±2 window has no label and is excluded from training and
from the backtest. Those are the big-move rows. Here every exit is looked up in the full
chain at the same (exit_datetime, dte_flag, expiry, strike), and the targets and quality
flags are recomputed with the same formulas as `feature_pipeline.construct_option_day`.
"""
from pathlib import Path
import json, shutil
import os
import duckdb, numpy as np, pandas as pd
from feature_pipeline import combine_call_put_iv

HERE = Path(__file__).resolve().parent
SRC = HERE / "artifacts" / "combined_atm_iv_convergence_v2_features.csv"
FULL = Path(os.environ.get(
    "VOLTR_43W_FULL_CHAIN_SOURCE",
    Path(__file__).resolve().parents[2] / "data" / "icharts_option_banknifty.csv",
))
OUT_DIR = HERE / "artifacts_fullchain"
OUT = OUT_DIR / "combined_atm_iv_convergence_v2_features_fullchain.csv"
EXIT = ["exit_fixed_strike", "exit_fixed_iv_call", "exit_fixed_iv_put", "exit_fixed_iv_call_converged",
        "exit_fixed_iv_put_converged", "exit_fixed_call_price", "exit_fixed_put_price", "matched_exit_timestamp"]


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def main():
    f = pd.read_csv(SRC, parse_dates=["datetime", "exit_datetime", "expiry", "matched_exit_timestamp"], low_memory=False)
    keys = f[["exit_datetime", "dte_flag", "expiry", "entry_strike"]].drop_duplicates().rename(columns={"entry_strike": "strike"})
    keys["strike"] = keys.strike.astype(float)
    con = duckdb.connect(); con.register("keys", keys)
    q = con.execute(f"""
        SELECT c.datetime AS exit_datetime, c.dte_flag, c.expiry, c.strike,
               c.iv_call, c.iv_put, CAST(c.iv_call_converged AS VARCHAR) AS iv_call_converged,
               CAST(c.iv_put_converged AS VARCHAR) AS iv_put_converged, c.call_price, c.put_price
        FROM read_csv('{FULL.as_posix()}', header=true) c
        JOIN keys k ON c.datetime = k.exit_datetime AND c.dte_flag = k.dte_flag
                   AND c.expiry = k.expiry AND c.strike = k.strike
    """).df()
    q["exit_datetime"] = pd.to_datetime(q.exit_datetime); q["expiry"] = pd.to_datetime(q.expiry)
    q = q.drop_duplicates(["exit_datetime", "dte_flag", "expiry", "strike"], keep="last")
    q = q.rename(columns={"strike": "entry_strike", "iv_call": "fc_iv_call", "iv_put": "fc_iv_put",
                          "iv_call_converged": "fc_call_conv", "iv_put_converged": "fc_put_conv",
                          "call_price": "fc_call_price", "put_price": "fc_put_price"})
    f["entry_strike"] = f.entry_strike.astype(float)
    f = f.merge(q, on=["exit_datetime", "dte_flag", "expiry", "entry_strike"], how="left", validate="many_to_one")

    had = f.exit_fixed_atm_iv_combined.notna() & f.fc_iv_call.notna()
    agree = {c: float(np.isclose(f.loc[had, a], f.loc[had, b], atol=1e-9).mean())
             for c, a, b in [("price", "exit_fixed_call_price", "fc_call_price"), ("iv_call", "exit_fixed_iv_call", "fc_iv_call"),
                             ("iv_put", "exit_fixed_iv_put", "fc_iv_put")]}
    fill = f.exit_fixed_atm_iv_combined.isna() & f.fc_iv_call.notna()
    f.loc[fill, "exit_fixed_strike"] = f.loc[fill, "entry_strike"]
    f.loc[fill, "matched_exit_timestamp"] = f.loc[fill, "exit_datetime"]
    for dst, src in [("exit_fixed_iv_call", "fc_iv_call"), ("exit_fixed_iv_put", "fc_iv_put"),
                     ("exit_fixed_call_price", "fc_call_price"), ("exit_fixed_put_price", "fc_put_price")]:
        f.loc[fill, dst] = f.loc[fill, src]
    f["exit_fixed_iv_call_converged"] = np.where(fill, as_bool(f.fc_call_conv), as_bool(f.exit_fixed_iv_call_converged))
    f["exit_fixed_iv_put_converged"] = np.where(fill, as_bool(f.fc_put_conv), as_bool(f.exit_fixed_iv_put_converged))
    f["exit_fixed_atm_iv_combined"] = combine_call_put_iv(f.exit_fixed_iv_call, f.exit_fixed_iv_put)

    # same target / quality formulas as feature_pipeline.construct_option_day
    f["target_fixed_combined_iv_change_120m"] = f.exit_fixed_atm_iv_combined - f.entry_atm_iv_combined
    f["target_fixed_combined_iv_return_120m"] = f.exit_fixed_atm_iv_combined / f.entry_atm_iv_combined - 1.0
    eg = f.entry_atm_iv_combined - f.target_rv_forward_120m
    xg = f.exit_fixed_atm_iv_combined - f.target_rv_forward_120m
    ok = eg.notna() & xg.notna()
    f["target_combined_converged_to_actual_rv"] = (xg.abs() < eg.abs()).where(ok)
    f["target_combined_actual_rv_gap_change"] = (xg.abs() - eg.abs()).where(ok)
    f["quality_exact_exit"] = f.exit_fixed_atm_iv_combined.notna()
    f["quality_exit_iv_valid"] = f.exit_fixed_iv_call_converged & f.exit_fixed_iv_put_converged & f.quality_exact_exit
    f["quality_same_strike"] = np.isclose(f.entry_strike, f.exit_fixed_strike.astype(float), equal_nan=False)
    f["quality_exact_120m_timestamp"] = (f.matched_exit_timestamp - f.datetime == pd.Timedelta(minutes=120)).fillna(False)
    f["quality_exact_spot_window"] = as_bool(f.quality_exact_spot_window)
    f["quality_entry_ok"] = as_bool(f.quality_entry_ok)
    old_label = as_bool(f.label_available)
    f["label_available"] = (f.quality_exact_exit & f.quality_exit_iv_valid & f.quality_exact_spot_window
                            & f.quality_same_strike & f.quality_exact_120m_timestamp)
    f["training_row_eligible"] = f.quality_entry_ok & f.label_available
    f["label_source"] = np.where(fill, "full_chain", np.where(f.exit_fixed_atm_iv_combined.notna(), "five_strike", "none"))
    f = f.drop(columns=[c for c in f if c.startswith("fc_")])

    OUT_DIR.mkdir(exist_ok=True)
    f.to_csv(OUT, index=False)
    man = SRC.with_name("combined_atm_iv_convergence_v2_features_column_manifest.json")
    shutil.copy(man, OUT_DIR / "combined_atm_iv_convergence_v2_features_fullchain_column_manifest.json")
    entry_ok = f.quality_entry_ok
    in_session = f.exit_datetime.dt.date.eq(f.datetime.dt.date) & f.exit_datetime.dt.strftime("%H:%M:%S").le("15:29:59")
    audit = {
        "rows": len(f), "full_chain_matches": int(f.exit_fixed_atm_iv_combined.notna().sum()),
        "agreement_where_both_sources_exist": agree,
        "labels_filled_from_full_chain": int(fill.sum()),
        "training_row_eligible_before": int((entry_ok & old_label).sum()),
        "training_row_eligible_after": int(f.training_row_eligible.sum()),
        "entry_ok_in_session_rows": int((entry_ok & in_session).sum()),
        "entry_ok_in_session_still_unlabelled": {
            "no_full_chain_quote": int((entry_ok & in_session & ~f.quality_exact_exit).sum()),
            "exit_iv_not_converged": int((entry_ok & in_session & f.quality_exact_exit & ~f.quality_exit_iv_valid).sum()),
            "spot_window_incomplete": int((entry_ok & in_session & f.quality_exit_iv_valid & ~f.quality_exact_spot_window).sum()),
        },
    }
    (OUT_DIR / "label_repair_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
