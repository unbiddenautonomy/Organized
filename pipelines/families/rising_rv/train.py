"""Train LightGBM without 120-minute RV-derived model features.

Run from any directory: py -3.13 train.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler

DATA_NAME = "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / "data" / DATA_NAME).is_file()), None)
SOURCE = ROOT / "data" / DATA_NAME if ROOT else None
OUT = Path(__file__).resolve().parent
NS_MIN = 60_000_000_000
ANNUAL = np.sqrt(375 * 252) * 100
H = 120
COLS = ["datetime", "Date", "expiry", "strike", "strike_position", "spot_price",
        "iv_call", "iv_put", "iv_call_converged", "iv_put_converged",
        "delta_call", "delta_put", "gamma_call", "gamma_put", "vega_call", "vega_put",
        "call_oi", "put_oi", "future_open", "days_to_expiry"]
FEATURES = ["current_iv", "rv_30", "rv_60",
            "iv_skew", "spot_price", "strike", "moneyness", "days_to_expiry",
            "minute_of_day", "weekday", "delta_call", "delta_put", "gamma_call",
            "gamma_put", "vega_call", "vega_put", "call_oi", "put_oi",
            "future_open"]


def rv(ps, ps2, left, right):
    n = right - left
    total = ps[right] - ps[left]
    variance = np.full(len(n), np.nan)
    valid = n >= 2
    variance[valid] = (ps2[right[valid]] - ps2[left[valid]]
                       - total[valid] ** 2 / n[valid]) / (n[valid] - 1)
    return np.sqrt(np.maximum(variance, 0)) * ANNUAL


def make_events():
    df = pd.read_csv(SOURCE, usecols=COLS, low_memory=False, parse_dates=["datetime", "Date", "expiry"])
    df = df.dropna(subset=["datetime", "Date", "expiry", "strike", "spot_price"])
    df = df.sort_values(["expiry", "Date", "strike", "datetime"])
    df = df.drop_duplicates(["expiry", "Date", "strike", "datetime"])
    events = []
    for (expiry, date), day in df.groupby(["expiry", "Date"], sort=True):
        # The ATM row supplies the unique underlying path; option IV comes from
        # the original entry strike, even when its position changes later.
        spot = day.loc[day.strike_position.eq(0), ["datetime", "spot_price"]]
        spot = spot.drop_duplicates("datetime").sort_values("datetime")
        if len(spot) < 240:
            continue
        st = spot.datetime.to_numpy(dtype="datetime64[ns]").astype("int64")
        price = spot.spot_price.to_numpy(float)
        returns = np.diff(np.log(price))
        ps = np.r_[0, returns.cumsum()]
        ps2 = np.r_[0, (returns ** 2).cumsum()]
        for strike, g in day.groupby("strike", sort=False):
            g = g.sort_values("datetime")
            starts = g.loc[g.strike_position.eq(0)]
            if starts.empty:
                continue
            times = g.datetime.to_numpy(dtype="datetime64[ns]").astype("int64")
            t0 = starts.datetime.to_numpy(dtype="datetime64[ns]").astype("int64")
            end_idx = np.searchsorted(times, t0 + H * NS_MIN)
            now_spot = np.searchsorted(st, t0)
            end_spot = np.searchsorted(st, t0 + H * NS_MIN)
            past = np.searchsorted(st, t0 - H * NS_MIN)
            keep = (end_idx < len(times)) & (end_spot < len(st)) & (now_spot < len(st)) & (past < len(st))
            if not keep.any():
                continue
            s = starts.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True)
            ei, a, b, p = end_idx[keep], now_spot[keep], end_spot[keep], past[keep]
            t0 = t0[keep]
            e = g.iloc[ei].reset_index(drop=True)
            good = ((times[ei] - (t0 + H * NS_MIN) <= 2 * NS_MIN)
                    & (st[b] - (t0 + H * NS_MIN) <= 2 * NS_MIN)
                    & (st[p] - (t0 - H * NS_MIN) <= 2 * NS_MIN)
                    & (np.abs(st[a] - t0) <= 2 * NS_MIN)
                    & s.iv_call_converged.eq(True).to_numpy()
                    & s.iv_put_converged.eq(True).to_numpy()
                    & e.iv_call_converged.eq(True).to_numpy()
                    & e.iv_put_converged.eq(True).to_numpy())
            if not good.any():
                continue
            s, e = s.loc[good].reset_index(drop=True), e.loc[good].reset_index(drop=True)
            a, b, p, t0 = a[good], b[good], p[good], t0[good]
            current_iv = 50 * (s.iv_call.to_numpy(float) + s.iv_put.to_numpy(float))
            future_iv = 50 * (e.iv_call.to_numpy(float) + e.iv_put.to_numpy(float))
            current_rv = rv(ps, ps2, p, a)
            future_rv = rv(ps, ps2, a, b)
            gap = current_iv - current_rv
            future_gap = future_iv - future_rv
            q = s[["datetime", "expiry", "Date", "strike", "spot_price", "days_to_expiry",
                   "delta_call", "delta_put", "gamma_call", "gamma_put", "vega_call", "vega_put",
                   "call_oi", "put_oi", "future_open"]].copy()
            q["current_iv"], q["current_rv"], q["current_gap"] = current_iv, current_rv, gap
            q["future_iv"], q["future_rv"], q["future_gap"] = future_iv, future_rv, future_gap
            q["n_trailing_returns"], q["n_forward_returns"] = a - p, b - a
            q["rv_30"] = rv(ps, ps2, np.searchsorted(st, t0 - 30 * NS_MIN), a)
            q["rv_60"] = rv(ps, ps2, np.searchsorted(st, t0 - 60 * NS_MIN), a)
            q["iv_skew"] = 100 * (s.iv_put.to_numpy(float) - s.iv_call.to_numpy(float))
            q["moneyness"] = (q.spot_price - q.strike) / q.spot_price
            q["minute_of_day"] = q.datetime.dt.hour * 60 + q.datetime.dt.minute
            q["weekday"] = q.datetime.dt.weekday
            # The intended signal requires realized volatility itself to rise
            # while the IV/RV distance shrinks at the fixed entry strike.
            q["converged"] = (future_rv > current_rv) & (np.abs(future_gap) < np.abs(gap))
            q = q[np.isfinite(q[["current_iv", "current_rv", "future_iv", "future_rv"]]).all(axis=1)]
            q = q[(q.n_trailing_returns >= 60) & (q.n_forward_returns >= 60)]
            q = q[q.current_gap > 0]
            events.append(q)
    if not events:
        raise RuntimeError("No qualifying events found")
    return pd.concat(events, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def scores(y, probability):
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {"n": len(y), "prevalence": float(np.mean(y)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "brier": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability)),
            "accuracy_at_0_5": float(accuracy_score(y, probability >= 0.5))}


def fit_model(name, params, x, y):
    if name != "lightgbm":
        raise ValueError(f"Unsupported model: {name}")
    model = LGBMClassifier(n_jobs=4, random_state=42, verbosity=-1, **params)
    model.fit(x, y)
    return model


def main():
    if SOURCE is None or not SOURCE.is_file():
        raise FileNotFoundError(f"Supply --source with {DATA_NAME} or place it in a parent data/ directory")
    OUT.mkdir(parents=True, exist_ok=True)
    d = make_events()
    d.to_csv(OUT / "events.csv", index=False)
    expiries = np.array(sorted(d.expiry.unique()))
    n = len(expiries)
    train_end, valid_end = int(n * 0.6), int(n * 0.8)
    train_exp, valid_exp, test_exp = expiries[:train_end], expiries[train_end:valid_end], expiries[valid_end:]
    train = d[d.expiry.isin(train_exp)]
    valid = d[d.expiry.isin(valid_exp)]
    test = d[d.expiry.isin(test_exp)]
    if min(len(train), len(valid), len(test)) == 0:
        raise RuntimeError("Chronological expiry split is empty")
    # Three expanding folds drawn only from the training era.
    folds = []
    for fraction in [0.4, 0.6, 0.8]:
        a = int(len(train_exp) * fraction)
        b = min(len(train_exp), a + max(3, int(len(train_exp) * 0.2)))
        tr = np.flatnonzero(train.expiry.isin(train_exp[:a]).to_numpy())
        va = np.flatnonzero(train.expiry.isin(train_exp[a:b]).to_numpy())
        if len(tr) and len(va):
            folds.append((tr, va))
    spaces = {
        "lightgbm": {"n_estimators": [150, 300], "num_leaves": [7, 15], "learning_rate": [0.03, 0.1],
                     "min_child_samples": [40, 100], "reg_lambda": [0, 2]},
    }
    xtr, ytr = train[FEATURES], train.converged.astype(int)
    xv, yv = valid[FEATURES], valid.converged.astype(int)
    xt, yt = test[FEATURES], test.converged.astype(int)
    results = []
    baseline = DummyClassifier(strategy="prior").fit(xtr, ytr)
    results.append({"model": "prevalence_baseline", "params": {},
                    "validation": scores(yv, baseline.predict_proba(xv)[:, 1]),
                    "test": scores(yt, baseline.predict_proba(xt)[:, 1])})
    for name, space in spaces.items():
        trials = list(ParameterSampler(space, n_iter=12, random_state=42))
        best = None
        for params in trials:
            fold_scores = []
            for a, b in folds:
                model = fit_model(name, params, xtr.iloc[a], ytr.iloc[a])
                prob = model.predict_proba(xtr.iloc[b])[:, 1]
                fold_scores.append(roc_auc_score(ytr.iloc[b], prob))
            mean_auc = float(np.mean(fold_scores))
            if best is None or mean_auc > best[0]:
                best = (mean_auc, params, fold_scores)
        model = fit_model(name, best[1], xtr, ytr)
        results.append({"model": name, "params": best[1], "cv_auc": best[0],
                        "cv_fold_auc": best[2], "validation": scores(yv, model.predict_proba(xv)[:, 1])})
        print(name, "CV AUC", round(best[0], 4), "validation AUC", round(results[-1]["validation"]["roc_auc"], 4), flush=True)
    # Pick by validation ROC AUC. The held-out test is touched once after selection.
    winner = max(results[1:], key=lambda r: r["validation"]["roc_auc"])
    name = winner["model"]
    combined = pd.concat([train, valid]).sort_values("datetime")
    model = fit_model(name, winner["params"], combined[FEATURES], combined.converged.astype(int))
    test_prob = model.predict_proba(xt)[:, 1]
    winner["test"] = scores(yt, test_prob)
    joblib.dump({"model": model, "features": FEATURES,
                 "target": "future_rv>current_rv and abs(future_iv-future_rv)<abs(current_iv-current_rv)"},
                OUT / "best_model.joblib")
    prediction = test[["datetime", "expiry", "Date", "strike", "current_iv", "current_rv",
                       "future_iv", "future_rv", "current_gap", "future_gap", "converged"]].copy()
    prediction["predicted_probability"] = test_prob
    prediction.to_csv(OUT / "test_predictions.csv", index=False)
    by_expiry = {}
    for expiry, group in prediction.groupby("expiry"):
        target = group.converged.astype(int).to_numpy()
        item = {"n": len(group), "prevalence": float(target.mean()),
                "mean_probability": float(group.predicted_probability.mean())}
        if len(np.unique(target)) == 2:
            item["roc_auc"] = float(roc_auc_score(target, group.predicted_probability))
        by_expiry[str(expiry.date())] = item
    # Resample whole expiries to reflect correlation between neighboring minutes.
    groups = [g for _, g in prediction.groupby("expiry")]
    rng = np.random.default_rng(42)
    cluster_auc = []
    for _ in range(2000):
        sampled = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if sampled.converged.nunique() == 2:
            cluster_auc.append(roc_auc_score(sampled.converged, sampled.predicted_probability))
    report = {"definition": "At t, average call/put IV > trailing 120-minute spot RV. At t+120, RV has risen and the absolute IV/RV gap is smaller. Option IV at both times uses the strike ATM at t.",
              "units": "annualized volatility percentage points", "sampling": "all eligible entry minutes; adjacent labels overlap",
              "coverage_rule": "At least 60 observed spot returns in each 120-minute window; source has missing minutes",
              "source": str(SOURCE), "features": FEATURES, "event_count": len(d),
              "expiry_count": n, "split": {"train": [str(train_exp[0]), str(train_exp[-1]), len(train)],
                                        "validation": [str(valid_exp[0]), str(valid_exp[-1]), len(valid)],
                                        "test": [str(test_exp[0]), str(test_exp[-1]), len(test)]},
              "selection_metric": "training-fold mean ROC AUC for LightGBM hyperparameters",
              "excluded_features": ["current_rv", "current_gap"],
              "winner": name, "results": results,
              "test_by_expiry": by_expiry,
              "test_expiry_bootstrap_auc_95pct": np.quantile(cluster_auc, [0.025, 0.975]).tolist()}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"winner": name, "test": winner["test"], "events": len(d)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="large historical five-strike CSV")
    parser.add_argument("--output", type=Path, help="isolated directory for trained artifacts")
    args = parser.parse_args()
    if args.source:
        SOURCE = args.source.expanduser().resolve()
    if args.output:
        OUT = args.output.expanduser().resolve()
    main()
