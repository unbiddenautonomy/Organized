"""Live BANKNIFTY spot-magnitude inference from an updating one-minute feed.

The feed may be a local CSV written by another process or an HTTP(S) CSV/JSON
endpoint. It must contain a timestamp and spot/close price. The runner validates
bar continuity, builds leakage-safe realized-volatility features, clips them to
training bounds, and emits mean and upper-tail forecasts per new bar.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "enhanced_rv" / "artifacts"
DEFAULT_OUTPUT = ROOT / "runtime" / "predictions.jsonl"
DEFAULT_BAR_OUTPUT = ROOT / "runtime" / "live_spot_bars.csv"
DEFAULT_PID_FILE = ROOT / "runtime" / "inference.pid"
EPS = 1e-12
HORIZONS = (15, 30, 60, 120)
YAHOO_BANKNIFTY = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEBANK?interval=1m&range=1d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Updating CSV/JSON path or HTTP(S) URL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bars-output", type=Path, default=DEFAULT_BAR_OUTPUT)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_FILE)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-age-seconds", type=float, default=180.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--allow-stale", action="store_true", help="Testing only: accept historical bars")
    return parser.parse_args()


def read_source(source: str) -> pd.DataFrame:
    if source == "yahoo-banknifty":
        source = YAHOO_BANKNIFTY
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "volpred-live/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
        if "json" in content_type or raw.lstrip().startswith((b"[", b"{")):
            payload = json.loads(raw)
            if isinstance(payload, dict) and isinstance(payload.get("chart"), dict):
                results = payload["chart"].get("result") or []
                if not results:
                    raise ValueError(f"Market-data response has no chart result: {payload['chart'].get('error')}")
                result = results[0]
                timestamps = result.get("timestamp", [])
                quote = (result.get("indicators", {}).get("quote") or [{}])[0]
                local_time = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata")
                frame = pd.DataFrame({"datetime": local_time, "spot_price": quote.get("close", [])})
                return normalize_bars(frame)
            if isinstance(payload, dict):
                for key in ("data", "bars", "candles", "results"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
            frame = pd.DataFrame(payload)
        else:
            frame = pd.read_csv(io.BytesIO(raw))
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Live source does not exist yet: {path}")
        frame = pd.read_json(path) if path.suffix.lower() in {".json", ".jsonl"} else pd.read_csv(path)
    return normalize_bars(frame)


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    lower = {str(column).lower(): column for column in frame.columns}
    time_column = next((lower[name] for name in ("datetime", "timestamp", "time", "date") if name in lower), None)
    price_column = next((lower[name] for name in ("spot_price", "close", "ltp", "price") if name in lower), None)
    if time_column is None or price_column is None:
        raise ValueError("Feed needs datetime/timestamp and spot_price/close columns")

    out = frame[[time_column, price_column]].rename(columns={time_column: "datetime", price_column: "spot_price"})
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["spot_price"] = pd.to_numeric(out["spot_price"], errors="coerce")
    out = out.dropna().loc[lambda x: x["spot_price"].gt(0)]
    out["datetime"] = out["datetime"].dt.floor("min")
    out = out.drop_duplicates("datetime", keep="last").sort_values("datetime")
    if out.empty:
        raise ValueError("Feed has no valid timestamp/price rows")
    out["date"] = out["datetime"].dt.date
    return out.reset_index(drop=True)


def latest_features(bars: pd.DataFrame) -> tuple[pd.Timestamp, float, pd.DataFrame, dict]:
    current_date = bars.iloc[-1]["date"]
    session = bars[bars["date"].eq(current_date)].set_index("datetime").between_time("09:15", "15:30").reset_index()
    if len(session) < 61:
        raise ValueError(f"Need 61 same-session prices; received {len(session)}")
    recent = session.tail(61)
    gaps = recent["datetime"].diff().dropna()
    if not gaps.eq(pd.Timedelta(minutes=1)).all():
        bad = gaps[gaps.ne(pd.Timedelta(minutes=1))].iloc[-1]
        raise ValueError(f"Non-contiguous one-minute bars in feature window; latest bad gap is {bad}")
    session["return"] = np.log(session["spot_price"] / session["spot_price"].shift(1))
    returns = session["return"].dropna()
    nonzero_fraction = float(returns.tail(60).abs().gt(1e-10).mean())
    if nonzero_fraction < 0.25:
        raise ValueError(f"Frozen or sparse feed: only {nonzero_fraction:.1%} nonzero returns")
    r2 = returns.pow(2)
    rv15 = float(r2.tail(15).sum())
    rq15 = float((15.0 / 3.0) * returns.tail(15).pow(4).sum())
    downside15 = float(r2.tail(15)[returns.tail(15).lt(0)].sum())
    products = returns.abs() * returns.abs().shift(1)
    bv15 = float((np.pi / 2.0) * products.tail(14).sum())
    minute = int(session.iloc[-1]["datetime"].hour * 60 + session.iloc[-1]["datetime"].minute)
    fraction = (minute - 555) / 375.0
    dow = int(session.iloc[-1]["datetime"].dayofweek)
    features = {
        f"log_rv_past_{window}": float(np.log(max(float(returns.tail(window).pow(2).sum()), EPS)))
        for window in (5, 15, 60)
    }
    features.update({
        "log_current_r2": float(np.log(max(float(r2.iloc[-1]), EPS))),
        "log_ewma_r2": float(np.log(max(float(r2.ewm(alpha=0.06, adjust=False).mean().iloc[-1]), EPS))),
        "log_rq_15": float(np.log(max(rq15, EPS))),
        "downside_share_15": float(np.clip(downside15 / max(rv15, EPS), 0, 1)),
        "jump_share_15": float(np.clip(max(rv15 - bv15, 0) / max(rv15, EPS), 0, 1)),
        "time_sin": float(np.sin(2 * np.pi * fraction)),
        "time_cos": float(np.cos(2 * np.pi * fraction)),
        "minutes_to_close_fraction": float(max(930 - minute, 0) / 375.0),
        "dow_sin": float(np.sin(2 * np.pi * dow / 5.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 5.0)),
        "time_bucket": minute // 30,
    })
    row = session.iloc[-1]
    quality = {"contiguous_bars": True, "nonzero_return_fraction": nonzero_fraction}
    return pd.Timestamp(row["datetime"]), float(row["spot_price"]), pd.DataFrame([features]), quality


def load_models() -> dict[int, dict]:
    models = {}
    for horizon in HORIZONS:
        path = MODEL_DIR / f"enhanced_rv_{horizon}m_model.joblib"
        payload = joblib.load(path)
        models[horizon] = payload
    return models


def predict(models: dict[int, dict], timestamp: pd.Timestamp, spot: float, features: pd.DataFrame, quality: dict) -> dict:
    forecasts = {}
    minute = timestamp.hour * 60 + timestamp.minute
    minutes_to_close = max(930 - minute, 0)
    for horizon, payload in models.items():
        names = payload["features"]
        raw_x = features[names]
        low, high = pd.Series(payload["clip_lower"]), pd.Series(payload["clip_upper"])
        clipped_x = raw_x.clip(lower=low, upper=high, axis=1)
        clipped_names = [name for name in names if float(raw_x.iloc[0][name]) != float(clipped_x.iloc[0][name])]
        log_variance = float(payload["model"].predict(clipped_x)[0])
        bucket = int(features.iloc[0]["time_bucket"])
        smear = payload["bucket_smear"].get(bucket, payload["bucket_smear"][-1])
        residual_q90 = payload["bucket_residual_q90"].get(bucket, payload["bucket_residual_q90"][-1])
        variance = float(np.exp(log_variance) * smear)
        variance_q90 = float(np.exp(log_variance + residual_q90))
        one_sigma_points = spot * np.sqrt(variance)
        multipliers = payload["absolute_return_multiplier"]
        available = minutes_to_close >= horizon
        forecasts[str(horizon)] = {
            "available": available,
            "trade_eligible": bool(available and not clipped_names),
            "algorithm": payload["algorithm"],
            "clipped_features": clipped_names,
            "future_variance": variance,
            "future_variance_q90": variance_q90,
            "annualized_rv": float(np.sqrt(variance * 252 * 375 / horizon)),
            "annualized_rv_q90": float(np.sqrt(variance_q90 * 252 * 375 / horizon)),
            "one_sigma_move_points": float(one_sigma_points),
            "expected_absolute_move_points": float(multipliers["mean"] * one_sigma_points),
            "absolute_move_q80_points": float(multipliers["q80"] * one_sigma_points),
            "absolute_move_q90_points": float(multipliers["q90"] * one_sigma_points),
        }
    return {
        "bar_timestamp": timestamp.isoformat(),
        "inference_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spot_price": spot,
        "quality": {**quality, "minutes_to_close": minutes_to_close},
        "features": {key: float(value) for key, value in features.iloc[0].items()},
        "forecasts": forecasts,
    }


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def store_bars(path: Path, bars: pd.DataFrame) -> None:
    """Persist source bars so matured live forecasts can be scored later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    current = bars[["datetime", "spot_price"]].copy()
    if path.exists():
        try:
            previous = pd.read_csv(path, parse_dates=["datetime"])
            current = pd.concat([previous, current], ignore_index=True)
        except (ValueError, OSError):
            pass
    current = current.dropna().drop_duplicates("datetime", keep="last").sort_values("datetime")
    temporary = path.with_suffix(path.suffix + ".tmp")
    current.to_csv(temporary, index=False)
    temporary.replace(path)


def bar_age_seconds(timestamp: pd.Timestamp) -> float:
    stamp = timestamp
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("Asia/Kolkata")
    return (pd.Timestamp.now(tz="UTC") - stamp.tz_convert("UTC")).total_seconds()


def main() -> None:
    args = parse_args()
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()), encoding="ascii")
    models = load_models()
    last_timestamp: pd.Timestamp | None = None
    print(f"Loaded {len(models)} enhanced RV models; polling {args.source}", flush=True)

    while True:
        try:
            bars = read_source(args.source)
            timestamp, spot, features, quality = latest_features(bars)
            age = bar_age_seconds(timestamp)
            if not args.allow_stale and (age < -60 or age > args.max_age_seconds):
                raise ValueError(f"Latest bar is stale or future-dated: {timestamp} (age {age:.0f}s)")
            if last_timestamp is None or timestamp > last_timestamp:
                record = predict(models, timestamp, spot, features, quality)
                store_bars(args.bars_output, bars)
                append_record(args.output, record)
                print(json.dumps(record), flush=True)
                last_timestamp = timestamp
        except Exception as exc:
            print(f"feed_error: {type(exc).__name__}: {exc}", flush=True)
            if args.once:
                raise SystemExit(1) from exc

        if args.once:
            return
        time.sleep(max(args.poll_seconds, 0.5))


if __name__ == "__main__":
    main()
