"""IVRV entry rules: which model outputs qualify for a paper short straddle, the statutory cost estimate,
and the JSON form of published values."""
from __future__ import annotations

from datetime import date, datetime
import math
from typing import Any

import numpy as np
import pandas as pd


def current_costs(entry_premium: float, exit_premium: float, quantity: int, config: dict) -> dict[str, float]:
    """Current exchange/statutory costs with brokerage forced to zero."""
    sell_turnover = max(entry_premium, 0.0) * quantity
    buy_turnover = max(exit_premium, 0.0) * quantity
    total_turnover = sell_turnover + buy_turnover
    brokerage = 0.0
    stt = sell_turnover * float(config["stt_sell_premium_rate"])
    exchange = total_turnover * float(config["nse_options_transaction_rate"])
    sebi = total_turnover * float(config["sebi_turnover_rate"])
    stamp = buy_turnover * float(config["stamp_buy_premium_rate"])
    gst = float(config["gst_rate"]) * (exchange + sebi)
    total = brokerage + stt + exchange + sebi + stamp + gst
    return {
        "brokerage": brokerage,
        "stt": stt,
        "nse_transaction_charges": exchange,
        "sebi_charges": sebi,
        "stamp_duty": stamp,
        "gst": gst,
        "estimated_total": total,
        "assumed_exit_premium": exit_premium,
        "quantity": quantity,
    }


def convergence_entry_eligible(prediction: dict[str, Any], threshold: float) -> bool:
    """Return whether the classifier probability strictly clears the paper-entry gate."""
    try:
        probability = float(prediction["pred_convergence_probability_classifier"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(probability) and probability > float(threshold)


def paper_entry_route(
    prediction: dict[str, Any],
    convergence_threshold: float,
    near_gap_decimal: float,
    rv_decline_score_threshold: float,
) -> dict[str, Any]:
    """Route a paper entry to the model assigned to the current IV/RV regime."""
    try:
        gap = float(prediction["pred_no120_gate_iv_minus_rv"])
    except (KeyError, TypeError, ValueError):
        gap = math.nan
    try:
        event_score = float(prediction["pred_no120_event_score"])
    except (KeyError, TypeError, ValueError):
        event_score = math.nan
    try:
        probability = float(prediction["pred_convergence_probability_classifier"])
    except (KeyError, TypeError, ValueError):
        probability = math.nan

    result = {
        "eligible": False,
        "route": None,
        "reason": "missing_or_non_finite_iv_rv_gap",
        "iv_minus_rv": _json_value(gap),
        "near_gap_decimal": float(near_gap_decimal),
        "convergence_probability": _json_value(probability),
        "convergence_probability_threshold": float(convergence_threshold),
        "rv_decline_event_score": _json_value(event_score),
        "rv_decline_event_score_threshold": float(rv_decline_score_threshold),
    }
    if not math.isfinite(gap):
        return result

    if not math.isfinite(probability) or probability <= float(convergence_threshold):
        result["reason"] = "convergence_probability_below_entry_threshold"
        return result

    if gap > float(near_gap_decimal):
        result["route"] = "iv_above_rv_convergence"
        convergence_rule = bool(
            prediction.get("pred_strategy_rule_iv_above_rv_iv_declines_to_rv", False)
        )
        result["eligible"] = convergence_rule
        result["reason"] = (
            "iv_above_rv_convergence_passed"
            if result["eligible"]
            else "iv_above_rv_convergence_rule_or_probability_failed"
        )
        return result

    score_ok = math.isfinite(event_score) and event_score >= float(rv_decline_score_threshold)
    if gap < -float(near_gap_decimal):
        result["route"] = "iv_below_rv_sharp_rv_decline"
        result["reason"] = (
            "iv_below_rv_sharp_rv_decline_passed"
            if score_ok else "iv_below_rv_event_score_failed"
        )
    else:
        result["route"] = "iv_near_rv_sharp_rv_decline"
        result["reason"] = (
            "iv_near_rv_sharp_rv_decline_passed"
            if score_ok else "iv_near_rv_event_score_failed"
        )
    result["eligible"] = score_ok
    return result


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, (pd.Timestamp, datetime)) else value.isoformat()
    return value
