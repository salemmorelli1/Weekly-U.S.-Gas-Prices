#!/usr/bin/env python3
"""Shared release, provenance, and strict-serialization contracts.

The EIA observation is a Monday price that is normally published Tuesday and
occasionally Wednesday after a holiday.  A live forecast is eligible only when
it is created after that observation is available and targets the following
Monday.  This module deliberately contains no network access.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")
PROTOCOL_SCHEMA = "V2_RELEASE_LEDGER"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def eastern_now() -> datetime:
    return utc_now().astimezone(EASTERN)


def eastern_today() -> pd.Timestamp:
    return pd.Timestamp(eastern_now().date())


def normalize_date(value: Any, field: str = "date") -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field} is not a valid date: {value!r}")
    if isinstance(parsed, pd.DatetimeIndex):
        raise ValueError(f"{field} must be scalar")
    if getattr(parsed, "tzinfo", None) is not None:
        parsed = parsed.tz_convert(EASTERN).tz_localize(None)
    return pd.Timestamp(parsed).normalize()


def next_target_monday(anchor_week: Any) -> pd.Timestamp:
    anchor = normalize_date(anchor_week, "anchor_week")
    if anchor.weekday() != 0:
        raise ValueError(f"anchor_week must be Monday, got {anchor.date()}")
    return anchor + pd.Timedelta(days=7)


def validate_prospective_forecast(
    *,
    decision_date: Any,
    anchor_week: Any,
    target_date: Any,
    is_live_forecast: Any,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    decision = normalize_date(decision_date, "decision_date")
    anchor = normalize_date(anchor_week, "anchor_week")
    target = normalize_date(target_date, "target_date")

    if not bool(int(is_live_forecast)):
        raise ValueError("prospective publication requires is_live_forecast=1")
    if anchor.weekday() != 0:
        raise ValueError(f"anchor_week must be Monday, got {anchor.date()}")
    if target.weekday() != 0:
        raise ValueError(f"target_date must be Monday, got {target.date()}")
    if target != anchor + pd.Timedelta(days=7):
        raise ValueError("target_date must be exactly seven days after anchor_week")
    if target <= decision:
        raise ValueError(
            f"target_date {target.date()} is not after decision_date {decision.date()}"
        )
    return decision, anchor, target


def protocol_eligible_record(row: Mapping[str, Any]) -> bool:
    try:
        validate_prospective_forecast(
            decision_date=row.get("decision_date"),
            anchor_week=row.get("anchor_week", row.get("week_date")),
            target_date=row.get("target_date"),
            is_live_forecast=row.get("is_live_forecast", 0),
        )
        prediction = float(row.get("pred_fusion"))
        return math.isfinite(prediction)
    except (TypeError, ValueError):
        return False


def pipeline_identity() -> dict[str, str]:
    run_id = (
        os.environ.get("GASPRICE_PIPELINE_RUN_ID", "").strip()
        or os.environ.get("GITHUB_RUN_ID", "").strip()
        or utc_now().strftime("local-%Y%m%dT%H%M%S%fZ")
    )
    return {
        "pipeline_run_id": run_id,
        "pipeline_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1").strip() or "1",
        "source_code_sha": os.environ.get("GITHUB_SHA", "").strip() or "LOCAL",
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def strict_json_dump(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
