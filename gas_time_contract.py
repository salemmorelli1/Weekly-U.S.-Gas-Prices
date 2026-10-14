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

# CSV inference must never treat these labels as numbers or NA markers.
PREDICTION_TEXT_COLUMNS = (
    "decision_date", "target_date", "anchor_week", "week_date", "actual_date",
    "confidence", "publication_mode", "regime_label", "schema_version", "run_utc",
    "pipeline_run_id", "pipeline_run_attempt", "source_code_sha",
)
REALIZED_COLUMNS = frozenset({
    "actual", "actual_date", "mae", "rmse", "mape", "direction_correct",
})


def read_prediction_log(path: str | Path) -> pd.DataFrame:
    """Read local ledger CSV without changing opaque labels or float values.

    Converters retain empty and NA-looking text tokens as well as leading zeros
    and old labels ending in '.0'. Numeric columns still use normal NA handling.
    The round-trip float parser permits exact checks after CSV serialization.
    """
    return pd.read_csv(
        Path(path), engine="c", encoding="utf-8", float_precision="round_trip",
        converters={column: str for column in PREDICTION_TEXT_COLUMNS},
    )


def assert_immutable_prediction_inputs(prior: pd.DataFrame, updated: pd.DataFrame) -> None:
    """Allow only realized-evidence changes, with no numerical tolerance."""
    immutable = [column for column in prior if column not in REALIZED_COLUMNS]
    updated_immutable = [column for column in updated if column not in REALIZED_COLUMNS]
    if immutable != updated_immutable:
        raise ValueError("Backfill cannot rewrite immutable forecast input columns")
    try:
        pd.testing.assert_frame_equal(
            prior[immutable].reset_index(drop=True),
            updated[immutable].reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError("Backfill cannot rewrite immutable forecast inputs") from exc


def write_prediction_log_atomic(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Verify every serialized cell before replacing the canonical ledger."""
    expected = frame.copy(deep=True).reset_index(drop=True)

    def text_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if not isinstance(value, str):
            raise ValueError("Prediction log text fields require strings, not inferred numbers")
        return value

    for column in PREDICTION_TEXT_COLUMNS:
        if column in expected:
            expected[column] = expected[column].map(text_cell)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        expected.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        verified = read_prediction_log(temporary)
        try:
            pd.testing.assert_frame_equal(expected, verified, check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise ValueError("Prediction log CSV round-trip changed values") from exc
        temporary.replace(destination)
        return verified
    finally:
        temporary.unlink(missing_ok=True)


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

    if float(is_live_forecast) != 1:
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
    if decision <= anchor:
        raise ValueError("decision_date must follow the Monday source observation")
    return decision, anchor, target


def protocol_eligible_record(row: Mapping[str, Any]) -> bool:
    try:
        validate_prospective_forecast(
            decision_date=row.get("decision_date"),
            anchor_week=row.get("anchor_week", row.get("week_date")),
            target_date=row.get("target_date"),
            is_live_forecast=row.get("is_live_forecast", 0),
        )
        if row.get("schema_version") != PROTOCOL_SCHEMA:
            return False
        issued = pd.Timestamp(row.get("run_utc"))
        if pd.isna(issued) or issued.tzinfo is None:
            return False
        decision = normalize_date(row.get("decision_date"))
        if issued.tz_convert(EASTERN).date() != decision.date():
            return False
        if any(str(row.get(key, "")).strip().lower() in {"", "nan", "none", "<na>"}
               for key in ("pipeline_run_id", "source_code_sha")):
            return False
        prices = [float(row.get(key)) for key in ("pred_fusion", "pred_persistence")]
        return all(math.isfinite(value) and value > 0 for value in prices)
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
    # Encode explicitly so Windows newline translation cannot change the
    # content-addressed publication bytes produced on Linux CI.
    temporary.write_bytes(
        (json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    temporary.replace(destination)
