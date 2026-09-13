import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import gas_part0_data_infrastructure as part0
from gas_time_contract import (
    PROTOCOL_SCHEMA,
    pipeline_identity,
    protocol_eligible_record,
    strict_json_dump,
    validate_prospective_forecast,
)


def test_prospective_release_contract_accepts_tuesday_publication():
    decision, anchor, target = validate_prospective_forecast(
        decision_date="2026-09-08",
        anchor_week="2026-09-07",
        target_date="2026-09-14",
        is_live_forecast=1,
    )
    assert (decision.weekday(), anchor.weekday(), target.weekday()) == (1, 0, 0)


@pytest.mark.parametrize(
    ("decision", "anchor", "target", "is_live"),
    [
        ("2026-09-14", "2026-09-07", "2026-09-14", 1),
        ("2026-09-08", "2026-09-07", "2026-09-21", 1),
        ("2026-09-08", "2026-09-08", "2026-09-15", 1),
        ("2026-09-08", "2026-09-07", "2026-09-14", 0),
    ],
)
def test_prospective_release_contract_rejects_hindsight_and_bad_dates(
    decision, anchor, target, is_live
):
    with pytest.raises(ValueError):
        validate_prospective_forecast(
            decision_date=decision,
            anchor_week=anchor,
            target_date=target,
            is_live_forecast=is_live,
        )


def test_legacy_rows_do_not_enter_live_cohort():
    legacy = {
        "decision_date": "2026-08-18",
        "anchor_week": "2026-08-10",
        "target_date": "2026-08-17",
        "is_live_forecast": 1,
        "pred_fusion": 3.77,
    }
    valid = {
        "decision_date": "2026-09-08",
        "anchor_week": "2026-09-07",
        "target_date": "2026-09-14",
        "is_live_forecast": 1,
        "pred_fusion": 3.91,
    }
    assert not protocol_eligible_record(legacy)
    assert protocol_eligible_record(valid)


def test_strict_json_replaces_nonfinite_values(tmp_path):
    path = tmp_path / "strict.json"
    strict_json_dump(
        {
            "nan": np.nan,
            "inf": float("inf"),
            "when": datetime(2026, 9, 8, tzinfo=timezone.utc),
        },
        path,
    )
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    loaded = json.loads(raw)
    assert loaded["nan"] is None and loaded["inf"] is None


def test_pipeline_identity_is_stable_inside_run(monkeypatch):
    monkeypatch.setenv("GASPRICE_PIPELINE_RUN_ID", "run-123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    assert pipeline_identity() == {
        "pipeline_run_id": "run-123",
        "pipeline_run_attempt": "2",
        "source_code_sha": "abc123",
    }


def test_part0_freshness_uses_required_gas_observation(monkeypatch):
    monkeypatch.setattr(part0, "eastern_today", lambda: pd.Timestamp("2026-09-08"))
    cfg = part0.Part0Config(min_history_weeks=2)
    frame = pd.DataFrame(
        {
            "week_date": pd.to_datetime(["2026-08-31", "2026-09-07"]),
            "gas_us_avg": [3.80, 3.90],
        }
    )
    result = part0.check_freshness(frame, cfg)
    assert result["data_freshness_ok"] is True
    assert result["latest_week"] == "2026-09-07"
    assert result["forecast_target_week"] == "2026-09-14"


def test_part0_freshness_fails_without_required_series(monkeypatch):
    monkeypatch.setattr(part0, "eastern_today", lambda: pd.Timestamp("2026-09-08"))
    frame = pd.DataFrame(
        {"week_date": pd.to_datetime(["2026-09-07"]), "wti_crude": [62.0]}
    )
    result = part0.check_freshness(frame, part0.Part0Config(min_history_weeks=1))
    assert result["data_freshness_ok"] is False
    assert "GASREGCOVW" in result["message"]


def test_part0_uses_verified_core_fred_series():
    series = part0.get_fred_series()
    assert series["gas_us_avg"] == "GASREGCOVW"
    assert "gas_gulf" not in series
    assert PROTOCOL_SCHEMA == "V2_RELEASE_LEDGER"
