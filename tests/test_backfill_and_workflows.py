from pathlib import Path

import numpy as np
import pandas as pd

import gas_backfill_realized as backfill
import gas_part9_live_attribution as part9


def _ledger_row(**overrides):
    row = {
        "decision_date": "2026-09-08",
        "target_date": "2026-09-14",
        "anchor_week": "2026-09-07",
        "week_date": "2026-09-07",
        "is_live_forecast": 1,
        "protocol_eligible": 1,
        "pred_fusion": 4.0,
        "pred_persistence": 3.9,
        "actual": np.nan,
    }
    row.update(overrides)
    return row


def test_backfill_requires_exact_target_observation(monkeypatch):
    monkeypatch.setattr(backfill, "eastern_today", lambda: pd.Timestamp("2026-09-15"))
    frame = pd.DataFrame([_ledger_row()])
    wrong_date = pd.Series(
        [4.1], index=pd.DatetimeIndex([pd.Timestamp("2026-09-15")])
    )
    result, matured, newly = backfill.backfill(frame.copy(), wrong_date)
    assert matured == 1 and newly == 0
    assert pd.isna(result.loc[0, "actual"])

    exact = pd.Series(
        [3.9, 4.1],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-09-07"), pd.Timestamp("2026-09-14")]
        ),
    )
    result, matured, newly = backfill.backfill(frame.copy(), exact)
    assert matured == 1 and newly == 1
    assert result.loc[0, "actual"] == 4.1
    assert result.loc[0, "actual_date"] == "2026-09-14"


def test_backfill_excludes_hindsight_rows(monkeypatch):
    monkeypatch.setattr(backfill, "eastern_today", lambda: pd.Timestamp("2026-09-15"))
    legacy = pd.DataFrame(
        [
            _ledger_row(
                decision_date="2026-09-14",
                target_date="2026-09-14",
            )
        ]
    )
    prices = pd.Series(
        [4.1], index=pd.DatetimeIndex([pd.Timestamp("2026-09-14")])
    )
    result, matured, newly = backfill.backfill(legacy, prices)
    assert matured == 0 and newly == 0 and pd.isna(result.loc[0, "actual"])


def test_part9_cohort_contains_only_eligible_realized_rows(tmp_path):
    path = tmp_path / "prediction_log.csv"
    valid = _ledger_row(actual=4.1, actual_date="2026-09-14")
    legacy = _ledger_row(
        decision_date="2026-09-14",
        actual=4.1,
        actual_date="2026-09-14",
    )
    pd.DataFrame([valid, legacy]).to_csv(path, index=False)
    cohort = part9.load_realized_rows(path)
    assert len(cohort) == 1
    assert cohort.iloc[0]["decision_date"] == "2026-09-08"


def test_attribution_preserves_integer_counts_and_stored_direction():
    frame = pd.DataFrame(
        {
            "actual": [4.0, 4.1],
            "pred_fusion": [3.9, 4.2],
            "direction_correct": [1.0, 0.0],
            "pred_persistence": [3.8, 4.0],
        }
    )
    all_time = part9.compute_all_time_metrics(frame)
    rolling = part9.compute_rolling_metrics(frame, (4,))
    assert all_time["dir_acc"] == 0.5
    assert all_time["n_realized"] == 2
    assert rolling["rolling_4w"]["n"] == 2
    assert isinstance(rolling["rolling_4w"]["n"], int)


def test_workflow_contracts_are_release_aware_and_fail_fast():
    root = Path(__file__).resolve().parents[1]
    production = (root / ".github/workflows/weekly-production.yml").read_text()
    backfill_text = (root / ".github/workflows/weekly-backfill.yml").read_text()

    assert 'timezone: "America/New_York"' in production
    assert "* * 2" in production and "* * 3" in production
    assert "in_window" not in production
    assert "strategy-option=ours" not in production
    assert "git pull" not in production
    assert "requirements-production-lock.txt" in production
    assert "gas_oof_predictions.csv" in production

    assert 'timezone: "America/New_York"' in backfill_text
    assert "in_window" not in backfill_text
    assert "strategy-option=ours" not in backfill_text
    assert "requirements-backfill-lock.txt" in backfill_text


def test_ci_workflow_runs_compile_lint_and_tests():
    root = Path(__file__).resolve().parents[1]
    ci = (root / ".github/workflows/ci.yml").read_text()
    assert "requirements-ci-lock.txt" in ci
    assert "py_compile" in ci
    assert "ruff check" in ci
    assert "pytest -q" in ci
