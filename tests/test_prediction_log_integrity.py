"""Regression fixtures based on the seven-row October 14 publication."""
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

import gas_backfill_realized as backfill
import gas_part9_live_attribution as attribution
import gas_release_manifest as release
import gas_part3_governance as governance
from gas_replay_backfill import prepare_replay, replay
from gas_time_contract import (
    assert_immutable_prediction_inputs, read_prediction_log, write_prediction_log_atomic,
)

ROOT = Path(__file__).resolve().parents[1]


def raw_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_raw_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def published_copy(tmp_path, monkeypatch):
    prepare_replay(tmp_path, ROOT)
    monkeypatch.setenv("GASPRICE_ROOT", str(tmp_path))
    monkeypatch.setenv("GASPRICE_PIPELINE_RUN_ID", "offline-identifier-regression")
    monkeypatch.setattr(backfill, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(backfill, "PREDLOG_PATH", tmp_path / "artifacts_part3/prediction_log.csv")
    monkeypatch.setattr(backfill, "eastern_today", lambda: pd.Timestamp("2026-10-14"))
    monkeypatch.setattr(attribution, "eastern_today", lambda: pd.Timestamp("2026-10-14"))
    monkeypatch.setattr(backfill, "fetch_gas_history_master", lambda *args: pd.Series(
        [4.150], index=pd.to_datetime(["2026-10-12"])))
    return tmp_path


def test_published_ledger_backfill_preserves_identifier_text(published_copy):
    path = published_copy / "artifacts_part3/prediction_log.csv"
    before = raw_rows(path)
    assert backfill.main([]) == 0
    after = raw_rows(path)
    for column in ("pipeline_run_id", "pipeline_run_attempt", "source_code_sha"):
        assert [row[column] for row in after] == [row[column] for row in before]


def test_published_ledger_backfill_attribution_and_staging(published_copy):
    assert backfill.main([]) == 0
    assert attribution.main() == 0
    release.stage_backfill(published_copy)
    release.verify_manifest(published_copy)
    rows = raw_rows(published_copy / "data/prediction_log.csv")
    target = next(row for row in rows if row["target_date"] == "2026-10-12")
    assert float(target["actual"]) == 4.150
    assert target["actual_date"] == "2026-10-12"
    report = json.loads((published_copy / "data/live_attribution_report.json").read_text())
    assert report["n_realized"] == 4
    assert report["model_health"]["min_required"] == 52


def test_manifest_rejects_tiny_frozen_forecast_rewrite(published_copy):
    path = published_copy / "artifacts_part3/prediction_log.csv"
    rows = raw_rows(path)
    rows[-1]["pred_fusion"] = str(float(rows[-1]["pred_fusion"]) + 1e-9)
    write_raw_rows(path, rows)
    with pytest.raises(ValueError, match="immutable"):
        release.stage_backfill(published_copy)


@pytest.mark.parametrize("token", [
    "", "38624403253", "35011816348.0", "000123", "9007199254740993",
    "NA", "N/A", "NULL", "None", "nan", "<NA>", "1e8", " padded ",
])
def test_opaque_text_survives_read_and_write(tmp_path, token):
    path = tmp_path / "ledger.csv"
    row = {
        "pipeline_run_id": token, "pipeline_run_attempt": token,
        "source_code_sha": token, "pred_fusion": "0.6100000000000001", "actual": "",
    }
    write_raw_rows(path, [row])
    frame = read_prediction_log(path)
    assert frame.iloc[0]["pred_fusion"] == float("0.6100000000000001")
    assert pd.isna(frame.iloc[0]["actual"])
    write_prediction_log_atomic(frame, path)
    for column in ("pipeline_run_id", "pipeline_run_attempt", "source_code_sha"):
        assert raw_rows(path)[0][column] == token


@pytest.mark.parametrize("column, replacement", [
    ("pipeline_run_id", "38624403253.0"),
    ("pipeline_run_attempt", "01"),
    ("source_code_sha", "0" * 40),
    ("schema_version", "REWRITTEN"),
    ("run_utc", "2026-10-14T19:34:03.569303+00:00"),
    ("pred_persistence", "4.150000001"),
])
def test_staging_rejects_changed_frozen_cells_without_resigning(published_copy, column, replacement):
    path = published_copy / "artifacts_part3/prediction_log.csv"
    published = {p: p.read_bytes() for p in (published_copy / "data").iterdir()}
    rows = raw_rows(path)
    rows[-1][column] = replacement
    write_raw_rows(path, rows)
    with pytest.raises(ValueError, match="immutable"):
        release.stage_backfill(published_copy)
    assert all(p.read_bytes() == value for p, value in published.items())


@pytest.mark.parametrize("change", ["add", "remove", "reorder"])
def test_backfill_rejects_row_changes(published_copy, change):
    path = published_copy / "artifacts_part3/prediction_log.csv"
    prior = read_prediction_log(path)
    if change == "add":
        updated = pd.concat([prior, prior.iloc[[-1]]], ignore_index=True)
    elif change == "remove":
        updated = prior.iloc[:-1]
    else:
        updated = prior.iloc[::-1]
    with pytest.raises(ValueError, match="immutable"):
        assert_immutable_prediction_inputs(prior, updated)


@pytest.mark.parametrize("corruption", ["identifier", "forecast", "row_count", "write_error"])
def test_temporary_csv_failure_preserves_canonical_ledger(published_copy, monkeypatch, corruption):
    path = published_copy / "artifacts_part3/prediction_log.csv"
    before = path.read_bytes()
    original_write = pd.DataFrame.to_csv

    def corrupt_write(frame, temporary, *args, **kwargs):
        assert Path(temporary) == path.with_suffix(".csv.tmp")
        original_write(frame, temporary, *args, **kwargs)
        if corruption == "write_error":
            raise OSError("injected disk write failure")
        rows = raw_rows(temporary)
        if corruption == "identifier":
            rows[-1]["source_code_sha"] = "0" * 40
        elif corruption == "forecast":
            rows[-1]["pred_fusion"] = str(float(rows[-1]["pred_fusion"]) + 1e-9)
        else:
            rows.pop()
        write_raw_rows(temporary, rows)

    monkeypatch.setattr(pd.DataFrame, "to_csv", corrupt_write)
    error = OSError if corruption == "write_error" else ValueError
    with pytest.raises(error):
        backfill.main([])
    assert path.read_bytes() == before
    assert not path.with_suffix(".csv.tmp").exists()


def test_writer_rejects_already_inferred_identifiers(published_copy):
    path = published_copy / "artifacts_part3/prediction_log.csv"
    before = path.read_bytes()
    inferred = pd.read_csv(path)
    with pytest.raises(ValueError, match="strings"):
        write_prediction_log_atomic(inferred, path)
    assert path.read_bytes() == before


def test_governance_append_preserves_prior_provenance(published_copy):
    path = published_copy / "artifacts_part3/prediction_log.csv"
    before = raw_rows(path)
    row = read_prediction_log(path).iloc[-1].copy()
    row.update(pd.Series({
        "decision_date": "2026-10-20", "target_date": "2026-10-26",
        "anchor_week": "2026-10-19", "week_date": "2026-10-19",
        "run_utc": "2026-10-20T19:00:00+00:00", "pipeline_run_id": "000123",
        "pipeline_run_attempt": "0002", "source_code_sha": "0" * 63 + "1",
    }))
    frame = governance.upsert_prediction_log(path, row)
    write_prediction_log_atomic(frame, path)
    after = raw_rows(path)
    for column in ("pipeline_run_id", "pipeline_run_attempt", "source_code_sha"):
        assert [r[column] for r in after[:-1]] == [r[column] for r in before]
        assert after[-1][column] == row[column]
    assert_immutable_prediction_inputs(read_prediction_log(ROOT / "tests/fixtures/prediction_log_2026-10-14.csv"), frame.iloc[:-1])


def test_offline_replay_runs_real_pipeline_and_preserves_source():
    result = replay()
    assert result["status"] == "PASS"
    assert result["eligible_realized"] == 4
    assert result["repeat_ledger_identical"] is True
