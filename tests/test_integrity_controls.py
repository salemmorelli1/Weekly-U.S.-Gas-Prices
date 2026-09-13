import json

import numpy as np
import pandas as pd
import pytest

import gas_part0c_eia_fetcher as part0c
import gas_part1_feature_builder as part1
import gas_part2_forecaster as part2
import gas_part2a_lstm_sleeve as part2a
import gas_part3_governance as part3
import gas_part6_regime_engine as part6
import gas_release_manifest as release
from gas_time_contract import sha256_file, strict_json_dump


def test_eia_fundamentals_are_lagged_to_release_availability():
    frame = pd.DataFrame(
        {
            "week_date": pd.date_range("2026-08-03", periods=3, freq="W-MON"),
            "eia_gas_us_regular": [3.1, 3.2, 3.3],
            "eia_gas_stocks_total": [100.0, 110.0, 120.0],
            "eia_gas_demand": [70.0, 77.0, 84.0],
        }
    )
    result = part0c.compute_eia_derived_features(frame, part0c.Part0cConfig())
    assert result["eia_gas_us_regular"].tolist() == [3.1, 3.2, 3.3]
    assert np.isnan(result.loc[0, "eia_gas_stocks_total"])
    assert result.loc[1, "eia_gas_stocks_total"] == 100.0
    assert result.loc[1, "eia_gas_demand"] == 70.0


def test_part1_keeps_current_price_and_excludes_descriptive_regime_features():
    dates = pd.date_range("2024-01-01", periods=80, freq="W-MON")
    master = pd.DataFrame(
        {
            "week_date": dates,
            "gas_us_avg": np.linspace(3.0, 4.0, len(dates)),
            "all_empty_candidate": np.nan,
        }
    )
    regime = pd.DataFrame(
        {
            "week_date": dates,
            "regime_label": "NORMAL__STATE_0",
            "regime_prob_0": 1.0,
        }
    )
    features, target = part1.build_feature_matrix(
        master,
        regime,
        part1.Part1Config(min_clean_rows=10),
    )
    assert "gas_us_avg_current" in features
    assert "all_empty_candidate" not in features
    assert not any(name.startswith("regime_") for name in features)
    assert features.iloc[-1]["is_live"] == 1
    assert pd.isna(target.iloc[-1])


def test_sklearn_models_handle_an_empty_feature_column():
    cfg = part2.Part2Config(
        hgb_max_iter=5,
        rf_n_estimators=5,
        gbm_n_estimators=5,
    )
    values = np.column_stack(
        [np.linspace(1.0, 2.0, 20), np.full(20, np.nan)]
    )
    target = np.linspace(3.0, 3.5, 20)
    for model in part2.build_models(cfg).values():
        model.fit(values, target)
        assert np.isfinite(model.predict(values[-2:])).all()


def test_lstm_feature_list_excludes_live_marker():
    frame = pd.DataFrame(
        {"week_date": [pd.Timestamp("2026-09-07")], "signal": [1.0], "is_live": [1]}
    )
    assert part2a.get_feature_cols(frame) == ["signal"]


def test_regime_semantic_labels_remain_state_unique():
    labels = np.array([0, 1, 0, 1])
    probs = np.full((4, 2), 0.5)
    features = pd.DataFrame({"gas_ret_4w": [0.0, 0.0, 0.0, 0.0]})
    cfg = part6.Part6Config(n_regimes=2)
    result = part6.label_regimes(
        labels, probs, cfg, features, ["gas_ret_4w"]
    )
    assert set(result) == {"NORMAL__STATE_0", "NORMAL__STATE_1"}


def test_master_lineage_accepts_part0c_hash_and_rejects_tamper(tmp_path, monkeypatch):
    monkeypatch.setenv("GASPRICE_PIPELINE_RUN_ID", "run-one")
    part0_dir = tmp_path / "artifacts_part0"
    part0_dir.mkdir()
    master = part0_dir / "gas_weekly_master.parquet"
    pd.DataFrame({"week_date": [pd.Timestamp("2026-09-07")], "gas_us_avg": [3.9]}).to_parquet(master)
    strict_json_dump(
        {"pipeline_run_id": "run-one", "master_parquet_sha256": "pre-merge"},
        part0_dir / "part0_summary.json",
    )
    strict_json_dump(
        {"pipeline_run_id": "run-one", "master_parquet_sha256": sha256_file(master)},
        part0_dir / "part0c_summary.json",
    )
    part1.validate_master_lineage(part0_dir, master)
    master.write_bytes(master.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        part1.validate_master_lineage(part0_dir, master)


def test_feature_lineage_rejects_mixed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GASPRICE_PIPELINE_RUN_ID", "current")
    matrix = tmp_path / "gas_feature_matrix.parquet"
    target = tmp_path / "gas_target.parquet"
    pd.DataFrame({"week_date": [pd.Timestamp("2026-09-07")], "x": [1.0]}).to_parquet(matrix)
    pd.DataFrame({"target_gas_price": [3.9]}).to_parquet(target)
    strict_json_dump(
        {
            "pipeline_run_id": "old",
            "feature_matrix_sha256": sha256_file(matrix),
            "target_sha256": sha256_file(target),
        },
        tmp_path / "part1_summary.json",
    )
    with pytest.raises(ValueError, match="different pipeline run"):
        part2.validate_feature_lineage(tmp_path)


def test_part3_core_lineage_verifies_run_anchor_and_tape(tmp_path, monkeypatch):
    monkeypatch.setenv("GASPRICE_PIPELINE_RUN_ID", "run-two")
    cfg = part3.Part3Config()
    part0_dir = tmp_path / cfg.part0_dir_name
    part2_dir = tmp_path / cfg.part2_dir_name
    part0_dir.mkdir()
    part2_dir.mkdir()
    tape = part2_dir / "gas_forecast_tape.parquet"
    pd.DataFrame({"week_date": [pd.Timestamp("2026-09-07")]}).to_parquet(tape)
    strict_json_dump(
        {
            "pipeline_run_id": "run-two",
            "freshness": {"latest_week": "2026-09-07"},
        },
        part0_dir / "part0_summary.json",
    )
    summary = {
        "pipeline_run_id": "run-two",
        "forecast_anchor_week": "2026-09-07",
        "forecast_tape_sha256": sha256_file(tape),
    }
    part3.validate_core_lineage(tmp_path, cfg, {"part2": summary})
    summary["forecast_tape_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        part3.validate_core_lineage(tmp_path, cfg, {"part2": summary})


def _prospective_row(value=3.9):
    return pd.Series(
        {
            "decision_date": "2026-09-08",
            "target_date": "2026-09-14",
            "anchor_week": "2026-09-07",
            "week_date": "2026-09-07",
            "is_live_forecast": 1,
            "protocol_eligible": 1,
            "pred_fusion": value,
            "schema_version": "V2_RELEASE_LEDGER",
            "pipeline_run_id": "run-two",
        }
    )


def test_prediction_ledger_is_immutable_and_detects_duplicates(tmp_path):
    path = tmp_path / "prediction_log.csv"
    first = part3.upsert_prediction_log(path, _prospective_row())
    first.to_csv(path, index=False)
    unchanged = part3.upsert_prediction_log(path, _prospective_row())
    assert len(unchanged) == 1
    with pytest.raises(RuntimeError, match="different forecast"):
        part3.upsert_prediction_log(path, _prospective_row(4.1))

    duplicated = pd.concat([first, first], ignore_index=True)
    duplicated.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="duplicate immutable"):
        part3.upsert_prediction_log(path, _prospective_row())


def test_build_prediction_row_requires_a_future_target(monkeypatch):
    monkeypatch.setattr(part3, "eastern_today", lambda: pd.Timestamp("2026-09-08"))
    fusion = pd.DataFrame(
        {
            "week_date": [pd.Timestamp("2026-09-07")],
            "is_live": [1],
            "pred_fusion": [3.9],
            "pred_candidate": [3.8],
            "pred_persistence": [3.9],
            "pred_part2": [3.8],
            "operator_validated": [False],
            "publication_mode": ["FAIL_CLOSED_PERSISTENCE"],
        }
    )
    row = part3.build_prediction_log_row(fusion, part3.Part3Config())
    assert row["target_date"] == "2026-09-14"
    assert row["protocol_eligible"] == 1

    monkeypatch.setattr(part3, "eastern_today", lambda: pd.Timestamp("2026-09-14"))
    with pytest.raises(ValueError, match="not after"):
        part3.build_prediction_log_row(fusion, part3.Part3Config())


def test_release_manifest_detects_tamper(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    evidence = data / "evidence.json"
    strict_json_dump({"ok": True}, evidence)
    strict_json_dump(
        {
            "schema_version": release.SCHEMA,
            "files": {
                evidence.name: {
                    "sha256": sha256_file(evidence),
                    "size_bytes": evidence.stat().st_size,
                }
            },
        },
        data / "release_manifest.json",
    )
    release.verify_manifest(tmp_path)
    evidence.write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        release.verify_manifest(tmp_path)
