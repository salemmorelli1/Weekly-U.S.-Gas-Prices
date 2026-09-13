#!/usr/bin/env python3
"""Stage and verify the small, reviewable dashboard release surface."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gas_time_contract import sha256_file, strict_json_dump

SCHEMA = "GAS_RELEASE_MANIFEST_V1"

REQUIRED_SOURCES = (
    "artifacts_part0/part0_summary.json",
    "artifacts_part0/part0c_summary.json",
    "artifacts_part1/part1_summary.json",
    "artifacts_part2/gas_part2_summary.json",
    "artifacts_part2/gas_forecast_tape.csv",
    "artifacts_part2/gas_oof_predictions.csv",
    "artifacts_part3/gas_part3_summary.json",
    "artifacts_part3/pipeline_status.json",
    "artifacts_part3/prediction_log.csv",
    "artifacts_part6/gas_regime_meta.json",
    "artifacts_part9/live_attribution_report.json",
    "artifacts_part9/live_attribution_tape.csv",
)

OPTIONAL_SOURCES = (
    "artifacts_part2a/gas_part2a_summary.json",
    "artifacts_part2b/gas_part2b_summary.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {token} in {path}")
            ),
        )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _require_equal(label: str, values: list[str]) -> str:
    cleaned = [str(value).strip() for value in values]
    if not cleaned[0] or len(set(cleaned)) != 1:
        raise ValueError(f"{label} disagree: {cleaned}")
    return cleaned[0]


def validate_release(root: Path) -> dict[str, str]:
    """Validate run, date, and immutable-ledger agreement before staging."""
    p0 = _load_json(root / "artifacts_part0/part0_summary.json")
    p0c = _load_json(root / "artifacts_part0/part0c_summary.json")
    p1 = _load_json(root / "artifacts_part1/part1_summary.json")
    p2 = _load_json(root / "artifacts_part2/gas_part2_summary.json")
    p3 = _load_json(root / "artifacts_part3/gas_part3_summary.json")
    p6 = _load_json(root / "artifacts_part6/gas_regime_meta.json")
    p9 = _load_json(root / "artifacts_part9/live_attribution_report.json")
    status = _load_json(root / "artifacts_part3/pipeline_status.json")

    run_id = _require_equal(
        "pipeline run IDs",
        [
            p0.get("pipeline_run_id", ""),
            p0c.get("pipeline_run_id", ""),
            p1.get("pipeline_run_id", ""),
            p2.get("pipeline_run_id", ""),
            p3.get("pipeline_run_id", ""),
            p6.get("pipeline_run_id", ""),
            p9.get("pipeline_run_id", ""),
            status.get("pipeline_run_id", ""),
        ],
    )

    latest = p3.get("latest_prediction", {})
    source_date = _require_equal(
        "source observation dates",
        [
            p0.get("freshness", {}).get("latest_week", ""),
            p2.get("forecast_anchor_week", ""),
            latest.get("anchor_week", ""),
            status.get("source_observation_date", ""),
        ],
    )
    target_date = _require_equal(
        "forecast target dates",
        [
            p0.get("freshness", {}).get("forecast_target_week", ""),
            p2.get("forecast_target_week", ""),
            latest.get("target_date", ""),
            status.get("target_date", ""),
        ],
    )

    log_path = root / "artifacts_part3/prediction_log.csv"
    ledger = pd.read_csv(log_path, dtype={"pipeline_run_id": "string"})
    target_rows = ledger.loc[
        ledger["target_date"].astype(str).str.strip().eq(target_date)
    ]
    if len(target_rows) != 1:
        raise ValueError(
            f"immutable ledger must contain one row for {target_date}; found {len(target_rows)}"
        )
    row = target_rows.iloc[0]
    prediction_run_id = _require_equal(
        "published prediction run IDs",
        [
            row.get("pipeline_run_id", ""),
            latest.get("pipeline_run_id", ""),
            status.get("prediction_pipeline_run_id", ""),
        ],
    )
    if str(row.get("anchor_week", "")) != source_date:
        raise ValueError("ledger anchor does not match release source date")
    if int(row.get("protocol_eligible", 0)) != 1:
        raise ValueError("latest ledger row is not protocol eligible")

    return {
        "pipeline_run_id": run_id,
        "prediction_pipeline_run_id": prediction_run_id,
        "source_observation_date": source_date,
        "target_date": target_date,
        "operator_validated": str(bool(p3.get("operator_validated", False))).lower(),
        "publication_mode": str(p3.get("publication_mode", "")),
    }


def stage_release(root: Path) -> Path:
    metadata = validate_release(root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    staged: list[Path] = []
    for relative in REQUIRED_SOURCES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required release file missing: {relative}")
        destination = data_dir / source.name
        shutil.copy2(source, destination)
        staged.append(destination)

    for relative in OPTIONAL_SOURCES:
        source = root / relative
        destination = data_dir / source.name
        if not source.is_file():
            destination.unlink(missing_ok=True)
            continue
        if source.suffix == ".json":
            summary = _load_json(source)
            if str(summary.get("pipeline_run_id", "")) != metadata["pipeline_run_id"]:
                destination.unlink(missing_ok=True)
                continue
        shutil.copy2(source, destination)
        staged.append(destination)

    manifest = {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staged)
        },
    }
    manifest_path = data_dir / "release_manifest.json"
    strict_json_dump(manifest, manifest_path)
    verify_manifest(root)
    return manifest_path


def stage_backfill(root: Path) -> Path:
    """Refresh only realized-evidence files and re-hash the release surface."""
    manifest_path = root / "data/release_manifest.json"
    manifest = verify_manifest(root)
    data_dir = manifest_path.parent
    sources = (
        root / "artifacts_part3/prediction_log.csv",
        root / "artifacts_part9/live_attribution_report.json",
        root / "artifacts_part9/live_attribution_tape.csv",
    )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"backfill release file missing: {source}")
        shutil.copy2(source, data_dir / source.name)

    report = _load_json(root / "artifacts_part9/live_attribution_report.json")
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["last_backfill_run_id"] = str(report.get("pipeline_run_id", ""))
    files = {
        path.name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.name != manifest_path.name
    }
    manifest["files"] = files
    strict_json_dump(manifest, manifest_path)
    verify_manifest(root)
    return manifest_path


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "data/release_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported release-manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("release manifest has no files")
    for filename, record in files.items():
        path = root / "data" / filename
        if not path.is_file():
            raise FileNotFoundError(f"manifest file missing: data/{filename}")
        if sha256_file(path) != str(record.get("sha256", "")):
            raise ValueError(f"manifest hash mismatch: data/{filename}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"manifest size mismatch: data/{filename}")
        if path.suffix == ".json":
            _load_json(path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "backfill", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if args.command == "stage":
        path = stage_release(root)
        print(f"Release staged and verified: {path}")
    elif args.command == "backfill":
        path = stage_backfill(root)
        print(f"Backfill release staged and verified: {path}")
    else:
        verify_manifest(root)
        print(f"Release manifest verified: {root / 'data/release_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
