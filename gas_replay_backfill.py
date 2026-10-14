#!/usr/bin/env python3
"""Replay the October 14 backfill failure offline in a disposable release copy.

The historical ledger is pinned to publication 8b1a88a6072fb17387a971faf2a0c352c6f3a6d2.
The exact October 12 GASREGCOVW observation, 4.150, comes from the failed run's
successful FRED fetch. No API requests, model imports, or source-data writes are
needed. This is regression evidence, not a new published release.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import gas_backfill_realized as backfill
import gas_part9_live_attribution as attribution
import gas_release_manifest as release
from gas_time_contract import (
    assert_immutable_prediction_inputs, read_prediction_log, sha256_file, strict_json_dump,
)

ROOT = Path(__file__).resolve().parent
FIXTURE_SHA256 = "43e7058cd95bab936493f82e5bf06770dde86c18e5c295175d33bcaaa56e8154"
MODEL_MODULES = ("torch", "sklearn", "xgboost", "hmmlearn", "yfinance")


def prepare_replay(root: Path, source: Path = ROOT) -> Path:
    """Validate published bytes, then install a pinned ledger in a temporary copy.

    Only the disposable manifest's ledger hash is refreshed to identify the
    fixture. Later real backfill staging must preserve every frozen fixture cell.
    This keeps the reproduction stable when main publishes subsequent weeks.
    """
    release.verify_manifest(source)
    fixture = source / "tests/fixtures/prediction_log_2026-10-14.csv"
    if sha256_file(fixture) != FIXTURE_SHA256:
        raise ValueError("Historical ledger fixture hash mismatch")
    shutil.copytree(source / "data", root / "data")
    (root / "artifacts_part3").mkdir()
    (root / "artifacts_part9").mkdir()
    ledger = root / "artifacts_part3/prediction_log.csv"
    shutil.copyfile(fixture, ledger)
    shutil.copyfile(fixture, root / "data/prediction_log.csv")
    manifest = release.verify_manifest(source)
    manifest["files"]["prediction_log.csv"] = {
        "sha256": sha256_file(ledger), "size_bytes": ledger.stat().st_size,
    }
    strict_json_dump(manifest, root / "data/release_manifest.json")
    release.verify_manifest(root)
    return ledger


def _identifiers(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [tuple(row[key] for key in (
            "pipeline_run_id", "pipeline_run_attempt", "source_code_sha",
        )) for row in csv.DictReader(handle)]


def replay(*, source: Path = ROOT, require_minimal: bool = False, verbose: bool = False) -> dict:
    """Run the same three Python stages as production without touching source data."""
    if require_minimal:
        present = [name for name in MODEL_MODULES if importlib.util.find_spec(name) is not None]
        if present:
            raise ValueError(f"Minimal backfill environment contains model modules: {present}")
    published = sorted((source / "data").glob("*")) + [
        source / "artifacts_part3/prediction_log.csv",
    ]
    hashes = {path: sha256_file(path) for path in published if path.is_file()}
    diagnostics = io.StringIO()
    try:
        with tempfile.TemporaryDirectory(prefix="gas-backfill-replay-") as directory:
            root = Path(directory)
            ledger = prepare_replay(root, source)
            before = read_prediction_log(ledger)
            identifiers = _identifiers(ledger)
            prices = pd.Series([4.150], index=pd.to_datetime(["2026-10-12"]))
            clock = lambda: pd.Timestamp("2026-10-14")
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(diagnostics))
                stack.enter_context(patch.dict(os.environ, {
                    "GASPRICE_ROOT": str(root),
                    "GASPRICE_PIPELINE_RUN_ID": "offline-backfill-identifier-replay",
                }))
                stack.enter_context(patch.object(backfill, "PROJECT_DIR", root))
                stack.enter_context(patch.object(backfill, "PREDLOG_PATH", ledger))
                stack.enter_context(patch.object(backfill, "eastern_today", clock))
                stack.enter_context(patch.object(attribution, "eastern_today", clock))
                stack.enter_context(patch.object(backfill, "fetch_gas_history_master", return_value=prices))
                if backfill.main([]) != 0 or attribution.main() != 0:
                    raise ValueError("Offline backfill or attribution failed")
                release.stage_backfill(root)
                release.verify_manifest(root)
                after = read_prediction_log(root / "data/prediction_log.csv")
                assert_immutable_prediction_inputs(before, after)
                if _identifiers(ledger) != identifiers:
                    raise ValueError("Offline replay changed identifier text")
                realized = after.loc[after["target_date"].eq("2026-10-12")].iloc[0]
                future = after.loc[after["target_date"].eq("2026-10-19")].iloc[0]
                if realized["actual"] != 4.150 or realized["actual_date"] != "2026-10-12":
                    raise ValueError("Offline replay did not score the exact target observation")
                if not pd.isna(future["actual"]):
                    raise ValueError("Offline replay scored an unavailable future observation")
                report = json.loads((root / "data/live_attribution_report.json").read_text(encoding="utf-8"))
                health = report["model_health"]
                if (report["n_realized"], health["min_required"], health["health_status"]) != (
                    4, 52, "INSUFFICIENT_DATA",
                ):
                    raise ValueError("Offline replay changed eligible-cohort health requirements")
                first_bytes = ledger.read_bytes()
                if backfill.main([]) != 0 or ledger.read_bytes() != first_bytes:
                    raise ValueError("Repeated backfill did not preserve ledger bytes")
            result = {
                "status": "PASS", "offline": True, "rows": len(after),
                "actual_date": "2026-10-12", "actual": 4.150,
                "identifiers_preserved": True, "frozen_inputs_preserved": True,
                "repeat_ledger_identical": True, "manifest_verified": True,
                "eligible_realized": 4, "min_required": 52,
                "health_status": "INSUFFICIENT_DATA", "minimal_runtime": require_minimal,
            }
    except Exception:
        print(diagnostics.getvalue(), file=sys.stderr)
        raise
    finally:
        if any(sha256_file(path) != digest for path, digest in hashes.items()):
            raise ValueError("Offline replay changed source publication files")
    if verbose:
        print(diagnostics.getvalue(), end="")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-minimal", action="store_true", help="Reject environments containing model dependencies")
    parser.add_argument("--verbose", action="store_true", help="Print the disposable pipeline diagnostics")
    args = parser.parse_args()
    print(json.dumps(replay(require_minimal=args.require_minimal, verbose=args.verbose), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
