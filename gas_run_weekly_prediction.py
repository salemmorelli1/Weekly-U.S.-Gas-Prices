#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_run_weekly_prediction.py
=============================
Canonical weekly production runner for the GasPriceForecast stack.

Authoritative weekly execution order
--------------------------------------
  gas_part0  (data infrastructure + FRED)
  gas_part0b (CollectAPI live prices + weather)
  gas_part0c (EIA API direct fetcher)
  gas_part6  (regime engine)
  gas_part1  (feature builder)
  gas_part2  (sklearn ensemble forecaster)
  gas_part2b (XGBoost sleeve — optional, non-blocking)
  gas_part2a (LSTM sleeve — optional, non-blocking)
  gas_part3  (governance + fusion + prediction log)
  gas_part9  (live attribution diagnostics)

Cadence
-------
  Weekly, Monday morning. EIA releases prior-week gas prices on Monday.
  Run after ~10am ET to ensure EIA data is live.

Usage
-----
  python gas_run_weekly_prediction.py
  python gas_run_weekly_prediction.py --direct      # skip validator if present
  python gas_run_weekly_prediction.py --with-backfill  # also run backfill_realized
  python gas_run_weekly_prediction.py --force       # run even if not Monday
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ── Colab / environment helpers ────────────────────────────────────────────────

def maybe_mount_drive() -> bool:
    try:
        from google.colab import drive
        mount_root = Path("/content/drive")
        if not (mount_root / "MyDrive").exists():
            drive.mount(str(mount_root), force_remount=False)
        else:
            print("Drive already mounted.")
        return True
    except Exception:
        return False


IN_COLAB = maybe_mount_drive()


def resolve_project_dir() -> Path:
    env_root = os.environ.get("GASPRICE_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_DIR = resolve_project_dir()
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GASPRICE_ROOT", str(PROJECT_DIR))

# ── Canonical files ────────────────────────────────────────────────────────────

CANONICAL_FILES: Dict[str, str] = {
    "PART0":  "gas_part0_data_infrastructure.py",
    "PART0B": "gas_part0b_collectapi_fetcher.py",
    "PART0C": "gas_part0c_eia_fetcher.py",
    "PART1":  "gas_part1_feature_builder.py",
    "PART2":  "gas_part2_forecaster.py",
    "PART2B": "gas_part2b_xgb_ensemble.py",    # optional sleeve
    "PART2A": "gas_part2a_lstm_sleeve.py",      # optional sleeve
    "PART3":  "gas_part3_governance.py",
    "PART6":  "gas_part6_regime_engine.py",
    "PART9":  "gas_part9_live_attribution.py",
}

BACKFILL_FILE = "gas_backfill_realized.py"

# Execution order — PART2B and PART2A are optional (non-blocking)
PIPELINE_ORDER: List[str] = [
    "PART0",
    "PART0B",
    "PART0C",
    "PART6",
    "PART1",
    "PART2",
    "PART2B",   # optional
    "PART2A",   # optional
    "PART3",
    "PART9",
]

OPTIONAL_PARTS = {"PART2B", "PART2A", "PART0B", "PART0C"}

# Required core parts for a valid run
REQUIRED_PARTS = [p for p in PIPELINE_ORDER if p not in OPTIONAL_PARTS]


# ── File helpers ───────────────────────────────────────────────────────────────

def check_files(project_dir: Path) -> Tuple[List[str], List[Tuple[str, Path, bool]]]:
    audit: List[Tuple[str, Path, bool]] = []
    missing: List[str] = []

    for label, filename in CANONICAL_FILES.items():
        path = (project_dir / filename).resolve()
        exists = path.exists()
        audit.append((label, path, exists))

    backfill_path = (project_dir / BACKFILL_FILE).resolve()
    audit.append(("BACKFILL", backfill_path, backfill_path.exists()))

    for label in REQUIRED_PARTS:
        path = (project_dir / CANONICAL_FILES[label]).resolve()
        if not path.exists():
            missing.append(CANONICAL_FILES[label])

    return missing, audit


# ── Subprocess runner ──────────────────────────────────────────────────────────

def run_subprocess(
    cmd: List[str],
    cwd: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"\n{'=' * 60}")
    print(f"Launching: {' '.join(str(x) for x in cmd)}")
    print(f"{'=' * 60}")

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print("\n--- STDERR ---")
        print(proc.stderr.rstrip())

    print(f"[exit={proc.returncode}]")
    return int(proc.returncode)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(project_dir: Path, with_backfill: bool = False) -> int:
    common_env = {
        "GASPRICE_ROOT": str(project_dir),
    }

    print("\n=== AUTHORITATIVE WEEKLY EXECUTION ORDER ===")
    print("Part0 -> Part0b* -> Part0c* -> Part6 -> Part1 -> Part2 -> "
          "Part2b* -> Part2a* -> Part3 -> Part9")
    print("(* optional / non-blocking)\n")

    for label in PIPELINE_ORDER:
        script_name = CANONICAL_FILES.get(label)
        if script_name is None:
            continue
        script = (project_dir / script_name).resolve()
        is_optional = label in OPTIONAL_PARTS

        if not script.exists():
            if is_optional:
                print(f"\n[INFO] {label} ({script_name}) not found — skipping (optional).")
                continue
            else:
                print(f"\n[ERROR] Required script {script_name} not found.")
                return 1

        rc = run_subprocess([sys.executable, str(script)], project_dir, extra_env=common_env)

        if rc != 0:
            if is_optional:
                print(f"\n[WARN] {label} exited with code {rc} — continuing (optional sleeve).")
            else:
                print(f"\n[ERROR] {label} failed with exit code {rc}.")
                return rc

    # Optional: backfill
    if with_backfill:
        backfill_script = (project_dir / BACKFILL_FILE).resolve()
        if backfill_script.exists():
            print("\n=== RUNNING BACKFILL ===")
            rc = run_subprocess(
                [sys.executable, str(backfill_script)],
                project_dir,
                extra_env=common_env,
            )
            if rc != 0:
                print(f"[WARN] Backfill exited with code {rc} — non-blocking.")
        else:
            print(f"\n[INFO] {BACKFILL_FILE} not found — backfill skipped.")

    return 0


# ── Day check ──────────────────────────────────────────────────────────────────

def is_monday() -> bool:
    return datetime.today().weekday() == 0  # Monday = 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical weekly GasPriceForecast production pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if today is not Monday.",
    )
    parser.add_argument(
        "--with-backfill",
        action="store_true",
        help="Also run gas_backfill_realized.py after the pipeline.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Accepted for compatibility; pipeline always runs directly.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring extra args: {' '.join(unknown)}")

    print(f"[Runner] ROOT: {PROJECT_DIR}")
    print(f"[Runner] IN_COLAB: {IN_COLAB}")
    today = datetime.today()
    print(f"[Runner] Today: {today.strftime('%A %Y-%m-%d')}")

    # Day-of-week guard
    if not is_monday() and not args.force:
        print("\n[Runner] Today is not Monday. The GasPriceForecast pipeline is designed")
        print("         to run on Monday after the EIA weekly release (~10am ET).")
        print("         Use --force to run on any day.")
        return 0

    # File audit
    print("\n=== CANONICAL FILE AUDIT ===")
    missing, audit = check_files(PROJECT_DIR)
    for label, path, exists in audit:
        status = "OK" if exists else "MISSING"
        print(f"  {label}: {status}")
        if not exists:
            print(f"    -> {path}")

    if missing:
        print(f"\n[ERROR] Missing required files: {missing}")
        return 1

    # Run
    rc = run_pipeline(PROJECT_DIR, with_backfill=args.with_backfill)

    if rc != 0:
        print(f"\n[Runner] Pipeline exited with code {rc}.")
        return rc

    print("\n[Runner] Weekly pipeline completed successfully.")
    print(f"[Runner] Forecast written to: "
          f"{PROJECT_DIR / 'artifacts_part3' / 'prediction_log.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
