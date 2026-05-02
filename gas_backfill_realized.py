#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_backfill_realized.py
=========================
Canonical realized price backfill for the GasPriceForecast stack.

Behavior
--------
- Mounts Google Drive when running in Colab
- Reads the canonical prediction_log.csv from artifacts_part3/
- For each matured row (target_date <= today), fetches the EIA actual
  weekly gas price (GASREGCOVW via FRED or EIA API)
- Computes MAE, MAPE, direction_correct for each backfilled row
- Writes updated prediction_log.csv back in place

Maturity rule
-------------
A prediction is considered matured when target_date <= today.
EIA releases weekly gas prices every Monday for the prior week.
Allow 2 extra days buffer (run Wednesday+ to ensure EIA data is live).

Usage
-----
  python gas_backfill_realized.py            # normal run
  python gas_backfill_realized.py --dry-run  # audit only, no write
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Colab / environment detection ─────────────────────────────────────────────
_IN_COLAB = "google.colab" in sys.modules
_DRIVE_ROOT = os.environ.get(
    "GASPRICE_ROOT",
    "/content/drive/MyDrive/GasPriceForecast" if _IN_COLAB
    else os.path.join(os.path.expanduser("~"), "GasPriceForecast"),
)


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

PREDLOG_PATH = PROJECT_DIR / "artifacts_part3" / "prediction_log.csv"

# ── API availability ───────────────────────────────────────────────────────────
try:
    from fredapi import Fred
    HAVE_FRED = True
except Exception:
    Fred = None
    HAVE_FRED = False

try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    requests = None
    HAVE_REQUESTS = False

# EIA FRED series for U.S. average regular gas price
EIA_FRED_SERIES = "GASREGCOVW"   # Weekly, Monday, $/gal
EIA_API_SERIES  = "PET.EMD_EPD2D_PTE_NUS_DPG.W"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _to_date(s) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(s).normalize()
    except Exception:
        return None


# ── Price history fetchers ─────────────────────────────────────────────────────

def fetch_gas_history_fred(start: str, end: str) -> pd.Series:
    """Fetch EIA weekly gas prices via FRED (GASREGCOVW)."""
    if not HAVE_FRED:
        print("[Backfill] fredapi not installed.")
        return pd.Series(dtype=float)

    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        print("[Backfill] FRED_API_KEY not set — cannot fetch via FRED.")
        return pd.Series(dtype=float)

    try:
        fred = Fred(api_key=key)
        raw = fred.get_series(EIA_FRED_SERIES, observation_start=start, observation_end=end)
        raw.index = pd.to_datetime(raw.index).normalize()
        print(f"[Backfill] FRED: {len(raw)} weekly gas price obs "
              f"({raw.index.min().date()} -> {raw.index.max().date()})")
        return raw
    except Exception as e:
        print(f"[Backfill] FRED fetch failed: {e}")
        return pd.Series(dtype=float)


def fetch_gas_history_eia_api(start: str, end: str) -> pd.Series:
    """Fetch EIA weekly gas prices via EIA API v2 (fallback)."""
    if not HAVE_REQUESTS:
        return pd.Series(dtype=float)

    key = os.environ.get("EIA_API_KEY", "").strip()
    if not key:
        print("[Backfill] EIA_API_KEY not set — skipping EIA API fallback.")
        return pd.Series(dtype=float)

    try:
        url = f"https://api.eia.gov/v2/seriesid/{EIA_API_SERIES}"
        params = {
            "api_key": key,
            "start": start,
            "end": end,
            "length": 5000,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("response", {}).get("data", [])
        if not rows:
            return pd.Series(dtype=float)

        records = []
        for row in rows:
            period = row.get("period", "")
            value  = row.get("value", None)
            if period and value is not None:
                try:
                    records.append((pd.to_datetime(period).normalize(), float(value)))
                except (ValueError, TypeError):
                    pass

        if not records:
            return pd.Series(dtype=float)

        idx, vals = zip(*records)
        s = pd.Series(vals, index=pd.DatetimeIndex(idx))
        s = s.sort_index()
        print(f"[Backfill] EIA API: {len(s)} weekly gas price obs")
        return s
    except Exception as e:
        print(f"[Backfill] EIA API fetch failed: {e}")
        return pd.Series(dtype=float)


def fetch_gas_history_master(start: str, end: str) -> pd.Series:
    """
    Try to load gas price history from the Part0 master parquet first
    (fastest, no API call), then fall back to FRED, then EIA API.
    """
    master_path = PROJECT_DIR / "artifacts_part0" / "gas_weekly_master.parquet"
    if master_path.exists():
        try:
            df = pd.read_parquet(master_path, columns=["week_date", "gas_us_avg"])
            df["week_date"] = pd.to_datetime(df["week_date"]).dt.normalize()
            s = df.set_index("week_date")["gas_us_avg"].dropna()
            s = s[s.index >= pd.to_datetime(start)]
            s = s[s.index <= pd.to_datetime(end)]
            if not s.empty:
                print(f"[Backfill] Master parquet: {len(s)} weekly prices "
                      f"({s.index.min().date()} -> {s.index.max().date()})")
                return s
        except Exception as e:
            print(f"[Backfill] Master parquet read failed: {e}")

    # FRED
    s = fetch_gas_history_fred(start, end)
    if not s.empty:
        return s

    # EIA API
    return fetch_gas_history_eia_api(start, end)


# ── Backfill logic ─────────────────────────────────────────────────────────────

def backfill(df: pd.DataFrame, price_history: pd.Series) -> Tuple[pd.DataFrame, int, int]:
    """
    Fill actual realized prices into matured prediction log rows.
    Returns (updated_df, matured_count, newly_backfilled_count).
    """
    today = pd.Timestamp.today().normalize()
    # Build date -> price lookup
    price_map: Dict[pd.Timestamp, float] = {
        idx.normalize(): float(val)
        for idx, val in price_history.items()
        if np.isfinite(float(val))
    }

    matured_count = 0
    newly_backfilled = 0

    for idx, row in df.iterrows():
        target_date = _to_date(row.get("target_date"))
        if target_date is None or target_date > today:
            continue

        matured_count += 1

        # Try exact date, then look within ±3 days
        realized_price: Optional[float] = price_map.get(target_date)
        if realized_price is None:
            for delta in [1, -1, 2, -2, 3, -3]:
                candidate = target_date + pd.Timedelta(days=delta)
                if candidate in price_map:
                    realized_price = price_map[candidate]
                    break

        if realized_price is None:
            continue

        already_done = np.isfinite(_safe_float(row.get("actual", np.nan)))
        df.at[idx, "actual"] = realized_price
        df.at[idx, "actual_date"] = target_date.strftime("%Y-%m-%d")

        # Compute error metrics
        pred_fusion = _safe_float(row.get("pred_fusion", np.nan))
        if np.isfinite(pred_fusion):
            abs_err = abs(realized_price - pred_fusion)
            df.at[idx, "mae"] = round(abs_err, 4)
            df.at[idx, "rmse"] = round(abs_err, 4)   # single-row RMSE = MAE
            if realized_price != 0:
                df.at[idx, "mape"] = round(abs_err / abs(realized_price) * 100, 4)

        # Direction accuracy (vs prior week)
        if idx > 0:
            prior_actual = _safe_float(df.at[idx - 1, "actual"] if "actual" in df.columns else np.nan)
            prior_pred   = _safe_float(df.at[idx - 1, "pred_fusion"] if "pred_fusion" in df.columns else np.nan)
            if np.isfinite(prior_actual) and np.isfinite(pred_fusion):
                true_dir = np.sign(realized_price - prior_actual)
                pred_dir = np.sign(pred_fusion - prior_actual)
                df.at[idx, "direction_correct"] = float(int(true_dir == pred_dir))

        if not already_done:
            newly_backfilled += 1

    return df, matured_count, newly_backfilled


# ── Audit summary ──────────────────────────────────────────────────────────────

def audit_paths() -> List[Tuple[str, Path, bool]]:
    artifacts = {
        "PREDLOG":         "artifacts_part3/prediction_log.csv",
        "PART3_FUSION":    "artifacts_part3/gas_fusion_tape.parquet",
        "PART2_SUMMARY":   "artifacts_part2/gas_part2_summary.json",
        "PART2B_SUMMARY":  "artifacts_part2b/gas_part2b_summary.json",
        "PART2A_SUMMARY":  "artifacts_part2a/gas_part2a_summary.json",
        "PART9_REPORT":    "artifacts_part9/live_attribution_report.json",
        "MASTER_PARQUET":  "artifacts_part0/gas_weekly_master.parquet",
    }
    rows = []
    for label, rel in artifacts.items():
        path = PROJECT_DIR / rel
        rows.append((label, path, path.exists()))
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backfill realized EIA gas prices into prediction log.")
    parser.add_argument("--dry-run", action="store_true", help="Audit only — no file write")
    parser.add_argument("--force", action="store_true", help="Re-backfill already-filled rows")
    args = parser.parse_args(argv)

    print(f"[Backfill] ROOT: {PROJECT_DIR}")
    print(f"[Backfill] IN_COLAB: {IN_COLAB}")
    print(f"[Backfill] Prediction log: {PREDLOG_PATH}")
    print(f"[Backfill] Dry run: {args.dry_run}")
    print()

    print("=== ARTIFACT AUDIT ===")
    for label, path, exists in audit_paths():
        status = "OK" if exists else "MISSING"
        print(f"  {label}: {status} ({path})")
    print()

    if not PREDLOG_PATH.exists():
        print("[Backfill] ERROR: prediction_log.csv not found. Run gas_part3 first.")
        return 1

    df = pd.read_csv(PREDLOG_PATH)
    if df.empty:
        print("[Backfill] WARN: prediction_log.csv is empty.")
        return 0

    print(f"[Backfill] Prediction log: {len(df)} rows")

    # Determine date range to fetch
    if "target_date" not in df.columns:
        print("[Backfill] ERROR: 'target_date' column missing from prediction log.")
        return 1

    dates = pd.to_datetime(df["target_date"], errors="coerce").dropna()
    if dates.empty:
        print("[Backfill] No valid target dates found.")
        return 0

    start_str = (dates.min() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp.today() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"[Backfill] Fetching price history {start_str} -> {end_str}...")

    price_history = fetch_gas_history_master(start_str, end_str)

    if price_history.empty:
        print("[Backfill] WARN: No price history fetched. Check API keys.")
        print("[Backfill] Set FRED_API_KEY or EIA_API_KEY environment variable.")
        return 1

    latest_price = pd.Timestamp(price_history.index.max()).date()
    print(f"[Backfill] Latest realized price: ${float(price_history.iloc[-1]):.3f}/gal "
          f"on {latest_price}")

    # Backfill
    df_updated, matured, newly = backfill(df, price_history)

    # Summary stats
    realized_mask = pd.to_numeric(df_updated.get("actual", pd.Series(dtype=float)),
                                  errors="coerce").notna()
    realized_count = int(realized_mask.sum())

    print()
    print("=== BACKFILL SUMMARY ===")
    print(f"  Total rows in log:     {len(df_updated)}")
    print(f"  Matured rows found:    {matured}")
    print(f"  Newly backfilled:      {newly}")
    print(f"  Total with actuals:    {realized_count}")
    print(f"  Latest price date:     {latest_price}")

    if realized_count > 0:
        actuals = pd.to_numeric(df_updated["actual"], errors="coerce").dropna()
        preds   = pd.to_numeric(df_updated.get("pred_fusion", pd.Series(dtype=float)),
                                 errors="coerce")
        mask = actuals.notna() & preds.notna()
        if mask.sum() >= 2:
            errors = (actuals[mask] - preds[mask]).abs()
            print(f"  Running MAE:           ${errors.mean():.4f}/gal")
            print(f"  Running MAPE:          "
                  f"{(errors / actuals[mask].abs()).mean() * 100:.2f}%")

    if args.dry_run:
        print()
        print("[Backfill] DRY RUN — no file written. Remove --dry-run to apply.")
        return 0

    # Write atomically
    tmp_path = PREDLOG_PATH.with_suffix(".csv.tmp")
    df_updated.to_csv(tmp_path, index=False)
    tmp_path.replace(PREDLOG_PATH)
    print(f"\n[Backfill] Prediction log written -> {PREDLOG_PATH}")

    # Verify round-trip
    verify = pd.read_csv(PREDLOG_PATH)
    assert len(verify) == len(df_updated), "Row count mismatch after write"
    print("[Backfill] Round-trip verification passed.")
    print("\n[Backfill] Backfill complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
