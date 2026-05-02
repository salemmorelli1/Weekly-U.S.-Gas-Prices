#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part0c_eia_fetcher.py
==========================
EIA (Energy Information Administration) data fetcher.

Responsibilities
----------------
- Fetch high-resolution EIA weekly petroleum supply data via the EIA API v2
  (free key at https://www.eia.gov/opendata/)
- Supplement / cross-validate FRED energy series with direct EIA sourcing
- Compute derived EIA features: days-of-supply, stock deviation, demand trend
- Write artifacts for downstream parts (part1 feature builder)

EIA API setup
-------------
1. Register at https://www.eia.gov/opendata/register.php (free)
2. Set your key: export EIA_API_KEY="your_key_here"

Key EIA series used
-------------------
PET.WGTSTUS1.W  — US Total Gasoline Stocks (weekly, thousand barrels)
PET.WGFUPUS2.W  — US Gasoline Finished Product Supplied (demand proxy)
PET.WCRSTUS1.W  — US Crude Oil Stocks
PET.WREFINER.W  — US Refinery Utilization Rate (%)
PET.EMD_EPD2D_PTE_NUS_DPG.W — US Regular Conventional Gas Price (weekly $/gal)
PET.WTTSTUS1.W  — US Total Petroleum Stocks

Pipeline position: THIRD — runs after gas_part0b.
"""
from __future__ import annotations

import sys as _sys
import os as _os

_IN_COLAB = "google.colab" in _sys.modules
_DRIVE_ROOT = _os.environ.get(
    "GASPRICE_ROOT",
    "/content/drive/MyDrive/GasPriceForecast" if _IN_COLAB
    else _os.path.join(_os.path.expanduser("~"), "GasPriceForecast"),
)


def _colab_init(extra_packages=None):
    if _IN_COLAB:
        if not _os.path.exists("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        _os.makedirs(_DRIVE_ROOT, exist_ok=True)
        _os.environ.setdefault("GASPRICE_ROOT", _DRIVE_ROOT)
    if extra_packages:
        import importlib, subprocess
        for pkg in extra_packages:
            mod = pkg.split("[")[0].replace("-", "_").split("==")[0]
            try:
                importlib.import_module(mod)
            except ImportError:
                subprocess.run([_sys.executable, "-m", "pip", "install", pkg, "-q"],
                               capture_output=True)


_colab_init(extra_packages=["requests", "pyarrow"])

import json, os, time, warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART0C_V1_CANONICAL"

EIA_API_BASE = "https://api.eia.gov/v2"

# EIA series IDs and clean names
EIA_SERIES: Dict[str, str] = {
    # Prices
    "eia_gas_us_regular":    "PET.EMD_EPD2D_PTE_NUS_DPG.W",
    "eia_gas_us_midgrade":   "PET.EMD_EPD2DM_PTE_NUS_DPG.W",
    "eia_gas_us_premium":    "PET.EMD_EPD2DP_PTE_NUS_DPG.W",
    "eia_gas_us_diesel":     "PET.EMD_EPM0_PTE_NUS_DPG.W",
    # Stocks & supply
    "eia_gas_stocks_total":  "PET.WGTSTUS1.W",
    "eia_crude_stocks":      "PET.WCRSTUS1.W",
    "eia_total_pet_stocks":  "PET.WTTSTUS1.W",
    # Demand
    "eia_gas_demand":        "PET.WGFUPUS2.W",
    "eia_total_pet_demand":  "PET.WTUPUUS2.W",
    # Refinery
    "eia_refinery_util":     "PET.WREFINER.W",
    "eia_crude_input_mbbl":  "PET.WCRRIUS2.W",
    # Imports/Exports
    "eia_crude_imports":     "PET.WCRIMUS2.W",
    "eia_gas_imports":       "PET.WGIMUS2.W",
}


@dataclass
class Part0cConfig:
    root_env_var: str = "GASPRICE_ROOT"
    out_dir_name: str = "artifacts_part0"
    history_start: str = "2000-01-01"
    history_end: str = date.today().strftime("%Y-%m-%d")
    request_timeout: int = 20
    request_retry: int = 3
    request_retry_delay: float = 2.0
    eia_freq: str = "weekly"
    # Days-of-supply window for derived features
    demand_avg_weeks: int = 4


def resolve_project_root() -> Path:
    env_root = os.environ.get("GASPRICE_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── EIA API client ─────────────────────────────────────────────────────────────

class EIAClient:
    """Thin wrapper around EIA API v2."""

    def __init__(self, api_key: Optional[str] = None, cfg: Part0cConfig = Part0cConfig()):
        self.key = api_key or os.environ.get("EIA_API_KEY", "").strip()
        self.cfg = cfg
        if not self.key:
            print("[Part0c] WARN: EIA_API_KEY not set. "
                  "Free key: https://www.eia.gov/opendata/register.php")

    def _fetch_series_v2(
        self,
        series_id: str,
        start: str,
        end: str,
    ) -> pd.Series:
        """
        Fetch a single EIA series via the v2 API.
        Falls back gracefully to empty Series on any error.
        """
        if not self.key:
            return pd.Series(dtype=float, name=series_id)

        # EIA v2 uses facets-based endpoint
        url = f"{EIA_API_BASE}/seriesid/{series_id}"
        params = {
            "api_key": self.key,
            "start": start,
            "end": end,
            "length": 5000,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }

        for attempt in range(self.cfg.request_retry):
            try:
                resp = requests.get(url, params=params, timeout=self.cfg.request_timeout)
                resp.raise_for_status()
                data = resp.json()
                rows = data.get("response", {}).get("data", [])
                if not rows:
                    return pd.Series(dtype=float, name=series_id)

                records = []
                for row in rows:
                    period = row.get("period", "")
                    value = row.get("value", None)
                    if period and value is not None:
                        try:
                            records.append((pd.to_datetime(period), float(value)))
                        except (ValueError, TypeError):
                            pass

                if not records:
                    return pd.Series(dtype=float, name=series_id)

                idx, vals = zip(*records)
                s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id)
                s = s.resample("W-MON").last().ffill()
                s.index = s.index.normalize()
                return s

            except requests.exceptions.HTTPError as e:
                print(f"[Part0c] HTTP error {e} for {series_id} attempt {attempt + 1}")
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 403:
                        print("[Part0c] FATAL: Invalid EIA API key.")
                        return pd.Series(dtype=float, name=series_id)
            except requests.exceptions.RequestException as e:
                print(f"[Part0c] Request error for {series_id}: {e} (attempt {attempt + 1})")

            if attempt < self.cfg.request_retry - 1:
                time.sleep(self.cfg.request_retry_delay)

        return pd.Series(dtype=float, name=series_id)

    def fetch_all_series(self, start: str, end: str) -> pd.DataFrame:
        """Fetch all EIA_SERIES and return a merged weekly DataFrame."""
        frames: List[pd.Series] = []
        print(f"[Part0c] Fetching {len(EIA_SERIES)} EIA series...")
        for clean_name, series_id in EIA_SERIES.items():
            s = self._fetch_series_v2(series_id, start, end)
            if not s.empty:
                s.name = clean_name
                frames.append(s)
                print(f"  [EIA] {clean_name}: {len(s)} obs")
            else:
                print(f"  [EIA] {clean_name}: EMPTY — skipping")

        if not frames:
            return pd.DataFrame()

        idx = pd.date_range(start=start, end=end, freq="W-MON")
        master = pd.DataFrame(index=idx)
        master.index.name = "week_date"
        for s in frames:
            master = master.join(s, how="left")

        return master.reset_index()


# ── Derived EIA features ───────────────────────────────────────────────────────

def compute_eia_derived_features(df: pd.DataFrame, cfg: Part0cConfig) -> pd.DataFrame:
    """
    Add derived supply/demand features on top of raw EIA series.
    All features are strictly backward-looking (no look-ahead bias).
    """
    df = df.copy()

    # --- Days-of-supply (stocks / (demand_per_day)) ---
    if "eia_gas_stocks_total" in df.columns and "eia_gas_demand" in df.columns:
        daily_demand = df["eia_gas_demand"] / 7.0  # weekly -> daily
        df["eia_gas_days_supply"] = df["eia_gas_stocks_total"] / daily_demand.replace(0, np.nan)

    # --- Stock deviation from rolling mean (z-score) ---
    for col in ["eia_gas_stocks_total", "eia_crude_stocks", "eia_total_pet_stocks"]:
        if col in df.columns:
            roll_mean = df[col].shift(1).rolling(52, min_periods=13).mean()
            roll_std  = df[col].shift(1).rolling(52, min_periods=13).std()
            df[f"{col}_zscore"] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)

    # --- Demand trend (4-week rolling vs 52-week rolling) ---
    if "eia_gas_demand" in df.columns:
        demand_4w  = df["eia_gas_demand"].shift(1).rolling(cfg.demand_avg_weeks, min_periods=2).mean()
        demand_52w = df["eia_gas_demand"].shift(1).rolling(52, min_periods=13).mean()
        df["eia_gas_demand_trend"] = (demand_4w / demand_52w.replace(0, np.nan)) - 1.0

    # --- Refinery utilization deviation ---
    if "eia_refinery_util" in df.columns:
        util_mean = df["eia_refinery_util"].shift(1).rolling(52, min_periods=13).mean()
        df["eia_refinery_util_dev"] = df["eia_refinery_util"] - util_mean

    # --- Price momentum (EIA price series) ---
    if "eia_gas_us_regular" in df.columns:
        df["eia_gas_chg_1w"]  = df["eia_gas_us_regular"].pct_change(1).shift(1)
        df["eia_gas_chg_4w"]  = df["eia_gas_us_regular"].pct_change(4).shift(1)
        df["eia_gas_chg_12w"] = df["eia_gas_us_regular"].pct_change(12).shift(1)

        # Crack spread proxy: gas vs crude price level ratio
        pass  # Handled in Part1 when both series are available

    print(f"[Part0c] Derived features added. Total columns: {df.shape[1]}")
    return df


# ── Merge into master parquet ──────────────────────────────────────────────────

def merge_eia_into_master(
    master_path: Path,
    eia_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Left-join EIA features into the master parquet on week_date."""
    if not master_path.exists():
        print(f"[Part0c] Master parquet not found at {master_path}. "
              "Run gas_part0 first.")
        return eia_df

    master = pd.read_parquet(master_path)
    master["week_date"] = pd.to_datetime(master["week_date"])
    eia_df["week_date"] = pd.to_datetime(eia_df["week_date"])

    # Drop any columns that already exist in master (avoid duplicates)
    overlap = [c for c in eia_df.columns if c in master.columns and c != "week_date"]
    eia_clean = eia_df.drop(columns=overlap)

    merged = master.merge(eia_clean, on="week_date", how="left")
    merged = merged.sort_values("week_date").reset_index(drop=True)

    parquet_path = out_dir / "gas_weekly_master.parquet"
    merged.to_parquet(parquet_path, index=False)
    merged.to_csv(out_dir / "gas_weekly_master.csv", index=False)
    print(f"[Part0c] Master parquet updated: {len(merged)} rows x {merged.shape[1]} cols -> {parquet_path}")
    return merged


# ── Summary ────────────────────────────────────────────────────────────────────

def write_part0c_summary(
    out_dir: Path,
    eia_df: pd.DataFrame,
    cfg: Part0cConfig,
) -> None:
    eia_cols = [c for c in eia_df.columns if c.startswith("eia_")]
    coverage: Dict[str, float] = {}
    for col in eia_cols:
        n = int(eia_df[col].notna().sum())
        coverage[col] = n

    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "n_weeks": len(eia_df),
        "eia_series_fetched": len(eia_cols),
        "eia_series_names": eia_cols,
        "coverage_n_obs": coverage,
        "history_start": cfg.history_start,
        "history_end": cfg.history_end,
    }
    path = out_dir / "part0c_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part0c] Summary -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part0cConfig()
    root = resolve_project_root()
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part0c] ROOT: {root}")
    print(f"[Part0c] Version: {SCRIPT_VERSION}\n")

    client = EIAClient(cfg=cfg)
    eia_df = client.fetch_all_series(cfg.history_start, cfg.history_end)

    if eia_df.empty:
        print("[Part0c] WARN: No EIA data fetched. Check EIA_API_KEY.")
        # Write empty summary and exit gracefully — non-blocking
        write_part0c_summary(out_dir, pd.DataFrame(), cfg)
        return 0

    # Compute derived features
    eia_df = compute_eia_derived_features(eia_df, cfg)

    # Write EIA-only parquet
    eia_path = out_dir / "gas_eia_features.parquet"
    eia_df.to_parquet(eia_path, index=False)
    print(f"[Part0c] EIA parquet -> {eia_path}")

    # Merge into master
    master_path = out_dir / "gas_weekly_master.parquet"
    merge_eia_into_master(master_path, eia_df, out_dir)

    write_part0c_summary(out_dir, eia_df, cfg)

    print("\n[Part0c] EIA data fetch and merge complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
