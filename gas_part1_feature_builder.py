#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part1_feature_builder.py
=============================
Feature engineering for the Gas Price Forecasting model.

Responsibilities
----------------
- Load the Part0 master parquet + Part6 regime tape
- Compute all model input features (lags, momentum, seasonality, fundamentals)
- Define the prediction target: next week's U.S. average regular gas price
- Write the canonical feature matrix and target series
- Write part1_summary.json for downstream consumers

Feature families
----------------
  LAG       — gas price lags (1w, 2w, 4w, 8w, 12w, 26w, 52w)
  MOMENTUM  — rolling returns and velocity
  VOLATILITY— realized vol, price range
  CRUDE     — WTI crude price, crude returns, crack spread
  RBOB      — RBOB gasoline futures signals
  EIA       — inventory, demand, refinery utilization
  SEASONAL  — month dummies, driving-season flag, hurricane-season flag
  MACRO     — USD index, Treasury yield, equity market
  REGIME    — HMM regime one-hot encoding

Pipeline position: FIFTH — after Part6.
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


_colab_init(extra_packages=["scikit-learn", "pyarrow"])

import json, os, warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART1_V1_CANONICAL"


@dataclass(frozen=True)
class Part1Config:
    root_env_var: str = "GASPRICE_ROOT"
    part0_dir_name: str = "artifacts_part0"
    part6_dir_name: str = "artifacts_part6"
    out_dir_name: str = "artifacts_part1"
    history_start: str = "2000-01-01"
    history_end: str = date.today().strftime("%Y-%m-%d")

    # Prediction horizon: 1 week forward
    horizon_weeks: int = 1

    # Lag windows (weeks)
    lag_windows: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 26, 52)

    # Rolling window sizes for vol/momentum features
    vol_windows: Tuple[int, ...] = (4, 8, 13, 26)
    momentum_windows: Tuple[int, ...] = (1, 2, 4, 8, 12, 26)

    # Minimum non-NaN rows required to write feature matrix
    min_clean_rows: int = 52

    # Feature imputation strategy for downstream models
    impute_strategy: str = "median"   # for sklearn SimpleImputer in Part2


def resolve_project_root(cfg: Part1Config) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── Feature builders ───────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, col: str, windows: Tuple[int, ...]) -> pd.DataFrame:
    """Add lagged level and return features for a given column."""
    for w in windows:
        df[f"{col}_lag_{w}w"] = df[col].shift(w)
        df[f"{col}_ret_{w}w"] = df[col].pct_change(w).shift(1)
    return df


def add_volatility_features(df: pd.DataFrame, col: str, windows: Tuple[int, ...]) -> pd.DataFrame:
    """Rolling realized volatility (std of weekly returns)."""
    ret = df[col].pct_change(1).shift(1)
    for w in windows:
        df[f"{col}_vol_{w}w"] = ret.rolling(w, min_periods=max(2, w // 2)).std()
    return df


def add_momentum_features(df: pd.DataFrame, col: str, windows: Tuple[int, ...]) -> pd.DataFrame:
    """Price momentum: current price relative to rolling mean."""
    for w in windows:
        roll_mean = df[col].shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
        df[f"{col}_mom_{w}w"] = (df[col].shift(1) / roll_mean.replace(0, np.nan)) - 1.0
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month dummies, driving season, hurricane season, year-end flags."""
    dt = pd.to_datetime(df["week_date"])
    month = dt.dt.month
    week_of_year = dt.dt.isocalendar().week.astype(int)

    # Month dummies (exclude December to avoid multicollinearity)
    for m in range(1, 12):
        df[f"month_{m:02d}"] = (month == m).astype(int)

    # Summer driving season: Memorial Day (late May) through Labor Day (early Sept)
    df["driving_season"] = ((month >= 5) & (month <= 9)).astype(int)

    # Peak driving (June–August)
    df["peak_driving"] = ((month >= 6) & (month <= 8)).astype(int)

    # Hurricane season (June 1 – November 30) — refinery disruption risk
    df["hurricane_season"] = ((month >= 6) & (month <= 11)).astype(int)

    # Winter heating demand (Dec–Feb)
    df["winter_demand"] = ((month == 12) | (month <= 2)).astype(int)

    # Spring refinery maintenance (Mar–Apr)
    df["spring_maintenance"] = ((month >= 3) & (month <= 4)).astype(int)

    # Week of year cyclical encoding (sin/cos)
    df["week_sin"] = np.sin(2 * np.pi * week_of_year / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week_of_year / 52.0)

    # Month cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return df


def add_crack_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crack spread and derived features."""
    if "wti_crude" in df.columns and "gas_us_avg" in df.columns:
        # Crude to gas ratio ($/gal vs $/bbl -> convert crude: $/bbl / 42 = $/gal)
        crude_per_gal = df["wti_crude"] / 42.0
        crack = df["gas_us_avg"].shift(1) - crude_per_gal.shift(1)
        df["crack_spread"] = crack
        roll_mean = crack.rolling(52, min_periods=13).mean()
        roll_std  = crack.rolling(52, min_periods=13).std()
        df["crack_spread_z"] = (crack - roll_mean) / roll_std.replace(0, np.nan)
        df["crack_spread_chg_4w"] = crack.diff(4)

    if "rbob_gasoline" in df.columns and "wti_crude" in df.columns:
        # RBOB crack: RBOB ($/gal) - crude ($/bbl)/42
        rbob_crack = df["rbob_gasoline"].shift(1) - (df["wti_crude"].shift(1) / 42.0)
        df["rbob_crack_spread"] = rbob_crack
        roll_mean = rbob_crack.rolling(26, min_periods=8).mean()
        roll_std  = rbob_crack.rolling(26, min_periods=8).std()
        df["rbob_crack_z"] = (rbob_crack - roll_mean) / roll_std.replace(0, np.nan)

    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Macro signal features."""
    if "treasury_10y" in df.columns:
        df["treasury_chg_4w"] = df["treasury_10y"].diff(4).shift(1)
        df["treasury_level"]  = df["treasury_10y"].shift(1)

    if "usd_index" in df.columns:
        df["usd_ret_4w"]  = df["usd_index"].pct_change(4).shift(1)
        df["usd_ret_12w"] = df["usd_index"].pct_change(12).shift(1)
        df["usd_z_26w"] = (
            (df["usd_index"].shift(1) - df["usd_index"].shift(1).rolling(26, min_periods=8).mean())
            / df["usd_index"].shift(1).rolling(26, min_periods=8).std().replace(0, np.nan)
        )

    if "sp500" in df.columns:
        df["sp500_ret_4w"]  = df["sp500"].pct_change(4).shift(1)
        df["sp500_vol_8w"]  = df["sp500"].pct_change(1).shift(1).rolling(8, min_periods=4).std()

    if "energy_xle" in df.columns:
        df["xle_ret_4w"] = df["energy_xle"].pct_change(4).shift(1)

    # CPI energy inflation signal
    if "cpi_energy" in df.columns:
        df["cpi_energy_chg_4w"] = df["cpi_energy"].pct_change(4).shift(1)

    return df


def add_regime_features(df: pd.DataFrame, regime_tape: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge regime labels and probabilities from Part6."""
    if regime_tape is None or regime_tape.empty:
        print("[Part1] WARN: No regime tape — regime features skipped.")
        return df

    regime_tape = regime_tape.copy()
    regime_tape["week_date"] = pd.to_datetime(regime_tape["week_date"])

    df = df.merge(regime_tape, on="week_date", how="left")

    # One-hot encode regime label
    if "regime_label" in df.columns:
        dummies = pd.get_dummies(
            df["regime_label"].shift(1),  # use prior week's regime
            prefix="regime",
            dtype=float,
        )
        df = pd.concat([df, dummies], axis=1)

    return df


def add_eia_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """EIA-derived ratio features."""
    if "eia_gas_stocks_total" in df.columns and "eia_gas_demand" in df.columns:
        daily_demand = df["eia_gas_demand"] / 7.0
        df["eia_days_supply"] = (
            df["eia_gas_stocks_total"].shift(1) / daily_demand.shift(1).replace(0, np.nan)
        )
        # Days supply z-score
        ds = df["eia_days_supply"]
        roll_mean = ds.rolling(52, min_periods=13).mean()
        roll_std  = ds.rolling(52, min_periods=13).std()
        df["eia_days_supply_z"] = (ds - roll_mean) / roll_std.replace(0, np.nan)

    return df


# ── Target variable ────────────────────────────────────────────────────────────

def build_target(df: pd.DataFrame, cfg: Part1Config) -> pd.Series:
    """
    Target: next week's U.S. average regular gas price ($/gal).
    This is a direct price level forecast.
    Also compute target_ret: the week-over-week return for diagnostics.
    """
    if "gas_us_avg" not in df.columns:
        print("[Part1] WARN: gas_us_avg not found — target will be empty.")
        return pd.Series(dtype=float, name="target_gas_price")

    target = df["gas_us_avg"].shift(-cfg.horizon_weeks)
    target.name = "target_gas_price"
    return target


# ── Assembly ───────────────────────────────────────────────────────────────────

def build_feature_matrix(
    master_df: pd.DataFrame,
    regime_tape: Optional[pd.DataFrame],
    cfg: Part1Config,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the full feature matrix and target series.
    Returns (X_df, y_series) both indexed by week_date.
    """
    df = master_df.copy()
    df["week_date"] = pd.to_datetime(df["week_date"])
    df = df.sort_values("week_date").reset_index(drop=True)

    print("[Part1] Building lag features...")
    df = add_lag_features(df, "gas_us_avg", cfg.lag_windows)

    print("[Part1] Building volatility features...")
    df = add_volatility_features(df, "gas_us_avg", cfg.vol_windows)

    print("[Part1] Building momentum features...")
    df = add_momentum_features(df, "gas_us_avg", cfg.momentum_windows)

    if "wti_crude" in df.columns:
        df = add_lag_features(df, "wti_crude", (1, 2, 4, 8, 12))
        df = add_volatility_features(df, "wti_crude", (4, 8))

    if "rbob_gasoline" in df.columns:
        df = add_lag_features(df, "rbob_gasoline", (1, 2, 4, 8))

    if "natural_gas" in df.columns:
        df = add_lag_features(df, "natural_gas", (1, 4, 8))

    print("[Part1] Building crack spread features...")
    df = add_crack_spread_features(df)

    print("[Part1] Building seasonal features...")
    df = add_seasonal_features(df)

    print("[Part1] Building macro features...")
    df = add_macro_features(df)

    print("[Part1] Building EIA ratio features...")
    df = add_eia_ratio_features(df)

    print("[Part1] Merging regime features...")
    df = add_regime_features(df, regime_tape)

    # Build target
    y = build_target(df, cfg)

    # Identify feature columns (exclude raw source cols and target)
    exclude_prefixes = ("week_date", "target_", "regime_label", "regime_int")
    exclude_cols = {
        "gas_us_avg", "wti_crude", "rbob_gasoline", "natural_gas",
        "brent_etf", "uso_etf", "ung_etf", "usd_index", "treasury_10y",
        "sp500", "energy_xle", "gas_midwest", "gas_gulf", "gas_east", "gas_west",
        "crude_wti", "crude_brent", "gas_stocks", "gas_demand", "crude_stocks",
        "cpi_energy", "cpi_gasoline", "unemployment", "gdp_growth",
        "eia_gas_us_regular", "eia_gas_us_midgrade", "eia_gas_us_premium",
        "eia_gas_us_diesel", "eia_gas_stocks_total", "eia_crude_stocks",
        "eia_total_pet_stocks", "eia_gas_demand", "eia_total_pet_demand",
        "eia_refinery_util", "eia_crude_input_mbbl", "eia_crude_imports",
        "eia_gas_imports", "gas_us_live",
        "gas_east_coast", "gas_midwest", "gas_gulf_coast", "gas_west_coast",
        "gas_rocky_mtn",
    }

    feature_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and c not in exclude_cols
        and not c.startswith("regime_prob_")
    ]

    X_df = df[["week_date"] + feature_cols].copy()
    X_df["target_gas_price"] = y.values

    # Drop rows where target is NaN (forecast horizon cutoff)
    n_before = len(X_df)
    X_df = X_df.dropna(subset=["target_gas_price"])
    print(f"[Part1] Dropped {n_before - len(X_df)} rows at forecast horizon cutoff")

    y_clean = X_df.pop("target_gas_price")

    print(f"[Part1] Feature matrix: {len(X_df)} rows x {len(feature_cols)} features")
    return X_df, y_clean


# ── Summary ────────────────────────────────────────────────────────────────────

def write_part1_summary(
    out_dir: Path,
    X_df: pd.DataFrame,
    y: pd.Series,
    cfg: Part1Config,
) -> None:
    feature_nan_rates = {
        col: float(X_df[col].isna().mean())
        for col in X_df.columns if col != "week_date"
    }
    high_nan = {k: v for k, v in feature_nan_rates.items() if v > 0.10}

    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "n_weeks": len(X_df),
        "n_features": len([c for c in X_df.columns if c != "week_date"]),
        "feature_names": [c for c in X_df.columns if c != "week_date"],
        "target_col": "target_gas_price",
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "date_range": {
            "start": str(X_df["week_date"].min().date()),
            "end": str(X_df["week_date"].max().date()),
        },
        "high_nan_features": high_nan,
        "horizon_weeks": cfg.horizon_weeks,
    }
    path = out_dir / "part1_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part1] Summary -> {path}")
    if high_nan:
        print(f"[Part1] WARN: {len(high_nan)} features with >10% NaN: "
              f"{list(high_nan.keys())[:5]}...")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part1Config()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    part0_dir = root / cfg.part0_dir_name
    part6_dir = root / cfg.part6_dir_name

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part1] ROOT: {root}")
    print(f"[Part1] Version: {SCRIPT_VERSION}\n")

    # Load master
    master_path = part0_dir / "gas_weekly_master.parquet"
    if not master_path.exists():
        print(f"[Part1] FATAL: {master_path} not found. Run gas_part0 first.")
        return 1

    master_df = pd.read_parquet(master_path)
    print(f"[Part1] Master: {len(master_df)} rows x {master_df.shape[1]} cols")

    # Load regime tape (optional)
    regime_tape: Optional[pd.DataFrame] = None
    regime_path = part6_dir / "gas_regime_tape.parquet"
    if regime_path.exists():
        regime_tape = pd.read_parquet(regime_path)
        print(f"[Part1] Regime tape: {len(regime_tape)} rows")
    else:
        print(f"[Part1] WARN: Regime tape not found at {regime_path}")

    # Build feature matrix
    X_df, y = build_feature_matrix(master_df, regime_tape, cfg)

    if len(X_df) < cfg.min_clean_rows:
        print(f"[Part1] FATAL: Only {len(X_df)} clean rows (min {cfg.min_clean_rows}).")
        return 1

    # Write outputs
    X_path = out_dir / "gas_feature_matrix.parquet"
    X_df.to_parquet(X_path, index=False)
    print(f"[Part1] Feature matrix -> {X_path}")

    y_path = out_dir / "gas_target.parquet"
    y_df = pd.DataFrame({"week_date": X_df["week_date"], "target_gas_price": y.values})
    y_df.to_parquet(y_path, index=False)
    print(f"[Part1] Target -> {y_path}")

    # Combined CSV
    combined = X_df.copy()
    combined["target_gas_price"] = y.values
    combined.to_csv(out_dir / "gas_feature_matrix.csv", index=False)

    write_part1_summary(out_dir, X_df, y, cfg)

    print(f"\n[Part1] Feature matrix: {len(X_df)} rows x "
          f"{len([c for c in X_df.columns if c != 'week_date'])} features")
    print(f"[Part1] Target range: ${y.min():.3f} - ${y.max():.3f}/gal")
    print(f"[Part1] Target mean: ${y.mean():.3f}/gal")
    print("\n[Part1] Feature builder complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
