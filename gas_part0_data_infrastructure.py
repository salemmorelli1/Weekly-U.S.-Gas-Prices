#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part0_data_infrastructure.py
=================================
Core data infrastructure for the Gas Price Forecasting model.

Responsibilities
----------------
- Google Drive mounting + Colab detection
- Project directory resolution and artifact folder creation
- DuckDB persistent data store (weekly gas price history)
- yfinance commodity price download (CL=F, RB=F, NG=F)
- FRED API client initialization
- Data freshness checks and schema versioning
- Canonical part0_summary.json for downstream consumers

Pipeline position: FIRST — all other parts depend on this.
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
                print(f"[setup] pip install {pkg}")
                subprocess.run([_sys.executable, "-m", "pip", "install", pkg, "-q"],
                               capture_output=True)


_colab_init(extra_packages=["yfinance", "fredapi", "duckdb", "pyarrow"])

import hashlib, json, os, warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from fredapi import Fred
    HAVE_FRED = True
except Exception:
    Fred = None
    HAVE_FRED = False

try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False

SCRIPT_VERSION = "GAS_PART0_V1_CANONICAL"


@dataclass(frozen=True)
class Part0Config:
    root_env_var: str = "GASPRICE_ROOT"
    history_start: str = "2000-01-01"
    history_end: str = date.today().strftime("%Y-%m-%d")

    commodity_tickers: Tuple[str, ...] = (
        "CL=F", "RB=F", "NG=F", "BNO", "USO", "UNG",
    )
    macro_tickers: Tuple[str, ...] = (
        "DX-Y.NYB", "^TNX", "^GSPC", "XLE",
    )

    duckdb_table: str = "gas_weekly_features"
    duckdb_filename: str = "gas_price_data.duckdb"
    freshness_warn_days: int = 14
    min_history_weeks: int = 104

    # FRED series IDs — populated in get_fred_series()
    # (dataclass field cannot be mutable dict, see helper below)


def get_fred_series() -> Dict[str, str]:
    return {
        "gas_us_avg":    "GASREGCOVW",
        "gas_midwest":   "GASMIDCOVW",
        "gas_gulf":      "GASGULFCOVW",
        "gas_east":      "GASEASTCOVW",
        "gas_west":      "GASWESTCOVW",
        "crude_wti":     "DCOILWTICO",
        "crude_brent":   "DCOILBRENTEU",
        "refinery_util": "WREFINER",
        "gas_stocks":    "WGTSTUS1",
        "gas_demand":    "WGFUPUS2",
        "crude_stocks":  "WCRSTUS1",
        "cpi_energy":    "CPIENGSL",
        "cpi_gasoline":  "CUSR0000SETB01",
        "unemployment":  "UNRATE",
        "gdp_growth":    "A191RL1Q225SBEA",
    }


def resolve_project_root(cfg: Part0Config) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def ensure_artifact_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "part0":  root / "artifacts_part0",
        "part1":  root / "artifacts_part1",
        "part2":  root / "artifacts_part2",
        "part2a": root / "artifacts_part2a",
        "part2b": root / "artifacts_part2b",
        "part3":  root / "artifacts_part3",
        "part6":  root / "artifacts_part6",
        "part9":  root / "artifacts_part9",
        "logs":   root / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_fred_client(api_key: Optional[str] = None) -> Optional[object]:
    if not HAVE_FRED:
        print("[Part0] fredapi not installed.")
        return None
    key = api_key or os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        print("[Part0] WARN: FRED_API_KEY not set. "
              "Free key: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    try:
        return Fred(api_key=key)
    except Exception as e:
        print(f"[Part0] FRED client init failed: {e}")
        return None


def fetch_fred_series(
    fred: object,
    series_id: str,
    start: str,
    end: str,
    freq: str = "W-MON",
) -> pd.Series:
    try:
        raw = fred.get_series(series_id, observation_start=start, observation_end=end)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.resample(freq).last().ffill()
        raw.name = series_id
        return raw
    except Exception as e:
        print(f"[Part0] FRED {series_id} failed: {e}")
        return pd.Series(dtype=float, name=series_id)


def fetch_yfinance_weekly(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(list(tickers), start=start, end=end,
                          progress=False, auto_adjust=True)
        close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        weekly = close.resample("W-MON").last()
        weekly.index = weekly.index.normalize()
        return weekly
    except Exception as e:
        print(f"[Part0] yfinance failed: {e}")
        return pd.DataFrame()


class GasPriceDuckDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def upsert(self, df: pd.DataFrame, table: str = "gas_weekly_features") -> None:
        if not HAVE_DUCKDB:
            return
        try:
            con = duckdb.connect(str(self.db_path))
            con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df LIMIT 0")
            con.execute(f"DELETE FROM {table} WHERE week_date IN (SELECT week_date FROM df)")
            con.execute(f"INSERT INTO {table} SELECT * FROM df")
            con.close()
        except Exception as e:
            print(f"[Part0] DuckDB upsert failed: {e}")

    def read(self, table: str = "gas_weekly_features") -> pd.DataFrame:
        if not HAVE_DUCKDB:
            return pd.DataFrame()
        try:
            con = duckdb.connect(str(self.db_path), read_only=True)
            df = con.execute(f"SELECT * FROM {table} ORDER BY week_date").df()
            con.close()
            return df
        except Exception as e:
            print(f"[Part0] DuckDB read failed: {e}")
            return pd.DataFrame()


def build_weekly_dataset(
    cfg: Part0Config,
    fred_client: Optional[object],
    out_dir: Path,
) -> pd.DataFrame:
    print("[Part0] Fetching commodity prices from yfinance...")
    all_tickers = list(cfg.commodity_tickers) + list(cfg.macro_tickers)
    yf_df = fetch_yfinance_weekly(tuple(all_tickers), cfg.history_start, cfg.history_end)

    rename_map = {
        "CL=F": "wti_crude", "RB=F": "rbob_gasoline", "NG=F": "natural_gas",
        "BNO": "brent_etf", "USO": "uso_etf", "UNG": "ung_etf",
        "DX-Y.NYB": "usd_index", "^TNX": "treasury_10y",
        "^GSPC": "sp500", "XLE": "energy_xle",
    }
    if not yf_df.empty:
        yf_df = yf_df.rename(columns={k: v for k, v in rename_map.items() if k in yf_df.columns})
        print(f"[Part0] yfinance: {len(yf_df)} weeks, {yf_df.shape[1]} columns")

    fred_frames: List[pd.Series] = []
    if fred_client is not None:
        print("[Part0] Fetching FRED series...")
        for name, series_id in get_fred_series().items():
            s = fetch_fred_series(fred_client, series_id, cfg.history_start, cfg.history_end)
            if not s.empty:
                s.name = name
                fred_frames.append(s)
                print(f"  [FRED] {name} ({series_id}): {len(s)} obs")
            else:
                print(f"  [FRED] {name}: EMPTY — skipping")
    else:
        print("[Part0] WARN: No FRED client — FRED series skipped.")

    idx = pd.date_range(start=cfg.history_start, end=cfg.history_end, freq="W-MON")
    master = pd.DataFrame(index=idx)
    master.index.name = "week_date"

    if not yf_df.empty:
        yf_df.index.name = "week_date"
        master = master.join(yf_df, how="left")

    for s in fred_frames:
        s.index.name = "week_date"
        master = master.join(s.rename(s.name), how="left")

    for col in ["cpi_energy", "cpi_gasoline", "unemployment", "gdp_growth"]:
        if col in master.columns:
            master[col] = master[col].ffill()

    if "gas_us_avg" in master.columns:
        n_before = len(master)
        master = master.dropna(subset=["gas_us_avg"])
        dropped = n_before - len(master)
        if dropped:
            print(f"[Part0] Dropped {dropped} rows with missing gas_us_avg")

    master = master.reset_index()
    master["week_date"] = pd.to_datetime(master["week_date"])
    print(f"[Part0] Master dataset: {len(master)} weeks x {master.shape[1]} columns")
    if len(master):
        print(f"[Part0] Range: {master['week_date'].min().date()} -> {master['week_date'].max().date()}")
    return master


def check_freshness(df: pd.DataFrame, cfg: Part0Config) -> Dict[str, object]:
    if df.empty or "week_date" not in df.columns:
        return {"status": "ERROR", "message": "Empty dataset"}
    latest = pd.to_datetime(df["week_date"]).max()
    age_days = (pd.Timestamp.today() - latest).days
    n_weeks = len(df)
    result = {
        "latest_week": latest.strftime("%Y-%m-%d"),
        "age_days": int(age_days),
        "n_weeks": int(n_weeks),
        "has_min_history": n_weeks >= cfg.min_history_weeks,
        "freshness_status": "OK" if age_days <= cfg.freshness_warn_days else "STALE",
    }
    print(f"[Part0] Freshness: {result['freshness_status']} | Latest: {latest.date()} "
          f"({age_days}d ago) | Weeks: {n_weeks}")
    return result


def compute_schema_hash(df: pd.DataFrame) -> str:
    return hashlib.md5("|".join(sorted(df.columns.tolist())).encode()).hexdigest()[:12]


def write_part0_summary(
    out_dir: Path, df: pd.DataFrame, cfg: Part0Config, freshness: Dict,
) -> None:
    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "n_weeks": len(df),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "schema_hash": compute_schema_hash(df),
        "freshness": freshness,
    }
    path = out_dir / "part0_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part0] Summary -> {path}")


def main() -> int:
    cfg = Part0Config()
    root = resolve_project_root(cfg)
    dirs = ensure_artifact_dirs(root)
    out_dir = dirs["part0"]
    os.environ.setdefault("GASPRICE_ROOT", str(root))

    print(f"[Part0] ROOT: {root}")
    print(f"[Part0] IN_COLAB: {_IN_COLAB}")
    print(f"[Part0] Version: {SCRIPT_VERSION}\n")

    fred_client = get_fred_client()
    df = build_weekly_dataset(cfg, fred_client, out_dir)

    if df.empty:
        print("[Part0] FATAL: Dataset empty. Check FRED_API_KEY and network.")
        return 1

    freshness = check_freshness(df, cfg)

    parquet_path = out_dir / "gas_weekly_master.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"[Part0] Parquet -> {parquet_path}")

    csv_path = out_dir / "gas_weekly_master.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Part0] CSV -> {csv_path}")

    db_path = out_dir / cfg.duckdb_filename
    GasPriceDuckDB(db_path).upsert(df, table=cfg.duckdb_table)
    print(f"[Part0] DuckDB -> {db_path}")

    write_part0_summary(out_dir, df, cfg, freshness)

    if not freshness.get("has_min_history"):
        print(f"[Part0] WARN: {freshness['n_weeks']} weeks of history "
              f"(min {cfg.min_history_weeks} required).")

    print("\n[Part0] Data infrastructure complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
