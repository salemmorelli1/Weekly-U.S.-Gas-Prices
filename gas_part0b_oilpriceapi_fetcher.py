#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part0b_oilpriceapi_fetcher.py
==================================
OilPriceAPI live commodity price fetcher.

REPLACES gas_part0b_collectapi_fetcher.py (Audit 2026-08 provider swap):
CollectAPI's gas price + weather bundles are no longer free-tier. OilPriceAPI
(https://www.oilpriceapi.com) offers a free tier (200 requests/month) with
live commodity prices. This part now stamps a Monday-morning LIVE snapshot of
the gasoline-complex commodities onto the current week's master row:

  live_wti_usd          — WTI crude, $/bbl
  live_brent_usd        — Brent crude, $/bbl
  live_rbob_usd         — RBOB gasoline futures, $/gal
  live_natgas_usd       — Henry Hub natural gas, $/MMBtu
  live_diesel_usd       — ULSD diesel, $/gal
  live_heating_oil_usd  — Heating oil, $/gal

plus two derived gap signals vs the prior weekly close from the master:

  live_wti_gap_pct      — live WTI vs last weekly WTI close (%)
  live_rbob_gap_pct     — live RBOB vs last weekly RBOB close (%)

All live_* columns are SUPPLEMENTARY (excluded from model features via the
"live_" prefix in Part 1, exactly like gas_us_live before them). They exist
for the dashboard, run summaries, and future research — the weekly feature
contract does not depend on them, so this part remains optional/non-blocking.

Budget: ONE batched API request per run (~5/month), far inside the free
tier's 200/month. A per-code fallback is used only if the batched call's
response shape is unrecognized (worst case 6 requests).

OilPriceAPI setup
-----------------
1. Free key at https://www.oilpriceapi.com (no credit card)
2. export OILPRICEAPI_KEY="your_key_here"
   (GitHub Actions: repo Settings -> Secrets -> Actions -> OILPRICEAPI_KEY)

Endpoint: GET https://api.oilpriceapi.com/v1/prices/latest?by_code=CODE[,CODE...]
Auth:     header "Authorization: Token <key>"

Pipeline position: SECOND — after gas_part0, before gas_part0c.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART0B_V2_OILPRICEAPI"

OILPRICEAPI_BASE = "https://api.oilpriceapi.com/v1"
LATEST_ENDPOINT = f"{OILPRICEAPI_BASE}/prices/latest"

# OilPriceAPI commodity code -> master column name
COMMODITY_CODES: Dict[str, str] = {
    "WTI_USD":           "live_wti_usd",
    "BRENT_CRUDE_USD":   "live_brent_usd",
    "GASOLINE_RBOB_USD": "live_rbob_usd",
    "NATURAL_GAS_USD":   "live_natgas_usd",
    "ULSD_DIESEL_USD":   "live_diesel_usd",
    "HEATING_OIL_USD":   "live_heating_oil_usd",
}

# Sanity bands (USD) per code — a price outside its band is discarded rather
# than stamped into the master (protects against unit/response-shape drift).
_SANITY_BANDS: Dict[str, Tuple[float, float]] = {
    "WTI_USD":           (10.0, 250.0),   # $/bbl
    "BRENT_CRUDE_USD":   (10.0, 250.0),   # $/bbl
    "GASOLINE_RBOB_USD": (0.4, 8.0),      # $/gal
    "NATURAL_GAS_USD":   (0.5, 30.0),     # $/MMBtu
    "ULSD_DIESEL_USD":   (0.4, 8.0),      # $/gal
    "HEATING_OIL_USD":   (0.4, 8.0),      # $/gal
}


@dataclass(frozen=True)
class Part0bConfig:
    root_env_var: str = "GASPRICE_ROOT"
    out_dir_name: str = "artifacts_part0"
    request_timeout: int = 20
    request_retry: int = 3
    request_retry_delay: float = 2.0


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


# ── OilPriceAPI client ─────────────────────────────────────────────────────────

class OilPriceAPIClient:
    """Thin wrapper around OilPriceAPI /v1/prices/latest."""

    def __init__(self, api_key: Optional[str] = None,
                 cfg: Part0bConfig = Part0bConfig()):
        self.key = api_key or os.environ.get("OILPRICEAPI_KEY", "").strip()
        self.cfg = cfg
        if not self.key:
            print("[Part0b] WARN: OILPRICEAPI_KEY not set. "
                  "Free key: https://www.oilpriceapi.com")

    def _get(self, params: Dict) -> Optional[Dict]:
        if not self.key:
            return None
        headers = {
            "Authorization": f"Token {self.key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.cfg.request_retry):
            try:
                resp = requests.get(LATEST_ENDPOINT, params=params,
                                    headers=headers,
                                    timeout=self.cfg.request_timeout)
                if resp.status_code in (401, 403):
                    print("[Part0b] FATAL: OilPriceAPI key rejected "
                          f"(HTTP {resp.status_code}). Check OILPRICEAPI_KEY.")
                    return None
                if resp.status_code == 429:
                    print("[Part0b] WARN: OilPriceAPI rate limit hit "
                          "(free tier: 200 requests/month).")
                    return None
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                print(f"[Part0b] Request error: {e} (attempt {attempt + 1})")
                if attempt < self.cfg.request_retry - 1:
                    time.sleep(self.cfg.request_retry_delay)
        return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_price_records(payload: Optional[Dict]) -> List[Dict]:
        """
        Normalize the /prices/latest payload into a list of
        {"code": ..., "price": ...} records.

        Handles the documented single-code shape:
            {"status":"success","data":{"price":79.22,"code":"BRENT_CRUDE_USD",...}}
        and defensively handles batched shapes where data is a list or holds a
        "prices" list — the multi-code response shape is not pinned down in
        the public docs, so the parser accepts all three rather than assuming.
        """
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("prices"), list):
                return [r for r in data["prices"] if isinstance(r, dict)]
            if "code" in data:
                return [data]
            return []
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    @staticmethod
    def _parse_price(record: Dict) -> Optional[Tuple[str, float]]:
        code = str(record.get("code", "")).strip().upper()
        if code not in COMMODITY_CODES:
            return None
        raw = record.get("price", record.get("value"))
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        lo, hi = _SANITY_BANDS.get(code, (0.0, float("inf")))
        if not (lo <= v <= hi):
            print(f"[Part0b] Discarding {code}={v} (outside sanity band "
                  f"{lo}-{hi})")
            return None
        return code, v

    # ------------------------------------------------------------------
    # Public fetchers
    # ------------------------------------------------------------------
    def fetch_live_prices(self) -> Dict[str, float]:
        """
        Fetch all tracked commodities. ONE batched request; per-code fallback
        only if the batched response can't be parsed (worst case 6 requests,
        still trivial against the 200/month free tier).
        Returns {master_column_name: price}.
        """
        if not self.key:
            return {}

        results: Dict[str, float] = {}

        batched = self._get({"by_code": ",".join(COMMODITY_CODES.keys())})
        for record in self._extract_price_records(batched):
            parsed = self._parse_price(record)
            if parsed:
                code, v = parsed
                results[COMMODITY_CODES[code]] = v

        if results:
            print(f"[Part0b] Batched fetch: {len(results)}/{len(COMMODITY_CODES)} "
                  "commodities")
            return results

        print("[Part0b] Batched response unrecognized — falling back to "
              "per-code requests.")
        for code, col in COMMODITY_CODES.items():
            payload = self._get({"by_code": code})
            for record in self._extract_price_records(payload):
                parsed = self._parse_price(record)
                if parsed:
                    results[col] = parsed[1]
        print(f"[Part0b] Per-code fetch: {len(results)}/{len(COMMODITY_CODES)} "
              "commodities")
        return results


# ── Derived gap signals ────────────────────────────────────────────────────────

def compute_gap_signals(df: pd.DataFrame,
                        live: Dict[str, float]) -> Dict[str, float]:
    """
    Live price vs the most recent weekly close in the master — the freshest
    read on where the gasoline complex has moved since Friday.
    """
    gaps: Dict[str, float] = {}

    def _last_close(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else None

    pairs = [
        ("live_wti_usd",  "wti_crude",     "live_wti_gap_pct"),
        ("live_rbob_usd", "rbob_gasoline", "live_rbob_gap_pct"),
    ]
    for live_col, close_col, gap_col in pairs:
        lv = live.get(live_col)
        cl = _last_close(close_col)
        if lv is not None and cl not in (None, 0):
            gaps[gap_col] = round((lv / cl - 1.0) * 100.0, 3)

    return gaps


# ── Master stamping (update-only, mirrors the CollectAPI-era contract) ─────────

def stamp_live_observation(
    df: pd.DataFrame,
    live_data: Dict[str, float],
) -> pd.DataFrame:
    """
    Stamp live commodity observations onto the current week's EXISTING row.

    Update-only by design (Audit 2026-08): appending a brand-new row here
    would create a second, price-less live anchor and degrade the forecast —
    the stack's single live anchor is the latest EIA-priced week from Part 0.
    If the current Monday's row does not exist yet (FRED lag), the live
    extras are simply skipped for this run.
    """
    today = pd.Timestamp.today().normalize()
    current_monday = today - pd.Timedelta(days=today.weekday())

    if "week_date" not in df.columns or not live_data:
        return df

    df["week_date"] = pd.to_datetime(df["week_date"])
    existing = df[df["week_date"] == current_monday]

    if existing.empty:
        print(f"[Part0b] Current Monday ({current_monday.date()}) not yet in "
              "master (FRED lag?) — live data NOT stamped this run. "
              "Update-only by design.")
        return df

    idx = existing.index[0]
    for col, val in live_data.items():
        if col not in df.columns:
            df[col] = np.nan
        df.at[idx, col] = val
    print(f"[Part0b] Stamped {len(live_data)} live values onto "
          f"{current_monday.date()}")

    return df.sort_values("week_date").reset_index(drop=True)


# ── Summary ────────────────────────────────────────────────────────────────────

def write_part0b_summary(
    out_dir: Path,
    live_prices: Dict[str, float],
    gaps: Dict[str, float],
    stamped: bool,
) -> None:
    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "oilpriceapi.com",
        "live_prices": {k: round(v, 4) for k, v in live_prices.items()},
        "gap_signals_pct": gaps,
        "n_commodities_fetched": len(live_prices),
        "stamped_into_master": bool(stamped),
    }
    path = out_dir / "part0b_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part0b] Summary -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part0bConfig()
    root = resolve_project_root()
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part0b] ROOT: {root}")
    print(f"[Part0b] Version: {SCRIPT_VERSION}\n")

    client = OilPriceAPIClient(cfg=cfg)
    if not client.key:
        print("[Part0b] No API key — live commodity snapshot skipped "
              "(non-blocking).")
        write_part0b_summary(out_dir, {}, {}, stamped=False)
        return 0

    live_prices = client.fetch_live_prices()
    if not live_prices:
        print("[Part0b] WARN: No live prices fetched. Continuing without "
              "live snapshot (non-blocking).")
        write_part0b_summary(out_dir, {}, {}, stamped=False)
        return 0

    for col, v in live_prices.items():
        print(f"  {col}: {v}")

    # Load master, compute gap signals, stamp, write back
    master_path = out_dir / "gas_weekly_master.parquet"
    stamped = False
    gaps: Dict[str, float] = {}
    if master_path.exists():
        df = pd.read_parquet(master_path)
        gaps = compute_gap_signals(df, live_prices)
        for k, v in gaps.items():
            print(f"  {k}: {v:+.2f}%")

        n_before = df.shape[1]
        df = stamp_live_observation(df, {**live_prices, **gaps})
        stamped = df.shape[1] >= n_before and any(
            c in df.columns for c in live_prices
        )
        if stamped:
            df.to_parquet(master_path, index=False)
            df.to_csv(out_dir / "gas_weekly_master.csv", index=False)
            print(f"[Part0b] Master updated -> {master_path}")
    else:
        print(f"[Part0b] Master parquet not found at {master_path}. "
              "Run gas_part0 first. Live prices recorded in summary only.")

    write_part0b_summary(out_dir, live_prices, gaps, stamped=stamped)

    print("\n[Part0b] Live commodity snapshot complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
