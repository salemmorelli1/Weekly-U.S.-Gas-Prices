#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part0b_collectapi_fetcher.py
=================================
Live gas price and weather data via CollectAPI.

Responsibilities
----------------
- Fetch live U.S. average gas prices by region from CollectAPI
  (https://collectapi.com/api/gasPrice/gas-prices-api)
- Fetch weather severity data (temperature, storm index) as a
  refinery/demand disruption signal
- Merge live data into the Part 0 master parquet, stamping the
  most-recent weekly observation
- Write part0b_summary.json for downstream consumers

CollectAPI setup
----------------
1. Register at https://collectapi.com and subscribe to:
   - Gas Prices API  (endpoint: gasPrice/gasPrice)
   - Weather API     (endpoint: weather/weather)
2. Set your API key:
   export COLLECTAPI_KEY="your_key_here"
   OR set it via os.environ or pass directly to GasCollectAPIClient(api_key=...)

Pipeline position: SECOND — runs after gas_part0, before gas_part0c.
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART0B_V1_CANONICAL"

# ── CollectAPI endpoints ───────────────────────────────────────────────────────
# FIX (Audit 2026-08): the previous endpoint "gasPrice/gasPrice" does not
# exist in CollectAPI's Gas Prices API. The state-level U.S. feed is
# "gasPrice/allUsaPrice"; single-state lookup is "gasPrice/stateUsaPrice".
# The weather endpoint is "weather/getWeather" (not "weather/getAdress").
COLLECTAPI_BASE = "https://api.collectapi.com"
GAS_PRICE_ENDPOINT       = f"{COLLECTAPI_BASE}/gasPrice/allUsaPrice"
GAS_PRICE_STATE_ENDPOINT = f"{COLLECTAPI_BASE}/gasPrice/stateUsaPrice"
WEATHER_ENDPOINT         = f"{COLLECTAPI_BASE}/weather/getWeather"

# FIX (Audit 2026-08): allUsaPrice items carry the FULL state name in the
# "name" field (e.g. "Alaska"), not a 2-letter "state" code. The previous
# parser looked for a nonexistent "state" field, so state_prices was always
# empty and every regional average silently came back missing. Map names to
# codes explicitly.
US_STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "washington dc": "DC", "washington d.c.": "DC",
}

# CollectAPI state codes for regional mapping
US_STATES_BY_REGION = {
    "east_coast":  ["NY", "PA", "NJ", "MA", "CT", "MD", "VA", "NC", "SC", "FL", "GA"],
    "midwest":     ["IL", "OH", "MI", "IN", "WI", "MN", "IA", "MO", "KS", "NE", "ND", "SD"],
    "gulf_coast":  ["TX", "LA", "MS", "AL"],
    "west_coast":  ["CA", "OR", "WA", "NV", "AZ"],
    "rocky_mtn":   ["CO", "UT", "ID", "MT", "WY", "NM"],
}

# Key weather cities for refinery/demand disruption signals
# FIX (Audit 2026-08): weather/getWeather takes a plain city name in
# data.city (e.g. "houston"), not "City,State".
WEATHER_CITIES = {
    "houston_tx":     "houston",
    "new_orleans_la": "new orleans",
    "los_angeles_ca": "los angeles",
    "new_york_ny":    "new york",
    "chicago_il":     "chicago",
}


@dataclass
class Part0bConfig:
    root_env_var: str = "GASPRICE_ROOT"
    part0_dir_name: str = "artifacts_part0"
    out_dir_name: str = "artifacts_part0"   # writes into same folder as Part0
    request_timeout: int = 15               # seconds
    request_retry: int = 3
    request_retry_delay: float = 2.0
    weather_enabled: bool = True


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


# ── CollectAPI client ──────────────────────────────────────────────────────────

class GasCollectAPIClient:
    """Thin wrapper around CollectAPI REST endpoints."""

    def __init__(self, api_key: Optional[str] = None, cfg: Part0bConfig = Part0bConfig()):
        self.key = api_key or os.environ.get("COLLECTAPI_KEY", "").strip()
        self.cfg = cfg
        self._price_cache: Optional[Dict] = None
        if not self.key:
            print("[Part0b] WARN: COLLECTAPI_KEY not set. "
                  "Get a free key at https://collectapi.com and set "
                  "os.environ['COLLECTAPI_KEY'] = 'your_key_here'")

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"apikey {self.key}",
        }

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        if not self.key:
            return None
        for attempt in range(self.cfg.request_retry):
            try:
                resp = requests.get(
                    url,
                    headers=self._headers,
                    params=params or {},
                    timeout=self.cfg.request_timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                print(f"[Part0b] HTTP error {e} on attempt {attempt + 1}")
                if resp.status_code == 401:
                    print("[Part0b] FATAL: Invalid API key.")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"[Part0b] Request error: {e} (attempt {attempt + 1})")
            if attempt < self.cfg.request_retry - 1:
                time.sleep(self.cfg.request_retry_delay)
        return None

    # ------------------------------------------------------------------
    # Gas price fetch
    # ------------------------------------------------------------------
    def fetch_gas_prices(self) -> Optional[Dict]:
        """
        Fetch current gas prices for all U.S. states (one API call).

        FIX (Audit 2026-08): the result is cached on the client so that
        get_us_avg_price() and get_regional_prices() share a SINGLE request.
        The previous version issued one call per method — doubling usage of
        the free-tier CollectAPI quota for no benefit.
        """
        if getattr(self, "_price_cache", None) is not None:
            return self._price_cache
        self._price_cache = self._get(GAS_PRICE_ENDPOINT)
        return self._price_cache

    @staticmethod
    def _parse_price(raw: object) -> Optional[float]:
        """Parse a CollectAPI price string like '$3.744' / '3,744' safely."""
        try:
            s = str(raw).replace("$", "").replace(",", ".").strip()
            v = float(s)
            # Sanity band for U.S. retail regular gasoline ($/gal)
            return v if 0.5 <= v <= 10.0 else None
        except (ValueError, TypeError):
            return None

    def _state_price_map(self) -> Dict[str, float]:
        """Build 2-letter state code -> regular gasoline price from allUsaPrice."""
        data = self.fetch_gas_prices()
        if data is None:
            return {}
        state_prices: Dict[str, float] = {}
        for item in data.get("result", []) or []:
            name = str(item.get("name", "")).strip().lower()
            code = US_STATE_NAME_TO_CODE.get(name)
            if code is None:
                # Some responses already use codes in a "state" field
                maybe = str(item.get("state", "")).upper().strip()
                code = maybe if len(maybe) == 2 else None
            if code is None:
                continue
            price = self._parse_price(item.get("gasoline"))
            if price is not None:
                state_prices[code] = price
        return state_prices

    def get_us_avg_price(self) -> Optional[float]:
        """Current U.S. national average regular gas price ($/gal) —
        unweighted mean across states (CollectAPI has no national row)."""
        state_prices = self._state_price_map()
        if not state_prices:
            return None
        return float(np.mean(list(state_prices.values())))

    def get_regional_prices(self) -> Dict[str, float]:
        """
        Average gas prices by region (east_coast, midwest, gulf_coast,
        west_coast, rocky_mtn). Averages state prices within each region.
        Reuses the cached allUsaPrice response — no extra API call.
        """
        state_prices = self._state_price_map()
        if not state_prices:
            return {}

        regional: Dict[str, float] = {}
        for region, states in US_STATES_BY_REGION.items():
            prices = [state_prices[s] for s in states if s in state_prices]
            if prices:
                regional[f"gas_{region}"] = float(np.mean(prices))

        regional["gas_us_live"] = float(np.mean(list(state_prices.values())))
        return regional

    # ------------------------------------------------------------------
    # Weather fetch
    # ------------------------------------------------------------------
    def fetch_weather(self, city: str) -> Optional[Dict]:
        params = {"data.lang": "en", "data.city": city}
        return self._get(WEATHER_ENDPOINT, params=params)

    def get_weather_severity_index(self) -> Dict[str, float]:
        """
        Compute a crude weather severity index for key refinery cities.
        High severity = cold snap or heat wave -> supply/demand disruption risk.

        FIX (Audit 2026-08): the previous parser assumed an OpenWeather-style
        payload (result.list[].main.temp in Kelvin, wind.speed in m/s).
        CollectAPI's weather/getWeather returns a flat daily list with a
        "degree" field in CELSIUS and no wind data. Parsed accordingly; the
        wind component of the index is dropped.
        """
        if not self.cfg.weather_enabled:
            return {}

        temps: List[float] = []

        for city_key, city_str in WEATHER_CITIES.items():
            data = self.fetch_weather(city_str)
            if data is None:
                continue
            try:
                forecasts = data.get("result", []) or []
                if not forecasts:
                    continue
                # First entry = today's forecast; "degree" is °C
                fc = forecasts[0]
                temp_c = float(fc.get("degree", np.nan))
                if not np.isfinite(temp_c):
                    continue
                temp_f = temp_c * 9 / 5 + 32
                temps.append(temp_f)
            except Exception as e:
                print(f"[Part0b] Weather parse error for {city_key}: {e}")
                continue

        result_dict: Dict[str, float] = {}
        if temps:
            avg_temp = float(np.mean(temps))
            result_dict["weather_avg_temp_f"] = avg_temp
            # Severity index: cold snaps (< 32F) or heat waves (> 90F) score high
            cold_severity = max(0.0, (32.0 - avg_temp) / 32.0)
            heat_severity = max(0.0, (avg_temp - 90.0) / 30.0)
            result_dict["weather_severity_index"] = float(
                min(1.0, cold_severity + heat_severity)
            )

        return result_dict


# ── Live data merging ──────────────────────────────────────────────────────────

def load_master_parquet(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "gas_weekly_master.parquet"
    if not path.exists():
        print(f"[Part0b] Master parquet not found at {path}. Run gas_part0 first.")
        return pd.DataFrame()
    return pd.read_parquet(path)


def stamp_live_observation(
    df: pd.DataFrame,
    regional_prices: Dict[str, float],
    weather: Dict[str, float],
    us_avg: Optional[float],
) -> pd.DataFrame:
    """
    Stamp the latest CollectAPI live observations onto the current week's
    EXISTING row of the master DataFrame.

    FIX (Audit 2026-08): the previous version APPENDED a brand-new row for
    the current Monday when it was absent from the master (i.e. whenever
    FRED had not yet propagated that morning's EIA release). That appended
    row had no gas_us_avg, so Part 1 flagged it as a second live anchor and
    the whole stack forecast off a week whose own price — the single most
    informative feature — was missing. Update-only keeps exactly one
    well-defined live anchor: the latest EIA-priced week. When the current
    Monday's row does not exist yet, the live extras are simply skipped for
    this run (they are supplementary signals, not pipeline inputs the
    feature contract depends on).
    """
    today = pd.Timestamp.today().normalize()
    # Round to current or previous Monday
    current_monday = today - pd.Timedelta(days=today.weekday())

    if "week_date" not in df.columns:
        return df

    df["week_date"] = pd.to_datetime(df["week_date"])
    existing = df[df["week_date"] == current_monday]

    live_data = {"week_date": current_monday}
    live_data.update(regional_prices)
    live_data.update(weather)
    if us_avg is not None:
        live_data["gas_us_live"] = us_avg

    if existing.empty:
        print(f"[Part0b] Current Monday ({current_monday.date()}) not yet in "
              "master (FRED lag?) — live data NOT stamped this run. "
              "Update-only by design; see Audit 2026-08 note.")
        return df

    idx = existing.index[0]
    for col, val in live_data.items():
        if col != "week_date":
            if col not in df.columns:
                df[col] = np.nan
            df.at[idx, col] = val
    print(f"[Part0b] Updated live row for {current_monday.date()}")

    return df.sort_values("week_date").reset_index(drop=True)


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_part0b_summary(
    out_dir: Path,
    regional_prices: Dict[str, float],
    weather: Dict[str, float],
    us_avg: Optional[float],
) -> None:
    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "fetch_date": date.today().isoformat(),
        "us_avg_price": us_avg,
        "regional_prices": regional_prices,
        "weather_metrics": weather,
        "live_data_available": us_avg is not None or bool(regional_prices),
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

    client = GasCollectAPIClient(cfg=cfg)

    # Fetch live gas prices
    print("[Part0b] Fetching live gas prices from CollectAPI...")
    us_avg = client.get_us_avg_price()
    regional_prices = client.get_regional_prices()

    if us_avg is not None:
        print(f"[Part0b] US average gas price: ${us_avg:.3f}/gal")
    else:
        print("[Part0b] WARN: Could not fetch live US average price.")

    for region, price in regional_prices.items():
        print(f"[Part0b]   {region}: ${price:.3f}/gal")

    # Fetch weather severity
    weather: Dict[str, float] = {}
    if cfg.weather_enabled:
        print("[Part0b] Fetching weather severity data...")
        weather = client.get_weather_severity_index()
        if weather:
            print(f"[Part0b] Weather severity index: {weather.get('weather_severity_index', 'N/A'):.3f}")
        else:
            print("[Part0b] WARN: Weather data unavailable.")

    # Merge into master parquet
    df = load_master_parquet(out_dir)
    if not df.empty:
        df = stamp_live_observation(df, regional_prices, weather, us_avg)
        parquet_path = out_dir / "gas_weekly_master.parquet"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(out_dir / "gas_weekly_master.csv", index=False)
        print(f"[Part0b] Master parquet updated -> {parquet_path}")
    else:
        print("[Part0b] WARN: Master parquet not found — live data not merged.")

    write_part0b_summary(out_dir, regional_prices, weather, us_avg)

    live_available = us_avg is not None or bool(regional_prices)
    if not live_available:
        print("[Part0b] WARN: No live data fetched. Check COLLECTAPI_KEY.")
    else:
        print("\n[Part0b] CollectAPI live data fetch complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
