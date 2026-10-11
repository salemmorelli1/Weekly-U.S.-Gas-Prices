"""Exact observed-price identities shared by ingestion and outcome scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

FRED_TARGET_SERIES = "GASREGCOVW"
EIA_TARGET_SERIES = "PET.EMM_EPMRU_PTE_NUS_DPG.W"
OBSERVATION_CONTRACT = "EXACT_MONDAY_NO_FILL_V1"


def observed_monday_prices(raw: pd.Series) -> pd.Series:
    """Keep actual Monday labels and missing values; never resample or fill.

    A missing survey is not a zero-change observation. Nonpositive or
    nonfinite prices remain missing and cannot establish release freshness.
    """
    result = pd.to_numeric(raw.copy(), errors="coerce")
    dates = pd.to_datetime(result.index, errors="coerce")
    if dates.isna().any() or dates.tz is not None:
        raise ValueError("Price observations require valid timezone-naive dates")
    dates = dates.normalize()
    if dates.duplicated().any() or (dates.dayofweek != 0).any():
        raise ValueError("Price observations require unique exact Monday dates")
    result.index = dates
    return result.where(np.isfinite(result) & result.gt(0)).sort_index()


def eia_price_records(rows: list[dict], series_id: str) -> pd.Series:
    """Reject a provider response naming a different product or unit."""
    expected = series_id.removeprefix("PET.").removesuffix(".W")
    records = []
    for row in rows:
        reported = row.get("series") or row.get("series_id")
        if reported and str(reported).removeprefix("PET.").removesuffix(".W") != expected:
            raise ValueError("EIA price response belongs to a different series")
        units = str(row.get("units", "")).lower().replace(" ", "")
        if units and units not in {"$/gal", "dollarspergallon"}:
            raise ValueError("EIA price response requires dollars per gallon")
        if row.get("period") and row.get("value") is not None:
            records.append((row["period"], row["value"]))
    if not records:
        return pd.Series(dtype=float)
    dates, values = zip(*records)
    return observed_monday_prices(pd.Series(values, index=pd.to_datetime(dates), name=series_id))
