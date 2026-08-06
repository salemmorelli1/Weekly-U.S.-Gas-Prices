#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part3_governance.py
========================
Governance, fusion engine, and canonical prediction logger.

Responsibilities
----------------
- Read all model sleeve outputs (Part2, Part2b, Part2a/LSTM)
- Apply sleeve gate logic (include only recommended sleeves)
- Compute dynamically-weighted fusion forecast
- Assess forecast confidence (HIGH_CONF / LOW_CONF)
- Write canonical prediction_log.csv with every weekly forecast
- Write part3_summary.json

Fusion logic
------------
  Base weight: Part2 ensemble (always included)
  + XGB sleeve weight if xgb_sleeve_recommended = true
  + LSTM sleeve weight if lstm_sleeve_recommended = true
  Weights are inverse-RMSE weighted from each sleeve's val metrics.

Confidence flags
----------------
  HIGH_CONF: sleeve agreement within 2% AND regime != SUPPLY_SHOCK
  LOW_CONF:  sleeves disagree > 2% OR regime = SUPPLY_SHOCK

Pipeline position: NINTH — after Part2a, writes the canonical prediction log.
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


_colab_init(extra_packages=["pyarrow"])

import json, os, warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART3_V1_CANONICAL"


@dataclass(frozen=True)
class Part3Config:
    root_env_var: str = "GASPRICE_ROOT"
    part0_dir_name: str = "artifacts_part0"
    part2_dir_name: str = "artifacts_part2"
    part2b_dir_name: str = "artifacts_part2b"
    part2a_dir_name: str = "artifacts_part2a"
    part6_dir_name: str = "artifacts_part6"
    out_dir_name: str = "artifacts_part3"

    # Confidence: sleeves must agree within this % of current gas price
    confidence_agreement_pct: float = 0.02   # 2 cents per $1.00

    # Prediction log column schema version
    schema_version: str = "V1_WEEKLY"


def resolve_project_root(cfg: Part3Config) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── Sleeve loaders ─────────────────────────────────────────────────────────────

def _safe_load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _safe_load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_sleeve_summaries(root: Path, cfg: Part3Config) -> Dict[str, Optional[Dict]]:
    return {
        "part2":  _safe_load_json(root / cfg.part2_dir_name / "gas_part2_summary.json"),
        "part2b": _safe_load_json(root / cfg.part2b_dir_name / "gas_part2b_summary.json"),
        "part2a": _safe_load_json(root / cfg.part2a_dir_name / "gas_part2a_summary.json"),
    }


def load_forecast_tapes(root: Path, cfg: Part3Config) -> Dict[str, Optional[pd.DataFrame]]:
    tapes: Dict[str, Optional[pd.DataFrame]] = {}
    for name, rel in [
        ("part2",  cfg.part2_dir_name + "/gas_forecast_tape.parquet"),
        ("part2b", cfg.part2b_dir_name + "/gas_xgb_tape.parquet"),
        ("part2a", cfg.part2a_dir_name + "/gas_lstm_tape.parquet"),
    ]:
        t = _safe_load_parquet(root / rel)
        if t is not None:
            t["week_date"] = pd.to_datetime(t["week_date"])
        tapes[name] = t
    return tapes


def load_regime_tape(root: Path, cfg: Part3Config) -> Optional[pd.DataFrame]:
    t = _safe_load_parquet(root / cfg.part6_dir_name / "gas_regime_tape.parquet")
    if t is not None:
        t["week_date"] = pd.to_datetime(t["week_date"])
    return t


# ── Sleeve gates ───────────────────────────────────────────────────────────────

def determine_active_sleeves(
    summaries: Dict[str, Optional[Dict]],
) -> Tuple[Dict[str, float], List[str]]:
    """
    Determine which sleeves are active and their inverse-RMSE weights.
    Returns (weights_dict, active_sleeve_names).
    """
    sleeve_rmse: Dict[str, float] = {}

    # Part2 base ensemble — always active
    p2 = summaries.get("part2")
    if p2:
        val = p2.get("val_metrics", {})
        for k, v in val.items():
            if "ensemble" in k and "rmse" in k and v is not None:
                sleeve_rmse["part2"] = float(v)
                break
    if "part2" not in sleeve_rmse:
        sleeve_rmse["part2"] = 0.10  # default fallback weight denominator

    # Part2b XGB — conditional
    p2b = summaries.get("part2b")
    if p2b and p2b.get("xgb_sleeve_recommended"):
        rmse = p2b.get("xgb_val_rmse")
        if rmse and np.isfinite(rmse):
            sleeve_rmse["part2b"] = float(rmse)
            print("[Part3] XGB sleeve INCLUDED (gate passed)")
        else:
            print("[Part3] XGB sleeve EXCLUDED (missing RMSE)")
    else:
        print("[Part3] XGB sleeve EXCLUDED (gate not passed)")

    # Part2a LSTM — conditional
    p2a = summaries.get("part2a")
    if p2a and p2a.get("lstm_sleeve_recommended"):
        rmse = p2a.get("lstm_val_rmse")
        if rmse and np.isfinite(rmse):
            sleeve_rmse["part2a"] = float(rmse)
            print("[Part3] LSTM sleeve INCLUDED (gate passed)")
        else:
            print("[Part3] LSTM sleeve EXCLUDED (missing RMSE)")
    else:
        print("[Part3] LSTM sleeve EXCLUDED (gate not passed)")

    # Inverse-RMSE weights
    inv = {k: 1.0 / v for k, v in sleeve_rmse.items() if v > 0}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    active = list(weights.keys())

    print(f"[Part3] Active sleeves: {active}")
    print(f"[Part3] Fusion weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")
    return weights, active


# ── Prediction column resolver ─────────────────────────────────────────────────

SLEEVE_PRED_COLS = {
    "part2":  "pred_ensemble",
    "part2b": "pred_xgb_ensemble",
    "part2a": "pred_lstm",
}


def fuse_forecasts(
    tapes: Dict[str, Optional[pd.DataFrame]],
    weights: Dict[str, float],
    active_sleeves: List[str],
) -> pd.DataFrame:
    """
    Build a fused weekly forecast DataFrame.
    Returns DataFrame with week_date + per-sleeve preds + fusion pred.
    """
    # Start with Part2 tape as base
    base = tapes.get("part2")
    if base is None:
        print("[Part3] FATAL: Part2 tape not found.")
        return pd.DataFrame()

    result = base[["week_date"]].copy()
    if "actual" in base.columns:
        result["actual"] = base["actual"].values
    # Live-row contract (Audit 2026-08): carry the live flag through so the
    # prediction log knows whether the latest row is a genuine forward forecast.
    if "is_live" in base.columns:
        result["is_live"] = base["is_live"].values
    else:
        result["is_live"] = 0

    # Add per-sleeve predictions
    for sleeve in active_sleeves:
        tape = tapes.get(sleeve)
        pred_col = SLEEVE_PRED_COLS.get(sleeve)
        if tape is None or pred_col is None or pred_col not in tape.columns:
            result[f"pred_{sleeve}"] = np.nan
            continue
        merged = result[["week_date"]].merge(
            tape[["week_date", pred_col]].rename(columns={pred_col: f"pred_{sleeve}"}),
            on="week_date",
            how="left",
        )
        result[f"pred_{sleeve}"] = merged[f"pred_{sleeve}"].values

    # Compute fusion forecast.
    # FIX (Audit 2026-08): the previous version used fillna(0) with a GLOBAL
    # weight sum. Any week where a sleeve had no prediction (e.g. the LSTM
    # tape starts seq_len weeks later than the sklearn tape) was dragged
    # toward $0/gal by the missing sleeve's zero. Weights now renormalize
    # PER ROW over the sleeves that actually have a finite prediction.
    sleeve_names = list(weights.keys())
    pred_matrix = np.column_stack([
        result[f"pred_{s}"].values if f"pred_{s}" in result.columns
        else np.full(len(result), np.nan)
        for s in sleeve_names
    ]) if sleeve_names else np.full((len(result), 1), np.nan)
    w_vec = np.array([weights[s] for s in sleeve_names]) if sleeve_names else np.array([0.0])

    finite = np.isfinite(pred_matrix)
    w_row = finite * w_vec[None, :]
    w_sum = w_row.sum(axis=1)
    fused = np.where(
        w_sum > 0,
        np.nansum(np.where(finite, pred_matrix, 0.0) * w_row, axis=1)
        / np.where(w_sum > 0, w_sum, 1.0),
        np.nan,
    )
    result["pred_fusion"] = fused

    return result


# ── Confidence assessment ──────────────────────────────────────────────────────

def assess_confidence(
    fusion_df: pd.DataFrame,
    active_sleeves: List[str],
    regime_tape: Optional[pd.DataFrame],
    cfg: Part3Config,
) -> pd.DataFrame:
    """
    Assign HIGH_CONF or LOW_CONF to each weekly forecast row.
    """
    df = fusion_df.copy()

    # Merge latest regime
    if regime_tape is not None and "week_date" in regime_tape.columns:
        reg = regime_tape[["week_date", "regime_label"]].copy()
        df = df.merge(reg, on="week_date", how="left")
    else:
        df["regime_label"] = "UNKNOWN"

    # Compute sleeve spread (max - min across active sleeves)
    pred_cols = [f"pred_{s}" for s in active_sleeves if f"pred_{s}" in df.columns]
    if len(pred_cols) >= 2:
        preds_matrix = df[pred_cols].values
        spread = np.nanmax(preds_matrix, axis=1) - np.nanmin(preds_matrix, axis=1)
        fusion_val = df["pred_fusion"].values
        spread_pct = np.where(fusion_val != 0, spread / np.abs(fusion_val), np.nan)
    else:
        spread_pct = np.zeros(len(df))

    supply_shock = df["regime_label"].str.upper().str.contains("SUPPLY_SHOCK", na=False)

    conditions_low = (
        (spread_pct > cfg.confidence_agreement_pct) | supply_shock
    )
    df["confidence"] = np.where(conditions_low, "LOW_CONF", "HIGH_CONF")
    df["sleeve_spread_pct"] = spread_pct

    return df


# ── Prediction log ─────────────────────────────────────────────────────────────

PREDLOG_COLUMNS = [
    "decision_date",
    "target_date",
    "week_date",
    "is_live_forecast",
    "pred_fusion",
    "pred_part2",
    "pred_part2b",
    "pred_part2a",
    "actual",
    "actual_date",
    "mae",
    "rmse",
    "mape",
    "direction_correct",
    "confidence",
    "regime_label",
    "sleeve_spread_pct",
    "schema_version",
    "run_utc",
]


def build_prediction_log_row(
    fusion_df: pd.DataFrame,
    cfg: Part3Config,
) -> pd.Series:
    """
    Build the new prediction log row from the latest week's fusion forecast.
    """
    if fusion_df.empty:
        return pd.Series()

    row = fusion_df.iloc[-1]

    # FIX (Audit 2026-08, date semantics): decision_date and target_date were
    # previously derived from TODAY'S calendar ("current Monday" / "next
    # Monday"), independent of what the model actually forecast. If the data
    # lagged by a week — or the pipeline ran on a Wednesday with --force —
    # the logged target_date no longer matched the week the fusion forecast
    # was predicting, and the backfill would score the forecast against the
    # wrong EIA release. Dates are now DATA-DERIVED:
    #   week_date    = the anchor week of the forecast row (EIA week W)
    #   target_date  = week_date + 7 days (the EIA release being predicted)
    #   decision_date = the run date (audit trail only, never a join key)
    anchor_week = pd.to_datetime(row.get("week_date"))
    target_date = anchor_week + pd.Timedelta(days=7)
    today = pd.Timestamp.today().normalize()
    is_live_row = int(row.get("is_live", 0)) if not pd.isna(row.get("is_live", 0)) else 0
    if not is_live_row:
        print("[Part3] WARN: Latest fusion row is NOT a live row — the logged "
              "forecast is retrospective (its target week is already realized).")

    log_row = {
        "decision_date":   today.strftime("%Y-%m-%d"),
        "target_date":     target_date.strftime("%Y-%m-%d"),
        "week_date":       anchor_week.strftime("%Y-%m-%d"),
        "is_live_forecast": is_live_row,
        "pred_fusion":     round(float(row.get("pred_fusion", np.nan)), 4),
        "pred_part2":      round(float(row.get("pred_part2", np.nan)), 4),
        "pred_part2b":     round(float(row.get("pred_part2b", np.nan)), 4),
        "pred_part2a":     round(float(row.get("pred_part2a", np.nan)), 4),
        "actual":          np.nan,            # filled by backfill_realized
        "actual_date":     "",
        "mae":             np.nan,
        "rmse":            np.nan,
        "mape":            np.nan,
        "direction_correct": np.nan,
        "confidence":      str(row.get("confidence", "UNKNOWN")),
        "regime_label":    str(row.get("regime_label", "UNKNOWN")),
        "sleeve_spread_pct": round(float(row.get("sleeve_spread_pct", np.nan)), 4),
        "schema_version":  cfg.schema_version,
        "run_utc":         datetime.now(timezone.utc).isoformat(),
    }
    return pd.Series(log_row)


REALIZED_COLUMNS = ("actual", "actual_date", "mae", "rmse", "mape", "direction_correct")


def upsert_prediction_log(
    predlog_path: Path,
    new_row: pd.Series,
) -> pd.DataFrame:
    """
    Append or update the prediction log with the new row.

    FIX (Audit 2026-08):
      - Keyed by target_date, not decision_date. The forecast IS the
        target-week price; decision_date is only the run timestamp, and a
        re-run on a different day must still upsert the same forecast row.
      - Realized fields written by the backfill (actual, mae, ...) are
        PRESERVED when a row is re-upserted: the fresh Part 3 row carries
        NaN for those fields, and overwriting a realized value with NaN
        would silently destroy backfilled history.
      - Missing schema columns are added to older logs (forward-compatible
        with the V1 schema migration pattern from the reference project).
    """
    if predlog_path.exists():
        df = pd.read_csv(predlog_path)
        for col in PREDLOG_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
    else:
        df = pd.DataFrame(columns=PREDLOG_COLUMNS)

    target_date = str(new_row.get("target_date", "")).strip()
    existing = df.index[df.get("target_date", pd.Series(dtype=str)).astype(str) == target_date]

    if target_date and len(existing):
        idx = existing[0]
        for col, val in new_row.items():
            if col not in df.columns:
                continue
            # Normalize empty-string placeholders to NaN so they hit the
            # realized-preserve rule and never poison numeric columns.
            if isinstance(val, str) and val.strip() == "":
                val = np.nan
            is_missing = val is None or (isinstance(val, float) and np.isnan(val))
            if col in REALIZED_COLUMNS:
                # Never clobber a realized value with a fresh NaN
                current = df.at[idx, col]
                if pd.notna(current) and is_missing:
                    continue
            # FIX (Audit 2026-08): pandas 2.x raises on assigning a string
            # into an all-NaN float64 column (e.g. confidence/regime read
            # back from CSV as float64 when previously empty). Upcast to
            # object before writing incompatible values.
            if (isinstance(val, str)
                    and df[col].dtype != object):
                df[col] = df[col].astype(object)
            df.at[idx, col] = val
        print(f"[Part3] Updated existing prediction log row (target {target_date})")
    else:
        new_df = pd.DataFrame([new_row])
        for col in PREDLOG_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df[[c for c in PREDLOG_COLUMNS if c in new_df.columns]]
        df = pd.concat([df, new_df], ignore_index=True)
        print(f"[Part3] Appended new prediction log row (target {target_date})")

    # Keep the log ordered by target_date
    df["_sort"] = pd.to_datetime(df["target_date"], errors="coerce")
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


# ── Summary ────────────────────────────────────────────────────────────────────

def write_part3_summary(
    out_dir: Path,
    new_row: pd.Series,
    weights: Dict[str, float],
    active_sleeves: List[str],
    predlog_len: int,
) -> None:
    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "active_sleeves": active_sleeves,
        "fusion_weights": {k: round(v, 4) for k, v in weights.items()},
        "latest_prediction": {
            "decision_date":     str(new_row.get("decision_date", "")),
            "target_date":       str(new_row.get("target_date", "")),
            "anchor_week":       str(new_row.get("week_date", "")),
            "is_live_forecast":  int(new_row.get("is_live_forecast", 0) or 0),
            "pred_fusion":       (float(new_row.get("pred_fusion"))
                                  if pd.notna(new_row.get("pred_fusion", np.nan)) else None),
            "confidence":        str(new_row.get("confidence", "")),
            "regime":            str(new_row.get("regime_label", "")),
        },
        "prediction_log_rows": predlog_len,
    }
    path = out_dir / "gas_part3_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part3] Summary -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part3Config()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part3] ROOT: {root}")
    print(f"[Part3] Version: {SCRIPT_VERSION}\n")

    # Load sleeve summaries and tapes
    summaries = load_sleeve_summaries(root, cfg)
    tapes     = load_forecast_tapes(root, cfg)
    regime    = load_regime_tape(root, cfg)

    if tapes.get("part2") is None:
        print("[Part3] FATAL: Part2 forecast tape not found. Run gas_part2 first.")
        return 1

    # Determine active sleeves and weights
    weights, active_sleeves = determine_active_sleeves(summaries)

    # Fuse forecasts
    fusion_df = fuse_forecasts(tapes, weights, active_sleeves)
    if fusion_df.empty:
        print("[Part3] FATAL: Fusion DataFrame is empty.")
        return 1

    # Confidence assessment
    fusion_df = assess_confidence(fusion_df, active_sleeves, regime, cfg)

    # Write full fusion tape
    tape_path = out_dir / "gas_fusion_tape.parquet"
    fusion_df.to_parquet(tape_path, index=False)
    fusion_df.to_csv(out_dir / "gas_fusion_tape.csv", index=False)
    print(f"[Part3] Fusion tape -> {tape_path} ({len(fusion_df)} rows)")

    # Latest week forecast
    latest = fusion_df.iloc[-1]
    latest_week = pd.to_datetime(latest["week_date"])
    target_week = latest_week + pd.Timedelta(days=7)
    fusion_val = latest.get("pred_fusion", float("nan"))
    fusion_str = f"${fusion_val:.3f}/gal" if np.isfinite(fusion_val) else "N/A"
    print(f"\n[Part3] Latest forecast — anchor week {latest_week.date()}, "
          f"target week {target_week.date()}:")
    print(f"  Fusion:     {fusion_str}")
    print(f"  Confidence: {latest.get('confidence', 'N/A')}")
    print(f"  Regime:     {latest.get('regime_label', 'N/A')}")
    for sleeve in active_sleeves:
        col = f"pred_{sleeve}"
        v = latest.get(col, float("nan"))
        if col in latest.index and np.isfinite(v):
            print(f"  {sleeve}: ${v:.3f}/gal")

    # Build and write prediction log row — atomically (tmp + rename), with a
    # round-trip verification, matching the backfill's write discipline.
    new_row = build_prediction_log_row(fusion_df, cfg)
    predlog_path = out_dir / "prediction_log.csv"
    df_log = upsert_prediction_log(predlog_path, new_row)
    tmp_path = predlog_path.with_suffix(".csv.tmp")
    df_log.to_csv(tmp_path, index=False)
    tmp_path.replace(predlog_path)
    verify = pd.read_csv(predlog_path)
    assert len(verify) == len(df_log), "Prediction log row count mismatch after write"
    print(f"[Part3] Prediction log -> {predlog_path} ({len(df_log)} rows)")

    write_part3_summary(out_dir, new_row, weights, active_sleeves, len(df_log))

    print("\n[Part3] Governance and fusion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
