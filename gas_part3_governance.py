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

from gas_time_contract import (
    PROTOCOL_SCHEMA,
    eastern_today,
    pipeline_identity,
    protocol_eligible_record,
    sha256_file,
    strict_json_dump,
    validate_prospective_forecast,
)

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART3_V2_IMMUTABLE_LEDGER"


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
    schema_version: str = PROTOCOL_SCHEMA


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
    """Select same-run sleeves using comparable validation RMSE values."""
    core = summaries.get("part2")
    if not core:
        print("[Part3] FATAL: Part2 summary is missing.")
        return {}, []

    core_rmse = core.get("val_metrics", {}).get("ensemble_rmse")
    if core_rmse is None or not np.isfinite(float(core_rmse)) or float(core_rmse) <= 0:
        print("[Part3] FATAL: Part2 ensemble RMSE is missing or invalid.")
        return {}, []

    run_id = str(core.get("pipeline_run_id", ""))
    sleeve_rmse: Dict[str, float] = {"part2": float(core_rmse)}

    candidates = (
        ("part2b", "xgb_sleeve_recommended", "xgb_val_rmse"),
        ("part2a", "lstm_sleeve_recommended", "lstm_val_rmse"),
    )
    for name, gate_key, rmse_key in candidates:
        summary = summaries.get(name)
        if not summary or not summary.get(gate_key):
            print(f"[Part3] {name} EXCLUDED (gate not passed)")
            continue
        if not run_id or str(summary.get("pipeline_run_id", "")) != run_id:
            print(f"[Part3] {name} EXCLUDED (stale or mixed-run provenance)")
            continue
        rmse = summary.get(rmse_key)
        if rmse is None or not np.isfinite(float(rmse)) or float(rmse) <= 0:
            print(f"[Part3] {name} EXCLUDED (invalid RMSE)")
            continue
        sleeve_rmse[name] = float(rmse)

    inverse = {name: 1.0 / value for name, value in sleeve_rmse.items()}
    total = sum(inverse.values())
    weights = {name: value / total for name, value in inverse.items()}
    active = list(weights)
    print(f"[Part3] Active sleeves: {active}")
    print(f"[Part3] Fusion weights: {weights}")
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
    if "pred_persistence" in base.columns:
        result["pred_persistence"] = base["pred_persistence"].values
    else:
        result["pred_persistence"] = np.nan
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
    "anchor_week",
    "week_date",
    "is_live_forecast",
    "protocol_eligible",
    "pred_fusion",
    "pred_candidate",
    "pred_persistence",
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
    "publication_mode",
    "operator_validated",
    "regime_label",
    "sleeve_spread_pct",
    "schema_version",
    "run_utc",
    "pipeline_run_id",
    "pipeline_run_attempt",
    "source_code_sha",
]



def build_prediction_log_row(
    fusion_df: pd.DataFrame,
    cfg: Part3Config,
) -> pd.Series:
    """Build and validate one genuinely prospective immutable ledger row."""
    if fusion_df.empty:
        raise ValueError("fusion forecast is empty")

    row = fusion_df.iloc[-1]
    anchor = pd.to_datetime(row.get("week_date"), errors="coerce")
    target = anchor + pd.Timedelta(days=7)
    decision = eastern_today()
    is_live = int(row.get("is_live", 0) or 0)

    validate_prospective_forecast(
        decision_date=decision,
        anchor_week=anchor,
        target_date=target,
        is_live_forecast=is_live,
    )

    governed = float(row.get("pred_fusion", np.nan))
    candidate = float(row.get("pred_candidate", np.nan))
    persistence = float(row.get("pred_persistence", np.nan))
    if not all(np.isfinite(value) for value in (governed, candidate, persistence)):
        raise ValueError("governed, candidate, and persistence forecasts must be finite")

    identity = pipeline_identity()
    log_row = {
        "decision_date": decision.strftime("%Y-%m-%d"),
        "target_date": target.strftime("%Y-%m-%d"),
        "anchor_week": anchor.strftime("%Y-%m-%d"),
        "week_date": anchor.strftime("%Y-%m-%d"),
        "is_live_forecast": 1,
        "protocol_eligible": 1,
        "pred_fusion": round(governed, 4),
        "pred_candidate": round(candidate, 4),
        "pred_persistence": round(persistence, 4),
        "pred_part2": round(float(row.get("pred_part2", np.nan)), 4),
        "pred_part2b": round(float(row.get("pred_part2b", np.nan)), 4),
        "pred_part2a": round(float(row.get("pred_part2a", np.nan)), 4),
        "actual": np.nan,
        "actual_date": "",
        "mae": np.nan,
        "rmse": np.nan,
        "mape": np.nan,
        "direction_correct": np.nan,
        "confidence": str(row.get("confidence", "NOT_VALIDATED")),
        "publication_mode": str(row.get("publication_mode", "FAIL_CLOSED")),
        "operator_validated": int(bool(row.get("operator_validated", False))),
        "regime_label": str(row.get("regime_label", "UNKNOWN")),
        "sleeve_spread_pct": round(
            float(row.get("sleeve_spread_pct", np.nan)), 4
        ),
        "schema_version": cfg.schema_version,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        **identity,
    }
    return pd.Series(log_row)


REALIZED_COLUMNS = ("actual", "actual_date", "mae", "rmse", "mape", "direction_correct")


def upsert_prediction_log(
    predlog_path: Path,
    new_row: pd.Series,
) -> pd.DataFrame:
    """Append a target once; never revise a published forecast or provenance."""
    if predlog_path.exists():
        df = pd.read_csv(predlog_path, dtype={"pipeline_run_id": "string"})
    else:
        df = pd.DataFrame(columns=PREDLOG_COLUMNS)

    for column in PREDLOG_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    # Reclassify legacy/hindsight rows without deleting historical evidence.
    if len(df):
        df["protocol_eligible"] = [
            int(protocol_eligible_record(record))
            for record in df.to_dict(orient="records")
        ]

    target = str(new_row.get("target_date", "")).strip()
    existing = df.index[
        df["target_date"].astype(str).str.strip().eq(target)
    ]
    if len(existing):
        if len(existing) != 1:
            raise RuntimeError(f"duplicate immutable ledger rows for target {target}")
        index = existing[0]
        old = pd.to_numeric(
            pd.Series([df.at[index, "pred_fusion"]]), errors="coerce"
        ).iloc[0]
        new = float(new_row["pred_fusion"])
        if not np.isfinite(old) or not np.isclose(float(old), new, atol=1e-12):
            raise RuntimeError(
                f"immutable target {target} already has a different forecast"
            )
        print(f"[Part3] Immutable target {target} already published; no rewrite.")
    else:
        addition = pd.DataFrame([new_row]).reindex(columns=PREDLOG_COLUMNS)
        if df.empty:
            df = addition.reset_index(drop=True)
        else:
            df = pd.concat(
                [df.reindex(columns=PREDLOG_COLUMNS), addition],
                ignore_index=True,
            )
        print(f"[Part3] Appended immutable prediction row for target {target}")

    sort_key = pd.to_datetime(df["target_date"], errors="coerce")
    df = (
        df.assign(_target_sort=sort_key)
        .sort_values("_target_sort", kind="stable")
        .drop(columns="_target_sort")
        .reset_index(drop=True)
    )
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
            "pred_candidate":    (float(new_row.get("pred_candidate"))
                                  if pd.notna(new_row.get("pred_candidate", np.nan)) else None),
            "pred_persistence":  (float(new_row.get("pred_persistence"))
                                  if pd.notna(new_row.get("pred_persistence", np.nan)) else None),
            "confidence":        str(new_row.get("confidence", "")),
            "regime":            str(new_row.get("regime_label", "")),
            "operator_validated": bool(new_row.get("operator_validated", 0)),
            "publication_mode": str(
                new_row.get("publication_mode", "FAIL_CLOSED")
            ),
            "pipeline_run_id": str(new_row.get("pipeline_run_id", "")),
            "pipeline_run_attempt": str(
                new_row.get("pipeline_run_attempt", "")
            ),
            "source_code_sha": str(new_row.get("source_code_sha", "")),
        },
        "prediction_log_rows": predlog_len,
        "operator_validated": bool(new_row.get("operator_validated", 0)),
        "publication_mode": str(new_row.get("publication_mode", "FAIL_CLOSED")),
        **pipeline_identity(),
    }
    path = out_dir / "gas_part3_summary.json"
    strict_json_dump(summary, path)
    print(f"[Part3] Summary -> {path}")


def validate_core_lineage(
    root: Path,
    cfg: Part3Config,
    summaries: Dict[str, Optional[Dict]],
) -> None:
    """Require Part 0, Part 2 summary, and Part 2 tape to be one run."""
    part0 = _safe_load_json(
        root / cfg.part0_dir_name / "part0_summary.json"
    )
    part2 = summaries.get("part2")
    if not part0 or not part2:
        raise ValueError("Part 0 and Part 2 summaries are required")

    run0 = str(part0.get("pipeline_run_id", ""))
    run2 = str(part2.get("pipeline_run_id", ""))
    if not run0 or run0 != run2:
        raise ValueError("Part 0 and Part 2 pipeline run IDs disagree")
    if run0 != pipeline_identity()["pipeline_run_id"]:
        raise ValueError("Core artifacts belong to a different pipeline run")

    source0 = str(part0.get("freshness", {}).get("latest_week", ""))
    source2 = str(part2.get("forecast_anchor_week", ""))
    if not source0 or source0 != source2:
        raise ValueError("Part 0 source week and Part 2 anchor disagree")

    tape_path = root / cfg.part2_dir_name / "gas_forecast_tape.parquet"
    expected_hash = str(part2.get("forecast_tape_sha256", ""))
    if not expected_hash or sha256_file(tape_path) != expected_hash:
        raise ValueError("Part 2 forecast tape hash does not match its summary")


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

    try:
        validate_core_lineage(root, cfg, summaries)
    except (OSError, ValueError) as exc:
        print(f"[Part3] FATAL: lineage validation failed: {exc}")
        return 1

    if tapes.get("part2") is None:
        print("[Part3] FATAL: Part2 forecast tape not found. Run gas_part2 first.")
        return 1

    # Determine active sleeves and weights
    weights, active_sleeves = determine_active_sleeves(summaries)

    # Fuse forecasts
    fusion_df = fuse_forecasts(tapes, weights, active_sleeves)
    if fusion_df.empty or not active_sleeves:
        print("[Part3] FATAL: Fusion DataFrame is empty or has no valid core sleeve.")
        return 1

    core_validation = (summaries.get("part2") or {}).get(
        "operator_validation", {}
    )
    operator_validated = bool(core_validation.get("operator_validated", False))
    fusion_df["pred_candidate"] = fusion_df["pred_fusion"]
    fusion_df["operator_validated"] = operator_validated
    if operator_validated:
        fusion_df["publication_mode"] = "VALIDATED_CANDIDATE"
    else:
        fusion_df["pred_fusion"] = fusion_df["pred_persistence"]
        fusion_df["publication_mode"] = "FAIL_CLOSED_PERSISTENCE"

    # Confidence assessment
    fusion_df = assess_confidence(fusion_df, active_sleeves, regime, cfg)
    if not operator_validated:
        fusion_df["confidence"] = "NOT_VALIDATED"

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

    target_mask = df_log["target_date"].astype(str).eq(
        str(new_row.get("target_date"))
    )
    published_row = df_log.loc[target_mask].iloc[0]
    write_part3_summary(
        out_dir, published_row, weights, active_sleeves, len(df_log)
    )
    strict_json_dump(
        {
            "status": "SUCCESS",
            "source_observation_date": str(published_row.get("anchor_week")),
            "target_date": str(published_row.get("target_date")),
            "operator_validated": bool(
                published_row.get("operator_validated", 0)
            ),
            "publication_mode": str(published_row.get("publication_mode")),
            "prediction_pipeline_run_id": str(
                published_row.get("pipeline_run_id", "")
            ),
            **pipeline_identity(),
        },
        out_dir / "pipeline_status.json",
    )

    print("\n[Part3] Governance and fusion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
