#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part9_live_attribution.py
==============================
Live attribution and model health diagnostics for gas price forecasting.

This is the system's truth-telling file.
No backtest metric matters. Only this file tells you if the model is real.

Responsibilities
----------------
1. Consume the canonical prediction_log.csv (live realized rows only)
2. Compute statistically valid live performance metrics
   (MAE, RMSE, MAPE rolling 4w/8w/all-time)
3. Diebold-Mariano test: fusion model vs. naive carry benchmark
4. Concept drift detection: has model accuracy degraded recently?
5. Model health diagnostics and stopping recommendations

Pipeline position: FINAL — after Part3.
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


_colab_init(extra_packages=["scipy", "pyarrow"])

import json, os, warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART9_V1_CANONICAL"


@dataclass(frozen=True)
class Part9Config:
    root_env_var: str = "GASPRICE_ROOT"
    part3_dir_name: str = "artifacts_part3"
    out_dir_name: str = "artifacts_part9"

    # Minimum realized observations required for significance
    min_realized_n: int = 8        # ~2 months of weekly data

    # Diebold-Mariano test significance threshold
    dm_t_stat_min: float = 1.64   # ~90% confidence (one-sided)

    # Rolling windows for performance tracking
    rolling_windows: Tuple[int, ...] = (4, 8, 13, 26)

    # Concept drift: compare recent RMSE vs historical
    drift_recent_weeks: int = 8
    drift_rmse_ratio_warn: float = 1.5    # 50% RMSE degradation = warning
    drift_rmse_ratio_suspend: float = 2.0  # 100% RMSE degradation = stop signal

    # MAPE threshold for model health flags
    mape_warn_pct: float = 3.0    # > 3% MAPE
    mape_suspend_pct: float = 6.0 # > 6% MAPE

    # Directional accuracy threshold
    dir_acc_warn: float = 0.50    # Not better than coin flip
    dir_acc_stop: float = 0.40    # Systematically worse


def resolve_project_root(cfg: Part9Config) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Direction accuracy: did we correctly call price-up vs price-down?"""
    if len(y_true) < 2:
        return np.nan
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    mask = np.isfinite(true_dir) & np.isfinite(pred_dir)
    if mask.sum() < 2:
        return np.nan
    return float(np.mean(true_dir[mask] == pred_dir[mask]))


# ── Load prediction log ────────────────────────────────────────────────────────

def load_realized_rows(predlog_path: Path) -> pd.DataFrame:
    """Load only rows that have actual realized gas prices."""
    if not predlog_path.exists():
        print(f"[Part9] Prediction log not found: {predlog_path}")
        return pd.DataFrame()

    df = pd.read_csv(predlog_path)
    if df.empty:
        return df

    # Filter to realized rows only
    actual_col = "actual"
    if actual_col not in df.columns:
        print("[Part9] WARN: 'actual' column not found in prediction log.")
        return pd.DataFrame()

    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
    realized = df[df[actual_col].notna()].copy()

    if "decision_date" in realized.columns:
        realized["decision_date"] = pd.to_datetime(realized["decision_date"])
        realized = realized.sort_values("decision_date").reset_index(drop=True)

    return realized


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_all_time_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["actual"].values
    y_pred = df["pred_fusion"].values
    return {
        "mae":         _mae(y_true, y_pred),
        "rmse":        _rmse(y_true, y_pred),
        "mape":        _mape(y_true, y_pred),
        "dir_acc":     _dir_acc(y_true, y_pred),
        "n_realized":  int(np.sum(np.isfinite(y_true))),
    }


def compute_rolling_metrics(df: pd.DataFrame, windows: Tuple[int, ...]) -> Dict[str, Dict]:
    """Compute metrics over rolling windows from the most recent observation."""
    result = {}
    for w in windows:
        tail = df.tail(w)
        y_true = tail["actual"].values
        y_pred = tail["pred_fusion"].values
        result[f"rolling_{w}w"] = {
            "mae":     _mae(y_true, y_pred),
            "rmse":    _rmse(y_true, y_pred),
            "mape":    _mape(y_true, y_pred),
            "dir_acc": _dir_acc(y_true, y_pred),
            "n":       int(np.sum(np.isfinite(y_true))),
        }
    return result


# ── Naive benchmark ────────────────────────────────────────────────────────────

def compute_naive_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Naive carry: predict this week's actual = last week's actual."""
    y_true  = df["actual"].values
    y_naive = np.roll(y_true, 1)
    y_naive[0] = np.nan
    return {
        "mae":         _mae(y_true[1:], y_naive[1:]),
        "rmse":        _rmse(y_true[1:], y_naive[1:]),
        "mape":        _mape(y_true[1:], y_naive[1:]),
        "dir_acc":     _dir_acc(y_true[1:], y_naive[1:]),
    }


# ── Diebold-Mariano test ───────────────────────────────────────────────────────

def diebold_mariano_test(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_naive: np.ndarray,
) -> Dict[str, float]:
    """
    Diebold-Mariano test: is the model significantly better than naive?
    H0: forecast losses are equal.
    H1: model loss < naive loss (model is better).
    Returns t-stat, p-value (one-sided), and interpretation.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_model) & np.isfinite(y_naive)
    if mask.sum() < 8:
        return {"dm_t_stat": np.nan, "dm_p_value": np.nan, "dm_interpretation": "INSUFFICIENT_DATA"}

    yt, ym, yn = y_true[mask], y_model[mask], y_naive[mask]

    e_model = (yt - ym) ** 2
    e_naive = (yn - yt) ** 2
    d = e_naive - e_model   # positive = model better

    t_stat, p_value = stats.ttest_1samp(d, 0.0)
    # One-sided: p for model better than naive
    p_one_sided = p_value / 2 if t_stat > 0 else 1.0 - p_value / 2

    interpretation = (
        "MODEL_SIGNIFICANTLY_BETTER"
        if t_stat > 1.64 and p_one_sided < 0.10
        else "NOT_SIGNIFICANT"
        if t_stat > 0
        else "MODEL_WORSE_THAN_NAIVE"
    )

    return {
        "dm_t_stat":       round(float(t_stat), 3),
        "dm_p_value":      round(float(p_one_sided), 4),
        "dm_interpretation": interpretation,
    }


# ── Concept drift detection ────────────────────────────────────────────────────

def detect_concept_drift(
    df: pd.DataFrame,
    cfg: Part9Config,
) -> Dict[str, object]:
    """
    Compare recent RMSE to historical RMSE.
    A large ratio indicates the model has stopped working.
    """
    n = len(df)
    result = {
        "drift_detected": False,
        "status": "OK",
        "recent_rmse": np.nan,
        "historical_rmse": np.nan,
        "rmse_ratio": np.nan,
    }

    if n < cfg.drift_recent_weeks * 2:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    recent   = df.tail(cfg.drift_recent_weeks)
    historic = df.iloc[:-cfg.drift_recent_weeks]

    recent_rmse   = _rmse(recent["actual"].values, recent["pred_fusion"].values)
    historic_rmse = _rmse(historic["actual"].values, historic["pred_fusion"].values)

    result["recent_rmse"]   = round(recent_rmse, 4) if np.isfinite(recent_rmse) else np.nan
    result["historical_rmse"] = round(historic_rmse, 4) if np.isfinite(historic_rmse) else np.nan

    if np.isfinite(recent_rmse) and np.isfinite(historic_rmse) and historic_rmse > 0:
        ratio = recent_rmse / historic_rmse
        result["rmse_ratio"] = round(ratio, 3)

        if ratio >= cfg.drift_rmse_ratio_suspend:
            result["drift_detected"] = True
            result["status"] = "STOP_SIGNAL"
            print(f"[Part9] STOP_SIGNAL: Recent RMSE {recent_rmse:.4f} is "
                  f"{ratio:.1f}x historical {historic_rmse:.4f}")
        elif ratio >= cfg.drift_rmse_ratio_warn:
            result["drift_detected"] = True
            result["status"] = "WARNING"
            print(f"[Part9] WARNING: Recent RMSE {recent_rmse:.4f} is "
                  f"{ratio:.1f}x historical {historic_rmse:.4f}")
        else:
            result["status"] = "OK"

    return result


# ── Model health assessment ────────────────────────────────────────────────────

def assess_model_health(
    metrics: Dict[str, float],
    drift: Dict[str, object],
    dm_result: Dict[str, float],
    cfg: Part9Config,
) -> Dict[str, object]:
    """
    Synthesize all diagnostics into a model health summary.
    """
    issues: List[str] = []
    status = "HEALTHY"

    mape = metrics.get("mape", np.nan)
    dir_acc = metrics.get("dir_acc", np.nan)
    n = metrics.get("n_realized", 0)

    if n < cfg.min_realized_n:
        return {
            "health_status": "INSUFFICIENT_DATA",
            "n_realized": n,
            "min_required": cfg.min_realized_n,
            "issues": [f"Only {n} realized observations (min {cfg.min_realized_n})"],
            "recommendation": "Continue accumulating live data before evaluation.",
        }

    if np.isfinite(mape):
        if mape > cfg.mape_suspend_pct:
            issues.append(f"MAPE {mape:.2f}% > suspend threshold {cfg.mape_suspend_pct}%")
            status = "STOP_SIGNAL"
        elif mape > cfg.mape_warn_pct:
            issues.append(f"MAPE {mape:.2f}% > warning threshold {cfg.mape_warn_pct}%")
            if status == "HEALTHY":
                status = "WARNING"

    if np.isfinite(dir_acc):
        if dir_acc < cfg.dir_acc_stop:
            issues.append(f"Direction accuracy {dir_acc:.1%} < stop threshold")
            status = "STOP_SIGNAL"
        elif dir_acc < cfg.dir_acc_warn:
            issues.append(f"Direction accuracy {dir_acc:.1%} < warning threshold")
            if status == "HEALTHY":
                status = "WARNING"

    if drift.get("drift_detected"):
        issues.append(f"Concept drift: {drift.get('status')} "
                      f"(RMSE ratio {drift.get('rmse_ratio')})")
        if drift.get("status") == "STOP_SIGNAL" and status != "STOP_SIGNAL":
            status = "STOP_SIGNAL"

    dm_interp = dm_result.get("dm_interpretation", "")
    if dm_interp == "MODEL_WORSE_THAN_NAIVE":
        issues.append("Model performs WORSE than naive carry benchmark (DM test)")
        if status == "HEALTHY":
            status = "WARNING"

    recommendation = {
        "HEALTHY":            "Continue normal operation. Model performing as expected.",
        "WARNING":            "Review model. Consider retraining if issues persist.",
        "STOP_SIGNAL":        "Model degradation detected. Retrain or suspend until resolved.",
        "INSUFFICIENT_DATA":  "Accumulate more live data before making health judgments.",
    }.get(status, "Unknown status.")

    return {
        "health_status": status,
        "n_realized": n,
        "issues": issues,
        "recommendation": recommendation,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part9Config()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    part3_dir = root / cfg.part3_dir_name

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part9] ROOT: {root}")
    print(f"[Part9] Version: {SCRIPT_VERSION}\n")

    # Load realized rows
    predlog_path = part3_dir / "prediction_log.csv"
    df = load_realized_rows(predlog_path)

    if df.empty:
        print("[Part9] No realized observations found in prediction log.")
        print("[Part9] Run gas_backfill_realized.py after receiving EIA data.")
        summary = {
            "script_version": SCRIPT_VERSION,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "n_realized": 0,
            "health_status": "INSUFFICIENT_DATA",
            "message": "No realized rows in prediction log yet.",
        }
        path = out_dir / "live_attribution_report.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Part9] Report -> {path}")
        return 0

    # Ensure pred_fusion column
    if "pred_fusion" not in df.columns:
        print("[Part9] FATAL: pred_fusion column not found in prediction log.")
        return 1

    df["actual"]     = pd.to_numeric(df["actual"], errors="coerce")
    df["pred_fusion"] = pd.to_numeric(df["pred_fusion"], errors="coerce")
    df = df.dropna(subset=["actual", "pred_fusion"])

    n = len(df)
    print(f"[Part9] Realized observations: {n}")

    if n < cfg.min_realized_n:
        print(f"[Part9] Only {n} realized rows (min {cfg.min_realized_n}). "
              "Metrics will have limited statistical power.")

    # All-time metrics
    all_time = compute_all_time_metrics(df)
    print(f"\n[Part9] All-time metrics:")
    print(f"  MAE:     ${all_time['mae']:.4f}/gal")
    print(f"  RMSE:    ${all_time['rmse']:.4f}/gal")
    print(f"  MAPE:    {all_time['mape']:.2f}%")
    print(f"  Dir Acc: {all_time['dir_acc']:.1%}")

    # Rolling metrics
    rolling = compute_rolling_metrics(df, cfg.rolling_windows)
    print(f"\n[Part9] Rolling metrics:")
    for window, m in rolling.items():
        n_w = m.get("n", 0)
        if n_w > 0:
            print(f"  {window} | n={n_w} | MAE=${m['mae']:.4f} | "
                  f"MAPE={m['mape']:.2f}% | DirAcc={m['dir_acc']:.1%}")

    # Naive benchmark
    naive = compute_naive_metrics(df)
    print(f"\n[Part9] Naive carry benchmark:")
    print(f"  RMSE: ${naive['rmse']:.4f}/gal | MAPE: {naive['mape']:.2f}%")

    # Diebold-Mariano test
    y_true = df["actual"].values
    y_pred = df["pred_fusion"].values
    y_naive = np.concatenate([[np.nan], y_true[:-1]])
    dm_result = diebold_mariano_test(y_true, y_pred, y_naive)
    print(f"\n[Part9] Diebold-Mariano test: {dm_result['dm_interpretation']} "
          f"(t={dm_result['dm_t_stat']}, p={dm_result['dm_p_value']})")

    # Concept drift detection
    drift = detect_concept_drift(df, cfg)
    print(f"\n[Part9] Concept drift: {drift['status']}")

    # Model health
    health = assess_model_health(all_time, drift, dm_result, cfg)
    print(f"\n[Part9] Model health: {health['health_status']}")
    print(f"  Recommendation: {health['recommendation']}")
    if health["issues"]:
        for issue in health["issues"]:
            print(f"  ⚠  {issue}")

    # Write detailed attribution tape (row-level errors)
    attr_df = df.copy()
    attr_df["error"]    = attr_df["actual"] - attr_df["pred_fusion"]
    attr_df["abs_error"] = attr_df["error"].abs()
    attr_df["ape"]      = (attr_df["abs_error"] / attr_df["actual"].abs()).clip(0, 10)

    naive_arr = np.concatenate([[np.nan], attr_df["actual"].values[:-1]])
    attr_df["naive_pred"]     = naive_arr
    attr_df["naive_error"]    = attr_df["actual"] - attr_df["naive_pred"]
    attr_df["naive_abs_error"] = attr_df["naive_error"].abs()
    attr_df["beats_naive"]    = attr_df["abs_error"] < attr_df["naive_abs_error"]

    attr_path = out_dir / "live_attribution_tape.csv"
    attr_df.to_csv(attr_path, index=False)
    print(f"\n[Part9] Attribution tape -> {attr_path}")

    # Write report
    report = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "n_realized": n,
        "all_time_metrics": {k: round(v, 4) if np.isfinite(v) else None
                              for k, v in all_time.items()},
        "naive_metrics": {k: round(v, 4) if isinstance(v, float) and np.isfinite(v) else None
                          for k, v in naive.items()},
        "rolling_metrics": {
            window: {mk: round(mv, 4) if isinstance(mv, float) and np.isfinite(mv) else None
                     for mk, mv in m.items()}
            for window, m in rolling.items()
        },
        "diebold_mariano": dm_result,
        "concept_drift": {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
                          for k, v in drift.items()},
        "model_health": health,
        "beats_naive_pct": round(float(attr_df["beats_naive"].mean() * 100), 1)
            if "beats_naive" in attr_df.columns else None,
    }
    report_path = out_dir / "live_attribution_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[Part9] Report -> {report_path}")

    print(f"\n[Part9] Beats naive: {report['beats_naive_pct']}% of weeks")
    print("\n[Part9] Live attribution complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
