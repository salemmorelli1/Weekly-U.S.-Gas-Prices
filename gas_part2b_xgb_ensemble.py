#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part2b_xgb_ensemble.py
===========================
XGBoost ensemble sleeve for the Gas Price Forecasting model.

This is an optional experimental sleeve that runs after gas_part2.
It trains multiple XGBoost models with different hyperparameter configurations
and writes a summary that Part3 governance uses to decide whether to include
this sleeve's forecast in the final ensemble.

Gate condition
--------------
If val_rmse_xgb < val_rmse_base (Part2 ensemble RMSE), the XGB sleeve is
recommended for inclusion. Part3 reads xgb_sleeve_recommended from
part2b_summary.json to decide.

Outputs
-------
  artifacts_part2b/gas_xgb_tape.parquet       — week_date + xgb predictions
  artifacts_part2b/gas_part2b_summary.json    — metrics + gate result

Pipeline position: SEVENTH (optional) — after Part2, before Part2a.
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


_colab_init(extra_packages=["xgboost", "scikit-learn", "pyarrow"])

import json, os, pickle, warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    xgb = None
    HAVE_XGB = False
    print("[Part2b] XGBoost not installed. Install: pip install xgboost")

SCRIPT_VERSION = "GAS_PART2B_V1_CANONICAL"


@dataclass(frozen=True)
class Part2bConfig:
    root_env_var: str = "GASPRICE_ROOT"
    part1_dir_name: str = "artifacts_part1"
    part2_dir_name: str = "artifacts_part2"
    out_dir_name: str = "artifacts_part2b"
    seed: int = 42

    initial_train_weeks: int = 156
    val_weeks: int = 52

    # XGBoost configurations to train (mini hyperparameter search)
    xgb_configs: Tuple[Dict, ...] = (
        {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03,
         "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.10,
         "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1},
    )


def resolve_project_root(cfg: Part2bConfig) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def load_part2_baseline_rmse(part2_dir: Path) -> Optional[float]:
    """Read Part2 ensemble val RMSE for gate comparison."""
    summary_path = part2_dir / "gas_part2_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    # Look for ensemble RMSE in val_metrics
    val = summary.get("val_metrics", {})
    # Key pattern: model_rmse
    for k, v in val.items():
        if "ensemble" in k and "rmse" in k and v is not None:
            return float(v)
    return None


def load_features(part1_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    matrix_path = part1_dir / "gas_feature_matrix.parquet"
    target_path  = part1_dir / "gas_target.parquet"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {matrix_path}")
    X = pd.read_parquet(matrix_path)
    y_df = pd.read_parquet(target_path)
    X["week_date"] = pd.to_datetime(X["week_date"])
    if "is_live" not in X.columns:   # live-row contract (Audit 2026-08)
        X = X.copy()
        X["is_live"] = 0
    return X, y_df["target_gas_price"]


NON_FEATURE_COLS = ("week_date", "is_live")


def get_feature_cols(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c not in NON_FEATURE_COLS]


def split_labeled_live(
    X: pd.DataFrame, y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split into (labeled X, labeled y, live X). Live rows have no target yet."""
    labeled_mask = (X["is_live"].values == 0) & y.notna().values
    X_lab = X.loc[labeled_mask].reset_index(drop=True)
    y_lab = y.loc[labeled_mask].reset_index(drop=True)
    X_live = X.loc[X["is_live"].values == 1].reset_index(drop=True)
    return X_lab, y_lab, X_live


def train_xgb_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Part2bConfig,
) -> Tuple[List[object], np.ndarray, np.ndarray, Dict[str, float], Optional[object]]:
    """
    Train multiple XGB configs, evaluate on the common gate window, then
    re-fit every member on ALL labeled rows for the live forecast.
    Returns (models, val_preds_ensemble, val_actuals, val_metrics, imputer).

    FIX (Audit 2026-08) — three defects corrected here:
      1. Return-arity crash: the no-XGB path returned a 4-tuple while the
         caller unpacked 5 names. All paths now return the same 5-tuple.
      2. Early-stopping refit crash: XGBRegressor was constructed with
         early_stopping_rounds=50 and then re-fit WITHOUT an eval_set, which
         raises in xgboost >= 1.6. The refit now uses a fresh estimator with
         the validated best_iteration as n_estimators and no early stopping.
      3. Imputer leakage: the median imputer was fit on the full matrix
         (train + val + future rows). It is now fit on training rows only.

    The validation split is the last cfg.val_weeks labeled rows — the COMMON
    GATE WINDOW shared with Part 2 and Part 2a so gate RMSEs are comparable.
    X and y must already be labeled-rows-only.
    """
    if not HAVE_XGB:
        return [], np.array([]), np.array([]), {}, None

    feature_cols = get_feature_cols(X)
    n = len(y)
    val_len = min(cfg.val_weeks, max(1, int(n * 0.25)))
    train_end = n - val_len
    val_end   = n
    print(f"[Part2b] Common gate window: train 0:{train_end}, "
          f"val {train_end}:{val_end} (last {val_len} labeled weeks)")

    # Impute NaN values — fit on TRAIN rows only (no leakage), then transform.
    imputer = SimpleImputer(strategy="median")
    Xtr  = imputer.fit_transform(X[feature_cols].values[:train_end])
    Xval = imputer.transform(X[feature_cols].values[train_end:val_end])
    ytr, yval = y.values[:train_end], y.values[train_end:val_end]

    models: List[object] = []
    all_val_preds: List[np.ndarray] = []

    for i, xgb_params in enumerate(cfg.xgb_configs):
        print(f"[Part2b] Training XGB config {i + 1}/{len(cfg.xgb_configs)}: {xgb_params}")
        es_model = xgb.XGBRegressor(
            **xgb_params,
            random_state=cfg.seed,
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbosity=0,
        )
        es_model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        val_pred = es_model.predict(Xval)
        rmse = float(np.sqrt(mean_squared_error(yval, val_pred)))
        mae  = float(mean_absolute_error(yval, val_pred))

        # Best boosting round found by early stopping on the gate window
        best_iter = getattr(es_model, "best_iteration", None)
        n_final = int(best_iter) + 1 if best_iter is not None else int(xgb_params["n_estimators"])
        n_final = max(n_final, 10)
        print(f"  Config {i + 1} val RMSE: {rmse:.4f} | MAE: {mae:.4f} | "
              f"best_iteration -> n_estimators={n_final}")

        # Re-fit a FRESH estimator (no early stopping) on ALL labeled rows.
        final_params = dict(xgb_params)
        final_params["n_estimators"] = n_final
        final_model = xgb.XGBRegressor(
            **final_params,
            random_state=cfg.seed,
            eval_metric="rmse",
            verbosity=0,
        )
        X_all = imputer.transform(X[feature_cols].values)
        final_model.fit(X_all, y.values)
        models.append(final_model)
        all_val_preds.append(val_pred)

    # Simple average ensemble on the gate window
    val_ensemble = np.mean(all_val_preds, axis=0)
    val_actuals  = yval

    xgb_rmse = float(np.sqrt(mean_squared_error(val_actuals, val_ensemble)))
    xgb_mae  = float(mean_absolute_error(val_actuals, val_ensemble))
    mape     = float(np.nanmean(np.abs((val_actuals - val_ensemble) /
                                        np.where(val_actuals != 0, val_actuals, np.nan)))) * 100

    metrics = {
        "val_rmse": xgb_rmse,
        "val_mae": xgb_mae,
        "val_mape": mape,
        "n_configs": len(models),
    }
    print(f"[Part2b] XGB ensemble val RMSE: {xgb_rmse:.4f} | MAE: {xgb_mae:.4f}")
    return models, val_ensemble, val_actuals, metrics, imputer


def predict_all(
    X: pd.DataFrame,
    models: List[object],
    imputer: object,
) -> np.ndarray:
    """Predict on all rows, return ensemble average."""
    feature_cols = get_feature_cols(X)
    X_imp = imputer.transform(X[feature_cols].values)
    all_preds = np.stack([m.predict(X_imp) for m in models], axis=1)
    return np.mean(all_preds, axis=1)


def main() -> int:
    cfg = Part2bConfig()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    part1_dir = root / cfg.part1_dir_name
    part2_dir = root / cfg.part2_dir_name

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part2b] ROOT: {root}")
    print(f"[Part2b] Version: {SCRIPT_VERSION}\n")

    if not HAVE_XGB:
        print("[Part2b] XGBoost not available. Skipping sleeve.")
        summary = {
            "script_version": SCRIPT_VERSION,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "xgb_sleeve_recommended": False,
            "reason": "xgboost_not_installed",
        }
        with open(out_dir / "gas_part2b_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return 0

    # Load features
    try:
        X, y = load_features(part1_dir)
    except FileNotFoundError as e:
        print(f"[Part2b] FATAL: {e}")
        return 1

    X_lab, y_lab, X_live = split_labeled_live(X, y)
    print(f"[Part2b] Features: {len(X_lab)} labeled + {len(X_live)} live rows")

    # Train XGB ensemble on labeled rows only
    models, val_ensemble, val_actuals, xgb_metrics, imputer = train_xgb_ensemble(
        X_lab, y_lab, cfg
    )

    if not models:
        print("[Part2b] No models trained — skipping.")
        return 0

    # Gate: compare against Part2 baseline on the SAME trailing window.
    # FIX (Audit 2026-08): a missing baseline previously defaulted the gate to
    # recommended=True. An optional experimental sleeve must fail CLOSED —
    # no baseline means no evidence, so the gate does not pass.
    baseline_rmse = load_part2_baseline_rmse(part2_dir)
    xgb_rmse = xgb_metrics["val_rmse"]
    if baseline_rmse is not None and np.isfinite(baseline_rmse):
        recommended = bool(np.isfinite(xgb_rmse) and xgb_rmse < baseline_rmse)
        print(f"[Part2b] XGB RMSE: {xgb_rmse:.4f} vs Baseline: {baseline_rmse:.4f} "
              f"-> recommended={recommended}")
    else:
        recommended = False
        print("[Part2b] No Part2 baseline RMSE found — gate FAILS CLOSED "
              "(recommended=False). Run gas_part2 first.")

    # Full prediction tape (labeled + live rows; live actual = NaN)
    n_live = len(X_live)
    all_X = pd.concat([X_lab, X_live], ignore_index=True) if n_live else X_lab
    all_preds = predict_all(all_X, models, imputer)
    actual = (np.concatenate([y_lab.values, np.full(n_live, np.nan)])
              if n_live else y_lab.values)
    tape = pd.DataFrame({
        "week_date": all_X["week_date"],
        "actual": actual,
        "pred_xgb_ensemble": all_preds,
        "is_live": np.concatenate([np.zeros(len(X_lab), dtype=int),
                                   np.ones(n_live, dtype=int)]),
    })

    # Live next-week forecast (live row preferred; labeled fallback is retrospective)
    latest_pred = float(all_preds[-1])
    latest_week = pd.to_datetime(all_X["week_date"].iloc[-1])
    target_week = latest_week + pd.Timedelta(weeks=1)
    live_tag = "LIVE" if n_live else "RETROSPECTIVE"
    print(f"[Part2b] {live_tag} forecast — anchored {latest_week.date()}, "
          f"targeting week of {target_week.date()}: ${latest_pred:.3f}/gal")

    # Write artifacts
    tape_path = out_dir / "gas_xgb_tape.parquet"
    tape.to_parquet(tape_path, index=False)
    tape.to_csv(out_dir / "gas_xgb_tape.csv", index=False)
    print(f"[Part2b] XGB tape -> {tape_path}")

    model_path = out_dir / "gas_xgb_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "imputer": imputer}, f)
    print(f"[Part2b] XGB models -> {model_path}")

    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "xgb_sleeve_recommended": bool(recommended),
        "xgb_val_rmse": xgb_rmse,
        "baseline_val_rmse": baseline_rmse,
        "xgb_metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in xgb_metrics.items()},
        "latest_forecast": {
            "anchor_week": latest_week.strftime("%Y-%m-%d"),
            "target_week": target_week.strftime("%Y-%m-%d"),
            "pred_xgb": round(latest_pred, 4),
            "is_true_live_forecast": bool(n_live),
        },
        "n_configs_trained": len(models),
        "gate_window": "last_val_weeks_labeled_rows",
    }
    summary_path = out_dir / "gas_part2b_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part2b] Summary -> {summary_path}")

    status = "RECOMMENDED" if recommended else "NOT_RECOMMENDED"
    print(f"\n[Part2b] XGB sleeve gate: {status}")
    print("[Part2b] XGBoost sleeve complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
