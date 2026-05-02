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
    return X, y_df["target_gas_price"]


def get_feature_cols(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c != "week_date"]


def train_xgb_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Part2bConfig,
) -> Tuple[List[object], np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Train multiple XGB configs on train split, evaluate on val.
    Returns (models, val_preds_ensemble, val_actuals, val_metrics).
    """
    if not HAVE_XGB:
        return [], np.array([]), np.array([]), {}

    feature_cols = get_feature_cols(X)
    n = len(y)
    train_end = min(cfg.initial_train_weeks, int(n * 0.75))
    val_end   = min(train_end + cfg.val_weeks, n)

    # Impute NaN values
    imputer = SimpleImputer(strategy="median")
    X_vals = imputer.fit_transform(X[feature_cols].values)

    Xtr, ytr   = X_vals[:train_end], y.values[:train_end]
    Xval, yval = X_vals[train_end:val_end], y.values[train_end:val_end]

    models: List[object] = []
    all_val_preds: List[np.ndarray] = []

    for i, xgb_params in enumerate(cfg.xgb_configs):
        print(f"[Part2b] Training XGB config {i + 1}/{len(cfg.xgb_configs)}: {xgb_params}")
        model = xgb.XGBRegressor(
            **xgb_params,
            random_state=cfg.seed,
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbosity=0,
        )
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            verbose=False,
        )
        val_pred = model.predict(Xval)
        rmse = float(np.sqrt(mean_squared_error(yval, val_pred)))
        mae  = float(mean_absolute_error(yval, val_pred))
        print(f"  Config {i + 1} val RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        # Re-fit on full data
        model.fit(X_vals[:val_end], y.values[:val_end])
        models.append(model)
        all_val_preds.append(val_pred)

    # Simple average ensemble
    val_ensemble = np.mean(all_val_preds, axis=0)
    val_actuals  = yval

    xgb_rmse = float(np.sqrt(mean_squared_error(val_actuals, val_ensemble)))
    xgb_mae  = float(mean_absolute_error(val_actuals, val_ensemble))
    mape     = float(np.mean(np.abs((val_actuals - val_ensemble) /
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

    # Train XGB ensemble
    result = train_xgb_ensemble(X, y, cfg)
    models, val_ensemble, val_actuals, xgb_metrics, imputer = result

    if not models:
        print("[Part2b] No models trained — skipping.")
        return 0

    # Gate: compare against Part2 baseline
    baseline_rmse = load_part2_baseline_rmse(part2_dir)
    xgb_rmse = xgb_metrics["val_rmse"]
    recommended = True
    if baseline_rmse is not None:
        recommended = xgb_rmse < baseline_rmse
        print(f"[Part2b] XGB RMSE: {xgb_rmse:.4f} vs Baseline: {baseline_rmse:.4f} "
              f"-> recommended={recommended}")
    else:
        print(f"[Part2b] No baseline RMSE found — XGB sleeve recommended by default.")

    # Full prediction tape
    all_preds = predict_all(X, models, imputer)
    tape = pd.DataFrame({
        "week_date": X["week_date"],
        "actual": y.values,
        "pred_xgb_ensemble": all_preds,
    })

    # Latest week prediction
    latest_pred = float(all_preds[-1])
    latest_week = pd.to_datetime(X["week_date"].iloc[-1])
    print(f"[Part2b] Latest ({latest_week.date()}) XGB forecast: ${latest_pred:.3f}/gal")

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
            "week": latest_week.strftime("%Y-%m-%d"),
            "pred_xgb": round(latest_pred, 4),
        },
        "n_configs_trained": len(models),
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
