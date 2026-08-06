#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part2_forecaster.py
========================
Sklearn ensemble forecaster for weekly U.S. average gas prices.

Models trained
--------------
  1. HistGradientBoostingRegressor  (primary)
  2. RandomForestRegressor
  3. ElasticNet (linear baseline)
  4. GradientBoostingRegressor (secondary GBM)

Ensemble method: weighted average (weights optimized on validation set).

Outputs
-------
  artifacts_part2/gas_forecast_tape.parquet  — week_date + predictions per model + ensemble
  artifacts_part2/gas_part2_summary.json     — metrics, weights, model health

Metrics computed
----------------
  MAE, RMSE, MAPE, Directional Accuracy, R^2
  Walk-forward cross-validation (no look-ahead bias)

Pipeline position: SIXTH — after Part1.
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

import json, os, pickle, warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART2_V1_CANONICAL"


@dataclass(frozen=True)
class Part2Config:
    root_env_var: str = "GASPRICE_ROOT"
    part1_dir_name: str = "artifacts_part1"
    out_dir_name: str = "artifacts_part2"
    seed: int = 42

    # Validation settings — the last `val_weeks` labeled rows form the
    # COMMON GATE WINDOW shared by Part 2 / 2b / 2a (see walk_forward_train).
    val_weeks: int = 52              # 1 year trailing validation window
    min_train_weeks: int = 156       # warn below 3 years of training data

    # Model hyperparameters
    hgb_max_iter: int = 500
    hgb_learning_rate: float = 0.05
    hgb_max_depth: int = 5
    hgb_l2: float = 1.0

    rf_n_estimators: int = 300
    rf_max_depth: int = 10
    rf_max_features: str = "sqrt"

    gbm_n_estimators: int = 300
    gbm_learning_rate: float = 0.05
    gbm_max_depth: int = 4

    elasticnet_alpha: float = 0.1
    elasticnet_l1_ratio: float = 0.5

    # Ensemble weight optimization: "equal" or "val_rmse"
    ensemble_weighting: str = "val_rmse"


def resolve_project_root(cfg: Part2Config) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan,
                "r2": np.nan, "dir_acc": np.nan}

    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / np.where(yt != 0, yt, np.nan)))) * 100
    r2   = float(r2_score(yt, yp))

    # Directional accuracy: did we correctly predict week-over-week direction?
    # Requires at least 2 data points; use index-based lag
    dir_acc = np.nan
    if len(yt) > 1:
        true_dir = np.sign(np.diff(yt))
        pred_dir = np.sign(np.diff(yp))
        dir_acc  = float(np.mean(true_dir == pred_dir))

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "dir_acc": dir_acc}


def naive_baseline_metrics(y: np.ndarray) -> Dict[str, float]:
    """Naive forecast: last week's price = next week's price."""
    y_naive = np.roll(y, 1)
    y_naive[0] = np.nan
    return compute_metrics(y[1:], y_naive[1:])


# ── Model factory ──────────────────────────────────────────────────────────────

def build_models(cfg: Part2Config) -> Dict[str, object]:
    """Return dict of sklearn pipeline models."""
    models = {
        "hgb": HistGradientBoostingRegressor(
            max_iter=cfg.hgb_max_iter,
            learning_rate=cfg.hgb_learning_rate,
            max_depth=cfg.hgb_max_depth,
            l2_regularization=cfg.hgb_l2,
            random_state=cfg.seed,
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                max_features=cfg.rf_max_features,
                random_state=cfg.seed,
                n_jobs=-1,
            ),
        ),
        "gbm": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                n_estimators=cfg.gbm_n_estimators,
                learning_rate=cfg.gbm_learning_rate,
                max_depth=cfg.gbm_max_depth,
                random_state=cfg.seed,
            ),
        ),
        "elasticnet": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(
                alpha=cfg.elasticnet_alpha,
                l1_ratio=cfg.elasticnet_l1_ratio,
                max_iter=5000,
            ),
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
    }
    return models


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features(part1_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    matrix_path = part1_dir / "gas_feature_matrix.parquet"
    target_path  = part1_dir / "gas_target.parquet"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {matrix_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    X = pd.read_parquet(matrix_path)
    y_df = pd.read_parquet(target_path)

    X["week_date"] = pd.to_datetime(X["week_date"])
    y = y_df["target_gas_price"]

    # Live-row contract (Audit 2026-08): Part 1 now ships trailing rows whose
    # targets are not yet realized, flagged is_live=1. Older matrices without
    # the flag are treated as all-labeled for backward compatibility.
    if "is_live" not in X.columns:
        X = X.copy()
        X["is_live"] = 0
    return X, y


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


# ── Walk-forward training ──────────────────────────────────────────────────────

def walk_forward_train(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Part2Config,
) -> Tuple[Dict[str, object], Dict[str, float], pd.DataFrame]:
    """
    Out-of-sample validation on the COMMON GATE WINDOW, then refit on all
    labeled rows for the live forecast.

    FIX (Audit 2026-08):
      - The docstring/signature previously disagreed (declared 3-tuple,
        returned 4). Corrected: returns
        (fitted_models, ensemble_weights, flat_val_metrics, oof_df, val_start_idx).
      - The validation window is now the LAST cfg.val_weeks labeled rows.
        This window is the shared sleeve-gate contract: Part 2, Part 2b, and
        Part 2a all score on the same trailing 52 labeled weeks, so the
        inverse-RMSE gates in Part 3 compare like with like. The previous
        version validated on an early-2000s window while the sleeves used
        different splits — the gate RMSEs were not comparable.

    X and y must already be labeled-rows-only (no live rows).
    """
    feature_cols = get_feature_cols(X)
    X_vals = X[feature_cols].values
    y_vals = y.values
    dates  = X["week_date"].values
    n = len(y_vals)

    models = build_models(cfg)
    val_preds: Dict[str, List[float]] = {m: [] for m in models}

    # Common gate window: last val_weeks labeled rows.
    val_len = min(cfg.val_weeks, max(1, int(n * 0.25)))
    train_end = n - val_len
    val_end   = n
    if train_end < cfg.min_train_weeks:
        print(f"[Part2] WARN: Only {train_end} training rows "
              f"(recommended >= {cfg.min_train_weeks}).")

    print(f"[Part2] Common gate window: train 0:{train_end}, val {train_end}:{val_end} "
          f"(last {val_len} labeled weeks)")

    Xtr = X_vals[:train_end]
    ytr = y_vals[:train_end]
    Xval = X_vals[train_end:val_end]
    yval = y_vals[train_end:val_end]

    for name, model in models.items():
        print(f"[Part2] Training {name}...")
        model.fit(Xtr, ytr)
        val_pred = model.predict(Xval)
        val_preds[name] = val_pred.tolist()

    val_actuals = yval.tolist()
    val_dates   = [pd.Timestamp(d) for d in dates[train_end:val_end]]

    # Compute per-model validation metrics
    val_metrics: Dict[str, Dict] = {}
    val_rmses: Dict[str, float] = {}
    for name, preds in val_preds.items():
        m = compute_metrics(np.array(val_actuals), np.array(preds))
        val_metrics[name] = m
        val_rmses[name] = m["rmse"]
        print(f"  [Part2] {name} val RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | "
              f"MAPE: {m['mape']:.2f}% | DirAcc: {m['dir_acc']:.3f}")

    # Compute ensemble weights
    if cfg.ensemble_weighting == "val_rmse":
        # Inverse RMSE weighting
        inv_rmse = {
            k: 1.0 / v for k, v in val_rmses.items()
            if v is not None and np.isfinite(v) and v > 0
        }
        total = sum(inv_rmse.values())
        weights = {k: v / total for k, v in inv_rmse.items()}
    else:
        n_m = len(models)
        weights = {k: 1.0 / n_m for k in models}

    print(f"[Part2] Ensemble weights: { {k: f'{v:.3f}' for k, v in weights.items()} }")

    # Re-fit on ALL labeled data so the live forecast uses every realized week.
    print("[Part2] Re-fitting on all labeled data for the live forecast...")
    for name, model in models.items():
        model.fit(X_vals, y_vals)

    # OOF predictions DataFrame (val window)
    oof_df = pd.DataFrame({
        "week_date": val_dates,
        "actual": val_actuals,
    })
    for name, preds in val_preds.items():
        oof_df[f"pred_{name}"] = preds

    oof_ensemble = np.zeros(len(val_actuals))
    for name, w in weights.items():
        oof_ensemble += w * np.array(val_preds[name])
    oof_df["pred_ensemble"] = oof_ensemble
    oof_df["weights_json"] = json.dumps(weights)

    # Ensemble metrics on the common gate window — this is the number Part 2b
    # and Part 2a gate against, so it must exist under an "ensemble" key.
    ens_metrics = compute_metrics(np.array(val_actuals), oof_ensemble)
    print(f"[Part2] ENSEMBLE val RMSE: {ens_metrics['rmse']:.4f} | "
          f"MAE: {ens_metrics['mae']:.4f} | MAPE: {ens_metrics['mape']:.2f}%")

    # Flatten val_metrics for return
    flat_val_metrics: Dict[str, float] = {}
    for model_name, m in val_metrics.items():
        for metric_name, val in m.items():
            flat_val_metrics[f"{model_name}_{metric_name}"] = float(val) if val is not None else np.nan
    for metric_name, val in ens_metrics.items():
        flat_val_metrics[f"ensemble_{metric_name}"] = float(val) if val is not None else np.nan

    return models, weights, flat_val_metrics, oof_df, train_end


# ── Live prediction ────────────────────────────────────────────────────────────

def predict_latest(
    X_lab: pd.DataFrame,
    X_live: pd.DataFrame,
    models: Dict[str, object],
    weights: Dict[str, float],
) -> Tuple[Dict[str, float], pd.Timestamp, bool]:
    """
    Predict the live row — the genuine next-week forecast.

    FIX (Live-row contract, Audit 2026-08): the previous version predicted on
    the last row of the labeled matrix, i.e. a week whose target price was
    already realized. It never produced an actual forward forecast. Now the
    live row (target not yet realized) is preferred; the last labeled row is
    only used as a retrospective fallback when no live row exists.

    Returns (preds, forecast_anchor_week, is_true_live_forecast).
    """
    feature_cols = get_feature_cols(X_lab)
    if len(X_live):
        anchor = X_live.iloc[[-1]]
        is_true_live = True
    else:
        anchor = X_lab.iloc[[-1]]
        is_true_live = False
        print("[Part2] WARN: No live row — forecast is retrospective "
              "(its target week is already realized).")

    anchor_week = pd.to_datetime(anchor["week_date"].iloc[0])
    latest_row = anchor[feature_cols].values

    preds: Dict[str, float] = {}
    for name, model in models.items():
        try:
            preds[f"pred_{name}"] = float(model.predict(latest_row)[0])
        except Exception as e:
            print(f"[Part2] {name} prediction failed: {e}")
            preds[f"pred_{name}"] = np.nan

    # Per-forecast weight renormalization over the models that succeeded —
    # a single failed model must not drag the ensemble toward zero.
    num = 0.0
    den = 0.0
    for name in models:
        v = preds.get(f"pred_{name}", np.nan)
        if np.isfinite(v):
            w = weights.get(name, 0.0)
            num += w * v
            den += w
    preds["pred_ensemble"] = float(num / den) if den > 0 else np.nan
    return preds, anchor_week, is_true_live


# ── Full forecast tape ─────────────────────────────────────────────────────────

def build_forecast_tape(
    X_lab: pd.DataFrame,
    y_lab: pd.Series,
    X_live: pd.DataFrame,
    models: Dict[str, object],
    weights: Dict[str, float],
    val_start_idx: int,
) -> pd.DataFrame:
    """
    Generate predictions for all rows (labeled + live) for historical analysis.

    FIX (Audit 2026-08):
      - Live rows are included with actual=NaN and in_sample=0.
      - in_sample flags rows the final models were trained on. Since the
        final refit uses ALL labeled rows, every labeled row is in-sample —
        honest OOS history lives in the OOF file (common gate window, flagged
        here too via oos_val=1 for convenience).
      - Ensemble fusion previously used fillna(0), which dragged the fused
        value toward zero wherever a model failed. Now weights renormalize
        per row over finite predictions only.
    """
    feature_cols = get_feature_cols(X_lab)
    n_lab, n_live = len(X_lab), len(X_live)
    all_X = pd.concat([X_lab, X_live], ignore_index=True) if n_live else X_lab
    actual = np.concatenate([y_lab.values, np.full(n_live, np.nan)]) if n_live else y_lab.values

    tape = pd.DataFrame({"week_date": all_X["week_date"], "actual": actual})
    tape["in_sample"] = np.concatenate([np.ones(n_lab, dtype=int),
                                        np.zeros(n_live, dtype=int)])
    oos_val = np.zeros(len(tape), dtype=int)
    oos_val[val_start_idx:n_lab] = 1  # common gate window rows
    tape["oos_val"] = oos_val
    tape["is_live"] = np.concatenate([np.zeros(n_lab, dtype=int),
                                      np.ones(n_live, dtype=int)])

    for name, model in models.items():
        try:
            tape[f"pred_{name}"] = model.predict(all_X[feature_cols].values)
        except Exception as e:
            print(f"[Part2] Full tape {name} failed: {e}")
            tape[f"pred_{name}"] = np.nan

    # Per-row renormalized weighted ensemble over finite predictions
    pred_matrix = np.column_stack([
        tape[f"pred_{name}"].values if f"pred_{name}" in tape.columns
        else np.full(len(tape), np.nan)
        for name in models
    ])
    w_vec = np.array([weights.get(name, 0.0) for name in models])
    finite = np.isfinite(pred_matrix)
    w_row = finite * w_vec[None, :]
    w_sum = w_row.sum(axis=1)
    fused = np.where(
        w_sum > 0,
        np.nansum(np.where(finite, pred_matrix, 0.0) * w_row, axis=1) / np.where(w_sum > 0, w_sum, 1.0),
        np.nan,
    )
    tape["pred_ensemble"] = fused

    return tape


# ── Summary ────────────────────────────────────────────────────────────────────

def write_part2_summary(
    out_dir: Path,
    latest_preds: Dict[str, float],
    val_metrics: Dict[str, float],
    weights: Dict[str, float],
    latest_week: pd.Timestamp,
    cfg: Part2Config,
    naive_metrics: Dict[str, float],
    is_true_live: bool,
) -> None:
    target_week = latest_week + pd.Timedelta(weeks=1)
    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_anchor_week": latest_week.strftime("%Y-%m-%d"),
        "forecast_target_week": target_week.strftime("%Y-%m-%d"),
        "is_true_live_forecast": bool(is_true_live),
        "latest_predictions": {k: (round(v, 4) if np.isfinite(v) else None)
                               for k, v in latest_preds.items()},
        "ensemble_weights": weights,
        "val_metrics": {k: round(v, 4) if np.isfinite(v) else None
                        for k, v in val_metrics.items()},
        "naive_baseline_metrics": {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else None)
                                   for k, v in naive_metrics.items()},
        "config": {
            "val_weeks": cfg.val_weeks,
            "min_train_weeks": cfg.min_train_weeks,
            "ensemble_weighting": cfg.ensemble_weighting,
            "gate_window": "last_val_weeks_labeled_rows",
        },
    }
    path = out_dir / "gas_part2_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part2] Summary -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part2Config()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    part1_dir = root / cfg.part1_dir_name

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part2] ROOT: {root}")
    print(f"[Part2] Version: {SCRIPT_VERSION}\n")

    # Load features
    try:
        X, y = load_features(part1_dir)
    except FileNotFoundError as e:
        print(f"[Part2] FATAL: {e}. Run gas_part1 first.")
        return 1

    X_lab, y_lab, X_live = split_labeled_live(X, y)
    print(f"[Part2] Features: {len(X_lab)} labeled + {len(X_live)} live rows x "
          f"{len(get_feature_cols(X))} features")
    print(f"[Part2] Target: ${y_lab.min():.3f} - ${y_lab.max():.3f}/gal\n")

    # Out-of-sample validation on the common gate window + final refit
    models, weights, val_metrics, oof_df, val_start_idx = walk_forward_train(
        X_lab, y_lab, cfg
    )

    # Full forecast tape (labeled + live rows)
    tape = build_forecast_tape(X_lab, y_lab, X_live, models, weights, val_start_idx)

    # Live next-week forecast
    latest_preds, latest_week, is_true_live = predict_latest(
        X_lab, X_live, models, weights
    )
    target_week = latest_week + pd.Timedelta(weeks=1)
    live_tag = "LIVE" if is_true_live else "RETROSPECTIVE"
    print(f"\n[Part2] {live_tag} forecast — anchored on week of {latest_week.date()}, "
          f"targeting week of {target_week.date()}:")
    for k, v in latest_preds.items():
        if np.isfinite(v):
            print(f"  {k}: ${v:.3f}/gal")

    # Naive baseline for comparison (labeled rows)
    naive_metrics = naive_baseline_metrics(y_lab.values)
    print(f"\n[Part2] Naive baseline RMSE: {naive_metrics['rmse']:.4f} | "
          f"MAE: {naive_metrics['mae']:.4f}")

    # Write artifacts
    tape_path = out_dir / "gas_forecast_tape.parquet"
    tape.to_parquet(tape_path, index=False)
    tape.to_csv(out_dir / "gas_forecast_tape.csv", index=False)
    print(f"[Part2] Forecast tape -> {tape_path}")

    oof_path = out_dir / "gas_oof_predictions.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"[Part2] OOF predictions -> {oof_path}")

    # Save models
    model_path = out_dir / "gas_part2_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "weights": weights}, f)
    print(f"[Part2] Models -> {model_path}")

    write_part2_summary(out_dir, latest_preds, val_metrics, weights,
                        latest_week, cfg, naive_metrics, is_true_live)

    print("\n[Part2] Forecaster ensemble complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
