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

from gas_forecast_statistics import TEST_METHOD, compare_squared_errors
from gas_time_contract import pipeline_identity, sha256_file, strict_json_dump

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GAS_PART2_V3_HAC_BLOCKED_OOF"


@dataclass(frozen=True)
class Part2Config:
    root_env_var: str = "GASPRICE_ROOT"
    part1_dir_name: str = "artifacts_part1"
    out_dir_name: str = "artifacts_part2"
    seed: int = 42
    horizon_weeks: int = 1

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
        "hgb": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=cfg.hgb_max_iter,
                learning_rate=cfg.hgb_learning_rate,
                max_depth=cfg.hgb_max_depth,
                l2_regularization=cfg.hgb_l2,
                random_state=cfg.seed,
            ),
        ),
        "rf": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                max_features=cfg.rf_max_features,
                random_state=cfg.seed,
                n_jobs=-1,
            ),
        ),
        "gbm": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            GradientBoostingRegressor(
                n_estimators=cfg.gbm_n_estimators,
                learning_rate=cfg.gbm_learning_rate,
                max_depth=cfg.gbm_max_depth,
                random_state=cfg.seed,
            ),
        ),
        "elasticnet": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            ElasticNet(
                alpha=cfg.elasticnet_alpha,
                l1_ratio=cfg.elasticnet_l1_ratio,
                max_iter=5000,
            ),
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
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


def validate_feature_lineage(part1_dir: Path) -> None:
    """Require Part 1 inputs to be complete outputs from this pipeline run."""
    summary_path = part1_dir / "part1_summary.json"
    if not summary_path.exists():
        raise ValueError("Part 1 summary is missing")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    current_run = pipeline_identity()["pipeline_run_id"]
    if str(summary.get("pipeline_run_id", "")) != current_run:
        raise ValueError("Part 1 summary belongs to a different pipeline run")

    expected = {
        "gas_feature_matrix.parquet": summary.get("feature_matrix_sha256"),
        "gas_target.parquet": summary.get("target_sha256"),
    }
    for filename, digest in expected.items():
        path = part1_dir / filename
        if not digest or sha256_file(path) != str(digest):
            raise ValueError(f"{filename} hash does not match Part 1 summary")


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
) -> Tuple[Dict[str, object], Dict[str, float], Dict[str, float], pd.DataFrame, int]:
    """Run expanding-origin blocked validation, then refit on all labels.

    Each fold's ensemble weights are derived only from earlier validation
    folds. The scored row therefore cannot influence its own model or weight.
    The mandatory comparator is same-window price persistence.
    """
    feature_cols = get_feature_cols(X)
    if "gas_us_avg_current" not in feature_cols:
        raise ValueError(
            "Part 1 must provide gas_us_avg_current for the persistence baseline"
        )

    X_vals = X[feature_cols].replace([np.inf, -np.inf], np.nan).values
    y_vals = pd.to_numeric(y, errors="coerce").values.astype(float)
    dates = pd.to_datetime(X["week_date"]).values
    n_rows = len(y_vals)
    if n_rows < 12:
        raise ValueError("At least 12 labeled rows are required")

    val_len = min(cfg.val_weeks, max(8, int(n_rows * 0.25)))
    val_start = n_rows - val_len
    if val_start < 4:
        raise ValueError("Insufficient pre-validation history")
    if val_start < cfg.min_train_weeks:
        print(
            f"[Part2] WARN: Only {val_start} initial training rows "
            f"(recommended >= {cfg.min_train_weeks})."
        )

    fold_count = min(4, val_len)
    folds = [part for part in np.array_split(np.arange(val_start, n_rows), fold_count)
             if len(part)]
    model_names = list(build_models(cfg))
    model_oof = {name: np.full(val_len, np.nan) for name in model_names}
    ensemble_oof = np.full(val_len, np.nan)
    fold_ids = np.zeros(val_len, dtype=int)
    fold_weights: list[dict[str, float]] = []

    for fold_number, positions in enumerate(folds, start=1):
        train_end = int(positions[0])
        fold_models = build_models(cfg)

        prior_stop = train_end - val_start
        if prior_stop:
            inverse = {}
            for name in model_names:
                metric = compute_metrics(
                    y_vals[val_start:train_end], model_oof[name][:prior_stop]
                )["rmse"]
                if np.isfinite(metric) and metric > 0:
                    inverse[name] = 1.0 / metric
            total = sum(inverse.values())
            weights_now = (
                {name: inverse.get(name, 0.0) / total for name in model_names}
                if total > 0
                else {name: 1.0 / len(model_names) for name in model_names}
            )
        else:
            weights_now = {name: 1.0 / len(model_names) for name in model_names}

        print(
            f"[Part2] Fold {fold_number}/{len(folds)}: "
            f"train 0:{train_end}, score {positions[0]}:{positions[-1] + 1}"
        )
        offset = positions - val_start
        for name, model in fold_models.items():
            model.fit(X_vals[:train_end], y_vals[:train_end])
            model_oof[name][offset] = model.predict(X_vals[positions])

        matrix = np.column_stack([model_oof[name][offset] for name in model_names])
        vector = np.array([weights_now[name] for name in model_names])
        ensemble_oof[offset] = matrix @ vector
        fold_ids[offset] = fold_number
        fold_weights.append(weights_now)

    actual = y_vals[val_start:]
    persistence = pd.to_numeric(
        X.iloc[val_start:]["gas_us_avg_current"], errors="coerce"
    ).values.astype(float)

    metrics: Dict[str, float] = {}
    inverse_final: Dict[str, float] = {}
    for name in model_names:
        values = compute_metrics(actual, model_oof[name])
        for metric_name, value in values.items():
            metrics[f"{name}_{metric_name}"] = float(value)
        rmse = values["rmse"]
        if np.isfinite(rmse) and rmse > 0:
            inverse_final[name] = 1.0 / rmse

    total_final = sum(inverse_final.values())
    weights = (
        {name: inverse_final.get(name, 0.0) / total_final for name in model_names}
        if total_final > 0
        else {name: 1.0 / len(model_names) for name in model_names}
    )

    ensemble_metrics = compute_metrics(actual, ensemble_oof)
    persistence_metrics = compute_metrics(actual, persistence)
    for metric_name, value in ensemble_metrics.items():
        metrics[f"ensemble_{metric_name}"] = float(value)
    for metric_name, value in persistence_metrics.items():
        metrics[f"persistence_{metric_name}"] = float(value)

    paired_test = compare_squared_errors(
        actual,
        ensemble_oof,
        persistence,
        horizon=cfg.horizon_weeks,
    )
    p_value = float(paired_test["p_value"])

    positive_folds = 0
    for fold_number in sorted(set(fold_ids)):
        fold_mask = fold_ids == fold_number
        model_rmse = compute_metrics(actual[fold_mask], ensemble_oof[fold_mask])["rmse"]
        base_rmse = compute_metrics(actual[fold_mask], persistence[fold_mask])["rmse"]
        if np.isfinite(model_rmse) and np.isfinite(base_rmse) and model_rmse < base_rmse:
            positive_folds += 1

    required_positive = max(1, int(np.ceil(len(folds) * 0.75)))
    operator_validated = bool(
        np.isfinite(ensemble_metrics["rmse"])
        and np.isfinite(persistence_metrics["rmse"])
        and ensemble_metrics["rmse"] < persistence_metrics["rmse"]
        and np.isfinite(p_value)
        and p_value < 0.10
        and positive_folds >= required_positive
    )
    metrics.update(
        {
            "paired_p_value": p_value,
            "paired_test_statistic": float(paired_test["statistic"]),
            "paired_mean_loss_advantage": float(
                paired_test["mean_loss_advantage"]
            ),
            "paired_hac_lags": float(
                paired_test["hac_lags"]
                if paired_test["hac_lags"] is not None
                else np.nan
            ),
            "positive_folds": float(positive_folds),
            "required_positive_folds": float(required_positive),
            "operator_validated": float(operator_validated),
        }
    )

    oof = pd.DataFrame(
        {
            "week_date": pd.to_datetime(dates[val_start:]),
            "actual": actual,
            "pred_ensemble": ensemble_oof,
            "pred_persistence": persistence,
            "fold": fold_ids,
        }
    )
    for name in model_names:
        oof[f"pred_{name}"] = model_oof[name]
    oof["weights_json"] = [
        json.dumps(fold_weights[max(0, fold_id - 1)], sort_keys=True)
        for fold_id in fold_ids
    ]

    final_models = build_models(cfg)
    for name, model in final_models.items():
        print(f"[Part2] Re-fitting {name} on all labeled rows...")
        model.fit(X_vals, y_vals)

    print(
        f"[Part2] Ensemble RMSE={ensemble_metrics['rmse']:.4f}; "
        f"persistence RMSE={persistence_metrics['rmse']:.4f}; "
        f"paired p={p_value:.4f}; validated={operator_validated}"
    )
    return final_models, weights, metrics, oof, val_start


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
    persistence = pd.to_numeric(
        anchor.get("gas_us_avg_current", pd.Series([np.nan])), errors="coerce"
    ).iloc[0]
    preds["pred_persistence"] = (
        float(persistence) if np.isfinite(persistence) else np.nan
    )
    return preds, anchor_week, is_true_live


# ── Full forecast tape ─────────────────────────────────────────────────────────

def build_forecast_tape(
    X_lab: pd.DataFrame,
    y_lab: pd.Series,
    X_live: pd.DataFrame,
    models: Dict[str, object],
    weights: Dict[str, float],
    val_start_idx: int,
    oof_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate predictions for all rows (labeled + live) for historical analysis.

    FIX (Audit 2026-08):
      - Live rows are included with actual=NaN and in_sample=0.
      - in_sample flags rows predicted by the final refit. The validation
        window is then replaced with its genuine expanding-origin blocked
        predictions, so rows marked oos_val=1 are honest out-of-sample values.
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
    tape["pred_persistence"] = pd.to_numeric(
        all_X["gas_us_avg_current"], errors="coerce"
    ).values

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

    if oof_df is not None and not oof_df.empty:
        oof = oof_df.copy()
        oof["week_date"] = pd.to_datetime(oof["week_date"])
        oof = oof.set_index("week_date")
        tape_dates = pd.to_datetime(tape["week_date"])
        overlay = tape_dates.isin(oof.index)
        for column in (
            "pred_hgb",
            "pred_rf",
            "pred_elasticnet",
            "pred_gbm",
            "pred_ridge",
            "pred_ensemble",
            "pred_persistence",
        ):
            if column in tape.columns and column in oof.columns:
                mapped = tape_dates.map(oof[column])
                tape.loc[overlay, column] = mapped.loc[overlay].to_numpy()
        tape.loc[overlay, "in_sample"] = 0

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
        "operator_validation": {
            "operator_validated": bool(val_metrics.get("operator_validated", 0.0)),
            "ensemble_rmse": val_metrics.get("ensemble_rmse"),
            "persistence_rmse": val_metrics.get("persistence_rmse"),
            "paired_p_value": val_metrics.get("paired_p_value"),
            "paired_test_method": TEST_METHOD,
            "paired_test_statistic": val_metrics.get("paired_test_statistic"),
            "paired_mean_loss_advantage": val_metrics.get(
                "paired_mean_loss_advantage"
            ),
            "paired_hac_lags": (
                int(val_metrics["paired_hac_lags"])
                if np.isfinite(val_metrics.get("paired_hac_lags", np.nan))
                else None
            ),
            "positive_folds": int(val_metrics.get("positive_folds", 0)),
            "required_positive_folds": int(
                val_metrics.get("required_positive_folds", 0)
            ),
        },
        "forecast_tape_sha256": sha256_file(
            out_dir / "gas_forecast_tape.parquet"
        ),
        **pipeline_identity(),
        "config": {
            "val_weeks": cfg.val_weeks,
            "horizon_weeks": cfg.horizon_weeks,
            "min_train_weeks": cfg.min_train_weeks,
            "ensemble_weighting": cfg.ensemble_weighting,
            "gate_window": "expanding_origin_blocked_oof",
            "operator_gate": (
                "rmse<persistence, one-sided DM-HLN-HAC p<0.10, "
                ">=75% positive folds"
            ),
        },
    }
    path = out_dir / "gas_part2_summary.json"
    strict_json_dump(summary, path)
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
        validate_feature_lineage(part1_dir)
        X, y = load_features(part1_dir)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as e:
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
    tape = build_forecast_tape(
        X_lab, y_lab, X_live, models, weights, val_start_idx, oof_df
    )

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
    oof_df.to_csv(out_dir / "gas_oof_predictions.csv", index=False)
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
