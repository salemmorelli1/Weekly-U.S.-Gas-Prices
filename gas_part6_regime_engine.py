#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part6_regime_engine.py
===========================
Hidden Markov Model regime detector for the gas price market.

Regimes detected
----------------
  0  NORMAL       — Balanced supply/demand, moderate volatility
  1  SUPPLY_SHOCK — Refinery outages, hurricane season, pipeline disruptions
                    (high gas stocks deviation, spiking prices)
  2  DEMAND_SURGE — Summer driving season, economic expansion, low inventories
                    (low stocks, high demand trend, rising prices)
  3  DEFLATION    — Demand collapse, oil glut, recession signal
                    (falling crude, falling demand, stocks piling up)

Outputs
-------
- artifacts_part6/gas_regime_tape.parquet   — week_date + regime label + probabilities
- artifacts_part6/gas_regime_meta.json      — model metadata
- artifacts_part6/gas_regime_model.pkl      — serialized HMM (optional)

Pipeline position: FOURTH — after Part0/0b/0c, before Part1.
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


_colab_init(extra_packages=["hmmlearn", "scikit-learn", "pyarrow"])

import json, os, pickle, warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm
    HAVE_HMM = True
except ImportError:
    hmm = None
    HAVE_HMM = False
    print("[Part6] hmmlearn not available — GMM fallback active. "
          "Install: pip install hmmlearn")

SCRIPT_VERSION = "GAS_PART6_V1_CANONICAL"

# Features with NaN rate > this threshold are excluded from HMM training
_NAN_COVERAGE_THRESHOLD: float = 0.40


@dataclass(frozen=True)
class Part6Config:
    root_env_var: str = "GASPRICE_ROOT"
    part0_dir_name: str = "artifacts_part0"
    out_dir_name: str = "artifacts_part6"
    version: str = "V1_WEEKLY_CANONICAL"

    n_regimes: int = 4
    hmm_covariance_type: str = "full"
    hmm_n_iter: int = 500
    hmm_min_train_rows: int = 104   # 2 years of weekly data
    seed: int = 42

    # Primary regime features — NaN-heavy features auto-dropped
    regime_features: Tuple[str, ...] = (
        "gas_ret_1w",           # gas price 1-week return
        "gas_ret_4w",           # gas price 4-week return
        "gas_vol_8w",           # 8-week realized volatility
        "crude_ret_1w",         # crude oil 1-week return
        "crude_ret_4w",
        "rbob_ret_1w",          # RBOB gasoline futures return
        "eia_gas_stocks_zscore",    # stock deviation
        "eia_gas_demand_trend",     # demand trend
        "eia_refinery_util_dev",    # refinery utilization deviation
        "crack_spread_z",           # crack spread z-score
        "gas_price_level_z",        # gas price z-score vs history
        "usd_index_ret_4w",         # dollar index 4-week return
    )

    # Regime labels (for human-readable output)
    regime_labels: Tuple[str, ...] = (
        "NORMAL",
        "SUPPLY_SHOCK",
        "DEMAND_SURGE",
        "DEFLATION",
    )


def resolve_project_root(cfg: Part6Config) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── Feature engineering for regime model ──────────────────────────────────────

def build_regime_features(df: pd.DataFrame, cfg: Part6Config) -> pd.DataFrame:
    """
    Compute regime features from the master dataset.
    All features are strictly backward-looking.
    """
    df = df.copy()
    df = df.sort_values("week_date").reset_index(drop=True)

    # Gas price returns
    if "gas_us_avg" in df.columns:
        g = df["gas_us_avg"]
        df["gas_ret_1w"]  = g.pct_change(1).shift(1)
        df["gas_ret_4w"]  = g.pct_change(4).shift(1)
        df["gas_ret_12w"] = g.pct_change(12).shift(1)
        df["gas_vol_8w"]  = g.pct_change(1).shift(1).rolling(8, min_periods=4).std()
        # Price level z-score
        roll_mean = g.shift(1).rolling(52, min_periods=13).mean()
        roll_std  = g.shift(1).rolling(52, min_periods=13).std()
        df["gas_price_level_z"] = (g - roll_mean) / roll_std.replace(0, np.nan)

    # Crude oil returns
    if "wti_crude" in df.columns:
        c = df["wti_crude"]
        df["crude_ret_1w"] = c.pct_change(1).shift(1)
        df["crude_ret_4w"] = c.pct_change(4).shift(1)

    # RBOB gasoline futures returns
    if "rbob_gasoline" in df.columns:
        r = df["rbob_gasoline"]
        df["rbob_ret_1w"] = r.pct_change(1).shift(1)
        df["rbob_ret_4w"] = r.pct_change(4).shift(1)

    # Crack spread (rough proxy: gas price / crude price)
    if "gas_us_avg" in df.columns and "wti_crude" in df.columns:
        crack = (df["gas_us_avg"] / df["wti_crude"].replace(0, np.nan)).shift(1)
        crack_mean = crack.rolling(52, min_periods=13).mean()
        crack_std  = crack.rolling(52, min_periods=13).std()
        df["crack_spread_z"] = (crack - crack_mean) / crack_std.replace(0, np.nan)

    # USD index returns
    if "usd_index" in df.columns:
        df["usd_index_ret_4w"] = df["usd_index"].pct_change(4).shift(1)

    return df


def _select_features(df: pd.DataFrame, cfg: Part6Config) -> List[str]:
    """Return features from cfg.regime_features that have enough coverage."""
    selected = []
    for feat in cfg.regime_features:
        if feat not in df.columns:
            continue
        nan_rate = df[feat].isna().mean()
        if nan_rate > _NAN_COVERAGE_THRESHOLD:
            print(f"[Part6] Dropping {feat}: NaN rate {nan_rate:.1%} > threshold")
            continue
        selected.append(feat)
    print(f"[Part6] Using {len(selected)} features: {selected}")
    return selected


# ── Model training ─────────────────────────────────────────────────────────────

def train_hmm(
    X: np.ndarray,
    cfg: Part6Config,
) -> Tuple[object, bool]:
    """Train HMM or fall back to GMM. Returns (model, used_hmm)."""
    if HAVE_HMM:
        try:
            model = hmm.GaussianHMM(
                n_components=cfg.n_regimes,
                covariance_type=cfg.hmm_covariance_type,
                n_iter=cfg.hmm_n_iter,
                random_state=cfg.seed,
            )
            model.fit(X)
            print(f"[Part6] HMM trained. Converged: {model.monitor_.converged}")
            return model, True
        except Exception as e:
            print(f"[Part6] HMM failed ({e}) — falling back to GMM")

    # GMM fallback
    model = GaussianMixture(
        n_components=cfg.n_regimes,
        covariance_type="full",
        n_init=5,
        random_state=cfg.seed,
    )
    model.fit(X)
    print(f"[Part6] GMM trained (HMM fallback). BIC: {model.bic(X):.1f}")
    return model, False


def predict_regimes(
    model: object,
    X: np.ndarray,
    used_hmm: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (regime_labels, regime_probabilities)."""
    if used_hmm:
        labels = model.predict(X)
        probs  = model.predict_proba(X)
    else:
        labels = model.predict(X)
        probs  = model.predict_proba(X)
    return labels, probs


def label_regimes(
    labels: np.ndarray,
    probs: np.ndarray,
    cfg: Part6Config,
    feature_matrix: pd.DataFrame,
    selected_features: List[str],
) -> np.ndarray:
    """
    Re-label HMM/GMM regime indices to semantic names using
    the mean values of interpretable features within each regime.
    """
    n = cfg.n_regimes
    named = ["NORMAL"] * n

    gas_ret_feat = "gas_ret_4w"
    vol_feat      = "gas_vol_8w"
    stock_feat    = "eia_gas_stocks_zscore"
    demand_feat   = "eia_gas_demand_trend"

    regime_stats: Dict[int, Dict] = {}
    for i in range(n):
        mask = labels == i
        stats: Dict[str, float] = {}
        for feat in [gas_ret_feat, vol_feat, stock_feat, demand_feat]:
            if feat in selected_features:
                col_idx = selected_features.index(feat)
                vals = feature_matrix.iloc[mask, col_idx]
                stats[feat] = float(vals.mean()) if len(vals) > 0 else 0.0
        regime_stats[i] = stats

    # Assign semantic labels based on sign patterns
    for i, stats in regime_stats.items():
        ret  = stats.get(gas_ret_feat, 0.0)
        vol  = stats.get(vol_feat, 0.0)
        stk  = stats.get(stock_feat, 0.0)
        dem  = stats.get(demand_feat, 0.0)

        if ret > 0.01 and vol > 0.015:
            named[i] = "SUPPLY_SHOCK"
        elif ret > 0.005 and dem > 0.01:
            named[i] = "DEMAND_SURGE"
        elif ret < -0.005 and stk > 0.3:
            named[i] = "DEFLATION"
        else:
            named[i] = "NORMAL"

    # Ensure uniqueness — fallback if two regimes get same label
    seen = {}
    for i, lbl in enumerate(named):
        if lbl in seen:
            named[i] = f"{lbl}_{i}"
        seen[lbl] = i

    print("[Part6] Regime label assignment:")
    for i, lbl in enumerate(named):
        stats = regime_stats.get(i, {})
        print(f"  Regime {i} -> {lbl} | stats: {stats}")

    # Apply naming to the integer label array
    named_arr = np.array([named[l] for l in labels])
    return named_arr


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part6Config()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    part0_dir = root / cfg.part0_dir_name

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part6] ROOT: {root}")
    print(f"[Part6] Version: {SCRIPT_VERSION}\n")

    # Load master dataset
    parquet_path = part0_dir / "gas_weekly_master.parquet"
    if not parquet_path.exists():
        print(f"[Part6] FATAL: {parquet_path} not found. Run gas_part0 first.")
        return 1

    df = pd.read_parquet(parquet_path)
    df["week_date"] = pd.to_datetime(df["week_date"])
    df = df.sort_values("week_date").reset_index(drop=True)
    print(f"[Part6] Loaded {len(df)} weeks from master parquet")

    # Build regime features
    df = build_regime_features(df, cfg)

    # Select features
    selected = _select_features(df, cfg)
    if len(selected) < 2:
        print(f"[Part6] FATAL: Only {len(selected)} features available (need >= 2).")
        return 1

    # Build feature matrix — drop rows with any NaN
    feat_df = df[["week_date"] + selected].copy()
    feat_df = feat_df.dropna(subset=selected)
    print(f"[Part6] Training rows after NaN drop: {len(feat_df)}")

    if len(feat_df) < cfg.hmm_min_train_rows:
        print(f"[Part6] WARN: Only {len(feat_df)} rows (min {cfg.hmm_min_train_rows}). "
              "Proceeding with reduced confidence.")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df[selected].values)

    # Train model
    model, used_hmm = train_hmm(X, cfg)

    # Predict
    int_labels, probs = predict_regimes(model, X, used_hmm)

    # Semantic labeling
    named_labels = label_regimes(int_labels, probs, cfg, feat_df[selected], selected)

    # Build output tape
    tape = feat_df[["week_date"]].copy()
    tape["regime_int"]   = int_labels
    tape["regime_label"] = named_labels
    for i in range(cfg.n_regimes):
        tape[f"regime_prob_{i}"] = probs[:, i]

    # Merge back to full index via left join
    full_tape = df[["week_date"]].merge(tape, on="week_date", how="left")
    full_tape["regime_label"] = full_tape["regime_label"].fillna("UNKNOWN")

    # Write tape
    tape_path = out_dir / "gas_regime_tape.parquet"
    full_tape.to_parquet(tape_path, index=False)
    full_tape.to_csv(out_dir / "gas_regime_tape.csv", index=False)
    print(f"[Part6] Regime tape -> {tape_path}")

    # Serialize model
    model_path = out_dir / "gas_regime_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "used_hmm": used_hmm,
                     "selected_features": selected}, f)
    print(f"[Part6] Model -> {model_path}")

    # Regime distribution
    dist = full_tape["regime_label"].value_counts().to_dict()
    print(f"[Part6] Regime distribution: {dist}")

    # Meta
    meta = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "HMM" if used_hmm else "GMM",
        "n_regimes": cfg.n_regimes,
        "selected_features": selected,
        "n_train_rows": len(feat_df),
        "regime_distribution": {k: int(v) for k, v in dist.items()},
        "latest_regime": str(full_tape["regime_label"].iloc[-1]),
        "latest_week": str(full_tape["week_date"].iloc[-1].date()),
    }
    meta_path = out_dir / "gas_regime_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[Part6] Meta -> {meta_path}")
    print(f"\n[Part6] Latest regime: {meta['latest_regime']} "
          f"(week of {meta['latest_week']})")

    print("\n[Part6] Regime engine complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
