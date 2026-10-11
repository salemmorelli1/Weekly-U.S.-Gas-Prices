#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_part2a_lstm_sleeve.py
==========================
LSTM (Long Short-Term Memory) deep learning sleeve for gas price forecasting.

Architecture
------------
  Input:  [sequence_length x n_features]  (sliding window of weekly observations)
  Layer1: LSTM(128 units) + Dropout(0.2)
  Layer2: LSTM(64 units) + Dropout(0.2)
  Layer3: Dense(32, ReLU)
  Output: Dense(1) — next week's gas price ($/gal)

Gate condition
--------------
If val_rmse_lstm < baseline_rmse AND val_rmse_lstm < xgb_rmse, the LSTM sleeve
is recommended for inclusion in the Part3 fusion.

Activate this sleeve only after Part2b's xgb_sleeve_recommended = true.
The gate ensures the LSTM genuinely adds signal before paying its compute cost.

Outputs
-------
  artifacts_part2a/gas_lstm_tape.parquet       — week_date + lstm predictions
  artifacts_part2a/gas_part2a_summary.json     — metrics + gate result
  artifacts_part2a/gas_lstm_model.pt           — saved PyTorch model weights

Pipeline position: EIGHTH (optional) — after Part2b, before Part3.
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


_colab_init(extra_packages=["torch", "scikit-learn", "pyarrow"])

import json, os, warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from gas_time_contract import pipeline_identity, sha256_file, strict_json_dump
from gas_validation import NESTED_PROTOCOL, aligned_oof, blocked_folds, candidate_gate
from gas_part2_forecaster import validate_feature_lineage

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAVE_TORCH = False
    print("[Part2a] PyTorch not installed. Install: pip install torch")
    print("[Part2a] Uncomment 'torch' in requirements.txt to enable LSTM sleeve.")

SCRIPT_VERSION = "GAS_PART2A_LSTM_V3_NESTED_OOF"


@dataclass
class Part2aConfig:
    root_env_var: str = "GASPRICE_ROOT"
    part1_dir_name: str = "artifacts_part1"
    part2_dir_name: str = "artifacts_part2"
    part2b_dir_name: str = "artifacts_part2b"
    out_dir_name: str = "artifacts_part2a"
    seed: int = 42

    # Sequence length: how many past weeks the LSTM sees
    sequence_length: int = 16

    # Architecture
    lstm_hidden_size_1: int = 128
    lstm_hidden_size_2: int = 64
    dense_hidden_size: int = 32
    dropout_rate: float = 0.20

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15          # Early stopping patience
    lr_patience: int = 7        # LR scheduler patience

    # Data split
    initial_train_weeks: int = 156
    val_weeks: int = 52

    # Device
    device: str = "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "cpu"


def resolve_project_root(cfg: Part2aConfig) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    if _IN_COLAB:
        return Path("/content/drive/MyDrive/GasPriceForecast")
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── LSTM Model ─────────────────────────────────────────────────────────────────
# FIX (Audit 2026-08): GasPriceLSTM subclassed nn.Module at MODULE level, so
# when torch was absent the file crashed at import time with
# "AttributeError: 'NoneType' object has no attribute 'Module'" — before
# main()'s HAVE_TORCH graceful-skip could ever run. The torch-dependent
# definitions are now guarded so the sleeve degrades gracefully as designed.

if HAVE_TORCH:

  class GasPriceLSTM(nn.Module):
      """
      Two-layer LSTM for weekly gas price forecasting.
      Input shape: (batch, seq_len, n_features)
      Output shape: (batch, 1) — next week's price
      """

      def __init__(
          self,
          n_features: int,
          hidden_size_1: int = 128,
          hidden_size_2: int = 64,
          dense_size: int = 32,
          dropout: float = 0.20,
      ):
          super().__init__()
          self.lstm1 = nn.LSTM(
              input_size=n_features,
              hidden_size=hidden_size_1,
              num_layers=1,
              batch_first=True,
              dropout=0.0,
          )
          self.dropout1 = nn.Dropout(dropout)
          self.lstm2 = nn.LSTM(
              input_size=hidden_size_1,
              hidden_size=hidden_size_2,
              num_layers=1,
              batch_first=True,
              dropout=0.0,
          )
          self.dropout2 = nn.Dropout(dropout)
          self.fc1 = nn.Linear(hidden_size_2, dense_size)
          self.relu = nn.ReLU()
          self.fc2 = nn.Linear(dense_size, 1)

      def forward(self, x: "torch.Tensor") -> "torch.Tensor":
          # x: (batch, seq_len, n_features)
          out1, _ = self.lstm1(x)                   # (batch, seq_len, hidden1)
          out1    = self.dropout1(out1)
          out2, _ = self.lstm2(out1)                 # (batch, seq_len, hidden2)
          out2    = self.dropout2(out2)
          last    = out2[:, -1, :]                   # use last timestep
          dense   = self.relu(self.fc1(last))
          pred    = self.fc2(dense)                  # (batch, 1)
          return pred.squeeze(-1)


# ── Sequence builder ───────────────────────────────────────────────────────────

def build_sequences(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X_seq, y_seq) sliding window sequences.

    FIX (Audit 2026-08, horizon alignment): the previous builder used
    y_seq[i] = y[i + seq_len] with a window ending at row i + seq_len - 1.
    The anchor row (whose lagged features encode the most recent week) was
    EXCLUDED from its own window, silently turning the H=1 problem into an
    effective H=2 forecast. The window now ENDS at the row whose target it
    predicts:
        X_seq[i] = X_arr[i : i + seq_len]        (rows i .. i+seq_len-1)
        y_seq[i] = y_arr[i + seq_len - 1]        (target of the LAST row)
    This matches the sklearn/XGB sleeves, which predict row t's target from
    row t's features.
    """
    n = len(y_arr)
    X_seq = []
    y_seq = []
    for i in range(n - seq_len + 1):
        X_seq.append(X_arr[i : i + seq_len])
        y_seq.append(y_arr[i + seq_len - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_lstm(
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: "DataLoader",
    cfg: Part2aConfig,
    device: str,
) -> Tuple["nn.Module", List[float], List[float]]:
    """Train the LSTM with early stopping. Returns (model, train_losses, val_losses)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    # FIX (Audit 2026-08): `verbose` kwarg removed — deprecated in torch 2.2
    # and rejected by newer releases.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.lr_patience, factor=0.5
    )
    criterion = nn.HuberLoss(delta=0.1)  # robust to outliers

    train_losses: List[float] = []
    val_losses:   List[float] = []
    best_val_loss = float("inf")
    best_weights  = None
    patience_count = 0

    for epoch in range(cfg.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(yb)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{cfg.epochs} | "
                  f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(best val_loss: {best_val_loss:.4f})")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, train_losses, val_losses


# ── Data loading ───────────────────────────────────────────────────────────────

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


def part2b_gate_passed(part2b_dir: Path) -> Tuple[bool, str]:
    """
    FIX (Audit 2026-08, mirrors PriceCall Part-46 audit):
    The module docstring has always said 'Activate this sleeve only after
    Part2b's xgb_sleeve_recommended = true' — but nothing ENFORCED it. Once
    torch was installed, the LSTM ran unconditionally every week regardless
    of the upstream gate, burning CI wall time on a sleeve its own contract
    says is not yet justified. This helper reads Part 2b's fresh summary
    (Part 2b always runs earlier in the pipeline order) and fails CLOSED on
    any error: missing file, bad JSON, or missing key all mean 'do not run'.
    Override for experimentation with GASPRICE_FORCE_LSTM=1.
    """
    if os.environ.get("GASPRICE_FORCE_LSTM", "").strip() == "1":
        return True, "GASPRICE_FORCE_LSTM=1 override"
    summary_path = part2b_dir / "gas_part2b_summary.json"
    if not summary_path.exists():
        return False, f"gas_part2b_summary.json not found at {summary_path}"
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception as exc:
        return False, f"could not parse {summary_path}: {exc}"
    rec = bool(s.get("xgb_sleeve_recommended", False)
               and s.get("validation_protocol") == NESTED_PROTOCOL
               and str(s.get("pipeline_run_id")) == pipeline_identity()["pipeline_run_id"])
    return rec, f"xgb_sleeve_recommended={rec} (source={summary_path})"


def load_baseline_rmse(part2_dir: Path, part2b_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    base_rmse = None
    xgb_rmse  = None
    p2 = part2_dir / "gas_part2_summary.json"
    if p2.exists():
        with open(p2) as f:
            s = json.load(f)
        for k, v in s.get("val_metrics", {}).items():
            if "ensemble" in k and "rmse" in k and v is not None:
                base_rmse = float(v)
    p2b = part2b_dir / "gas_part2b_summary.json"
    if p2b.exists():
        with open(p2b) as f:
            s = json.load(f)
        xgb_rmse = s.get("xgb_val_rmse")
    return base_rmse, xgb_rmse


def get_feature_cols(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c not in ("week_date", "is_live")]


# ── Main ───────────────────────────────────────────────────────────────────────

def _new_model(n_features: int, cfg: Part2aConfig):
    return GasPriceLSTM(n_features, cfg.lstm_hidden_size_1, cfg.lstm_hidden_size_2,
                        cfg.dense_hidden_size, cfg.dropout_rate)


def _scaled_history(values: np.ndarray, actual: np.ndarray):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputer.fit_transform(values))
    mean, std = float(np.mean(actual)), float(np.std(actual)) + 1e-8
    return scaled, (actual - mean) / std, (imputer, scaler, mean, std)


def _fixed_lstm_fit(values: np.ndarray, actual: np.ndarray, epochs: int, cfg: Part2aConfig):
    """Refit only on labels available at this forecast origin."""
    torch.manual_seed(cfg.seed)
    scaled, target, bundle = _scaled_history(values, actual)
    sequences, targets = build_sequences(scaled, target, cfg.sequence_length)
    if not len(targets):
        raise ValueError("Insufficient LSTM training sequences")
    loader = DataLoader(TensorDataset(torch.tensor(sequences), torch.tensor(targets)),
                        batch_size=cfg.batch_size, shuffle=False)
    model = _new_model(values.shape[1], cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.HuberLoss(delta=.1)
    model.train()
    for _ in range(epochs):
        for inputs, outcomes in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(cfg.device)), outcomes.to(cfg.device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()
    return model, bundle


def _lstm_predict(model, bundle, values: np.ndarray, positions: np.ndarray, cfg: Part2aConfig):
    imputer, scaler, mean, std = bundle
    scaled = scaler.transform(imputer.transform(values))
    inputs = np.array([scaled[i-cfg.sequence_length+1:i+1] for i in positions], dtype=np.float32)
    if inputs.ndim != 3 or inputs.shape[1] != cfg.sequence_length:
        raise ValueError("Incomplete LSTM prediction window")
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), cfg.batch_size):
            batch = torch.tensor(inputs[start:start+cfg.batch_size]).to(cfg.device)
            predictions.extend(model(batch).cpu().numpy().tolist())
    return np.asarray(predictions) * std + mean


def nested_lstm_forecasts(X: pd.DataFrame, y: pd.Series, cfg: Part2aConfig):
    """Choose epoch counts on inner history; score later untouched blocks."""
    values = X[get_feature_cols(X)].replace([np.inf, -np.inf], np.nan).to_numpy()
    actual = pd.to_numeric(y).to_numpy(dtype=float)
    dates = pd.to_datetime(X["week_date"])
    if not dates.diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("LSTM requires consecutive calendar-week observations")
    start, folds = blocked_folds(len(actual), cfg.val_weeks)
    predictions = np.full(len(actual) - start, np.nan)
    fold_ids = np.zeros(len(predictions), dtype=int)
    epoch_counts = []
    for fold_id, positions in enumerate(folds, start=1):
        stop = int(positions[0])
        inner_len = min(cfg.val_weeks, max(8, stop // 5))
        inner_stop = stop - inner_len
        if inner_stop - cfg.sequence_length + 1 < 8:
            raise ValueError("Insufficient inner LSTM history")
        scaled, inner_targets, bundle = _scaled_history(values[:inner_stop], actual[:inner_stop])
        imputer, scaler, mean, std = bundle
        entire = scaler.transform(imputer.transform(values[:stop]))
        target = (actual[:stop] - mean) / std
        sequences, targets = build_sequences(entire, target, cfg.sequence_length)
        cutoff = inner_stop - cfg.sequence_length + 1
        loaders = [DataLoader(TensorDataset(torch.tensor(sequences[a:b]), torch.tensor(targets[a:b])),
                              batch_size=cfg.batch_size, shuffle=False)
                   for a, b in [(0, cutoff), (cutoff, len(targets))]]
        torch.manual_seed(cfg.seed)
        tuned, _, losses = train_lstm(_new_model(values.shape[1], cfg), *loaders, cfg, cfg.device)
        del tuned
        epochs = int(np.argmin(losses)) + 1
        epoch_counts.append(epochs)
        model, bundle = _fixed_lstm_fit(values[:stop], actual[:stop], epochs, cfg)
        predictions[positions - start] = _lstm_predict(model, bundle, values, positions, cfg)
        fold_ids[positions - start] = fold_id
    final_epochs = max(1, int(np.median(epoch_counts)))
    model, bundle = _fixed_lstm_fit(values, actual, final_epochs, cfg)
    oof = pd.DataFrame({"week_date": dates.iloc[start:].to_numpy(), "actual": actual[start:],
                        "pred_persistence": X["gas_us_avg_current"].iloc[start:].to_numpy(),
                        "pred_lstm": predictions, "fold": fold_ids})
    return model, bundle, oof, epoch_counts


def main() -> int:
    cfg = Part2aConfig()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GASPRICE_ROOT", str(root))
    identity = pipeline_identity()
    def unavailable(reason):
        strict_json_dump({"script_version": SCRIPT_VERSION, "run_utc": datetime.now(timezone.utc).isoformat(),
                          "lstm_sleeve_recommended": False, "reason": reason, **identity},
                         out_dir / "gas_part2a_summary.json")
        return 0
    if not HAVE_TORCH:
        return unavailable("torch_not_installed")
    gate, reason = part2b_gate_passed(root / cfg.part2b_dir_name)
    if not gate:
        return unavailable("part2b_gate_not_passed: " + reason)
    part1_dir = root / cfg.part1_dir_name
    validate_feature_lineage(part1_dir)
    X, y = load_features(part1_dir)
    n_lab = int((X["is_live"].eq(0) & y.notna()).sum())
    try:
        model, bundle, oof, epochs = nested_lstm_forecasts(X.iloc[:n_lab], y.iloc[:n_lab], cfg)
    except ValueError as exc:
        return unavailable(str(exc))
    core_oof = pd.read_csv(root / cfg.part2_dir_name / "gas_oof_predictions.csv")
    xgb_oof_path = root / cfg.part2b_dir_name / "gas_xgb_oof.csv"
    comparable = aligned_oof(core_oof, oof)
    xgb_oof = pd.read_csv(xgb_oof_path) if xgb_oof_path.exists() else pd.DataFrame()
    comparable = comparable and aligned_oof(oof, xgb_oof)
    validation = candidate_gate(oof, "pred_lstm")
    lstm_rmse = float(np.sqrt(np.mean((oof.actual - oof.pred_lstm) ** 2)))
    base_rmse, xgb_rmse = load_baseline_rmse(root / cfg.part2_dir_name, root / cfg.part2b_dir_name)
    core = json.loads((root / cfg.part2_dir_name / "gas_part2_summary.json").read_text())
    recommended = bool(comparable and validation["operator_validated"]
                       and core.get("operator_validation", {}).get("operator_validated", False)
                       and base_rmse is not None and xgb_rmse is not None
                       and lstm_rmse < min(base_rmse, xgb_rmse))
    positions = np.arange(cfg.sequence_length - 1, len(X))
    values = X[get_feature_cols(X)].replace([np.inf, -np.inf], np.nan).to_numpy()
    all_predictions = _lstm_predict(model, bundle, values, positions, cfg)
    tape = pd.DataFrame({"week_date": X.week_date.iloc[positions].to_numpy(),
                         "actual": y.iloc[positions].to_numpy(), "pred_lstm": all_predictions,
                         "is_live": X.is_live.iloc[positions].to_numpy()})
    tape_path, oof_path = out_dir / "gas_lstm_tape.parquet", out_dir / "gas_lstm_oof.csv"
    tape.to_parquet(tape_path, index=False)
    tape.to_csv(out_dir / "gas_lstm_tape.csv", index=False)
    oof.to_csv(oof_path, index=False)
    imputer, scaler, mean, std = bundle
    torch.save({"model_state_dict": model.state_dict(), "imputer": imputer, "scaler": scaler,
                "y_mean": mean, "y_std": std, "cfg": asdict(cfg)}, out_dir / "gas_lstm_model.pt")
    anchor = pd.Timestamp(tape.week_date.iloc[-1])
    strict_json_dump({
        "script_version": SCRIPT_VERSION, "run_utc": datetime.now(timezone.utc).isoformat(),
        "lstm_sleeve_recommended": recommended, "lstm_val_rmse": lstm_rmse,
        "baseline_val_rmse": base_rmse, "xgb_val_rmse": xgb_rmse,
        "validation_protocol": NESTED_PROTOCOL, "gate_window": "expanding_origin_blocked_oof",
        "operator_validation": validation, "inner_selected_epochs": epochs,
        "oof_sha256": sha256_file(oof_path), "forecast_tape_sha256": sha256_file(tape_path),
        "feature_matrix_sha256": sha256_file(part1_dir / "gas_feature_matrix.parquet"),
        "latest_forecast": {"anchor_week": str(anchor.date()),
                            "target_week": str((anchor + pd.Timedelta(days=7)).date()),
                            "pred_lstm": float(all_predictions[-1]), "is_true_live_forecast": bool(len(X) > n_lab)},
        **identity,
    }, out_dir / "gas_part2a_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
