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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

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

SCRIPT_VERSION = "GAS_PART2A_LSTM_V1_CANONICAL"


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
    X_seq[i] = X_arr[i : i+seq_len]
    y_seq[i] = y_arr[i + seq_len]
    """
    n = len(y_arr)
    X_seq = []
    y_seq = []
    for i in range(n - seq_len):
        X_seq.append(X_arr[i : i + seq_len])
        y_seq.append(y_arr[i + seq_len])
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.lr_patience, factor=0.5, verbose=False
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
    return X, y_df["target_gas_price"]


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
    return [c for c in X.columns if c != "week_date"]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = Part2aConfig()
    root = resolve_project_root(cfg)
    out_dir = root / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    part1_dir  = root / cfg.part1_dir_name
    part2_dir  = root / cfg.part2_dir_name
    part2b_dir = root / cfg.part2b_dir_name

    os.environ.setdefault("GASPRICE_ROOT", str(root))
    print(f"[Part2a] ROOT: {root}")
    print(f"[Part2a] Version: {SCRIPT_VERSION}")
    print(f"[Part2a] Device: {cfg.device}\n")

    if not HAVE_TORCH:
        print("[Part2a] PyTorch not available. LSTM sleeve skipped.")
        summary = {
            "script_version": SCRIPT_VERSION,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "lstm_sleeve_recommended": False,
            "reason": "torch_not_installed",
        }
        with open(out_dir / "gas_part2a_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return 0

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load features
    try:
        X, y = load_features(part1_dir)
    except FileNotFoundError as e:
        print(f"[Part2a] FATAL: {e}")
        return 1

    feature_cols = get_feature_cols(X)
    n = len(y)
    print(f"[Part2a] Features: {n} rows x {len(feature_cols)} features")

    # Preprocess: impute then scale
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp   = imputer.fit_transform(X[feature_cols].values)
    X_scaled = scaler.fit_transform(X_imp)

    # Scale target too (for stable LSTM training; rescale predictions back)
    y_arr    = y.values.astype(np.float32)
    y_mean   = float(y_arr.mean())
    y_std    = float(y_arr.std())
    y_scaled = (y_arr - y_mean) / (y_std + 1e-8)

    # Build sequences
    seq_len = cfg.sequence_length
    X_seq, y_seq = build_sequences(X_scaled, y_scaled, seq_len)
    # Corresponding dates for y_seq[i] = y[i + seq_len]
    dates = X["week_date"].values[seq_len:]

    n_seq = len(y_seq)
    train_end = max(cfg.initial_train_weeks - seq_len, int(n_seq * 0.70))
    val_end   = min(train_end + cfg.val_weeks, n_seq)

    X_tr, y_tr   = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]

    print(f"[Part2a] Sequences: total={n_seq} | train={train_end} | val={val_end - train_end}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Build model
    n_features = X_scaled.shape[1]
    model = GasPriceLSTM(
        n_features=n_features,
        hidden_size_1=cfg.lstm_hidden_size_1,
        hidden_size_2=cfg.lstm_hidden_size_2,
        dense_size=cfg.dense_hidden_size,
        dropout=cfg.dropout_rate,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Part2a] LSTM parameters: {n_params:,}")

    # Train
    print(f"\n[Part2a] Training LSTM ({cfg.epochs} epochs max, "
          f"patience={cfg.patience})...")
    model, train_losses, val_losses = train_lstm(model, train_loader, val_loader, cfg, cfg.device)

    # Evaluate on validation set
    model.eval()
    device = cfg.device
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_pred_scaled = model(X_val_t).cpu().numpy()

    # Rescale predictions and actuals
    val_pred  = val_pred_scaled * y_std + y_mean
    val_actuals = y_val * y_std + y_mean

    lstm_rmse = float(np.sqrt(mean_squared_error(val_actuals, val_pred)))
    lstm_mae  = float(mean_absolute_error(val_actuals, val_pred))
    lstm_mape = float(np.mean(np.abs((val_actuals - val_pred) /
                                      np.where(val_actuals != 0, val_actuals, np.nan)))) * 100
    print(f"[Part2a] LSTM val RMSE: {lstm_rmse:.4f} | MAE: {lstm_mae:.4f} | MAPE: {lstm_mape:.2f}%")

    # Gate check
    base_rmse, xgb_rmse = load_baseline_rmse(part2_dir, part2b_dir)
    recommended = True
    if base_rmse is not None:
        recommended = lstm_rmse < base_rmse
        print(f"[Part2a] LSTM vs Base: {lstm_rmse:.4f} < {base_rmse:.4f} -> {recommended}")
    if xgb_rmse is not None:
        recommended = recommended and (lstm_rmse < xgb_rmse)
        print(f"[Part2a] LSTM vs XGB:  {lstm_rmse:.4f} < {xgb_rmse:.4f} -> {recommended}")

    # Full prediction tape
    model.eval()
    all_preds_scaled = []
    with torch.no_grad():
        for i in range(0, len(X_seq), cfg.batch_size):
            Xb = torch.tensor(X_seq[i : i + cfg.batch_size], dtype=torch.float32).to(device)
            preds = model(Xb).cpu().numpy()
            all_preds_scaled.extend(preds.tolist())

    all_preds = np.array(all_preds_scaled) * y_std + y_mean

    tape = pd.DataFrame({
        "week_date": pd.to_datetime(dates),
        "actual": y_arr[seq_len:],
        "pred_lstm": all_preds,
    })

    # Latest week forecast
    latest_pred = float(all_preds[-1])
    latest_week = pd.to_datetime(dates[-1])
    print(f"[Part2a] Latest ({latest_week.date()}) LSTM forecast: ${latest_pred:.3f}/gal")

    # Write artifacts
    tape_path = out_dir / "gas_lstm_tape.parquet"
    tape.to_parquet(tape_path, index=False)
    tape.to_csv(out_dir / "gas_lstm_tape.csv", index=False)
    print(f"[Part2a] LSTM tape -> {tape_path}")

    model_path = out_dir / "gas_lstm_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "y_mean": y_mean,
        "y_std": y_std,
        "sequence_length": seq_len,
        "cfg": cfg,
    }, model_path)
    print(f"[Part2a] LSTM model -> {model_path}")

    summary = {
        "script_version": SCRIPT_VERSION,
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "lstm_sleeve_recommended": bool(recommended),
        "lstm_val_rmse": round(lstm_rmse, 4),
        "lstm_val_mae": round(lstm_mae, 4),
        "lstm_val_mape": round(lstm_mape, 4),
        "baseline_val_rmse": base_rmse,
        "xgb_val_rmse": xgb_rmse,
        "latest_forecast": {
            "week": latest_week.strftime("%Y-%m-%d"),
            "pred_lstm": round(latest_pred, 4),
        },
        "architecture": {
            "sequence_length": seq_len,
            "n_features": n_features,
            "lstm_units": [cfg.lstm_hidden_size_1, cfg.lstm_hidden_size_2],
            "dense_units": cfg.dense_hidden_size,
            "dropout": cfg.dropout_rate,
            "n_parameters": n_params,
        },
        "training": {
            "epochs_run": len(train_losses),
            "final_train_loss": round(train_losses[-1], 6),
            "final_val_loss": round(val_losses[-1], 6),
        },
    }
    summary_path = out_dir / "gas_part2a_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[Part2a] Summary -> {summary_path}")

    status = "RECOMMENDED" if recommended else "NOT_RECOMMENDED"
    print(f"\n[Part2a] LSTM sleeve gate: {status}")
    print("[Part2a] LSTM sleeve complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
