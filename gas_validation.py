"""Comparable blocked forecasts and gates for the core and optional sleeves."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast_statistics import compare_squared_errors

NESTED_PROTOCOL = "NESTED_BLOCKED_OOF_V1"


def blocked_folds(n_rows: int, val_weeks: int) -> tuple[int, list[np.ndarray]]:
    if n_rows < 12:
        raise ValueError("At least 12 labeled rows are required")
    val_len = min(val_weeks, max(8, int(n_rows * 0.25)))
    start = n_rows - val_len
    if start < 4:
        raise ValueError("Insufficient pre-validation history")
    return start, list(np.array_split(np.arange(start, n_rows), min(4, val_len)))


def aligned_oof(core: pd.DataFrame, sleeve: pd.DataFrame) -> bool:
    """Require the exact same scored dates, folds, targets and comparator."""
    keys = ["week_date", "actual", "pred_persistence", "fold"]
    if any(key not in frame for frame in (core, sleeve) for key in keys):
        return False
    if len(core) != len(sleeve) or len(core) < 8:
        return False
    for frame in (core, sleeve):
        dates = pd.to_datetime(frame["week_date"], errors="coerce")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            return False
    if not pd.to_datetime(core["week_date"]).reset_index(drop=True).equals(
        pd.to_datetime(sleeve["week_date"]).reset_index(drop=True)
    ):
        return False
    return all(np.allclose(pd.to_numeric(core[key]), pd.to_numeric(sleeve[key]),
                           rtol=0, atol=1e-12, equal_nan=False) for key in keys[1:])


def candidate_gate(oof: pd.DataFrame, column: str) -> dict:
    """Apply the existing RMSE, HAC p<.10 and 75%-positive-fold criteria."""
    required = ["actual", "pred_persistence", "fold", column]
    if any(key not in oof for key in required) or len(oof) < 8:
        return {"operator_validated": False, "reason": "missing_paired_oof"}
    numbers = oof[required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numbers.to_numpy()).all():
        return {"operator_validated": False, "reason": "nonfinite_paired_oof"}
    actual, model, baseline = (numbers[key].to_numpy() for key in ("actual", column, "pred_persistence"))
    test = compare_squared_errors(actual, model, baseline)
    rmse = float(np.sqrt(np.mean((actual - model) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((actual - baseline) ** 2)))
    fold_ids = numbers["fold"].to_numpy()
    folds = np.unique(fold_ids)
    positive = sum(np.mean((actual[fold_ids == fold] - model[fold_ids == fold]) ** 2)
                   < np.mean((actual[fold_ids == fold] - baseline[fold_ids == fold]) ** 2)
                   for fold in folds)
    required_positive = max(1, int(np.ceil(0.75 * len(folds))))
    return {
        "operator_validated": bool(rmse < baseline_rmse and np.isfinite(test["p_value"])
                                   and test["p_value"] < .10 and positive >= required_positive),
        "ensemble_rmse": rmse, "persistence_rmse": baseline_rmse,
        "paired_p_value": test["p_value"], "paired_test_method": test["test_method"],
        "positive_folds": int(positive), "required_positive_folds": required_positive,
        "n_paired": int(test["n"]),
    }


def prequential_fusion(oof: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Choose each block's weights exclusively from earlier scored blocks."""
    result = oof.copy()
    result["pred_fusion"] = np.nan
    for fold in sorted(result["fold"].unique()):
        prior = result.loc[result["fold"] < fold]
        weights = np.ones(len(columns))
        if len(prior):
            errors = np.sqrt(np.mean((prior[columns].to_numpy() - prior["actual"].to_numpy()[:, None]) ** 2, axis=0))
            weights = 1 / np.maximum(errors, 1e-12)
        weights /= weights.sum()
        mask = result["fold"] == fold
        result.loc[mask, "pred_fusion"] = result.loc[mask, columns].to_numpy() @ weights
    return result
