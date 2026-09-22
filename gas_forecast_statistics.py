"""Statistical tests shared by validation and live forecast attribution."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

TEST_METHOD = "DIEBOLD_MARIANO_HLN_HAC"


def _automatic_hac_lags(n_observations: int, horizon: int) -> int:
    """Return a conservative Newey-West bandwidth for weekly loss differences."""
    rule_of_thumb = int(np.floor(4 * (n_observations / 100.0) ** (2.0 / 9.0)))
    return min(n_observations - 1, max(horizon - 1, rule_of_thumb))


def one_sided_hac_mean_test(
    values: np.ndarray,
    *,
    horizon: int = 1,
    max_lag: int | None = None,
    min_observations: int = 8,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Test whether the mean of a serially correlated sequence is positive.

    The long-run variance uses a Bartlett-kernel Newey-West estimator. The
    statistic then receives the Harvey-Leybourne-Newbold small-sample
    correction used for Diebold-Mariano forecast comparisons.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    if max_lag is not None and max_lag < 0:
        raise ValueError("max_lag cannot be negative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    sample = np.asarray(values, dtype=float).reshape(-1)
    sample = sample[np.isfinite(sample)]
    n_observations = int(sample.size)
    result: dict[str, Any] = {
        "test_method": TEST_METHOD,
        "n": n_observations,
        "horizon": int(horizon),
        "alpha": float(alpha),
        "hac_lags": None,
        "mean_loss_advantage": np.nan,
        "statistic": np.nan,
        "p_value": np.nan,
        "interpretation": "INSUFFICIENT_DATA",
    }
    if n_observations < min_observations:
        return result

    mean_advantage = float(np.mean(sample))
    centered = sample - mean_advantage
    lag_count = (
        _automatic_hac_lags(n_observations, horizon)
        if max_lag is None
        else min(int(max_lag), n_observations - 1)
    )

    long_run_variance = float(np.dot(centered, centered) / n_observations)
    for lag in range(1, lag_count + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n_observations
        )
        bartlett_weight = 1.0 - lag / (lag_count + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance

    result["hac_lags"] = lag_count
    result["mean_loss_advantage"] = mean_advantage
    tolerance = np.finfo(float).eps * max(1.0, float(np.var(sample)))
    if not np.isfinite(long_run_variance) or long_run_variance <= tolerance:
        result["interpretation"] = "DEGENERATE_LOSS_DIFFERENTIAL"
        return result

    standard_error = float(np.sqrt(long_run_variance / n_observations))
    raw_statistic = mean_advantage / standard_error
    correction_term = (
        n_observations
        + 1
        - 2 * horizon
        + horizon * (horizon - 1) / n_observations
    ) / n_observations
    if correction_term <= 0:
        result["interpretation"] = "INSUFFICIENT_DATA"
        return result

    statistic = float(raw_statistic * np.sqrt(correction_term))
    p_value = float(stats.t.sf(statistic, df=n_observations - 1))
    if statistic > 0 and p_value < alpha:
        interpretation = "MODEL_SIGNIFICANTLY_BETTER"
    elif statistic < 0:
        interpretation = "MODEL_WORSE_THAN_NAIVE"
    else:
        interpretation = "NOT_SIGNIFICANT"

    result.update(
        {
            "statistic": statistic,
            "p_value": p_value,
            "interpretation": interpretation,
        }
    )
    return result


def compare_squared_errors(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_baseline: np.ndarray,
    *,
    horizon: int = 1,
    max_lag: int | None = None,
    min_observations: int = 8,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Compare model and baseline squared errors with a one-sided DM test.

    Positive loss differences mean the learned model has lower squared error
    than the baseline. Non-finite triplets are removed pairwise.
    """
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    model = np.asarray(y_model, dtype=float).reshape(-1)
    baseline = np.asarray(y_baseline, dtype=float).reshape(-1)
    if not (actual.size == model.size == baseline.size):
        raise ValueError("actual, model, and baseline arrays must have equal length")

    mask = np.isfinite(actual) & np.isfinite(model) & np.isfinite(baseline)
    differential = (actual[mask] - baseline[mask]) ** 2 - (
        actual[mask] - model[mask]
    ) ** 2
    return one_sided_hac_mean_test(
        differential,
        horizon=horizon,
        max_lag=max_lag,
        min_observations=min_observations,
        alpha=alpha,
    )
