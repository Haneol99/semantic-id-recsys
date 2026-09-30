"""Percentile bootstrap CIs over users."""

import numpy as np

N_RESAMPLES = 1000
SEED = 0


def _resample_idx(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def bootstrap_ci(values, n_resamples: int = N_RESAMPLES, alpha: float = 0.05, seed: int = SEED) -> dict:
    """Mean of per-user values with a (1 - alpha) percentile bootstrap CI."""
    values = np.asarray(values, dtype=np.float64)
    means = values[_resample_idx(len(values), n_resamples, seed)].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_ci(a, b, n_resamples: int = N_RESAMPLES, alpha: float = 0.05, seed: int = SEED) -> dict:
    """CI for mean(a - b), where a and b are per-user values of two runs on the same users (same order)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have the same shape, got {a.shape} and {b.shape}")
    out = bootstrap_ci(a - b, n_resamples, alpha, seed)
    out["frac_resamples_diff_le_0"] = float(
        ((a - b)[_resample_idx(len(a), n_resamples, seed)].mean(axis=1) <= 0).mean()
    )
    return out
