"""Small, deterministic bootstrap utilities for the revision analyses.

The functions in this module resample complete rows.  In the knot analyses a
row is one canonical knot, so all invariant views and outcomes remain coupled.
They are intentionally agnostic about the scientific metric being evaluated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def percentile_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided percentile interval after dropping non-finite draws."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, (alpha, 1.0 - alpha))
    return float(low), float(high)


def bootstrap_rows(
    n_rows: int,
    statistic: Callable[[np.ndarray], float],
    *,
    n_reps: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap a row-index statistic by sampling rows with replacement."""

    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if n_reps < 1:
        raise ValueError("n_reps must be positive")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_reps, dtype=float)
    for replicate in range(n_reps):
        idx = rng.integers(0, n_rows, size=n_rows)
        draws[replicate] = statistic(idx)
    return draws


def summarize_bootstrap(
    observed: float,
    draws: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Summarize bootstrap draws with a percentile CI and bootstrap SE."""

    draws = np.asarray(draws, dtype=float)
    valid = draws[np.isfinite(draws)]
    low, high = percentile_interval(valid, confidence=confidence)
    return {
        "observed": float(observed),
        "bootstrap_mean": float(np.mean(valid)) if len(valid) else np.nan,
        "bootstrap_se": (
            float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
        ),
        "ci_level": float(confidence),
        "ci_low": low,
        "ci_high": high,
        "n_valid_bootstrap": int(len(valid)),
    }


def multinomial_jaccard_bootstrap(
    overlap: int,
    only_a: int,
    only_b: int,
    *,
    n_reps: int = 2_000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap Jaccard from the three categories inside the observed union."""

    counts = np.asarray([overlap, only_a, only_b], dtype=int)
    if np.any(counts < 0):
        raise ValueError("Jaccard category counts must be non-negative")
    union = int(counts.sum())
    if union == 0:
        return np.full(n_reps, np.nan)
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(union, counts / union, size=n_reps)
    return sampled[:, 0] / sampled.sum(axis=1)


def coarsen_ordered_bins(
    bin_codes: np.ndarray,
    target_bins: int,
) -> np.ndarray:
    """Collapse adjacent ordered bin codes into at most ``target_bins`` bins.

    The original conditional scores use ordered, approximately equal-frequency
    quantile bins.  Collapsing adjacent codes therefore gives a deterministic
    coarser quantile partition without needing to reload the raw norm values.
    """

    codes = np.asarray(bin_codes)
    if codes.ndim != 1:
        raise ValueError("bin_codes must be one-dimensional")
    if target_bins < 1:
        raise ValueError("target_bins must be positive")
    unique = np.unique(codes)
    if not len(unique):
        return np.empty(0, dtype=np.int32)
    rank = np.searchsorted(unique, codes)
    n_out = min(int(target_bins), len(unique))
    collapsed = np.floor(rank * n_out / len(unique)).astype(np.int32)
    return np.minimum(collapsed, n_out - 1)