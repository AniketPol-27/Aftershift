"""Metrics for the AFTERSHIFT protocol (pilot v0.1)."""
from __future__ import annotations

import numpy as np


def cliff_point(values, returns, nominal_return: float, threshold_frac: float = 0.5):
    """First sweep value (ordered away from nominal) with return < threshold * nominal.

    `values` must already be ordered by distance from nominal (callers guarantee this).
    Returns None if performance never crosses the threshold.
    """
    for v, r in zip(values, returns):
        if r < threshold_frac * max(nominal_return, 1e-9):
            return v
    return None


def max_adjacent_drop(returns):
    """Largest drop between adjacent sweep points (returns in sweep order)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    return float(np.max(np.maximum(0.0, r[:-1] - r[1:])))


def i95(budgets, returns, oracle_return: float, frac: float = 0.95):
    """Smallest interaction budget reaching `frac` of oracle return; None if never."""
    for b, r in zip(budgets, returns):
        if r >= frac * oracle_return:
            return int(b)
    return None


def lead_time_stats(leads: list[float]):
    """Median / IQR / count over detection lead times (positive = early warning)."""
    if not leads:
        return dict(n=0, median=None, iqr=None)
    a = np.asarray(leads, dtype=float)
    return dict(
        n=int(len(a)),
        median=float(np.median(a)),
        iqr=float(np.percentile(a, 75) - np.percentile(a, 25)),
    )
