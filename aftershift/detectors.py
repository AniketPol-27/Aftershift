"""Deployment monitors: nominal-physics one-step predictor + FA-calibrated CUSUM.

The monitor models the *deployed agent's beliefs*: it predicts the next state with the
NOMINAL context (the physics the system was trained/verified under). Under a dynamics
shift, the residual between predicted and actual next states grows; a CUSUM detector
turns the residual stream into a detection decision.

Calibration: `calibrate_cusum` chooses the threshold h so that the false-alarm rate on
nominal episodes is <= target_fpr (v0.1 heuristic; protocol v1.0 adds ROC curves and
threshold-sensitivity ablations).
"""
from __future__ import annotations

import numpy as np

from .envs import CARTPOLE_NOMINAL


class CartPolePhysicsPredictor:
    """One-step state predictor using a fixed (nominal by default) context."""

    def __init__(self, context: dict | None = None):
        self.ctx = dict(CARTPOLE_NOMINAL)
        if context:
            self.ctx.update(context)

    def predict(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, x_dot, th, th_dot = obs
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        force = a * self.ctx["force_mag"] - self.ctx["cart_friction"] * x_dot
        ct, st = np.cos(th), np.sin(th)
        mc, mp, l, g = self.ctx["masscart"], self.ctx["masspole"], self.ctx["pole_len"], self.ctx["gravity"]
        total = mc + mp
        mpl = mp * l
        temp = (force + mpl * th_dot ** 2 * st) / total
        th_acc = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct ** 2 / total))
        x_acc = temp - mpl * th_acc * ct / total
        return np.array([
            x + 0.02 * x_dot,
            x_dot + 0.02 * x_acc,
            th + 0.02 * th_dot,
            th_dot + 0.02 * th_acc,
        ], dtype=np.float64)


def residual_norm(pred: np.ndarray, actual: np.ndarray) -> float:
    """L2 norm of the one-step prediction residual."""
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(pred)))


class CUSUMDetector:
    """One-sided CUSUM on residual magnitude: s_t = max(0, s_{t-1} + r_t - nu - k)."""

    def __init__(self, nu: float, k: float, h: float, warmup: int = 5):
        self.nu, self.k, self.h, self.warmup = nu, k, h, warmup
        self.reset()

    def reset(self) -> None:
        self.s = 0.0
        self.t = 0

    def update(self, r: float) -> bool:
        """Feed residual; returns True latched once the statistic crosses h."""
        self.t += 1
        if self.t <= self.warmup:
            return False
        self.s = max(0.0, self.s + r - self.nu - self.k)
        return self.s > self.h


def calibrate_cusum(residual_episodes: list[list[float]], target_fpr: float = 0.05):
    """Choose (nu, k, h) from nominal-episode residual streams.

    nu/k are the mean/std of nominal residuals; h is the smallest grid value whose
    episode-level crossing rate on nominal data is <= target_fpr.
    Returns (nu, k, h, achieved_fpr).
    """
    flat = np.concatenate([np.asarray(e) for e in residual_episodes])
    nu = float(np.mean(flat))
    k = float(np.std(flat))
    # episode-max statistic for a grid of thresholds
    max_stats = []
    for eps_res in residual_episodes:
        s, best = 0.0, 0.0
        for i, r in enumerate(eps_res):
            if i + 1 <= 5:  # warmup
                continue
            s = max(0.0, s + r - nu - k)
            best = max(best, s)
        max_stats.append(best)
    max_stats = np.asarray(max_stats)
    grid = np.quantile(max_stats, np.linspace(0.30, 0.999, 60))
    h, fpr = None, None
    for cand in grid:
        fa = float(np.mean(max_stats > cand))
        if fa <= target_fpr:
            h = float(cand)
            fpr = fa
            break
    if h is None:  # fall back to the most conservative grid point
        h = float(grid[-1] * 1.5)
        fpr = float(np.mean(max_stats > h))
    return nu, k, h, fpr
